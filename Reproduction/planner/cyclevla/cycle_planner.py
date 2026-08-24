"""
CycleVLA Planning Pipeline.

This module implements the main CycleVLA planning pipeline that integrates:
- SubtaskManager: Task decomposition and progress tracking
- CycleVLMPredictor: VLM-based failure prediction at boundaries
- MBRActionSampler: Consensus-based action selection after backtrack
- CognitiveRollbackTool: Context-level backtracking

Reference: CycleVLA Paper (arxiv.org/abs/2601.02295)
The paper introduces proactive self-correction for VLAs via:
1. Progress-aware subtask execution
2. VLM-based failure prediction at subtask boundaries
3. Backtracking to earlier subtasks when failure is predicted
4. MBR decoding for robust retry after backtracking

This implementation adapts these concepts to LLM-based planning.
"""

import logging
import traceback
from typing import Any, Dict, Optional, TYPE_CHECKING

from habitat_llm.evaluation.cyclevla import SubtaskManager
from .vlm_predictor import CycleVLMPredictor
from .mbr_sampler import MBRActionSampler
from .rollback import CognitiveRollbackTool

if TYPE_CHECKING:
    from habitat_llm.planner.llm_planner import LLMPlanner

logger = logging.getLogger(__name__)


def run_cycle_pipeline(
    planner: "LLMPlanner",
    instruction: str,
    env_interface: Any,
    llm_config: Dict[str, Any],
) -> str:
    """
    CycleVLA-style planning pipeline.
    
    This is the main entry point for CycleVLA planning, called by
    LLMPlanner.replan() when use_cycle_vla=True.
    
    Flow:
    1. Initialize CycleVLA components (first call only)
    2. Get current subtask and save checkpoint if at subtask start
    3. Estimate progress within current subtask
    4. If at boundary (progress >= threshold):
       a. Call VLM predictor for transit/backtrack decision
       b. If backtrack: cognitive rollback + MBR sample
       c. If transit: mark subtask complete, advance to next
    5. Inject subtask guidance and generate LLM response
    
    Args:
        planner: LLMPlanner instance
        instruction: High-level task instruction
        env_interface: EnvironmentInterface for state access
        llm_config: LLM configuration dictionary
    
    Returns:
        LLM response string containing actions
        
    Reference: CycleVLA Algorithm 1 (Inference Procedure)
    """
    # Get CycleVLA config from planner
    # planner_config is OmegaConf DictConfig, use .get() method
    config = planner.planner_config.get("cyclevla", {})
    
    # Convert OmegaConf to dict if needed
    from omegaconf import OmegaConf, DictConfig
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    elif not isinstance(config, dict):
        config = {}
    
    # === 1. Initialize CycleVLA components (lazy initialization) ===
    if not hasattr(planner, "_cycle_initialized") or not planner._cycle_initialized:
        _initialize_cycle_components(planner, config, env_interface, instruction)
    
    # Get component references
    subtask_mgr: SubtaskManager = planner._cycle_subtask_mgr
    rollback_tool: CognitiveRollbackTool = planner._cycle_rollback
    vlm_predictor: CycleVLMPredictor = planner._cycle_vlm
    mbr_sampler: MBRActionSampler = planner._cycle_mbr
    
    # === 2. Check if all subtasks completed ===
    current_subtask = subtask_mgr.get_current_subtask()
    if current_subtask is None:
        # All subtasks done
        logger.info("CycleVLA: All subtasks completed")
        return _generate_done_response(planner)
    
    # === 3. Save checkpoint at subtask start ===
    if planner._cycle_is_subtask_start:
        world_state = _get_world_state(planner, env_interface)
        rollback_tool.save_checkpoint(
            subtask_mgr.current_idx,
            world_state,
            subtask_mgr.step_count
        )
        planner._cycle_is_subtask_start = False
        logger.debug(f"CycleVLA: Saved checkpoint for subtask {subtask_mgr.current_idx}")
    
    # === 4. Estimate progress ===
    world_state = _get_world_state(planner, env_interface)
    action_history = _get_action_history(env_interface, planner)
    progress = subtask_mgr.estimate_progress(current_subtask, action_history, world_state)
    
    # Advance step counter
    subtask_mgr.advance_step()
    
    logger.debug(
        f"CycleVLA: Subtask {current_subtask.idx} ({current_subtask.instruction}), "
        f"progress={progress:.2f}"
    )
    
    # === 5. Check subtask boundary ===
    threshold = config.get("progress_threshold", 0.8)
    if subtask_mgr.is_boundary(progress, threshold):
        logger.info(
            f"CycleVLA: At boundary for subtask {current_subtask.idx}, "
            f"progress={progress:.2f} >= {threshold}"
        )
        
        # Call VLM predictor
        observations = env_interface.get_observations() if hasattr(env_interface, 'get_observations') else {}
        decision, target_idx = vlm_predictor.predict_and_plan(
            observations=observations,
            current_subtask=current_subtask,
            subtask_list=subtask_mgr.subtasks,
            progress=progress,
            world_state=world_state,
            action_history=action_history
        )
        
        if decision == "backtrack" and target_idx is not None:
            # === BACKTRACK PATH ===
            logger.info(f"CycleVLA: Backtracking from {current_subtask.idx} to {target_idx}")
            
            # Get target subtask instruction for guidance
            target_instruction = ""
            if 0 <= target_idx < len(subtask_mgr.subtasks):
                target_instruction = subtask_mgr.subtasks[target_idx].instruction
            
            # Perform cognitive rollback
            success = rollback_tool.rollback_to(
                subtask_idx=target_idx,
                from_subtask_idx=current_subtask.idx,
                reason="VLM predicted potential failure",
                confidence="medium",
                subtask_instruction=target_instruction
            )
            
            if success:
                # Update subtask manager
                subtask_mgr.rollback_to(target_idx)
                rollback_tool.clear_checkpoints_after(target_idx)
                planner._cycle_is_subtask_start = True
                
                # Use MBR sampling for robust retry
                try:
                    response, selected_actions = mbr_sampler.sample_and_select(
                        instruction=instruction,
                        observations=observations,
                        world_graph=env_interface.world_graph.get(planner.agents[0].uid) if hasattr(env_interface, 'world_graph') else None,
                        current_subtask_instruction=target_instruction
                    )
                    
                    if response:
                        logger.info(f"CycleVLA: MBR selected response with {len(selected_actions)} actions")
                        return response
                except Exception as e:
                    logger.warning(f"CycleVLA: MBR sampling failed: {e}")
                    # Fall through to normal generation
            else:
                logger.warning(f"CycleVLA: Rollback to {target_idx} failed, continuing")
        
        elif decision == "transit":
            # === TRANSIT PATH ===
            logger.info(f"CycleVLA: Transit from subtask {current_subtask.idx} to next")
            subtask_mgr.mark_complete(current_subtask.idx)
            planner._cycle_is_subtask_start = True
            
            # Check if there are more subtasks
            next_subtask = subtask_mgr.get_current_subtask()
            if next_subtask is None:
                return _generate_done_response(planner)
            
            # Update current_subtask for guidance injection
            current_subtask = next_subtask
    
    # === 6. Inject subtask guidance and generate response ===
    # Check for action loops
    loop_warning = _detect_action_loop(action_history)
    
    guidance = _build_subtask_guidance(current_subtask, progress, subtask_mgr, world_state, loop_warning)
    planner.enqueue_context_update("CycleVLA Guidance", guidance)
    
    try:
        response = planner.generate_action_response()
        return response
    except Exception as e:
        # Check for connection errors specifically
        error_msg = str(e).lower()
        if "connection" in error_msg or "refused" in error_msg or "timeout" in error_msg:
            logger.error(f"CycleVLA: LLM server connection failed: {e}")
            logger.warning("CycleVLA: Falling back to Wait action due to server unavailability")
        else:
            logger.error(f"CycleVLA: Generation failed: {e}")
            traceback.print_exc()
        return _generate_wait_response(planner)


def _initialize_cycle_components(
    planner: "LLMPlanner",
    config: Dict[str, Any],
    env_interface: Any,
    instruction: str
):
    """Initialize CycleVLA components for a new episode."""
    logger.info("CycleVLA: Initializing components")
    
    # Create SubtaskManager
    planner._cycle_subtask_mgr = SubtaskManager(
        llm_client=planner.llm,
        config=config
    )
    
    # Decompose task into subtasks
    world_state = _get_world_state(planner, env_interface)
    available_tools = planner.tool_list if hasattr(planner, 'tool_list') else None
    
    subtasks = planner._cycle_subtask_mgr.decompose_task(
        instruction=instruction,
        world_state=world_state,
        available_tools=available_tools
    )
    
    logger.info(f"CycleVLA: Decomposed task into {len(subtasks)} subtasks")
    for st in subtasks:
        logger.debug(f"  [{st.idx}] {st.instruction}")
    
    # Create other components
    planner._cycle_rollback = CognitiveRollbackTool(planner)
    planner._cycle_vlm = CycleVLMPredictor(planner.llm, config)
    planner._cycle_mbr = MBRActionSampler(planner, config)
    
    # State flags
    planner._cycle_is_subtask_start = True
    planner._cycle_initialized = True


def _get_world_state(planner: "LLMPlanner", env_interface: Any) -> Dict[str, Any]:
    """Extract world state from planner's perception connector."""
    try:
        if hasattr(planner, 'perception_connector'):
            return planner.perception_connector.extract_world_state(env_interface)
    except Exception as e:
        logger.warning(f"Failed to extract world state: {e}")
    
    return {}


def _get_action_history(env_interface: Any, planner: "LLMPlanner") -> list:
    """Get action history from environment interface."""
    try:
        if hasattr(env_interface, 'agent_action_history'):
            agent_id = planner.agents[0].uid if planner.agents else 0
            return env_interface.agent_action_history.get(agent_id, [])
    except Exception as e:
        logger.warning(f"Failed to get action history: {e}")
    
    return []


def _detect_action_loop(action_history: list) -> Optional[str]:
    """Detect if the agent is repeating the same action."""
    if not action_history or len(action_history) < 4:
        return None
        
    # Get last 4 actions
    recent = action_history[-4:]
    
    # Check if all are identical (using string representation for loose comparison)
    # Action history items are typically tuples or objects
    recent_strs = [str(a) for a in recent]
    
    if all(s == recent_strs[0] for s in recent_strs):
        return "You have executed the same action multiple times recently. You might be stuck. Consider checking if the action is succeeding, or try a different approach (e.g. Move to a different location, or skipping to the next step)."
    
    return None


def _build_subtask_guidance(
    subtask: Any,
    progress: float,
    subtask_mgr: SubtaskManager,
    world_state: Dict[str, Any],
    loop_warning: Optional[str] = None
) -> str:
    """Build guidance text for current subtask."""
    lines = [
        f"Suggested Phase ({subtask.idx + 1}/{len(subtask_mgr.subtasks)}): {subtask.instruction}",
        f"Estimated Progress: {progress:.0%}",
    ]
    
    # Add world state context to help agent realize if postconditions are met
    if world_state:
        # Check holding
        held = world_state.get("agent_holding")
        if held:
             lines.append(f"Current Status: Holding {held}")
        
        # Check if we are in the right room if instruction contains room name
        current_room = world_state.get("current_room") if hasattr(world_state, "get") else None
        if current_room:
             lines.append(f"Current Location: {current_room}")

    if subtask.postconditions:
        lines.append(f"Phase Goal: {', '.join(subtask.postconditions)}")
    
    # Add adaptive hints
    if progress < 0.2:
        lines.append("Hint: Start by identifying necessary objects or locations.")
    elif progress > 0.7:
        lines.append("Hint: Phase seems nearly complete. Verify goals and decide if you should proceed.")
    
    if loop_warning:
        lines.append(f"\nWARNING: {loop_warning}")
        
    # Add disclaimer to encourage flexibility
    lines.append("\nNote: This phase is ONLY A SUGGESTION. Do NOT get stuck here. If you are repeating actions or making no progress, you MUST deviate or SKIP this phase. Focus on the main task goal.")
    
    return "\n".join(lines)


def _generate_done_response(planner: "LLMPlanner") -> str:
    """Generate a Done response when all subtasks complete."""
    assistant_tag = planner.planner_config.llm.assistant_tag
    
    # Build Done actions for all agents
    action_lines = []
    for agent in planner.agents:
        action_lines.append(f"Agent_{agent.uid}_Action: Done[]")
    
    return (
        f"{assistant_tag}Thought: All subtasks completed successfully. "
        "The task is done.\n" +
        "\n".join(action_lines) +
        "\nAssigned!"
    )


def _generate_wait_response(planner: "LLMPlanner") -> str:
    """Generate a Wait response on error."""
    assistant_tag = planner.planner_config.llm.assistant_tag
    
    action_lines = []
    for agent in planner.agents:
        action_lines.append(f"Agent_{agent.uid}_Action: Wait[]")
    
    return (
        f"{assistant_tag}Thought: Encountered an issue. Waiting to reassess.\n" +
        "\n".join(action_lines) +
        "\nAssigned!"
    )


def reset_cycle_state(planner: "LLMPlanner"):
    """
    Reset CycleVLA state for a new episode.
    
    Call this at the start of each new episode to clear
    any cached state from previous runs.
    """
    if hasattr(planner, '_cycle_initialized'):
        planner._cycle_initialized = False
    
    if hasattr(planner, '_cycle_subtask_mgr'):
        planner._cycle_subtask_mgr.reset()
    
    if hasattr(planner, '_cycle_rollback'):
        planner._cycle_rollback.reset()
    
    if hasattr(planner, '_cycle_vlm'):
        planner._cycle_vlm.reset_statistics()
    
    if hasattr(planner, '_cycle_mbr'):
        planner._cycle_mbr.reset_statistics()
    
    planner._cycle_is_subtask_start = True


def get_cycle_statistics(planner: "LLMPlanner") -> Dict[str, Any]:
    """
    Get CycleVLA statistics for logging/analysis.
    
    Returns aggregated statistics from all CycleVLA components.
    """
    stats = {"cycle_enabled": hasattr(planner, '_cycle_initialized') and planner._cycle_initialized}
    
    if hasattr(planner, '_cycle_subtask_mgr'):
        mgr = planner._cycle_subtask_mgr
        stats["subtask_count"] = len(mgr.subtasks)
        stats["current_subtask_idx"] = mgr.current_idx
        stats["step_count"] = mgr.step_count
        stats["subtask_summary"] = mgr.get_subtask_summary()
    
    if hasattr(planner, '_cycle_rollback'):
        stats.update(planner._cycle_rollback.get_statistics())
    
    if hasattr(planner, '_cycle_vlm'):
        vlm_stats = planner._cycle_vlm.get_statistics()
        stats["vlm_predictions"] = vlm_stats["prediction_count"]
        stats["vlm_backtracks"] = vlm_stats["backtrack_count"]
        stats["vlm_backtrack_rate"] = vlm_stats["backtrack_rate"]
    
    if hasattr(planner, '_cycle_mbr'):
        mbr_stats = planner._cycle_mbr.get_statistics()
        stats["mbr_sample_count"] = mbr_stats["sample_count"]
        stats["mbr_consensus_dist"] = mbr_stats["avg_consensus_distance"]
    
    return stats
