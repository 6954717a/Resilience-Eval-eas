"""
Phase-based Replan Pipeline.
Executes the replan logic using structured phases but standard LLM generation.
"""
from typing import Any, Dict, List, Optional
import traceback
import logging

logger = logging.getLogger(__name__)

def run_phase_based_replan(
    planner: Any, # LLMPlanner
    instruction: str,
    env_interface: Any,
    llm_config: Dict[str, Any],
) -> str:
    """
    Execute the phase-based replanning process.
    
    Args:
        planner: The calling LLMPlanner instance.
        instruction: The high-level instruction.
        env_interface: Environment interface.
        llm_config: LLM configuration.
        
    Returns:
        The generated LLM response string.
    """
    
    # 1. Update/Extract World State
    world_state = planner.perception_connector.extract_world_state(env_interface)
    
    # 2. Ensure Phase and Get Current Phase
    current_phase = planner.phase_planner.ensure_current_phase(
        instruction, env_interface, world_state, llm_config
    )
    
    if not current_phase:
        # No phase implies done or failure. Check if we really are done.
        # If phase manager says complete, we can probably return a "Stop" or let LLM decide based on empty phase?
        # Usually ensure_current_phase returns None if all phases done.
        return f"{planner.planner_config.llm.assistant_tag}Thought: All phases appear complete. I should verify if the task is done.\nAgent_0_Action: Done[]\nAgent_1_Action: Done[]\nAssigned!"

    # 3. Increment phase step count
    phase_manager = planner.perception_connector.phase_manager
    phase_manager.increment_phase_step()
    current_phase_steps = phase_manager.get_current_phase_step_count()
    
    # 4. Get action history for loop detection
    action_history = _get_action_history(env_interface, planner)
    loop_warning = _detect_action_loop(action_history)
    
    # 5. Check for forced phase transition (based on step count)
    # Get max steps per phase from config, default to 15
    max_steps_per_phase = llm_config.get("max_steps_per_phase", 15)
    force_advance = False
    
    if current_phase_steps >= max_steps_per_phase:
        logger.info(
            f"[Pipeline] Phase {current_phase.get('phase_id', 0)} has reached "
            f"{current_phase_steps} steps (threshold: {max_steps_per_phase}). "
            "Forcing advance to next phase."
        )
        force_advance = True
    
    # 6. Check Phase Completion Logic (Advance if needed)
    # Check if we should advance based on *previous* responses or forced transition
    should_advance = force_advance
    
    if not should_advance and hasattr(planner, 'last_responses') and planner.last_responses:
        if planner.perception_connector.is_current_phase_complete(planner.last_responses):
            logger.info("[Pipeline] Phase complete detected. Advancing...")
            should_advance = True
    
    if should_advance:
        if planner.perception_connector.advance_to_next_phase():
            # Recursively get next phase
            current_phase = planner.phase_planner.ensure_current_phase(
                instruction, env_interface, world_state, llm_config
            )
            if not current_phase:
                return f"{planner.planner_config.llm.assistant_tag}Thought: Task execution finished.\nAgent_0_Action: Done[]\nAgent_1_Action: Done[]\nAssigned!"
            # Reset step count for new phase
            current_phase_steps = phase_manager.get_current_phase_step_count()

    # 7. Construct Guidance for Phase Context (as SUGGESTION, not requirement)
    
    # Extract tasks for prompt
    tasks = current_phase.get('tasks', [])
    phase_id = current_phase.get('phase_id', 0)
    total_phases = len(phase_manager.task_execution_phases)
    
    guidance = _build_phase_guidance(
        current_phase, 
        tasks, 
        phase_id, 
        total_phases, 
        current_phase_steps,
        max_steps_per_phase,
        phase_manager.completed_tasks,
        world_state,
        loop_warning
    )

    # 8. Generate Response using Planner's existing logic
    # Inject guidance as an ephemeral context block instead of mutating prompt history.
    planner.enqueue_context_update("Phase Guidance", guidance)

    # Call planner's generation wrapper
    try:
        # We assume planner has generate_action_response as per plan
        response = planner.generate_action_response(prompt_override=None)
        return response
    except Exception as e:
        # Check for connection errors specifically
        error_msg = str(e).lower()
        if "connection" in error_msg or "refused" in error_msg or "timeout" in error_msg:
            logger.error(f"[Pipeline] LLM server connection failed: {e}")
            logger.warning("[Pipeline] Falling back to Wait action due to server unavailability")
        else:
            logger.error(f"[Pipeline] Error generating LLM response: {e}")
            traceback.print_exc()
        return f"{planner.planner_config.llm.assistant_tag}Thought: Error in generation. Call: Wait()"


def _get_action_history(env_interface: Any, planner: Any) -> list:
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
    recent_strs = [str(a) for a in recent]
    
    if all(s == recent_strs[0] for s in recent_strs):
        return "You have executed the same action multiple times recently. You might be stuck. Consider checking if the action is succeeding, or try a different approach (e.g. Move to a different location, or skipping to the next step)."
    
    return None


def _build_phase_guidance(
    current_phase: Dict[str, Any],
    tasks: List[Dict[str, Any]],
    phase_id: int,
    total_phases: int,
    current_steps: int,
    max_steps: int,
    completed_tasks: List[str],
    world_state: Dict[str, Any],
    loop_warning: Optional[str] = None
) -> str:
    """
    Build guidance text for current phase.
    This guidance is a SUGGESTION, not a requirement - agents can deviate if needed.
    """
    lines = [
        f"Suggested Phase ({phase_id + 1}/{total_phases}): {current_phase.get('description', '')}",
        f"Steps in this phase: {current_steps}/{max_steps}",
    ]
    
    if tasks:
        lines.append("\nSuggested Subtasks (these are GUIDELINES, not strict requirements):")
        for t in tasks:
            status = "(Pending)"
            if t['task_id'] in completed_tasks:
                status = "(Done)"
            lines.append(f"- {t.get('task_type')} -> {t.get('target')}: {t.get('description')} {status}")
    else:
        lines.append("No specific subtasks suggested for this phase.")
    
    # Add world state context if available
    if world_state:
        held = world_state.get("agent_holding")
        if held:
            lines.append(f"\nCurrent Status: Holding {held}")
        
        current_room = world_state.get("current_room")
        if current_room:
            lines.append(f"Current Location: {current_room}")
    
    # Add progress warning if approaching step limit
    if current_steps >= max_steps * 0.8:
        lines.append(f"\nWARNING: This phase has been active for {current_steps} steps. "
                    f"If you are not making progress, consider moving to the next phase or "
                    f"trying a different approach.")
    
    # Add loop warning if detected
    if loop_warning:
        lines.append(f"\n{loop_warning}")
    
    # CRITICAL: Add disclaimer that this is only a suggestion
    lines.append(
        "\n" + "="*60 +
        "\nIMPORTANT: This phase is ONLY A SUGGESTION, NOT A REQUIREMENT."
        "\n- You are allowed to deviate from this phase if needed"
        "\n- If you are stuck or making no progress, you MUST skip or modify this phase"
        "\n- Focus on the MAIN TASK GOAL, not strictly following these phases"
        "\n- If actions are repeating or failing, try a different approach"
        "\n- You can proceed to the next phase even if current phase tasks are incomplete"
        "\n" + "="*60
    )
    
    return "\n".join(lines)
