"""
CoPAL Planner Integration Module

Provides integration functions for CoPAL (Corrective Planning) with
the existing planner and evaluation framework.

This module connects CoPAL components to the evaluation runner
without modifying core architecture.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from habitat_llm.planner.llm_planner import LLMPlanner



def initialize_copal(planner: "LLMPlanner") -> None:
    """
    Initialize CoPAL components for the planner.
    
    Args:
        planner: The LLMPlanner instance
    """
    copal_config = planner.copal_config
    
    from habitat_llm.planner.copal import (
        CoPALContextBuilder,
        ErrorTranslator,
    )

    # Initialize Context Builder
    planner.copal_context_builder = CoPALContextBuilder(
        max_context_chars=copal_config.get("max_context_chars", 300),
        include_history=copal_config.get("include_history_in_context", True),
        history_steps=copal_config.get("context_history_steps", 3),
        config=copal_config,
    )
    
    # Initialize Error Translator
    planner.error_translator = ErrorTranslator(copal_config)
    
    # Initialize Recovery Metrics
    try:
        from habitat_llm.evaluation.copal.recovery_metrics import RecoveryMetrics
        planner.copal_recovery_metrics = RecoveryMetrics(copal_config)
    except ImportError:
        planner.copal_recovery_metrics = None


def process_copal_step(
    planner: "LLMPlanner",
    responses: Dict[int, str],
    is_replanning_step: bool = True
) -> None:
    """
    Process a simulation step for CoPAL: record history and update metrics.
    
    This function should be called at the end of a planning step,
    before generating the next prompt.
    
    Args:
        planner: The LLMPlanner instance
        responses: Dictionary of agent responses
        is_replanning_step: Whether this is a replanning step (vs simulation loop)
    """
    if not getattr(planner, "use_copal", False):
        return
        
    builder = getattr(planner, "copal_context_builder", None)
    if not builder:
        return
        
    # Only record at Planning Steps (replanning steps), not Simulation Steps
    # unless configured otherwise. This aligns with offline_analyzer.py.
    if not is_replanning_step:
        return

    # Process each agent's response
    for agent_uid, response_text in responses.items():
        if not response_text:
            continue
            
        # Get last action for this agent
        last_action = getattr(planner, "last_high_level_actions", {}).get(agent_uid)
        
        # Format action string
        action_str = ""
        if last_action:
            action_str = f"{last_action[0]}[{last_action[1]}]" if last_action[1] else last_action[0]
            
        # Determine success/failure
        # Simple heuristic: look for failure keywords
        is_success = not any(
            kw in response_text.lower() 
            for kw in ["fail", "error", "cannot", "unable"]
        )
        
        # 1. Record step in History Manager
        # Also extracts error type if failed
        builder.record_step(
            action=action_str,
            observation=response_text,
            success=is_success,
            info={"agent_uid": agent_uid}
        )
        
        # 2. Update Recovery Metrics
        metrics = getattr(planner, "copal_recovery_metrics", None)
        if metrics:
            # Record step count
            metrics.record_step(is_recovery_step=metrics.is_in_recovery())
            
            # Handle failure/success logic
            if not is_success:
                if not metrics.is_in_recovery():
                    # Start new recovery attempt
                    error_type = "unknown" # Could parse from translator
                    metrics.start_recovery(
                        step=getattr(planner, "replanning_count", 0),
                        action_type=last_action[0] if last_action else "Unknown",
                        error_type=error_type
                    )
            elif metrics.is_in_recovery():
                # End successful recovery
                metrics.end_recovery(
                    step=getattr(planner, "replanning_count", 0),
                    success=True
                )


def get_copal_statistics(planner: "LLMPlanner") -> Dict[str, Any]:
    """
    Get CoPAL statistics from the planner.
    
    Args:
        planner: The LLMPlanner instance
        
    Returns:
        Dictionary with CoPAL statistics
    """
    if not getattr(planner, "use_copal", False):
        return {}
    
    builder = getattr(planner, "copal_context_builder", None)
    if not builder:
        return {}
    
    stats = builder.get_statistics()
    
    result = {
        "copal_enabled": True,
        "copal_history_size": stats.get("history_size", 0),
        "copal_total_steps": stats.get("total_steps", 0),
        "copal_failure_count": stats.get("failure_count", 0),
        "copal_success_rate": stats.get("success_rate", 1.0),
        "copal_consecutive_failures": stats.get("consecutive_failures", 0),
    }
    
    # Add recovery metrics if available
    recovery_metrics = getattr(planner, "copal_recovery_metrics", None)
    if recovery_metrics:
        rm_stats = recovery_metrics.get_metrics()
        result.update({
            "copal_rsr": rm_stats.get("copal_rsr", 1.0),
            "copal_rtr": rm_stats.get("copal_rtr", 0.0),
            "copal_rstc": rm_stats.get("copal_rstc", 0.0),
            "copal_efficiency": rm_stats.get("copal_efficiency", 0.0),
        })
        
    return result


def log_copal_analysis(
    info: Dict[str, Any],
    planner: "LLMPlanner",
    output_dir: str,
    episode_filename: str,
) -> Optional[str]:
    """
    Log CoPAL analysis data to a file.
    
    Args:
        info: Info dict from episode
        planner: The LLMPlanner instance
        output_dir: Directory to save output
        episode_filename: Filename for the episode
        
    Returns:
        Path to saved file, or None if not saved
    """
    import json
    import os
    
    if not getattr(planner, "use_copal", False):
        return None
    
    builder = getattr(planner, "copal_context_builder", None)
    if not builder:
        return None
    
    # Get statistics
    stats = builder.get_statistics()
    
    # Build analysis data
    analysis_data = {
        "episode_id": info.get("episode_id", "unknown"),
        "episode_filename": episode_filename,
        "copal_statistics": stats,
        "task_percent_complete": info.get("task_percent_complete", 0.0),
        "task_state_success": info.get("task_state_success", False),
    }
    
    # Get history summary if available
    if hasattr(builder.history_manager, "_history"):
        history_items = []
        for step in list(builder.history_manager._history)[-10:]:
            history_items.append({
                "step_id": step.step_id,
                "action": step.action,
                "success": step.success,
                "observation": step.observation[:100] if step.observation else "",
            })
        analysis_data["recent_history"] = history_items
    
    # Save to file
    try:
        copal_dir = os.path.join(output_dir, "analyses", "copal")
        os.makedirs(copal_dir, exist_ok=True)
        
        filepath = os.path.join(
            copal_dir,
            f"copal_analysis-{episode_filename}.json"
        )
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        return filepath
        
    except Exception as e:
        print(f"[CoPAL] Failed to save analysis: {e}")
        return None


def inject_copal_context_if_needed(
    planner: "LLMPlanner",
    responses: Dict[int, str],
) -> Optional[str]:
    """
    Check if CoPAL context should be injected based on current responses.
    
    This function is called from the planner to determine if corrective
    context should be added to the prompt.
    
    Args:
        planner: The LLMPlanner instance
        responses: Dictionary of agent responses
        
    Returns:
        CoPAL context string if injection is needed, None otherwise
    """
    if not getattr(planner, "use_copal", False):
        return None
    
    builder = getattr(planner, "copal_context_builder", None)
    if not builder:
        return None
    
    # Check if any response indicates failure
    has_failure = False
    failed_agent_id = None
    failed_response = ""
    
    for agent_id, response in responses.items():
        if response and any(
            kw in response.lower() 
            for kw in ["fail", "error", "cannot", "unable"]
        ):
            has_failure = True
            failed_agent_id = agent_id
            failed_response = response
            break
    
    if not has_failure:
        return None
    
    # Check if context should be injected
    if not builder.should_inject_context():
        return None
    
    # Get the failed action
    failed_action = ""
    if failed_agent_id is not None and hasattr(planner, "last_high_level_actions"):
        action_tuple = planner.last_high_level_actions.get(failed_agent_id)
        if action_tuple:
            failed_action = f"{action_tuple[0]}[{action_tuple[1]}]" if action_tuple[1] else action_tuple[0]
    
    # Build corrective context
    copal_config = getattr(planner, "copal_config", {})
    context = builder.build(
        failed_action=failed_action,
        error_response=failed_response,
        info={"agent_uid": failed_agent_id},
        compact=copal_config.get("use_compact_format", False),
    )
    
    return context


def format_copal_metrics_for_log(planner: "LLMPlanner") -> str:
    """
    Format CoPAL metrics for logging output.
    
    Args:
        planner: The LLMPlanner instance
        
    Returns:
        Formatted string of CoPAL metrics
    """
    stats = get_copal_statistics(planner)
    
    if not stats.get("copal_enabled", False):
        return ""
    
    lines = [
        "CoPAL Metrics:",
        f"  Steps Tracked: {stats.get('copal_total_steps', 0)}",
        f"  Failures: {stats.get('copal_failure_count', 0)}",
        f"  Success Rate: {stats.get('copal_success_rate', 1.0):.2%}",
        f"  Consecutive Failures: {stats.get('copal_consecutive_failures', 0)}",
    ]
    
    return "\n".join(lines)


def generate_and_inject_copal_context(
    planner: "LLMPlanner",
    responses: Dict[int, str]
) -> None:
    """
    Generate CoPAL context and inject it into the planner's prompt/trace.
    
    Args:
        planner: The LLMPlanner instance
        responses: Dictionary of agent responses
    """
    if not getattr(planner, "use_copal", False):
        return

    # Use existing helper to get context string
    context = inject_copal_context_if_needed(planner, responses)
    
    if not context:
        return
        
    # Inject into prompt
    # Note: CoPAL context is typically injected before the LLM generates the next action
    # We append it to curr_prompt and trace
    
    copal_block = f"\n{context}\n"
    planner.curr_prompt += copal_block
    planner.trace += copal_block

