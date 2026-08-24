"""
SayCan Planner Integration

Provides integration functions to connect SayCan with LLMPlanner.
Keeps SayCan logic separate from LLMPlanner core.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from habitat_llm.planner.llm_planner import LLMPlanner
    from habitat_llm.planner.saycan.saycan_planner import SayCanPlanner


def apply_saycan_selection(
    planner: "LLMPlanner", candidates: Optional[List[Dict[str, Any]]] = None
) -> Optional[str]:
    """
    Apply SayCan selection logic if planner is a SayCanPlanner.
    
    Args:
        planner: LLMPlanner instance (may be SayCanPlanner)
        candidates: Optional pre-generated candidates (if None, will be generated)
        
    Returns:
        Selected action string, or None if not a SayCanPlanner
    """
    if not isinstance(planner, SayCanPlanner):
        return None
    
    if not planner.saycan_enabled:
        return None
    
    # SayCan selection is handled in generate_action_response()
    # This function is for future extensibility
    return None


def inject_saycan_context(planner: "LLMPlanner", guidance: str) -> None:
    """
    Inject SayCan guidance into planner prompt.
    
    Args:
        planner: LLMPlanner instance
        guidance: Guidance text to inject
    """
    if not guidance:
        return
    
    # Use planner's context update mechanism
    if hasattr(planner, 'enqueue_context_update'):
        planner.enqueue_context_update("SayCan Guidance", guidance)
    elif hasattr(planner, 'use_prompt_builder') and planner.use_prompt_builder:
        if hasattr(planner, 'prompt_builder') and planner.prompt_builder:
            planner.prompt_builder.add_user_turn(guidance, title="SayCan Guidance")
    else:
        # Fallback: append to current prompt
        user_tag = planner.planner_config.llm.user_tag
        eot_tag = planner.planner_config.llm.eot_tag
        planner.curr_prompt += f"\n{user_tag}[SayCan Guidance]\n{guidance}\n{eot_tag}"
