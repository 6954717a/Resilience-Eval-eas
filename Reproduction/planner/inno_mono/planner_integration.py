"""
Planner Integration for Inner Monologue

Provides integration functions to connect Inner Monologue with LLMPlanner.
"""

from typing import TYPE_CHECKING, Dict, Any, Optional, Tuple

if TYPE_CHECKING:
    from habitat_llm.planner.llm_planner import LLMPlanner


def generate_and_inject_feedback(
    planner: "LLMPlanner",
    responses: Dict[int, str],
    world_state: Dict[str, Any],
    critic_feedback: Optional[Dict[str, Any]] = None,
    rebound_feedback: Optional[Dict[str, Any]] = None
) -> None:
    """
    Generate Inner Monologue feedback and inject it into the planner's prompt.
    
    This is the main integration function that:
    1. Gets or creates FeedbackGenerator from planner
    2. Generates structured feedback
    3. Formats feedback as text
    4. Injects feedback into PromptBuilder or curr_prompt
    
    Args:
        planner: LLMPlanner instance
        responses: Dictionary mapping agent_id to response string
        world_state: World state dictionary
        critic_feedback: Optional feedback from Critic
        rebound_feedback: Optional feedback from Rebound
    """
    if not hasattr(planner, 'inner_monologue_enabled') or not planner.inner_monologue_enabled:
        return
    
    if not hasattr(planner, 'feedback_generator') or planner.feedback_generator is None:
        return
    
    # Generate feedback
    feedback = planner.feedback_generator.generate_feedback(
        agent_responses=responses,
        last_actions=planner.last_high_level_actions,
        world_state=world_state,
        critic_feedback=critic_feedback,
        rebound_feedback=rebound_feedback
    )
    
    # Format as text
    feedback_text = planner.feedback_generator.format_feedback_as_text(feedback)
    
    if not feedback_text or feedback_text.strip() == "":
        return
    
    # Inject into prompt
    if planner.use_prompt_builder and planner.prompt_builder:
        # Use PromptBuilder (recommended)
        planner.prompt_builder.add_user_turn(feedback_text, title="Feedback")
        planner._refresh_prompt()
    else:
        # Direct string concatenation (fallback)
        user_tag = planner.planner_config.llm.user_tag
        eot_tag = planner.planner_config.llm.eot_tag
        feedback_block = f"{user_tag}Feedback:\n{feedback_text}\n{eot_tag}"
        planner.curr_prompt += feedback_block
        planner.trace += f"\n[Feedback]\n{feedback_text}\n"


def ensure_thought_generation(planner: "LLMPlanner") -> None:
    """
    Ensure that Thought generation prompt is added after feedback.
    
    This function ensures that after Inner Monologue feedback is injected,
    the prompt ends with "Thought:" to force LLM to generate reasoning.
    
    Args:
        planner: LLMPlanner instance
    """
    if not hasattr(planner, 'inner_monologue_enabled') or not planner.inner_monologue_enabled:
        return
    
    assistant_tag = planner.planner_config.llm.assistant_tag
    thought_prompt = f"{assistant_tag}Thought:"
    
    if planner.use_prompt_builder and planner.prompt_builder:
        # PromptBuilder will handle assistant turn in build_prompt()
        # But we need to ensure the prompt ends with Thought: for next generation
        # This is handled in generate_action_response() or _refresh_prompt()
        pass
    else:
        # Ensure curr_prompt ends with Thought:
        if not planner.curr_prompt.rstrip().endswith("Thought:"):
            # Remove any trailing assistant tag first
            if planner.curr_prompt.rstrip().endswith(assistant_tag.rstrip()):
                planner.curr_prompt = planner.curr_prompt.rstrip()[:-len(assistant_tag.rstrip())].rstrip()
            planner.curr_prompt += f"\n{thought_prompt}"
        
        # Also update trace
        if not planner.trace.rstrip().endswith("Thought:"):
            planner.trace += "\nThought:"
