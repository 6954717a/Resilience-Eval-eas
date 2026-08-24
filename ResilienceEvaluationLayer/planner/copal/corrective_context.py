"""
Corrective Context Generator for CoPAL

Generates CoPAL-style corrective prompts that combine:
- Translated error descriptions
- Recent action history
- Recovery hints

These contexts are injected into the LLM prompt to guide corrective planning.
"""

from typing import Any, Dict, List, Optional

from habitat_llm.planner.copal.error_translator import ErrorTranslator
from habitat_llm.planner.copal.history_manager import ActionHistoryManager


# CoPAL Context Template
COPAL_CONTEXT_TEMPLATE = """[Corrective Planning Context]
Prior Action: {failed_action}
Failure: {translated_error}
{history_context}
Recovery Suggestion: {recovery_hint}"""

# Compact version for strict character limits
COPAL_COMPACT_TEMPLATE = """[CoPAL] {failed_action} failed: {error_summary}
{recovery_hint}"""


class CoPALContextBuilder:
    """
    Builds CoPAL-style corrective context for LLM prompt injection.
    
    This class combines ErrorTranslator and HistoryManager to produce
    structured corrective planning context that helps the LLM understand
    what went wrong and how to recover.
    
    Example:
        builder = CoPALContextBuilder()
        context = builder.build(
            failed_action="Pick[cup_0]",
            error_response="Failed: object not close enough",
            info={"skill_success": False}
        )
        # Returns formatted CoPAL context for prompt injection
    """
    
    def __init__(
        self,
        max_context_chars: int = 300,
        include_history: bool = True,
        history_steps: int = 3,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the CoPALContextBuilder.
        
        Args:
            max_context_chars: Maximum characters for context output
            include_history: Whether to include action history
            history_steps: Number of history steps to include
            config: Optional configuration dict
        """
        self.max_context_chars = max_context_chars
        self.include_history = include_history
        self.history_steps = history_steps
        self.config = config or {}
        
        # Initialize components
        self.error_translator = ErrorTranslator(config)
        self.history_manager = ActionHistoryManager(
            max_history=self.config.get("max_history", 10),
            config=config,
        )
    
    def reset(self) -> None:
        """Reset for a new episode."""
        self.history_manager.reset()
    
    def record_step(
        self,
        action: str,
        observation: str,
        success: bool,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record an action step (should be called after each action).
        
        Args:
            action: The action taken
            observation: The observation/response
            success: Whether the action succeeded
            info: Additional info from environment
        """
        self.history_manager.add_step(action, observation, success, info)
    
    def build(
        self,
        failed_action: str,
        error_response: str,
        info: Optional[Dict[str, Any]] = None,
        compact: bool = False,
    ) -> str:
        """
        Build CoPAL corrective context.
        
        Args:
            failed_action: The action that failed
            error_response: Error response from the action
            info: Info dict from environment
            compact: Use compact format for strict limits
            
        Returns:
            Formatted CoPAL context string
        """
        info = info or {}
        
        # Translate the error
        translated_error = self.error_translator.translate(
            failed_action, info, error_response
        )
        
        # Get recovery hint
        recovery_hint = self.error_translator.get_recovery_hint(
            failed_action, error_response
        )
        
        # Get history context if enabled
        history_context = ""
        if self.include_history:
            history_context = self.history_manager.get_copal_history(
                max_chars=100
            )
        
        if compact:
            # Use compact template
            error_summary = translated_error.split(":")[1].strip() if ":" in translated_error else translated_error
            error_summary = error_summary[:60] + "..." if len(error_summary) > 60 else error_summary
            
            context = COPAL_COMPACT_TEMPLATE.format(
                failed_action=self._shorten_action(failed_action),
                error_summary=error_summary,
                recovery_hint=recovery_hint[:80] if recovery_hint else "Verify and retry.",
            )
        else:
            # Use full template
            context = COPAL_CONTEXT_TEMPLATE.format(
                failed_action=failed_action,
                translated_error=translated_error,
                history_context=history_context,
                recovery_hint=recovery_hint or "Analyze the situation and try an alternative approach.",
            )
        
        # Enforce character limit
        if len(context) > self.max_context_chars:
            context = context[:self.max_context_chars - 3] + "..."
        
        return context
    
    def build_from_history(self) -> str:
        """
        Build context from the last failure in history.
        
        Returns:
            CoPAL context if there was a recent failure, empty string otherwise
        """
        last_failure = self.history_manager.get_last_failure()
        if not last_failure:
            return ""
        
        return self.build(
            failed_action=last_failure.action,
            error_response=last_failure.observation,
            info=last_failure.extra_info,
        )
    
    def should_inject_context(self) -> bool:
        """
        Determine if CoPAL context should be injected.
        
        Returns True if:
        - There was a recent failure
        - OR there are repeated failures of the same type
        """
        if not self.history_manager._history:
            return False
        
        last_step = self.history_manager._history[-1]
        
        # Inject if last action failed
        if not last_step.success:
            return True
        
        # Inject if there are repeated failures recently
        if self.history_manager._consecutive_failures >= 2:
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get builder statistics for metrics."""
        return self.history_manager.get_statistics()
    
    def _shorten_action(self, action: str, max_len: int = 25) -> str:
        """Shorten action string for compact output."""
        if len(action) <= max_len:
            return action
        
        # Keep action type and truncate target
        if "[" in action:
            action_type = action.split("[")[0]
            target = action.split("[")[1].rstrip("]")
            if len(target) > 10:
                target = target[:7] + "..."
            return f"{action_type}[{target}]"
        
        return action[:max_len - 3] + "..."


def build_copal_context(
    failed_action: str,
    error_response: str,
    info: Optional[Dict[str, Any]] = None,
    history_summary: str = "",
    recovery_hint: str = "",
    max_chars: int = 300,
) -> str:
    """
    Convenience function to build CoPAL context without instantiating a builder.
    
    Args:
        failed_action: The action that failed
        error_response: Error response text
        info: Optional info dict
        history_summary: Optional pre-built history summary
        recovery_hint: Optional recovery hint (auto-generated if empty)
        max_chars: Maximum output characters
        
    Returns:
        Formatted CoPAL context string
    """
    # Translate error
    translator = ErrorTranslator()
    translated_error = translator.translate(failed_action, info or {}, error_response)
    
    # Get recovery hint if not provided
    if not recovery_hint:
        recovery_hint = translator.get_recovery_hint(failed_action, error_response)
    
    # Build context
    parts = [f"[CoPAL] {failed_action} failed"]
    
    if translated_error:
        error_core = translated_error.split("failed:")[-1].strip() if "failed:" in translated_error else translated_error
        parts.append(f"Reason: {error_core[:80]}")
    
    if history_summary:
        parts.append(f"Context: {history_summary[:60]}")
    
    if recovery_hint:
        parts.append(f"Suggestion: {recovery_hint[:80]}")
    
    result = "\n".join(parts)
    
    if len(result) > max_chars:
        result = result[:max_chars - 3] + "..."
    
    return result


def format_copal_for_prompt(context: str, title: str = "Corrective Planning") -> str:
    """
    Format CoPAL context for prompt injection with proper tags.
    
    Args:
        context: The CoPAL context content
        title: Title for the context section
        
    Returns:
        Formatted context ready for prompt injection
    """
    if not context:
        return ""
    
    return f"[{title}]\n{context}"
