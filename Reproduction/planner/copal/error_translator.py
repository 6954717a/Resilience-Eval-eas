"""
Error Translator for CoPAL

Translates technical error codes and simulator feedback into
natural language descriptions that LLMs can understand and act upon.

This is a core component of the CoPAL (Corrective Planning) framework.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


class ErrorTranslator:
    """
    Translates technical errors from Habitat simulator into natural language.
    
    CoPAL requires that failure feedback be expressed in natural language
    so the LLM can reason about recovery strategies.
    
    Example:
        translator = ErrorTranslator()
        error_msg = translator.translate(
            action="Pick[cup_0]",
            info={"skill_done": True, "skill_success": False},
            response="Failed: object not close enough"
        )
        # Returns: "Pick[cup_0] failed: object is too far away. Navigate closer first."
    """
    
    # Error pattern to natural language mapping
    ERROR_PATTERNS = {
        # Distance/proximity errors
        r"not close enough|too far|out of reach|distance": {
            "template": "object is too far away (distance exceeded threshold)",
            "hint": "Navigate closer to the target first.",
        },
        # Occlusion/visibility errors
        r"occluded|blocked|obstructed|not visible|cannot see": {
            "template": "object is blocked or not visible",
            "hint": "Navigate to a different position for clear line of sight.",
        },
        # Object state errors
        r"already held|already holding|hands full": {
            "template": "agent is already holding an object",
            "hint": "Place the current object before picking another.",
        },
        r"not holding|empty hands|nothing to place": {
            "template": "agent is not holding any object",
            "hint": "Pick up an object first before placing.",
        },
        # Target not found
        r"not found|unknown|cannot find|does not exist": {
            "template": "target object or location not found",
            "hint": "Explore the area or check object name spelling.",
        },
        # Collision errors
        r"collision|collided|hit|crashed": {
            "template": "collision detected during movement",
            "hint": "Try a different path or navigate around the obstacle.",
        },
        # Navigation errors
        r"unreachable|no path|path blocked|cannot navigate": {
            "template": "destination is unreachable",
            "hint": "Choose an alternative destination or clear the path.",
        },
        # Action precondition failures
        r"precondition|invalid state|cannot perform": {
            "template": "action preconditions not met",
            "hint": "Verify the current state before retrying.",
        },
        # Receptacle errors
        r"receptacle full|no space|cannot place on": {
            "template": "target receptacle has no available space",
            "hint": "Choose a different receptacle or clear space first.",
        },
    }
    
    # Action-specific recovery hints
    ACTION_HINTS = {
        "Pick": "Ensure you are close to the object and it is visible.",
        "Place": "Navigate to the target receptacle before placing.",
        "Navigate": "Check if the destination exists and is accessible.",
        "Open": "Ensure the container is within reach and not already open.",
        "Close": "Ensure the container is within reach and not already closed.",
        "Rearrange": "Consider using Navigate + Pick + Place sequence instead.",
        "Explore": "Try exploring different rooms systematically.",
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ErrorTranslator.
        
        Args:
            config: Optional configuration dict with custom patterns
        """
        self.config = config or {}
        self.max_error_chars = self.config.get("max_error_chars", 150)
        self.include_hints = self.config.get("include_hints", True)
        
        # Compile regex patterns for efficiency
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), mapping)
            for pattern, mapping in self.ERROR_PATTERNS.items()
        ]
    
    def translate(
        self,
        action: str,
        info: Dict[str, Any],
        response: str = "",
    ) -> str:
        """
        Translate an action failure into natural language.
        
        Args:
            action: The action that failed (e.g., "Pick[cup_0]")
            info: Info dict from environment step
            response: Text response from skill execution
            
        Returns:
            Natural language description of the failure with recovery hint
        """
        # Extract action name
        action_name = self._extract_action_name(action)
        
        # Determine if this is a failure
        is_failure = self._is_failure(info, response)
        if not is_failure:
            return ""
        
        # Find matching error pattern
        error_desc, hint = self._match_error_pattern(response)
        
        # Fallback if no pattern matched
        if not error_desc:
            error_desc = self._extract_error_summary(response)
        
        # Get action-specific hint if available
        if not hint and action_name in self.ACTION_HINTS:
            hint = self.ACTION_HINTS[action_name]
        
        # Build translated message
        parts = [f"{action} failed: {error_desc}"]
        if self.include_hints and hint:
            parts.append(f"Recovery: {hint}")
        
        result = " ".join(parts)
        
        # Enforce character limit
        if len(result) > self.max_error_chars:
            result = result[:self.max_error_chars - 3] + "..."
        
        return result
    
    def get_recovery_hint(self, action: str, error_type: str = "") -> str:
        """
        Get a recovery hint for a specific action type.
        
        Args:
            action: Action name or full action string
            error_type: Optional error type for more specific hints
            
        Returns:
            Recovery hint string
        """
        action_name = self._extract_action_name(action)
        
        # Check for error-type specific hint first
        if error_type:
            for pattern, mapping in self._compiled_patterns:
                if pattern.search(error_type):
                    return mapping.get("hint", "")
        
        # Fall back to action-specific hint
        return self.ACTION_HINTS.get(action_name, "Verify preconditions before retrying.")
    
    def translate_batch(
        self,
        failures: List[Tuple[str, Dict[str, Any], str]]
    ) -> List[str]:
        """
        Translate multiple failures at once.
        
        Args:
            failures: List of (action, info, response) tuples
            
        Returns:
            List of translated error messages
        """
        return [self.translate(action, info, response) 
                for action, info, response in failures]
    
    def _extract_action_name(self, action: str) -> str:
        """Extract action name from action string like 'Pick[cup_0]'."""
        if "[" in action:
            return action.split("[")[0]
        return action
    
    def _is_failure(self, info: Dict[str, Any], response: str) -> bool:
        """Determine if the action was a failure."""
        # Check common failure indicators in info
        if info.get("skill_success") is False:
            return True
        if info.get("success") is False:
            return True
        
        # Check for failure keywords in response
        failure_keywords = ["failed", "fail", "error", "cannot", "unable", "not"]
        response_lower = response.lower()
        return any(keyword in response_lower for keyword in failure_keywords)
    
    def _match_error_pattern(self, response: str) -> Tuple[str, str]:
        """Match response against known error patterns."""
        for pattern, mapping in self._compiled_patterns:
            if pattern.search(response):
                return mapping.get("template", ""), mapping.get("hint", "")
        return "", ""
    
    def _extract_error_summary(self, response: str, max_len: int = 60) -> str:
        """Extract a summary from the response when no pattern matches."""
        if not response:
            return "action failed for unknown reason"
        
        # Clean up the response
        response = response.strip()
        
        # Try to extract the core error message
        # Often format is "Failed: <reason>" or "Error: <reason>"
        for prefix in ["Failed:", "Error:", "Cannot:", "Unable to:"]:
            if prefix.lower() in response.lower():
                idx = response.lower().index(prefix.lower())
                response = response[idx + len(prefix):].strip()
                break
        
        # Truncate if too long
        if len(response) > max_len:
            response = response[:max_len - 3] + "..."
        
        return response if response else "action failed"


def translate_error(action: str, info: Dict[str, Any], response: str = "") -> str:
    """
    Convenience function for one-off error translation.
    
    Args:
        action: The action that failed
        info: Info dict from environment
        response: Response text
        
    Returns:
        Translated error message
    """
    translator = ErrorTranslator()
    return translator.translate(action, info, response)
