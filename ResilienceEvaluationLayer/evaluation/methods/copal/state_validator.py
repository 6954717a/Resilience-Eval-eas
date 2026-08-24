"""
State Validator for CoPAL

Validates action execution results and detects implicit failures
that may not be caught by standard skill success flags.

This enables proactive correction before failures compound.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ValidationResult:
    """Result of state validation."""
    is_valid: bool
    failure_type: str = ""
    confidence: float = 1.0
    details: str = ""
    recovery_needed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "failure_type": self.failure_type,
            "confidence": self.confidence,
            "details": self.details,
            "recovery_needed": self.recovery_needed,
        }


class StateValidator:
    """
    Validates action execution results for CoPAL.
    
    Detects:
    - Explicit failures (skill_success = False)
    - Implicit failures (state didn't change as expected)
    - State drift (gradual deviation from expected state)
    
    Example:
        validator = StateValidator()
        result = validator.validate_action_result(
            action="Pick[cup_0]",
            pre_state={"holding": None},
            post_state={"holding": "cup_0"},
            info={"skill_success": True}
        )
    """
    
    # Expected state changes for each action type
    EXPECTED_CHANGES = {
        "Pick": {
            "holding": "should_change_to_target",
        },
        "Place": {
            "holding": "should_become_none",
        },
        "Navigate": {
            "position": "should_change",
        },
        "Open": {
            "target_state": "should_become_open",
        },
        "Close": {
            "target_state": "should_become_closed",
        },
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the StateValidator.
        
        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        self.drift_threshold = self.config.get("drift_threshold", 0.3)
        self.validation_history: List[ValidationResult] = []
    
    def reset(self) -> None:
        """Reset validation state for new episode."""
        self.validation_history.clear()
    
    def validate_action_result(
        self,
        action: str,
        pre_state: Optional[Dict[str, Any]] = None,
        post_state: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None,
        response: str = "",
    ) -> ValidationResult:
        """
        Validate an action's execution result.
        
        Args:
            action: The action that was executed
            pre_state: State before action (optional)
            post_state: State after action (optional)
            info: Info dict from environment
            response: Text response from action
            
        Returns:
            ValidationResult with validation details
        """
        info = info or {}
        
        # Check explicit failure first
        if self._is_explicit_failure(info, response):
            result = ValidationResult(
                is_valid=False,
                failure_type="explicit_failure",
                confidence=1.0,
                details=self._extract_failure_reason(response),
                recovery_needed=True,
            )
            self.validation_history.append(result)
            return result
        
        # Check implicit failure (state didn't change as expected)
        if pre_state and post_state:
            action_type = self._extract_action_type(action)
            implicit_check = self._check_implicit_failure(
                action_type, pre_state, post_state, action
            )
            if not implicit_check[0]:
                result = ValidationResult(
                    is_valid=False,
                    failure_type="implicit_failure",
                    confidence=implicit_check[2],
                    details=implicit_check[1],
                    recovery_needed=True,
                )
                self.validation_history.append(result)
                return result
        
        # Action succeeded
        result = ValidationResult(
            is_valid=True,
            failure_type="",
            confidence=1.0,
            details="Action completed successfully",
            recovery_needed=False,
        )
        self.validation_history.append(result)
        return result
    
    def detect_drift(
        self,
        expected_state: Dict[str, Any],
        actual_state: Dict[str, Any],
    ) -> float:
        """
        Detect state drift between expected and actual states.
        
        Args:
            expected_state: The expected state
            actual_state: The actual observed state
            
        Returns:
            Drift score (0.0 = no drift, 1.0 = complete deviation)
        """
        if not expected_state or not actual_state:
            return 0.0
        
        # Compare key state elements
        differences = 0
        total_keys = 0
        
        for key in expected_state:
            total_keys += 1
            if key not in actual_state:
                differences += 1
            elif expected_state[key] != actual_state[key]:
                differences += 0.5  # Partial difference
        
        if total_keys == 0:
            return 0.0
        
        return differences / total_keys
    
    def should_trigger_recovery(self) -> bool:
        """
        Check if recovery should be triggered based on validation history.
        
        Returns True if:
        - Last validation failed
        - Multiple recent validations showed implicit failures
        """
        if not self.validation_history:
            return False
        
        # Check last validation
        if self.validation_history[-1].recovery_needed:
            return True
        
        # Check for pattern of implicit failures
        recent = self.validation_history[-3:] if len(self.validation_history) >= 3 else self.validation_history
        implicit_failures = sum(
            1 for r in recent 
            if not r.is_valid and r.failure_type == "implicit_failure"
        )
        
        return implicit_failures >= 2
    
    def get_failure_summary(self) -> str:
        """Get a summary of recent validation failures."""
        recent_failures = [
            r for r in self.validation_history[-5:]
            if not r.is_valid
        ]
        
        if not recent_failures:
            return ""
        
        parts = [f"Recent Validation Issues ({len(recent_failures)}):"]
        for failure in recent_failures[-3:]:
            parts.append(f"  - {failure.failure_type}: {failure.details[:50]}")
        
        return "\n".join(parts)
    
    def _is_explicit_failure(self, info: Dict[str, Any], response: str) -> bool:
        """Check for explicit failure indicators."""
        # Check info flags
        if info.get("skill_success") is False:
            return True
        if info.get("success") is False:
            return True
        
        # Check response keywords
        failure_keywords = ["failed", "fail", "error", "cannot", "unable"]
        response_lower = response.lower()
        return any(kw in response_lower for kw in failure_keywords)
    
    def _check_implicit_failure(
        self,
        action_type: str,
        pre_state: Dict[str, Any],
        post_state: Dict[str, Any],
        action: str,
    ) -> Tuple[bool, str, float]:
        """
        Check for implicit failure (state didn't change as expected).
        
        Returns:
            (is_valid, failure_reason, confidence)
        """
        expectations = self.EXPECTED_CHANGES.get(action_type, {})
        
        for state_key, expected_change in expectations.items():
            pre_value = pre_state.get(state_key)
            post_value = post_state.get(state_key)
            
            if expected_change == "should_change":
                if pre_value == post_value:
                    return (False, f"{state_key} did not change as expected", 0.8)
            
            elif expected_change == "should_change_to_target":
                target = self._extract_target(action)
                if post_value != target:
                    return (False, f"{state_key} should be {target}, got {post_value}", 0.9)
            
            elif expected_change == "should_become_none":
                if post_value is not None:
                    return (False, f"{state_key} should be empty but is {post_value}", 0.9)
        
        return (True, "", 1.0)
    
    def _extract_action_type(self, action: str) -> str:
        """Extract action type from action string."""
        if "[" in action:
            return action.split("[")[0]
        return action
    
    def _extract_target(self, action: str) -> str:
        """Extract target from action string."""
        if "[" in action and "]" in action:
            return action.split("[")[1].rstrip("]").split(",")[0].strip()
        return ""
    
    def _extract_failure_reason(self, response: str) -> str:
        """Extract failure reason from response."""
        if not response:
            return "Unknown failure"
        
        # Try to extract after "Failed:" or similar
        for prefix in ["Failed:", "Error:", "Cannot:"]:
            if prefix.lower() in response.lower():
                idx = response.lower().index(prefix.lower())
                return response[idx:].strip()[:80]
        
        return response[:80] if response else "Action failed"
