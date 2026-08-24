"""
Action History Manager for CoPAL

Manages the action-observation history with a sliding window,
providing structured summaries for corrective planning.

CoPAL requires access to recent action history with failure context
to enable effective re-planning.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import deque


@dataclass
class ActionStep:
    """
    Represents a single action-observation step in history.
    """
    step_id: int
    action: str
    observation: str
    success: bool
    action_type: str = ""
    target: str = ""
    timestamp: float = 0.0
    extra_info: Dict[str, Any] = field(default_factory=dict)
    
    def to_summary(self, include_obs: bool = True) -> str:
        """Convert to a compact summary string."""
        status = "✓" if self.success else "✗"
        if include_obs and self.observation:
            obs_short = self.observation[:50] + "..." if len(self.observation) > 50 else self.observation
            return f"[{status}] {self.action} → {obs_short}"
        return f"[{status}] {self.action}"


class ActionHistoryManager:
    """
    Manages action-observation history for CoPAL corrective planning.
    
    Features:
    - Sliding window of recent actions
    - Failure pattern detection
    - Structured history summaries for prompts
    
    Example:
        manager = ActionHistoryManager(max_history=5)
        manager.add_step("Navigate[kitchen]", "Arrived at kitchen", success=True)
        manager.add_step("Pick[cup_0]", "Failed: too far", success=False)
        
        context = manager.get_recent_context(n=3)
        # Returns formatted recent history for prompt injection
    """
    
    def __init__(
        self,
        max_history: int = 10,
        failure_window: int = 5,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the ActionHistoryManager.
        
        Args:
            max_history: Maximum number of steps to keep in history
            failure_window: Window size for failure pattern detection
            config: Optional configuration dict
        """
        self.max_history = max_history
        self.failure_window = failure_window
        self.config = config or {}
        
        self._history: deque[ActionStep] = deque(maxlen=max_history)
        self._step_counter = 0
        self._failure_count = 0
        self._consecutive_failures = 0
        self._last_action_type = ""
    
    def reset(self) -> None:
        """Reset history for a new episode."""
        self._history.clear()
        self._step_counter = 0
        self._failure_count = 0
        self._consecutive_failures = 0
        self._last_action_type = ""
    
    def add_step(
        self,
        action: str,
        observation: str,
        success: bool,
        extra_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a new action-observation step to history.
        
        Args:
            action: The action taken (e.g., "Pick[cup_0]")
            observation: The observation/response received
            success: Whether the action succeeded
            extra_info: Additional info to store
        """
        self._step_counter += 1
        
        # Extract action type and target
        action_type, target = self._parse_action(action)
        
        step = ActionStep(
            step_id=self._step_counter,
            action=action,
            observation=observation,
            success=success,
            action_type=action_type,
            target=target,
            extra_info=extra_info or {},
        )
        
        self._history.append(step)
        
        # Update failure tracking
        if not success:
            self._failure_count += 1
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0
        
        self._last_action_type = action_type
    
    def get_recent_context(self, n: int = 3, include_obs: bool = True) -> str:
        """
        Get formatted recent history for prompt injection.
        
        Args:
            n: Number of recent steps to include
            include_obs: Whether to include observations
            
        Returns:
            Formatted string of recent action history
        """
        if not self._history:
            return "No previous actions."
        
        recent = list(self._history)[-n:]
        lines = ["Recent Actions:"]
        
        for step in recent:
            lines.append(f"  {step.to_summary(include_obs)}")
        
        return "\n".join(lines)
    
    def get_failure_summary(self) -> str:
        """
        Get a summary of recent failures for error analysis.
        
        Returns:
            Formatted string summarizing failure patterns
        """
        recent = list(self._history)[-self.failure_window:]
        failures = [s for s in recent if not s.success]
        
        if not failures:
            return ""
        
        # Analyze failure patterns
        failure_types = {}
        for f in failures:
            key = f.action_type
            if key not in failure_types:
                failure_types[key] = []
            failure_types[key].append(f)
        
        parts = [f"Recent Failures ({len(failures)}/{len(recent)} steps):"]
        
        for action_type, steps in failure_types.items():
            if len(steps) > 1:
                parts.append(f"  - {action_type}: {len(steps)} failures (repeated)")
            else:
                parts.append(f"  - {steps[0].action}: {steps[0].observation[:40]}...")
        
        return "\n".join(parts)
    
    def get_copal_history(self, max_chars: int = 200) -> str:
        """
        Get CoPAL-formatted history context for corrective planning.
        
        This format emphasizes recent failures and action sequences
        to help the LLM understand what went wrong and why.
        
        Args:
            max_chars: Maximum characters for the output
            
        Returns:
            CoPAL-formatted history string
        """
        if not self._history:
            return ""
        
        parts = []
        
        # Add recent action sequence (compact)
        recent = list(self._history)[-5:]
        action_sequence = " → ".join(
            f"{s.action_type}{'✓' if s.success else '✗'}" 
            for s in recent
        )
        parts.append(f"Sequence: {action_sequence}")
        
        # Add failure context if any
        if self._consecutive_failures > 0:
            last_failure = self._history[-1]
            parts.append(f"Last Failure: {last_failure.action}")
            if last_failure.observation:
                obs_short = last_failure.observation[:60]
                parts.append(f"Error: {obs_short}")
        
        # Add pattern warning if repeated failures
        if self._consecutive_failures >= 2:
            parts.append(f"⚠ {self._consecutive_failures} consecutive failures detected")
        
        result = "\n".join(parts)
        
        # Enforce character limit
        if len(result) > max_chars:
            result = result[:max_chars - 3] + "..."
        
        return result
    
    def has_repeated_failure(self, action_type: str = "") -> bool:
        """
        Check if the same action type has failed repeatedly.
        
        Args:
            action_type: Optional specific action type to check
            
        Returns:
            True if repeated failures detected
        """
        if not action_type:
            return self._consecutive_failures >= 2
        
        recent = list(self._history)[-3:]
        failures = [s for s in recent if not s.success and s.action_type == action_type]
        return len(failures) >= 2
    
    def get_last_failure(self) -> Optional[ActionStep]:
        """Get the most recent failed action, if any."""
        for step in reversed(self._history):
            if not step.success:
                return step
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get history statistics for metrics."""
        total = len(self._history)
        failures = sum(1 for s in self._history if not s.success)
        
        return {
            "total_steps": total,
            "failure_count": failures,
            "success_rate": (total - failures) / total if total > 0 else 1.0,
            "consecutive_failures": self._consecutive_failures,
            "history_size": len(self._history),
        }
    
    def _parse_action(self, action: str) -> tuple:
        """Parse action string into type and target."""
        if "[" in action and "]" in action:
            action_type = action.split("[")[0]
            target = action.split("[")[1].rstrip("]")
            return action_type, target
        return action, ""


# Global instance for convenience (reset per episode)
_default_manager: Optional[ActionHistoryManager] = None


def get_history_manager() -> ActionHistoryManager:
    """Get or create the default history manager."""
    global _default_manager
    if _default_manager is None:
        _default_manager = ActionHistoryManager()
    return _default_manager


def reset_history_manager() -> None:
    """Reset the default history manager."""
    global _default_manager
    if _default_manager is not None:
        _default_manager.reset()
