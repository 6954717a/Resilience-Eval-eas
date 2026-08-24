"""
Recovery Metrics for CoPAL

Tracks CoPAL-specific recovery metrics:
- RSR (Recovery Success Rate): Rate of successful recoveries
- RTR (Recovery Token Ratio): Efficiency of recovery in terms of LLM calls
- RSTC (Recovery Steps to Complete): Steps from failure to recovery

These metrics complement the existing Rebound MTTR/MTBF metrics.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time


@dataclass
class RecoveryAttempt:
    """Records a single recovery attempt."""
    start_step: int
    end_step: Optional[int] = None
    success: bool = False
    action_type: str = ""
    error_type: str = ""
    recovery_actions: int = 0
    duration_seconds: float = 0.0
    
    @property
    def steps_to_recover(self) -> int:
        if self.end_step is None:
            return 0
        return self.end_step - self.start_step


class RecoveryMetrics:
    """
    Tracks CoPAL-specific recovery metrics.
    
    Metrics:
    - RSR (Recovery Success Rate): Successful recoveries / Total attempts
    - RTR (Recovery Token Ratio): Recovery tokens / Total tokens
    - RSTC (Recovery Steps to Complete): Average steps from failure to success
    - Recovery Efficiency: RSR * (1 - RTR) - balances success with efficiency
    
    Example:
        metrics = RecoveryMetrics()
        metrics.start_recovery(step=10, action_type="Pick", error_type="too_far")
        # ... recovery actions ...
        metrics.end_recovery(step=13, success=True)
        
        stats = metrics.get_metrics()
        print(f"RSR: {stats['rsr']:.2f}")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize RecoveryMetrics.
        
        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        
        # Tracking
        self._attempts: List[RecoveryAttempt] = []
        self._active_recovery: Optional[RecoveryAttempt] = None
        self._recovery_start_time: Optional[float] = None
        
        # Token tracking (if available)
        self._total_tokens: int = 0
        self._recovery_tokens: int = 0
        
        # Step tracking
        self._total_steps: int = 0
        self._recovery_steps: int = 0
    
    def reset(self) -> None:
        """Reset metrics for a new episode."""
        self._attempts.clear()
        self._active_recovery = None
        self._recovery_start_time = None
        self._total_tokens = 0
        self._recovery_tokens = 0
        self._total_steps = 0
        self._recovery_steps = 0
    
    def start_recovery(
        self,
        step: int,
        action_type: str = "",
        error_type: str = "",
    ) -> None:
        """
        Mark the start of a recovery attempt.
        
        Args:
            step: Current planning step
            action_type: Type of action that failed
            error_type: Type of error that triggered recovery
        """
        if self._active_recovery is not None:
            # Previous recovery didn't complete - mark as failed
            self._active_recovery.end_step = step
            self._active_recovery.success = False
            self._attempts.append(self._active_recovery)
        
        self._active_recovery = RecoveryAttempt(
            start_step=step,
            action_type=action_type,
            error_type=error_type,
        )
        self._recovery_start_time = time.time()
    
    def end_recovery(
        self,
        step: int,
        success: bool,
        recovery_actions: int = 0,
    ) -> None:
        """
        Mark the end of a recovery attempt.
        
        Args:
            step: Current planning step
            success: Whether recovery was successful
            recovery_actions: Number of actions taken during recovery
        """
        if self._active_recovery is None:
            return
        
        self._active_recovery.end_step = step
        self._active_recovery.success = success
        self._active_recovery.recovery_actions = recovery_actions
        
        if self._recovery_start_time:
            self._active_recovery.duration_seconds = time.time() - self._recovery_start_time
        
        self._attempts.append(self._active_recovery)
        
        # Update step tracking
        recovery_steps = self._active_recovery.steps_to_recover
        self._recovery_steps += recovery_steps
        
        self._active_recovery = None
        self._recovery_start_time = None
    
    def record_step(self, is_recovery_step: bool = False) -> None:
        """
        Record a planning step for ratio calculations.
        
        Args:
            is_recovery_step: Whether this step is part of recovery
        """
        self._total_steps += 1
        if is_recovery_step or self._active_recovery is not None:
            self._recovery_steps += 1
    
    def record_tokens(self, tokens: int, is_recovery: bool = False) -> None:
        """
        Record token usage for RTR calculation.
        
        Args:
            tokens: Number of tokens used
            is_recovery: Whether tokens were used for recovery
        """
        self._total_tokens += tokens
        if is_recovery or self._active_recovery is not None:
            self._recovery_tokens += tokens
    
    def is_in_recovery(self) -> bool:
        """Check if currently in recovery mode."""
        return self._active_recovery is not None
    
    def get_rsr(self) -> float:
        """
        Calculate Recovery Success Rate.
        
        RSR = Successful Recoveries / Total Recovery Attempts
        """
        if not self._attempts:
            return 1.0  # No failures = perfect recovery
        
        successful = sum(1 for a in self._attempts if a.success)
        return successful / len(self._attempts)
    
    def get_rtr(self) -> float:
        """
        Calculate Recovery Token Ratio.
        
        RTR = Recovery Tokens / Total Tokens
        Lower is better (less overhead for recovery).
        """
        if self._total_tokens == 0:
            return 0.0
        
        return self._recovery_tokens / self._total_tokens
    
    def get_rstc(self) -> float:
        """
        Calculate Recovery Steps to Complete (average).
        
        RSTC = Average steps from failure to successful recovery.
        """
        successful = [a for a in self._attempts if a.success]
        if not successful:
            return 0.0
        
        return sum(a.steps_to_recover for a in successful) / len(successful)
    
    def get_recovery_efficiency(self) -> float:
        """
        Calculate Recovery Efficiency score.
        
        Efficiency = RSR * (1 - RTR)
        Balances success rate with token efficiency.
        """
        rsr = self.get_rsr()
        rtr = self.get_rtr()
        return rsr * (1 - rtr)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all CoPAL recovery metrics.
        
        Returns:
            Dictionary with all metrics
        """
        return {
            # Core CoPAL metrics
            "copal_rsr": self.get_rsr(),
            "copal_rtr": self.get_rtr(),
            "copal_rstc": self.get_rstc(),
            "copal_efficiency": self.get_recovery_efficiency(),
            
            # Raw counts
            "copal_total_attempts": len(self._attempts),
            "copal_successful_recoveries": sum(1 for a in self._attempts if a.success),
            "copal_failed_recoveries": sum(1 for a in self._attempts if not a.success),
            
            # Step/Token tracking
            "copal_total_steps": self._total_steps,
            "copal_recovery_steps": self._recovery_steps,
            "copal_total_tokens": self._total_tokens,
            "copal_recovery_tokens": self._recovery_tokens,
            
            # Breakdown by error type
            "copal_error_breakdown": self._get_error_breakdown(),
        }
    
    def get_summary(self) -> str:
        """
        Get a text summary of recovery metrics.
        
        Returns:
            Formatted summary string
        """
        metrics = self.get_metrics()
        
        lines = [
            "CoPAL Recovery Metrics:",
            f"  RSR (Recovery Success Rate): {metrics['copal_rsr']:.2%}",
            f"  RTR (Recovery Token Ratio): {metrics['copal_rtr']:.2%}",
            f"  RSTC (Avg Steps to Recover): {metrics['copal_rstc']:.1f}",
            f"  Efficiency Score: {metrics['copal_efficiency']:.2%}",
            f"  Total Attempts: {metrics['copal_total_attempts']}",
            f"  Successful: {metrics['copal_successful_recoveries']}",
            f"  Failed: {metrics['copal_failed_recoveries']}",
        ]
        
        return "\n".join(lines)
    
    def _get_error_breakdown(self) -> Dict[str, Dict[str, int]]:
        """Get breakdown of recovery attempts by error type."""
        breakdown = {}
        
        for attempt in self._attempts:
            error_type = attempt.error_type or "unknown"
            if error_type not in breakdown:
                breakdown[error_type] = {"total": 0, "success": 0, "failed": 0}
            
            breakdown[error_type]["total"] += 1
            if attempt.success:
                breakdown[error_type]["success"] += 1
            else:
                breakdown[error_type]["failed"] += 1
        
        return breakdown
