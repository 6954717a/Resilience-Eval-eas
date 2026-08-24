"""
Stability Monitor for SayCan

Monitors SayCan stability and filters low-score actions to prevent execution
of actions with low confidence (both Say and Can).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .candidate_scorer import ScoredCandidate

logger = logging.getLogger(__name__)


class StabilityMonitor:
    """
    Monitors SayCan stability and triggers fallback actions when scores are too low.
    
    This implements the stability check from SayCan: if the best candidate has
    a total score below the threshold, trigger exploration or wait instead of
    executing a low-confidence action.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize StabilityMonitor.
        
        Args:
            config: Configuration dictionary with:
                - stability_threshold: Minimum total score to accept (default: 0.2)
                - fallback_action: Action to take when threshold violated ("Explore" or "Wait")
                - track_variance: Whether to track score variance (default: True)
        """
        self.config = config or {}
        self.stability_threshold = self.config.get("stability_threshold", 0.2)
        self.fallback_action = self.config.get("fallback_action", "Explore")
        self.track_variance = self.config.get("track_variance", True)
        
        # Track score history for variance calculation
        self.score_history: List[float] = []

    def check_stability(
        self, best_candidate: ScoredCandidate
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Check if the best candidate meets stability threshold.
        
        Args:
            best_candidate: The highest-scoring candidate from CandidateScorer
            
        Returns:
            Tuple of (is_stable, action_to_take, reason):
                - is_stable: True if score >= threshold, False otherwise
                - action_to_take: Action string to execute
                - reason: Optional reason string for logging
        """
        if best_candidate.total_score >= self.stability_threshold:
            # Stable: execute the best candidate
            return True, best_candidate.action, None
        
        # Unstable: trigger fallback
        reason = (
            f"Best action '{best_candidate.action}' has low score "
            f"({best_candidate.total_score:.2f} < {self.stability_threshold:.2f}). "
            f"Say: {best_candidate.say_score:.2f}, Can: {best_candidate.can_score:.2f}"
        )
        
        logger.warning(f"SayCan Stability: {reason}. Triggering {self.fallback_action}.")
        
        return False, self.fallback_action, reason

    def track_score(self, total_score: float) -> None:
        """
        Track a score for variance calculation.
        
        Args:
            total_score: Total score from a planning step
        """
        if self.track_variance:
            self.score_history.append(total_score)
            # Keep only recent history (last 100 steps)
            if len(self.score_history) > 100:
                self.score_history = self.score_history[-100:]

    def compute_score_variance(self, window: Optional[int] = None) -> float:
        """
        Compute variance of total scores over a sliding window.
        
        Args:
            window: Number of recent steps to consider. If None, uses all history.
            
        Returns:
            Variance of scores (0.0 if insufficient data)
        """
        if not self.score_history:
            return 0.0
        
        scores_to_use = self.score_history
        if window is not None and len(self.score_history) > window:
            scores_to_use = self.score_history[-window:]
        
        if len(scores_to_use) < 2:
            return 0.0
        
        import numpy as np
        return float(np.var(scores_to_use))

    def get_stability_metrics(self) -> Dict[str, Any]:
        """
        Get stability metrics.
        
        Returns:
            Dictionary with:
                - threshold_violations: Number of times threshold was violated
                - mean_score: Mean total score
                - score_variance: Variance of scores
                - recent_variance: Variance of last 10 steps
        """
        if not self.score_history:
            return {
                "threshold_violations": 0,
                "mean_score": 0.0,
                "score_variance": 0.0,
                "recent_variance": 0.0,
            }
        
        import numpy as np
        
        violations = sum(1 for score in self.score_history if score < self.stability_threshold)
        mean_score = float(np.mean(self.score_history))
        variance = float(np.var(self.score_history))
        recent_variance = self.compute_score_variance(window=10)
        
        return {
            "threshold_violations": violations,
            "mean_score": mean_score,
            "score_variance": variance,
            "recent_variance": recent_variance,
        }

    def reset(self) -> None:
        """Reset score history for a new episode."""
        self.score_history = []
