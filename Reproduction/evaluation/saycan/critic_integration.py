"""
Critic Integration for SayCan

Integrates A2C Critic with SayCan to compare Affordance scores with learned value functions.
Provides analysis of consistency between heuristic Affordance and learned Critic values.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class CriticIntegration:
    """
    Integrates A2C Critic with SayCan for comparison and enhancement.
    
    Compares heuristic Affordance scores with learned Critic value functions
    to assess consistency and potentially enhance Affordance predictions.
    """

    def __init__(self, critic: Any, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CriticIntegration.
        
        Args:
            critic: A2CCritic instance
            config: Optional configuration dictionary
        """
        self.critic = critic
        self.config = config or {}
        self.comparison_history: List[Dict[str, Any]] = []

    def compare_affordance_vs_critic(
        self,
        affordance_score: float,
        critic_value: float,
        action: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare Affordance score with Critic value for an action.
        
        Args:
            affordance_score: Affordance (Can) score from AffordanceModel
            critic_value: Value function V(s) or Q(s,a) from Critic
            action: Action dictionary
            state: World state dictionary
            
        Returns:
            Dictionary with comparison results:
                - affordance_score: Original affordance score
                - critic_value: Critic value
                - difference: Absolute difference
                - correlation: Normalized correlation (if history available)
                - agreement: Whether both agree on action quality (both high or both low)
        """
        # Normalize both to [0, 1] for comparison
        # Critic values can be negative, so we need to handle that
        # For now, assume critic_value is already in reasonable range
        
        difference = abs(affordance_score - critic_value)
        
        # Agreement: both high (>0.5) or both low (<0.5)
        affordance_high = affordance_score > 0.5
        critic_high = critic_value > 0.5
        agreement = affordance_high == critic_high
        
        comparison = {
            "affordance_score": affordance_score,
            "critic_value": critic_value,
            "difference": difference,
            "agreement": agreement,
            "action": action,
        }
        
        # Store in history for correlation calculation
        self.comparison_history.append(comparison)
        
        # Keep only recent history
        if len(self.comparison_history) > 1000:
            self.comparison_history = self.comparison_history[-1000:]
        
        return comparison

    def compute_correlation(self, window: Optional[int] = None) -> float:
        """
        Compute correlation between Affordance and Critic values.
        
        Args:
            window: Number of recent comparisons to use. If None, uses all history.
            
        Returns:
            Pearson correlation coefficient (-1.0 to 1.0)
        """
        if len(self.comparison_history) < 2:
            return 0.0
        
        comparisons_to_use = self.comparison_history
        if window is not None and len(self.comparison_history) > window:
            comparisons_to_use = self.comparison_history[-window:]
        
        affordance_scores = [c["affordance_score"] for c in comparisons_to_use]
        critic_values = [c["critic_value"] for c in comparisons_to_use]
        
        if len(affordance_scores) < 2:
            return 0.0
        
        # Compute Pearson correlation
        correlation = np.corrcoef(affordance_scores, critic_values)[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0

    def enhance_affordance_with_critic(
        self,
        affordance_model: Any,
        weight: float = 0.3,
    ) -> None:
        """
        Enhance Affordance model with Critic values (optional future enhancement).
        
        This is a placeholder for future work where Critic values could be used
        to improve Affordance predictions.
        
        Args:
            affordance_model: AffordanceModel instance
            weight: Weight for Critic contribution (0.0 to 1.0)
        """
        # Future enhancement: could modify affordance_model to incorporate critic feedback
        logger.info(
            f"Critic enhancement not yet implemented. "
            f"Would enhance AffordanceModel with weight={weight}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about Affordance vs Critic comparison.
        
        Returns:
            Dictionary with statistics:
                - correlation: Correlation coefficient
                - mean_difference: Mean absolute difference
                - agreement_rate: Rate of agreement
                - num_comparisons: Number of comparisons
        """
        if not self.comparison_history:
            return {
                "correlation": 0.0,
                "mean_difference": 0.0,
                "agreement_rate": 0.0,
                "num_comparisons": 0,
            }
        
        correlation = self.compute_correlation()
        differences = [c["difference"] for c in self.comparison_history]
        agreements = [c["agreement"] for c in self.comparison_history]
        
        return {
            "correlation": correlation,
            "mean_difference": float(np.mean(differences)) if differences else 0.0,
            "agreement_rate": (
                sum(agreements) / len(agreements) if agreements else 0.0
            ),
            "num_comparisons": len(self.comparison_history),
        }

    def reset(self) -> None:
        """Reset comparison history for a new episode."""
        self.comparison_history = []
