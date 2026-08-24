"""
Critic Feedback Extractor for Inner Monologue

Extracts feedback information from A2CCritic for use in Inner Monologue.
This is an optional component that integrates Critic evaluation results
into the feedback loop.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CriticFeedbackExtractor:
    """
    Extracts feedback information from A2CCritic.
    
    This component extracts value estimates, reward shaping results, and other
    evaluation information from the Critic and formats them as natural language
    feedback for Inner Monologue.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CriticFeedbackExtractor.
        
        Args:
            config: Configuration dictionary with optional settings:
                - include_value_estimate: Whether to include V(s) estimate (default: True)
                - include_reward_shaping: Whether to include shaped reward (default: False)
                - value_threshold_low: Low value threshold for feedback (default: 0.3)
                - value_threshold_high: High value threshold for feedback (default: 0.7)
        """
        self.config = config or {}
        self.include_value_estimate = self.config.get("include_value_estimate", True)
        self.include_reward_shaping = self.config.get("include_reward_shaping", False)
        self.value_threshold_low = self.config.get("value_threshold_low", 0.3)
        self.value_threshold_high = self.config.get("value_threshold_high", 0.7)

    def extract_feedback(
        self,
        critic: Any,  # A2CCritic
        state: Dict[str, Any],
        action: Any,
        reward: float,
        next_state: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract feedback from Critic evaluation.
        
        Args:
            critic: A2CCritic instance
            state: Current world state dictionary
            action: Action taken
            reward: Environment reward
            next_state: Next world state (optional)
        
        Returns:
            Dictionary with feedback information, or None if extraction fails
        """
        if not critic:
            return None
        
        feedback = {}
        
        try:
            # 1. Get value estimate
            if self.include_value_estimate:
                try:
                    value = critic.evaluate(state, action)
                    feedback["value"] = float(value)
                    feedback["value_message"] = self._format_value_feedback(value)
                except Exception as e:
                    logger.debug(f"Could not extract value estimate: {e}")
            
            # 2. Get reward shaping result (if available)
            if self.include_reward_shaping and hasattr(critic, 'reward_shaper') and critic.reward_shaper:
                try:
                    # Note: reward_shaper.shape_reward() requires more context
                    # For now, we just note that shaping is enabled
                    feedback["reward_shaping_enabled"] = True
                    feedback["original_reward"] = reward
                except Exception as e:
                    logger.debug(f"Could not extract reward shaping: {e}")
            
            # 3. Generate overall feedback message
            if feedback:
                feedback["message"] = self._generate_feedback_message(feedback)
                return feedback
        
        except Exception as e:
            logger.warning(f"Error extracting Critic feedback: {e}")
            return None
        
        return None

    def _format_value_feedback(self, value: float) -> str:
        """Format value estimate as natural language feedback."""
        if value < self.value_threshold_low:
            return f"State value is low ({value:.2f}), indicating limited progress toward goal."
        elif value > self.value_threshold_high:
            return f"State value is high ({value:.2f}), indicating good progress toward goal."
        else:
            return f"State value is moderate ({value:.2f}), indicating steady progress."

    def _generate_feedback_message(self, feedback: Dict[str, Any]) -> str:
        """Generate overall feedback message from extracted information."""
        messages = []
        
        if "value_message" in feedback:
            messages.append(feedback["value_message"])
        
        if "reward_shaping_enabled" in feedback and feedback["reward_shaping_enabled"]:
            original_reward = feedback.get("original_reward", 0.0)
            messages.append(f"Environment reward: {original_reward:.2f}")
        
        return " ".join(messages) if messages else "Critic evaluation available."

