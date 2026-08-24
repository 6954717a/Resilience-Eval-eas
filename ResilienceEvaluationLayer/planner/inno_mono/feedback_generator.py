"""
Feedback Generator for Inner Monologue

Core component that integrates multiple feedback sources (Success Detection,
Scene Description, Self State, Critic Feedback) and generates structured
Inner Monologue feedback.
"""

from typing import Dict, Any, Optional, Tuple, List
from .success_detector import SuccessDetector
from .scene_describer import SceneDescriber
from .self_state_reporter import SelfStateReporter


class FeedbackGenerator:
    """
    Integrates multiple feedback sources and generates structured Inner Monologue feedback.
    
    This is the core component of Inner Monologue that:
    1. Collects feedback from SuccessDetector, SceneDescriber, SelfStateReporter
    2. Optionally integrates Critic feedback
    3. Formats everything into a structured feedback dictionary
    4. Can be formatted as natural language text for prompt injection
    """

    def __init__(
        self,
        config: Dict[str, Any],
        env_interface: Any,
        success_detector: Optional[SuccessDetector] = None,
        scene_describer: Optional[SceneDescriber] = None,
        self_state_reporter: Optional[SelfStateReporter] = None
    ):
        """
        Initialize FeedbackGenerator.
        
        Args:
            config: Configuration dictionary with:
                - feedback_sources: List of enabled sources (success_detection, scene_description, self_state)
                - use_critic_feedback: Whether to use Critic feedback (default: False)
                - use_rebound_feedback: Whether to use Rebound feedback (default: False)
            env_interface: Environment interface for accessing world state
            success_detector: SuccessDetector instance (created if None)
            scene_describer: SceneDescriber instance (created if None)
            self_state_reporter: SelfStateReporter instance (created if None)
        """
        self.config = config
        self.env_interface = env_interface
        
        # Initialize feedback source components
        feedback_sources = config.get("feedback_sources", ["success_detection", "scene_description", "self_state"])
        
        if success_detector is None and "success_detection" in feedback_sources:
            self.success_detector = SuccessDetector(config.get("success_detector", {}))
        else:
            self.success_detector = success_detector
        
        if scene_describer is None and "scene_description" in feedback_sources:
            self.scene_describer = SceneDescriber(config.get("scene_describer", {}))
        else:
            self.scene_describer = scene_describer
        
        if self_state_reporter is None and "self_state" in feedback_sources:
            self.self_state_reporter = SelfStateReporter(config.get("self_state_reporter", {}))
        else:
            self.self_state_reporter = self_state_reporter
        
        # Flags for optional feedback sources
        self.use_critic_feedback = config.get("use_critic_feedback", False)
        self.use_rebound_feedback = config.get("use_rebound_feedback", False)
        
        # Statistics tracking for resilience metrics
        self.feedback_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.thought_generation_count = 0
        self.correction_attempts = 0
        self.correction_successes = 0
        self._pending_corrections = 0  # Track pending corrections waiting for success
        self.feedback_history: List[Dict[str, Any]] = []

    def generate_feedback(
        self,
        agent_responses: Dict[int, str],
        last_actions: Dict[int, Tuple[str, str, str]],
        world_state: Dict[str, Any],
        critic_feedback: Optional[Dict[str, Any]] = None,
        rebound_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate structured Inner Monologue feedback.
        
        Args:
            agent_responses: Dictionary mapping agent_id to response string
            last_actions: Dictionary mapping agent_id to (action_name, arg1, arg2) tuple
            world_state: World state dictionary
            critic_feedback: Optional feedback from Critic (if use_critic_feedback=True)
            rebound_feedback: Optional feedback from Rebound (if use_rebound_feedback=True)
        
        Returns:
            Structured feedback dictionary with keys:
                - success_detection: Dict mapping agent_id to success/failure info
                - scene_description: Scene description string
                - self_state: Dict mapping agent_id to state description
                - critic_feedback: Optional Critic feedback
                - rebound_feedback: Optional Rebound feedback
        """
        feedback = {
            "success_detection": {},
            "scene_description": None,
            "self_state": {},
            "critic_feedback": None,
            "rebound_feedback": None
        }
        
        # 1. Success Detection
        if self.success_detector:
            for agent_id in agent_responses.keys():
                action = last_actions.get(agent_id, ("Unknown", "", ""))
                response = agent_responses.get(agent_id, "")
                success_info = self.success_detector.detect_success(
                    agent_id=agent_id,
                    action=action,
                    response=response,
                    world_state=world_state
                )
                feedback["success_detection"][agent_id] = success_info
        
        # 2. Scene Description
        if self.scene_describer:
            try:
                scene_desc = self.scene_describer.describe_scene(
                    world_state=world_state,
                    env_interface=self.env_interface
                )
                feedback["scene_description"] = scene_desc
            except Exception as e:
                # Fallback if scene description fails
                feedback["scene_description"] = "Scene state available but description generation failed."
        
        # 3. Self State
        if self.self_state_reporter:
            for agent_id in agent_responses.keys():
                try:
                    state_desc = self.self_state_reporter.report_state(
                        agent_id=agent_id,
                        world_state=world_state
                    )
                    feedback["self_state"][agent_id] = state_desc
                except Exception as e:
                    feedback["self_state"][agent_id] = f"Agent {agent_id} state unavailable."
        
        # 4. Critic Feedback (optional)
        if self.use_critic_feedback and critic_feedback:
            feedback["critic_feedback"] = critic_feedback
        
        # 5. Rebound Feedback (optional)
        if self.use_rebound_feedback and rebound_feedback:
            feedback["rebound_feedback"] = rebound_feedback
        
        # Track statistics
        self.feedback_count += 1
        
        # Count successes and failures from success_detection
        success_detection = feedback.get("success_detection", {})
        for agent_id, success_info in success_detection.items():
            if success_info.get("success"):
                self.success_count += 1
            else:
                self.failure_count += 1
        
        # Store feedback history (limit to last 100)
        feedback_record = {
            "feedback_id": self.feedback_count,
            "success_detection": {k: {"success": v.get("success")} for k, v in success_detection.items()},
            "has_scene_desc": feedback.get("scene_description") is not None,
            "has_self_state": len(feedback.get("self_state", {})) > 0,
            "has_critic": feedback.get("critic_feedback") is not None,
            "has_rebound": feedback.get("rebound_feedback") is not None,
        }
        self.feedback_history.append(feedback_record)
        if len(self.feedback_history) > 100:
            self.feedback_history.pop(0)
        
        return feedback

    def format_feedback_as_text(self, feedback: Dict[str, Any]) -> str:
        """
        Format structured feedback as natural language text for prompt injection.
        
        Args:
            feedback: Structured feedback dictionary from generate_feedback()
        
        Returns:
            Formatted text ready for injection into prompt
        """
        lines = []
        
        # 1. Success Detection
        success_detection = feedback.get("success_detection", {})
        if success_detection:
            for agent_id, success_info in sorted(success_detection.items()):
                if success_info.get("success"):
                    lines.append(f"Agent {agent_id}: Success - {success_info.get('message', 'Action completed')}")
                else:
                    lines.append(f"Agent {agent_id}: Failure - {success_info.get('message', 'Action failed')}")
        
        # 2. Scene Description
        scene_desc = feedback.get("scene_description")
        if scene_desc:
            lines.append(f"Scene: {scene_desc}")
        
        # 3. Self State
        self_state = feedback.get("self_state", {})
        if self_state:
            state_lines = []
            for agent_id, state_desc in sorted(self_state.items()):
                state_lines.append(state_desc)
            if state_lines:
                lines.append("State: " + "; ".join(state_lines))
        
        # 4. Critic Feedback (optional)
        critic_feedback = feedback.get("critic_feedback")
        if critic_feedback:
            critic_text = self._format_critic_feedback(critic_feedback)
            if critic_text:
                lines.append(f"Note: {critic_text}")
        
        # 5. Rebound Feedback (optional)
        rebound_feedback = feedback.get("rebound_feedback")
        if rebound_feedback:
            rebound_text = self._format_rebound_feedback(rebound_feedback)
            if rebound_text:
                lines.append(f"Note: {rebound_text}")
        
        return "\n".join(lines) if lines else "Feedback: No feedback available."

    def _format_critic_feedback(self, critic_feedback: Dict[str, Any]) -> str:
        """Format Critic feedback as text."""
        # Extract relevant information from Critic feedback
        # This is a placeholder - actual format depends on CriticFeedbackExtractor output
        if isinstance(critic_feedback, dict):
            value = critic_feedback.get("value")
            if value is not None:
                return f"State value estimate: {value:.2f}"
            message = critic_feedback.get("message")
            if message:
                return message
        return ""

    def _format_rebound_feedback(self, rebound_feedback: Dict[str, Any]) -> str:
        """Format Rebound feedback as text."""
        # Extract relevant information from Rebound feedback
        # This is a placeholder - actual format depends on Rebound integration
        if isinstance(rebound_feedback, dict):
            faults = rebound_feedback.get("faults", [])
            if faults:
                return f"Detected issues: {', '.join(faults)}"
            message = rebound_feedback.get("message")
            if message:
                return message
        return ""
    
    def record_thought_generation(self):
        """Record that a Thought was generated after feedback."""
        self.thought_generation_count += 1
    
    def record_correction_attempt(self, success: bool):
        """
        Record a correction attempt after failure feedback.
        
        Args:
            success: Whether the correction was successful
        """
        if not hasattr(self, "_pending_corrections"):
            self._pending_corrections = 0
        
        if success:
            # If we have pending corrections, mark one as success
            if self._pending_corrections > 0:
                self._pending_corrections -= 1
                self.correction_successes += 1
            # Note: If no pending corrections, this success is not a correction
            # (it's just a normal success)
        else:
            # Record a new correction attempt (pending evaluation)
            self.correction_attempts += 1
            self._pending_corrections += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get Inner Monologue statistics for resilience metrics.
        
        Returns:
            Dictionary with statistics:
                - feedback_count: Total feedback generated
                - success_count: Number of successful actions
                - failure_count: Number of failed actions
                - feedback_success_rate: Success rate
                - thought_generation_count: Number of Thoughts generated
                - correction_attempts: Number of correction attempts
                - correction_successes: Number of successful corrections
                - correction_effectiveness: Correction success rate
        """
        total_feedback = self.success_count + self.failure_count
        feedback_success_rate = (
            self.success_count / total_feedback if total_feedback > 0 else 0.0
        )
        
        correction_effectiveness = (
            self.correction_successes / self.correction_attempts
            if self.correction_attempts > 0 else 0.0
        )
        
        thought_generation_rate = (
            self.thought_generation_count / self.feedback_count
            if self.feedback_count > 0 else 0.0
        )
        
        return {
            "feedback_count": self.feedback_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "feedback_success_rate": feedback_success_rate,
            "thought_generation_count": self.thought_generation_count,
            "thought_generation_rate": thought_generation_rate,
            "correction_attempts": self.correction_attempts,
            "correction_successes": self.correction_successes,
            "correction_effectiveness": correction_effectiveness,
            "feedback_history_count": len(self.feedback_history),
        }
    
    def reset_statistics(self):
        """Reset statistics for new episode."""
        self.feedback_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.thought_generation_count = 0
        self.correction_attempts = 0
        self.correction_successes = 0
        self._pending_corrections = 0
        self.feedback_history.clear()
