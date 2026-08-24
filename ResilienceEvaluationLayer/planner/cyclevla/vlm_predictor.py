"""
VLM Failure Predictor for CycleVLA.

This module implements the VLM-based failure prediction and planning
component of CycleVLA. At subtask boundaries, it inspects the current
state and decides whether to transit to the next subtask or backtrack.

Reference: CycleVLA Paper Section IV-B
- "With a progress-aware VLA, we use an off-the-shelf VLM to predict
   failure and plan recovery at subtask boundaries."
- "The VLM outputs one of two decisions: transit to the next subtask,
   or backtrack to the earliest subtask that restores missing preconditions"
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from habitat_llm.evaluation.methods.cyclevla import Subtask

logger = logging.getLogger(__name__)


class CycleVLMPredictor:
    """
    VLM-based failure predictor and planner for CycleVLA.
    
    At subtask boundaries (when progress >= threshold), this predictor:
    1. Constructs a Chain-of-Thought prompt with current state info
    2. Calls VLM to analyze whether the current subtask succeeded
    3. Returns either "transit" (continue to next) or "backtrack" (retry earlier)
    
    This implements the proactive self-correction mechanism of CycleVLA,
    detecting potential failures before they fully manifest.
    """
    
    # Chain-of-Thought prompt template based on CycleVLA paper
    COT_PROMPT_TEMPLATE = '''You are a robotic task verification system. Analyze whether the current subtask has been completed successfully.

## Current Subtask
Index: {subtask_idx}
Instruction: {subtask_instruction}
Current Progress: {progress:.0%}

Required Postconditions:
{postconditions}

## All Subtasks
{subtask_list}

## Current Environment State
{state_summary}

## Recent Action History
{action_history}

## Analysis Task
Carefully analyze if the current subtask's postconditions are satisfied.

Think step by step:
1. What was the subtask trying to achieve?
2. Based on the current state, have the postconditions been met?
3. Are there any signs of failure (object dropped, wrong location, etc.)?
4. If there is a failure, which earlier subtask should we return to?

## Output Format
Provide your analysis in this exact format:

Reasoning: [Your step-by-step analysis]
Decision: [transit OR backtrack]
Target Subtask: [If backtrack, the subtask index (0-based) to return to; otherwise -1]
Confidence: [high/medium/low]'''
    
    def __init__(self, llm_client: Any, config: Dict[str, Any]):
        """
        Initialize VLM Predictor.
        
        Args:
            llm_client: LLM/VLM client for predictions
            config: Configuration dictionary:
                - vlm_model: Model name for VLM calls (default: "gpt-4-turbo")
                - confidence_threshold: Min confidence for backtrack (default: "medium")
                - use_images: Whether to include images in prompt (default: False)
        """
        self.llm_client = llm_client
        self.config = config
        self.vlm_model = config.get("vlm_model", "gpt-4-turbo")
        self.confidence_threshold = config.get("confidence_threshold", "medium")
        self.use_images = config.get("use_images", False)
        
        # Statistics
        self.prediction_count = 0
        self.transit_count = 0
        self.backtrack_count = 0
    
    def predict_and_plan(
        self,
        observations: Dict[str, Any],
        current_subtask: "Subtask",
        subtask_list: List["Subtask"],
        progress: float,
        world_state: Optional[Dict[str, Any]] = None,
        action_history: Optional[List[Any]] = None
    ) -> Tuple[str, Optional[int]]:
        """
        Predict whether to transit or backtrack at subtask boundary.
        
        This is the core method implementing CycleVLA's VLM-based decision.
        
        Args:
            observations: Agent observations (may include images)
            current_subtask: Current Subtask object
            subtask_list: All subtasks in the task
            progress: Estimated progress of current subtask
            world_state: Optional world state dictionary
            action_history: Optional list of recent actions
        
        Returns:
            Tuple of:
                - decision: "transit" or "backtrack"
                - target_idx: If backtrack, the subtask index to return to; else None
        """
        self.prediction_count += 1
        
        # Build CoT prompt
        prompt = self._build_cot_prompt(
            current_subtask, 
            subtask_list, 
            progress,
            observations,
            world_state,
            action_history
        )
        
        try:
            # Call VLM
            response = self._call_vlm(prompt, observations)
            
            # Parse decision
            decision, target_idx, confidence = self._parse_decision(response)
            
            # Log prediction
            logger.info(
                f"VLM Prediction: decision={decision}, target={target_idx}, "
                f"confidence={confidence}, subtask={current_subtask.idx}"
            )
            
            # Update statistics
            if decision == "transit":
                self.transit_count += 1
            else:
                self.backtrack_count += 1
            
            # Validate target index for backtrack
            if decision == "backtrack":
                if target_idx is None or target_idx < 0:
                    # Default to current subtask (retry in place)
                    target_idx = current_subtask.idx
                elif target_idx >= current_subtask.idx:
                    # Can't backtrack forward, retry current
                    target_idx = current_subtask.idx
            
            return decision, target_idx
            
        except Exception as e:
            logger.error(f"VLM prediction failed: {e}")
            # Default to transit on error
            return "transit", None
    
    def _build_cot_prompt(
        self,
        current_subtask: "Subtask",
        subtask_list: List["Subtask"],
        progress: float,
        observations: Dict[str, Any],
        world_state: Optional[Dict[str, Any]],
        action_history: Optional[List[Any]]
    ) -> str:
        """Build Chain-of-Thought prompt for VLM."""
        
        # Format postconditions
        postconds = current_subtask.postconditions
        postconds_str = "\n".join(f"- {p}" for p in postconds) if postconds else "- (none specified)"
        
        # Format subtask list
        subtask_lines = []
        for st in subtask_list:
            status_marker = {
                "pending": "○",
                "active": "●",
                "completed": "✓",
                "failed": "✗"
            }.get(st.status, "?")
            marker = "→" if st.idx == current_subtask.idx else " "
            subtask_lines.append(f"{marker} [{status_marker}] {st.idx}: {st.instruction}")
        subtask_list_str = "\n".join(subtask_lines)
        
        # Format state summary
        state_str = self._format_state_summary(observations, world_state)
        
        # Format action history
        history_str = self._format_action_history(action_history)
        
        # Fill template
        prompt = self.COT_PROMPT_TEMPLATE.format(
            subtask_idx=current_subtask.idx,
            subtask_instruction=current_subtask.instruction,
            progress=progress,
            postconditions=postconds_str,
            subtask_list=subtask_list_str,
            state_summary=state_str,
            action_history=history_str
        )
        
        return prompt
    
    def _format_state_summary(
        self, 
        observations: Dict[str, Any],
        world_state: Optional[Dict[str, Any]]
    ) -> str:
        """Format state summary for prompt."""
        parts = []
        
        if world_state:
            # Agent holding
            if world_state.get("agent_holding"):
                parts.append(f"Agent is holding: {world_state['agent_holding']}")
            else:
                parts.append("Agent is not holding anything")
            
            # Object positions (sample)
            objects = world_state.get("objects", {})
            if objects:
                obj_lines = []
                for name, info in list(objects.items())[:5]:
                    loc = info.get("location", "unknown")
                    obj_lines.append(f"  - {name}: at {loc}")
                parts.append("Objects:\n" + "\n".join(obj_lines))
            
            # Agent position
            poses = world_state.get("agent_poses", {})
            if poses:
                for agent_id, pose in poses.items():
                    pos = pose.get("position", [0, 0, 0])
                    parts.append(f"Agent {agent_id} position: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")
        
        # From observations
        if observations:
            if "task_observations" in observations:
                task_obs = observations["task_observations"]
                if isinstance(task_obs, dict):
                    for key, val in list(task_obs.items())[:3]:
                        parts.append(f"{key}: {val}")
        
        return "\n".join(parts) if parts else "No detailed state available"
    
    def _format_action_history(self, action_history: Optional[List[Any]]) -> str:
        """Format recent action history for prompt."""
        if not action_history:
            return "No recent actions"
        
        # Take last 5 actions
        recent = action_history[-5:] if len(action_history) > 5 else action_history
        
        lines = []
        for action in recent:
            if hasattr(action, 'action') and hasattr(action, 'response'):
                tool = action.action[0] if action.action else "Unknown"
                args = action.action[1] if len(action.action) > 1 else ""
                resp = str(action.response)[:100] if action.response else ""
                lines.append(f"- {tool}[{args}] -> {resp}")
            else:
                lines.append(f"- {str(action)[:80]}")
        
        return "\n".join(lines) if lines else "No recent actions"
    
    def _call_vlm(self, prompt: str, observations: Dict[str, Any]) -> str:
        """Call VLM with prompt (and optionally images)."""
        
        if hasattr(self.llm_client, 'generate'):
            # Use planner's LLM wrapper (text-only)
            # Note: VLLM wrapper uses 'stop' parameter, not 'stop_str'
            response = self.llm_client.generate(prompt, stop=None)
        elif hasattr(self.llm_client, 'chat'):
            # Use OpenAI-style client
            messages = [{"role": "user", "content": prompt}]
            
            # Add images if configured and available
            if self.use_images and observations:
                # This would need proper image encoding for multimodal models
                pass  # For now, text-only
            
            response = self.llm_client.chat.completions.create(
                model=self.vlm_model,
                messages=messages,
                temperature=0.0,
                max_tokens=500
            ).choices[0].message.content
        else:
            raise ValueError("LLM client type not recognized")
        
        return response
    
    def _parse_decision(self, response: str) -> Tuple[str, Optional[int], str]:
        """
        Parse VLM response to extract decision, target index, and confidence.
        
        Returns:
            Tuple of (decision, target_idx, confidence)
        """
        decision = "transit"  # Default
        target_idx = None
        confidence = "medium"
        
        # Parse Decision
        decision_match = re.search(r'Decision:\s*(transit|backtrack)', response, re.IGNORECASE)
        if decision_match:
            decision = decision_match.group(1).lower()
        
        # Parse Target Subtask
        target_match = re.search(r'Target Subtask:\s*(-?\d+)', response)
        if target_match:
            target_idx = int(target_match.group(1))
            if target_idx < 0:
                target_idx = None
        
        # Parse Confidence
        conf_match = re.search(r'Confidence:\s*(high|medium|low)', response, re.IGNORECASE)
        if conf_match:
            confidence = conf_match.group(1).lower()
        
        # Apply confidence threshold
        conf_levels = {"high": 3, "medium": 2, "low": 1}
        threshold_level = conf_levels.get(self.confidence_threshold, 2)
        response_level = conf_levels.get(confidence, 2)
        
        # If confidence too low for backtrack, default to transit
        if decision == "backtrack" and response_level < threshold_level:
            logger.info(f"Confidence {confidence} below threshold, overriding to transit")
            decision = "transit"
            target_idx = None
        
        return decision, target_idx, confidence
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get prediction statistics."""
        return {
            "prediction_count": self.prediction_count,
            "transit_count": self.transit_count,
            "backtrack_count": self.backtrack_count,
            "backtrack_rate": self.backtrack_count / max(1, self.prediction_count),
        }
    
    def reset_statistics(self):
        """Reset statistics for new episode."""
        self.prediction_count = 0
        self.transit_count = 0
        self.backtrack_count = 0
