"""
SayCan Planner

Main planner that combines LLM confidence ("Say") with Affordance scores ("Can").
Implements the SayCan pipeline: candidate generation → scoring → stability check → selection.
"""

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from habitat_llm.planner.llm_planner import LLMPlanner
from habitat_llm.evaluation.methods.saycan.affordance_model import AffordanceModel
from habitat_llm.planner.saycan.candidate_scorer import CandidateScorer, ScoredCandidate
from habitat_llm.planner.saycan.stability_monitor import StabilityMonitor
from .prompts import SAYCAN_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from habitat_llm.evaluation.methods.saycan.saycan_analyzer import SayCanAnalyzer


class SayCanPlanner(LLMPlanner):
    """
    SayCan Planner: Combines LLM confidence ("Say") with Affordance scores ("Can").
    
    Core Formula: Total Score = P(Say) × P(Can)
    
    Pipeline:
    1. "Say": Get top-K candidate actions from LLM with confidence scores
    2. "Can": Calculate affordance (feasibility) for each candidate
    3. Fuse: Multiply Say × Can to get total score
    4. Select: Choose action with highest total score
    5. Stability Check: If best score < threshold, trigger fallback action
    """

    def __init__(self, plan_config: Any, env_interface: Any):
        """
        Initialize SayCanPlanner.
        
        Args:
            plan_config: Planner configuration (DictConfig)
            env_interface: Environment interface
        """
        super().__init__(plan_config, env_interface)
        
        # Get SayCan configuration
        self.saycan_config = plan_config.get("saycan", {})
        self.saycan_enabled = self.saycan_config.get("enabled", False)
        
        if not self.saycan_enabled:
            logger.warning("SayCan enabled in config but saycan.enabled=False. SayCan will not be used.")
            return
        
        # Initialize Affordance Model
        self.affordance_model = AffordanceModel(
            env_interface, 
            config=self.saycan_config.get("affordance", {})
        )
        
        # Initialize Candidate Scorer
        self.candidate_scorer = CandidateScorer(
            affordance_model=self.affordance_model,
            config=self.saycan_config
        )
        
        # Initialize Stability Monitor
        self.stability_monitor = StabilityMonitor(
            config=self.saycan_config
        )
        
        # Configuration
        self.num_candidates = self.saycan_config.get("num_candidates", 5)
        self.stability_threshold = self.saycan_config.get("stability_threshold", 0.2)
        
        # Initialize SayCan Analyzer for data collection and analysis
        try:
            from habitat_llm.evaluation.methods.saycan.saycan_analyzer import SayCanAnalyzer
            analyzer_config = self.saycan_config.get("analyzer", {})
            analyzer_config["enabled"] = analyzer_config.get("enabled", True)
            self.saycan_analyzer = SayCanAnalyzer(config=analyzer_config)
            logger.info("SayCan Analyzer initialized for data collection")
        except Exception as e:
            logger.warning(f"Failed to initialize SayCan Analyzer: {e}")
            self.saycan_analyzer = None

    def generate_action_response(self, prompt_override: Optional[str] = None) -> str:
        """
        Override standard generation to run the SayCan pipeline.
        
        This method is called by LLMPlanner.replan() when replanning is required.
        
        Args:
            prompt_override: Optional prompt override (not used in SayCan mode)
            
        Returns:
            Formatted action response string with thought trace
        """
        if not self.saycan_enabled:
            # Fallback to standard LLM generation
            return super().generate_action_response(prompt_override)
        
        # 1. "Say": Get candidates from LLM
        candidates = self._get_say_candidates()
        
        if not candidates:
            logger.warning("SayCan: No candidates generated. Fallback to Wait.")
            agent_id = self.agents[0].uid if self.agents else 0
            return self._format_action("Wait", "", agent_id)
        
        agent_id = self.agents[0].uid if self.agents else 0
        
        # 2. "Can": Score candidates using CandidateScorer
        scored_candidates = self.candidate_scorer.score_candidates(
            candidates=candidates,
            agent_id=agent_id,
            env_interface=self.env_interface
        )
        
        if not scored_candidates:
            logger.warning("SayCan: No scored candidates. Fallback to Wait.")
            return self._format_action("Wait", "", agent_id)
        
        # 3. Select best candidate
        best_candidate = scored_candidates[0]
        
        # 4. Stability Check
        is_stable, action_to_take, reason = self.stability_monitor.check_stability(best_candidate)
        
        # Track score for variance calculation
        self.stability_monitor.track_score(best_candidate.total_score)
        
        # 5. Record step data for analysis
        if self.saycan_analyzer:
            self.saycan_analyzer.record_step(
                step=self.replanning_count,
                candidates=scored_candidates,
                selected=best_candidate if is_stable else None,
                stability_triggered=not is_stable,
                fallback_action=action_to_take if not is_stable else None,
            )
        
        # 6. Format response
        if not is_stable:
            # Stability violation: use fallback action
            thought = f"Thought: {reason}"
            action = self._format_action(action_to_take, "context", agent_id)
            return f"{thought}\n{action}"
        
        # Stable: return best candidate
        thought = (
            f"Thought: SayCan selected '{best_candidate.action}' "
            f"(Score: {best_candidate.total_score:.2f}, "
            f"Say: {best_candidate.say_score:.2f}, "
            f"Can: {best_candidate.can_score:.2f})."
        )
        formatted_action = self._format_action_from_str(best_candidate.action, agent_id)
        
        return f"{thought}\n{formatted_action}"

    def _get_say_candidates(self) -> List[Dict[str, Any]]:
        """
        Prompt LLM for top-K candidate actions with confidence scores.
        
        Returns:
            List of candidate dictionaries, each with:
                - action: Action string (e.g., "Navigate[kitchen]")
                - confidence: LLM confidence score (0.0 to 1.0)
        """
        prompt_addition = (
            f"\n[System]: Please list the top {self.num_candidates} best next actions "
            f"with confidence scores in JSON format. "
            f"Format: [{{\"action\": \"ActionName[target]\", \"confidence\": 0.9}}, ...]"
        )
        
        full_prompt = self.curr_prompt + prompt_addition
        
        try:
            response = self.llm.generate(full_prompt, stop=None)
            return self._parse_json_candidates(response)
        except Exception as e:
            logger.error(f"Error getting Say candidates: {e}")
            return []

    def _parse_json_candidates(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract JSON list of candidates from LLM response.
        
        Args:
            text: LLM response text
            
        Returns:
            List of candidate dictionaries
        """
        try:
            # Find JSON-like structure (array)
            match = re.search(r'\[.*?\]', text, re.DOTALL)
            if match:
                candidates = json.loads(match.group())
                # Validate format
                validated = []
                for cand in candidates:
                    if isinstance(cand, dict) and "action" in cand:
                        validated.append({
                            "action": str(cand["action"]),
                            "confidence": float(cand.get("confidence", 0.5))
                        })
                return validated
        except json.JSONDecodeError as e:
            logger.debug(f"Failed to parse JSON candidates: {e}")
        except Exception as e:
            logger.debug(f"Error parsing candidates: {e}")
        
        return []

    def _format_action(self, skill: str, args: str, agent_id: int) -> str:
        """
        Format action string for agent.
        
        Args:
            skill: Skill name (e.g., "Navigate")
            args: Action arguments (e.g., "kitchen")
            agent_id: Agent ID
            
        Returns:
            Formatted action string
        """
        return f"Agent_{agent_id}_Action: {skill}[{args}]\nAssigned!"

    def _format_action_from_str(self, action_str: str, agent_id: int) -> str:
        """
        Format action from action string (may already include Agent prefix).
        
        Args:
            action_str: Action string (e.g., "Navigate[kitchen]" or "Agent_0_Action: Navigate[kitchen]")
            agent_id: Agent ID
            
        Returns:
            Formatted action string
        """
        # Check if it already has Agent prefix
        if "Agent_" in action_str:
            if not action_str.endswith("\nAssigned!"):
                return action_str + "\nAssigned!"
            return action_str
        
        return f"Agent_{agent_id}_Action: {action_str}\nAssigned!"

    def reset(self) -> None:
        """Reset planner state, including SayCan components."""
        super().reset()
        
        if hasattr(self, 'stability_monitor'):
            self.stability_monitor.reset()
        
        if hasattr(self, 'saycan_analyzer') and self.saycan_analyzer:
            self.saycan_analyzer.reset()
