# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Candidate Scorer for SayCan

Implements the core SayCan scoring logic: Say × Can fusion.
Scores candidate actions by combining LLM confidence (Say) with Affordance feasibility (Can).
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from habitat_llm.evaluation.saycan.affordance_model import AffordanceModel

logger = logging.getLogger(__name__)


@dataclass
class ScoredCandidate:
    """
    Represents a candidate action with SayCan scores.
    
    Attributes:
        action: Action string (e.g., "Navigate[kitchen]")
        say_score: LLM confidence score (0.0 to 1.0)
        can_score: Affordance feasibility score (0.0 to 1.0)
        total_score: Fused score = say_score × can_score
        affordance_details: Detailed affordance calculation information
        rank: Rank after sorting by total_score (0 = best)
    """
    action: str
    say_score: float
    can_score: float
    total_score: float
    affordance_details: Dict[str, Any]
    rank: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action": self.action,
            "say_score": self.say_score,
            "can_score": self.can_score,
            "total_score": self.total_score,
            "affordance_details": self.affordance_details,
            "rank": self.rank,
        }


class CandidateScorer:
    """
    Scores candidate actions using SayCan formula: Total = Say × Can.
    
    This is the core component that implements the SayCan fusion logic.
    """

    def __init__(
        self,
        affordance_model: AffordanceModel,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize CandidateScorer.
        
        Args:
            affordance_model: AffordanceModel instance for computing Can scores
            config: Optional configuration dictionary
        """
        self.affordance_model = affordance_model
        self.config = config or {}

    def score_candidates(
        self,
        candidates: List[Dict[str, Any]],
        agent_id: int,
        env_interface: Any,
    ) -> List[ScoredCandidate]:
        """
        Score candidate actions using SayCan formula.
        
        Args:
            candidates: List of candidate actions from LLM, each with:
                - action: Action string (e.g., "Navigate[kitchen]")
                - confidence: LLM confidence score (0.0 to 1.0)
            agent_id: ID of the agent executing the action
            env_interface: Environment interface (for affordance calculation)
            
        Returns:
            List of ScoredCandidate objects, sorted by total_score (descending)
        """
        scored_candidates = []
        
        for cand in candidates:
            action_str = cand.get('action', '')
            say_score = float(cand.get('confidence', 0.0))
            
            # Parse action to extract skill and target
            skill, target = self._parse_skill_target(action_str)
            
            # Calculate Can score (affordance) with details
            can_score, affordance_details = self.affordance_model.get_affordance_with_details(
                skill, target, agent_id
            )
            
            # Fuse scores: Total = Say × Can
            total_score = say_score * can_score
            
            scored_candidate = ScoredCandidate(
                action=action_str,
                say_score=say_score,
                can_score=can_score,
                total_score=total_score,
                affordance_details=affordance_details,
                rank=0,  # Will be set after sorting
            )
            
            scored_candidates.append(scored_candidate)
            
            logger.debug(
                f"SayCan Candidate: {action_str} | "
                f"Say: {say_score:.2f} | Can: {can_score:.2f} | "
                f"Total: {total_score:.2f}"
            )
        
        # Sort by total_score (descending)
        scored_candidates.sort(key=lambda x: x.total_score, reverse=True)
        
        # Assign ranks
        for rank, candidate in enumerate(scored_candidates):
            candidate.rank = rank
        
        return scored_candidates

    def _parse_skill_target(self, action_str: str) -> Tuple[str, str]:
        """
        Parse action string to extract skill and target.
        
        Handles formats like:
        - "Navigate[kitchen]"
        - "Pick[apple]"
        - "Place[apple, on, table]"
        
        Args:
            action_str: Action string
            
        Returns:
            Tuple of (skill, target)
        """
        import re
        
        # Match pattern: Skill[Target, ...]
        match = re.match(r'(\w+)\[(.*?)\]', action_str)
        if match:
            skill = match.group(1)
            args = match.group(2)
            # Assume first arg is target (before comma or end)
            target = args.split(',')[0].strip() if args else ""
            return skill, target
        
        # If no brackets, assume entire string is skill
        return action_str, ""
