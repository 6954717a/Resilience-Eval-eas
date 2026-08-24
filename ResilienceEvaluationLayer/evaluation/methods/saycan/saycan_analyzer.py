"""
SayCan Analyzer

Collects and analyzes SayCan data for evaluation and debugging.
Similar to ADCA Analyzer, but focused on SayCan-specific metrics.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from habitat_llm.planner.saycan.candidate_scorer import ScoredCandidate

logger = logging.getLogger(__name__)


@dataclass
class SayCanStepData:
    """
    Represents SayCan data for a single planning step.
    
    Attributes:
        step: Planning step number
        candidates: List of scored candidates
        selected_action: Action that was selected (or fallback action)
        selected_rank: Rank of selected action in candidate list (0 = best)
        stability_triggered: Whether stability threshold was violated
        fallback_action: Fallback action if stability triggered
        say_score: Say score of selected action
        can_score: Can score of selected action
        total_score: Total score of selected action
    """
    step: int
    candidates: List[ScoredCandidate]
    selected_action: Optional[str] = None
    selected_rank: int = 0
    stability_triggered: bool = False
    fallback_action: Optional[str] = None
    say_score: float = 0.0
    can_score: float = 0.0
    total_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step": self.step,
            "candidates": [c.to_dict() for c in self.candidates],
            "selected_action": self.selected_action,
            "selected_rank": self.selected_rank,
            "stability_triggered": self.stability_triggered,
            "fallback_action": self.fallback_action,
            "say_score": self.say_score,
            "can_score": self.can_score,
            "total_score": self.total_score,
        }


class SayCanAnalyzer:
    """
    Analyzes SayCan performance by collecting and computing statistics.
    
    Collects data at each planning step and provides episode-level analysis.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SayCanAnalyzer.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.step_data: List[SayCanStepData] = []
        self.enabled = self.config.get("enabled", True)

    def record_step(
        self,
        step: int,
        candidates: List[ScoredCandidate],
        selected: Optional[ScoredCandidate] = None,
        stability_triggered: bool = False,
        fallback_action: Optional[str] = None,
    ) -> None:
        """
        Record SayCan data for a planning step.
        
        Args:
            step: Planning step number
            candidates: List of scored candidates
            selected: Selected candidate (None if stability triggered)
            stability_triggered: Whether stability threshold was violated
            fallback_action: Fallback action if stability triggered
        """
        if not self.enabled:
            return
        
        # Extract scores from selected candidate
        say_score = selected.say_score if selected else 0.0
        can_score = selected.can_score if selected else 0.0
        total_score = selected.total_score if selected else 0.0
        selected_action = selected.action if selected else fallback_action
        selected_rank = selected.rank if selected else -1
        
        step_data = SayCanStepData(
            step=step,
            candidates=candidates,
            selected_action=selected_action,
            selected_rank=selected_rank,
            stability_triggered=stability_triggered,
            fallback_action=fallback_action,
            say_score=say_score,
            can_score=can_score,
            total_score=total_score,
        )
        
        self.step_data.append(step_data)

    def analyze_episode(self) -> Dict[str, Any]:
        """
        Analyze collected step data and compute episode-level statistics.
        
        Returns:
            Dictionary with analysis results:
                - step_data: List of step data
                - summary: Summary statistics
                - candidate_analysis: Candidate-level analysis
                - affordance_analysis: Affordance score distribution
        """
        if not self.step_data:
            return {
                "step_data": [],
                "summary": {},
                "candidate_analysis": {},
                "affordance_analysis": {},
            }
        
        # Extract all scores
        say_scores = [s.say_score for s in self.step_data if s.say_score > 0]
        can_scores = [s.can_score for s in self.step_data if s.can_score > 0]
        total_scores = [s.total_score for s in self.step_data if s.total_score > 0]
        
        # Summary statistics
        summary = {
            "total_steps": len(self.step_data),
            "mean_say_score": float(np.mean(say_scores)) if say_scores else 0.0,
            "mean_can_score": float(np.mean(can_scores)) if can_scores else 0.0,
            "mean_total_score": float(np.mean(total_scores)) if total_scores else 0.0,
            "std_say_score": float(np.std(say_scores)) if say_scores else 0.0,
            "std_can_score": float(np.std(can_scores)) if can_scores else 0.0,
            "std_total_score": float(np.std(total_scores)) if total_scores else 0.0,
            "stability_violations": sum(1 for s in self.step_data if s.stability_triggered),
            "stability_violation_rate": (
                sum(1 for s in self.step_data if s.stability_triggered) / len(self.step_data)
                if self.step_data else 0.0
            ),
        }
        
        # Candidate analysis
        candidate_analysis = self._analyze_candidates()
        
        # Affordance analysis
        affordance_analysis = self._analyze_affordance()
        
        return {
            "step_data": [s.to_dict() for s in self.step_data],
            "summary": summary,
            "candidate_analysis": candidate_analysis,
            "affordance_analysis": affordance_analysis,
        }

    def _analyze_candidates(self) -> Dict[str, Any]:
        """
        Analyze candidate selection patterns.
        
        Returns:
            Dictionary with candidate analysis:
                - rank_changes: How often selected action rank differs from LLM top choice
                - llm_vs_affordance: Comparison of LLM top choice vs Affordance top choice
        """
        rank_changes = []
        llm_top_vs_final = []
        
        for step_data in self.step_data:
            if not step_data.candidates:
                continue
            
            # LLM top choice (highest Say score)
            llm_top = max(step_data.candidates, key=lambda x: x.say_score)
            
            # Affordance top choice (highest Can score)
            affordance_top = max(step_data.candidates, key=lambda x: x.can_score)
            
            # Final choice (highest Total score)
            final_top = step_data.candidates[0]  # Already sorted by total_score
            
            # Rank change: where did final choice rank in LLM's view?
            llm_ranked = sorted(step_data.candidates, key=lambda x: x.say_score, reverse=True)
            final_llm_rank = next(
                (i for i, c in enumerate(llm_ranked) if c.action == final_top.action),
                -1
            )
            
            rank_changes.append({
                "step": step_data.step,
                "llm_top_action": llm_top.action,
                "final_action": final_top.action,
                "final_llm_rank": final_llm_rank,
                "rank_change": final_llm_rank - 0,  # 0 = LLM top choice
            })
            
            llm_top_vs_final.append({
                "step": step_data.step,
                "llm_top_say": llm_top.say_score,
                "llm_top_can": llm_top.can_score,
                "llm_top_total": llm_top.total_score,
                "final_say": final_top.say_score,
                "final_can": final_top.can_score,
                "final_total": final_top.total_score,
                "same_action": llm_top.action == final_top.action,
            })
        
        return {
            "rank_changes": rank_changes,
            "llm_vs_final": llm_top_vs_final,
            "mean_rank_change": (
                float(np.mean([r["rank_change"] for r in rank_changes]))
                if rank_changes else 0.0
            ),
            "same_action_rate": (
                sum(1 for v in llm_top_vs_final if v["same_action"]) / len(llm_top_vs_final)
                if llm_top_vs_final else 0.0
            ),
        }

    def _analyze_affordance(self) -> Dict[str, Any]:
        """
        Analyze Affordance score distribution by skill type.
        
        Returns:
            Dictionary with affordance analysis:
                - by_skill: Mean Can scores by skill type
                - failure_reasons: Distribution of failure reasons
        """
        by_skill: Dict[str, List[float]] = {}
        failure_reasons: Dict[str, int] = {}
        
        for step_data in self.step_data:
            for candidate in step_data.candidates:
                # Extract skill type from affordance details
                skill_type = candidate.affordance_details.get("skill_type", "unknown")
                can_score = candidate.can_score
                
                if skill_type not in by_skill:
                    by_skill[skill_type] = []
                by_skill[skill_type].append(can_score)
                
                # Track failure reasons
                failure_reason = candidate.affordance_details.get("failure_reason")
                if failure_reason:
                    failure_reasons[failure_reason] = failure_reasons.get(failure_reason, 0) + 1
        
        # Compute mean scores by skill
        mean_by_skill = {
            skill: float(np.mean(scores)) if scores else 0.0
            for skill, scores in by_skill.items()
        }
        
        return {
            "by_skill": mean_by_skill,
            "failure_reasons": failure_reasons,
            "total_candidates": sum(len(s.candidates) for s in self.step_data),
        }

    def reset(self) -> None:
        """Reset analyzer state for a new episode."""
        self.step_data = []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics without full analysis.
        
        Returns:
            Dictionary with basic statistics
        """
        if not self.step_data:
            return {}
        
        total_scores = [s.total_score for s in self.step_data if s.total_score > 0]
        
        return {
            "num_steps": len(self.step_data),
            "mean_total_score": float(np.mean(total_scores)) if total_scores else 0.0,
            "stability_violations": sum(1 for s in self.step_data if s.stability_triggered),
        }
