# -*- coding: utf-8 -*-
"""
ADCA (Attribution-Driven Credit Assignment) Analyzer.

Integrates logic from AgentEvolver to provide step-level advantage estimation
and semantic attribution analysis for Habitat-LLM trajectories.
"""

import math
import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union

from .prompt_evaluation import (
    get_positive_mask,
    build_adca_prompt,
    call_llm_completion
)

logger = logging.getLogger(__name__)


@dataclass
class PRMHyper:
    """
    Hyperparameters for the Policy Reward Model (PRM) process.
    Adapted from AgentEvolver/adca_grpo.py.
    """
    # Weights: higher weight for consistent steps, lower weight for inconsistent steps
    consistent_scale: float = 1.0
    pos_unconsistent_scale: float = 0.2   # Weight for BAD steps in successful trajectories
    neg_unconsistent_scale: float = 0.2   # Weight for GOOD steps in failed trajectories
    eps: float = 1e-8
    do_batch_norm: bool = False           # Disabled for single trajectory evaluation
    fix_base: float = 0.2                 # Base magnitude for fix scheme (good=+base, bad=-base)
    alpha: float = 1.0                    # PRM weight balance coefficient
    orm_distribution: str = "last_step"   # ORM distribution method: "last_step" or "all_steps"
    enable_length_normalization: bool = False  # Whether to enable length normalization


def get_positive_mask(score: float, threshold: float = 0) -> bool:
    """Determines whether score is positive based on the given threshold."""
    return score > threshold


class ADCAAnalyzer:
    """
    Analyzes episode trajectories using ADCA (Attribution-Driven Credit Assignment).
    """

    def __init__(self, llm_client: Any, config: Dict[str, Any]):
        """
        Initialize ADCA Analyzer.

        Args:
            llm_client: LLM client (e.g., OpenAI client)
            config: Configuration dict.
        """
        self.llm_client = llm_client
        self.llm_model = config.get('llm_model', 'gpt-4')
        self.hyper = PRMHyper(
            alpha=config.get('adca_alpha', 1.0),
            fix_base=config.get('adca_fix_base', 0.2),
            orm_distribution=config.get('adca_orm_distribution', 'last_step')
        )
        self.enabled = config.get('enabled', False)

    def analyze(
        self,
        trajectory: List[Dict[str, str]],
        outcome_score: float,
        task_instruction: str
    ) -> Dict[str, Any]:
        """
        Analyze a trajectory to compute step-level advantages.

        Args:
            trajectory: List of steps, each a dict with 'action', 'observation'.
            outcome_score: The final reward/score of the episode.
            task_instruction: The task description.

        Returns:
            Dict containing:
                - step_flags: List of bool (True=GOOD, False=BAD)
                - step_advantages: List of float (computed advantages)
                - llm_response: Raw LLM response
                - analysis_stats: Dict of statistics
        """
        if not trajectory:
            logger.warning("ADCA analyze called with empty trajectory")
            return {}

        logger.info(f"Starting ADCA analysis for trajectory length: {len(trajectory)}, outcome: {outcome_score}")

        # 1. Evaluate steps using LLM
        step_flags, llm_response = self._evaluate_steps_llm(
            trajectory, outcome_score, task_instruction
        )

        # 2. Compute advantages
        step_advantages, stats = self._compute_advantages(
            trajectory, step_flags, outcome_score
        )

        return {
            "step_flags": step_flags,
            "step_advantages": step_advantages,
            "llm_response": llm_response,
            "analysis_stats": stats
        }

    def _evaluate_steps_llm(
        self,
        trajectory: List[Dict[str, str]],
        outcome_score: float,
        task_instruction: str
    ) -> Tuple[List[bool], str]:
        """Calls LLM to evaluate each step as GOOD or BAD."""
        
        # Use common prompt builder
        messages = build_adca_prompt(task_instruction, trajectory, outcome_score)
        
        # Use common LLM caller
        response_text = call_llm_completion(
            self.llm_client,
            self.llm_model,
            messages,
            temperature=0.0,
            max_tokens=4096
        )

        if response_text:
            # Parse results
            flags = self._parse_evaluation_result(response_text, len(trajectory))
            return flags, response_text
        else:
            logger.error("ADCA LLM evaluation returned None/Empty")
            # Fallback
            is_pos = get_positive_mask(outcome_score)
            return [is_pos] * len(trajectory), "ERROR: LLM Call Failed"

    def _parse_evaluation_result(self, response: str, num_steps: int) -> List[bool]:
        """Parses the LLM response to extract GOOD/BAD flags."""
        # Try parsing structured output "Step X Judgment: GOOD/BAD"
        numbered = {}
        for m in re.finditer(r"Step\s+(\d+)\s+Judgment:\s*(GOOD|BAD)", response, flags=re.I):
            numbered[int(m.group(1))] = m.group(2).upper() == "GOOD"
        
        if len(numbered) == num_steps:
            return [numbered.get(i, False) for i in range(num_steps)]
        
        # Fallback: sequential find
        flags = re.findall(r"\b(GOOD|BAD)\b", response.upper())
        # Filter out GOOD/BAD from instructions if any (simple heuristic)
        # Assuming the response mostly contains the judgments. 
        # This is a bit risky but common in simple parsers.
        
        # A better fallback might be needed if the LLM output is very verbose.
        # But let's try to match at least num_steps.
        if len(flags) >= num_steps:
            # Take the last num_steps occurrences if there are more (e.g. if reasoning mentions GOOD/BAD)
            # Or the first? The prompt asks for "Step 0 ... Step 1 ...". 
            # Usually the judgments appear in order.
            
            # Let's try to be stricter.
            pass

        # If strict parsing failed, let's look for "Judgment: GOOD/BAD" without step number
        judgments = re.findall(r"Judgment:\s*(GOOD|BAD)", response, flags=re.I)
        if len(judgments) == num_steps:
             return [j.upper() == "GOOD" for j in judgments]

        if len(flags) >= num_steps:
             # Just take the first N, hoping the reasoning doesn't spam keywords
             # Actually, usually reasoning might contain "good" or "bad". 
             # The regex \b(GOOD|BAD)\b is case sensitive in previous pattern if I didn't use flags=re.I
             # But here I used re.I in the strict parser.
             
             # Let's return best effort.
             return [f == "GOOD" for f in flags[:num_steps]]

        logger.warning(f"Could not parse {num_steps} steps from response. Found {len(numbered)} numbered judgments.")
        # Default to False (BAD) if parsing fails for safety, or use simple heuristic
        return [numbered.get(i, False) for i in range(num_steps)]

    def _compute_advantages(
        self,
        trajectory: List[Dict[str, str]],
        step_flags: List[bool],
        outcome_score: float
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Computes step-level advantages using the Decouple scheme logic.
        Adapted from _build_decouple in adca_grpo.py for single trajectory.
        """
        K = len(trajectory)
        if K == 0:
            return [], {}

        # 1. Base PRM rewards
        # fix_base for GOOD, -fix_base for BAD
        prm_rewards = [
            self.hyper.fix_base if f else -self.hyper.fix_base 
            for f in step_flags
        ]

        # 2. Standardization (Skip batch norm for single trajectory as per plan)
        # We just use the raw values as "standardized" values for single trajectory
        prm_std = prm_rewards 

        # 3. ORM Standardization (Skip for single trajectory)
        # We use the outcome score directly or normalized if needed.
        # adca_grpo uses (score - mean) / std. For single sample, mean=score, std=0 -> 0.
        # But we want to keep the signal. 
        # In single sample context, we might just want to use the score itself 
        # or maybe we assume a distribution (e.g. mean=0, std=1) if we don't have batch stats.
        # Let's use the raw outcome_score.
        orm_std = outcome_score

        # 4. Combine
        alpha = self.hyper.alpha
        orm_distribution = self.hyper.orm_distribution
        
        combined_step_rewards = []
        for j, prm_r in enumerate(prm_std):
            val = 0.0
            if orm_distribution == "last_step":
                if j == K - 1:
                    val = alpha * prm_r + orm_std
                else:
                    val = alpha * prm_r
            elif orm_distribution == "all_steps":
                val = alpha * prm_r + orm_std
            else:
                val = prm_r # Fallback
            
            combined_step_rewards.append(val)

        # 5. Suffix Sum (Advantage)
        # adv[t] = sum(reward[t:])
        advantages = []
        running_sum = 0.0
        for r in reversed(combined_step_rewards):
            running_sum += r
            advantages.insert(0, running_sum)

        stats = {
            "adca/prm_mean": sum(prm_rewards)/K,
            "adca/orm_score": outcome_score,
            "adca/good_steps": sum(step_flags),
            "adca/total_steps": K
        }

        # Cliff Detection (Degradation Metric)
        # Definition: Consistent GOOD performance followed by sudden failure (BAD)
        cliff_detected = False
        cliff_step = -1
        if outcome_score < 0.5 and K >= 3:
            try:
                first_bad_idx = step_flags.index(False)
                # If we had at least 3 GOOD steps before the first BAD step
                if first_bad_idx >= 3:
                    cliff_detected = True
                    cliff_step = first_bad_idx
            except ValueError:
                pass # No BAD steps found
        
        stats["adca/cliff_detected"] = 1.0 if cliff_detected else 0.0
        stats["adca/cliff_step"] = cliff_step

        return advantages, stats

