#!/usr/bin/env python3

"""
Stability Metrics Data Classes

Defines the unified data structures for policy stability metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StabilityMetrics:
    r"""Unified policy stability metrics.

    ``beta`` remains a single-run proxy when computed inside a single
    episode run. The authoritative :math:`\hat\beta` defined in §2 of the
    resilience design doc is computed *across* multiple perturbed rollouts
    via :func:`StabilityCollector.aggregate_beta_over_anchors`. The fields
    ``beta_neighborhood`` / ``beta_vv`` / ``beta_out`` carry the
    neighborhood aggregates; ``beta_scope`` records how ``beta`` was
    produced.
    """

    beta: float = 1.0
    value_variance: float = 0.0
    value_delta_variance: float = 0.0
    n_replan: int = 0
    p_cbf: float = 0.0
    collision_rate: float = 0.0
    safety_score: float = 1.0

    score_variance: Optional[float] = None
    stability_violations: Optional[int] = None
    stability_violation_rate: Optional[float] = None

    total_steps: int = 0
    episode_id: Optional[str] = None
    beta_mode: str = "proxy_delta_value_variance"
    beta_scope: str = "single_run_episode_proxy"

    # Neighborhood-level β (§2.1 of the design doc). Set via
    # :func:`StabilityCollector.aggregate_beta_over_anchors`.
    beta_neighborhood: Optional[float] = None
    beta_vv: Optional[float] = None
    beta_out: Optional[float] = None
    beta_action: Optional[float] = None
    beta_progress: Optional[float] = None
    beta_success: Optional[float] = None
    clean_ref_valid: bool = False
    clean_ref_strategy: str = ""
    n_clean_ref: int = 0
    beta_neighborhood_valid: bool = False
    beta_neighborhood_missing_reason: str = ""
    perturbation_type: Optional[str] = None
    perturbation_seed: Optional[int] = None

    value_history: list = field(default_factory=list)
    replan_steps: list = field(default_factory=list)

    # Critic-derived execution oscillation primitives. These are populated
    # from A2CCritic after a full episode is processed.
    td_error_mean_abs: float = 0.0
    td_error_p95_abs: float = 0.0
    gae_mean_abs: float = 0.0
    gae_p95_abs: float = 0.0
    return_target_variance: float = 0.0
    oscillation_loss: float = 0.0
    oscillation_loss_valid: bool = False
    oscillation_source: str = ""
    critic_trace_len: int = 0

    def compute_derived_metrics(
        self,
        task_progress: float = 1.0,
        task_success: bool = True,
        total_steps: int = 1,
        method: str = "hybrid"
    ) -> None:
        """
        Compute derived stability metrics including beta.

        Hybrid approach:
        - Component 1 (60%): Planning efficiency (ideal: 7.5 plans per 1000 steps)
        - Component 2 (40%): Value trajectory features (trend, smoothness, progress consistency)

        Args:
            task_progress: Task completion ratio (0-1)
            task_success: Whether task succeeded
            total_steps: Total simulation steps
            method: "hybrid" (default), "planning_only", or "value_only"
        """
        import numpy as np

        # === Compute value statistics ===
        if len(self.value_history) > 1:
            history = np.asarray(self.value_history, dtype=np.float32)
            self.value_variance = float(np.var(history))
            deltas = np.diff(history)
            self.value_delta_variance = float(np.var(deltas)) if deltas.size > 0 else 0.0
        else:
            self.value_variance = 0.0
            self.value_delta_variance = 0.0

        # === Component 1: Planning Efficiency (beta_planning) ===
        # Ideal: 7.5 plans per 1000 steps (0.0075 density)
        # Range: [0.005, 0.010] is acceptable
        ideal_planning_density = 0.0075
        acceptable_range = 0.0025  # ±0.0025 around ideal

        if total_steps > 0:
            actual_planning_density = self.n_replan / total_steps
            density_deviation = abs(actual_planning_density - ideal_planning_density)

            if density_deviation <= acceptable_range:
                # Within acceptable range: linear decay from 1.0
                beta_planning = 1.0 - (density_deviation / acceptable_range) * 0.2
            else:
                # Outside acceptable range: steeper penalty
                excess_deviation = density_deviation - acceptable_range
                beta_planning = 0.8 - min(excess_deviation / 0.02, 1.0) * 0.8

            beta_planning = max(0.0, min(beta_planning, 1.0))
        else:
            beta_planning = 1.0

        # === Component 2: Value Trajectory Features (beta_value) ===
        if len(self.value_history) > 2:
            history = np.asarray(self.value_history, dtype=np.float32)

            # Feature 1: Trend consistency (should increase for successful tasks)
            trend_score = 0.0
            if task_success and task_progress > 0.5:
                # Expect positive trend for successful tasks
                trend = np.polyfit(np.arange(len(history)), history, 1)[0]
                if trend > 0:
                    trend_score = min(trend * 100, 1.0)  # Scale to [0, 1]
                else:
                    trend_score = 0.0
            elif not task_success:
                # For failed tasks, penalize stagnation (near-zero variance)
                if self.value_delta_variance < 1e-5:
                    trend_score = 0.0  # Stagnation penalty
                else:
                    trend_score = 0.5  # Some activity
            else:
                trend_score = 0.5  # Neutral for incomplete tasks

            # Feature 2: Smoothness (low relative variance is good)
            mean_abs_value = float(np.mean(np.abs(history)))
            if mean_abs_value > 1e-6:
                relative_variance = self.value_delta_variance / mean_abs_value
                # Lower relative variance = smoother = better
                smoothness_score = 1.0 - min(relative_variance / 0.05, 1.0)
            else:
                smoothness_score = 0.0 if not task_success else 0.5

            # Feature 3: Progress consistency (value should align with task progress)
            if task_progress > 0.1:
                # Normalize value history to [0, 1]
                value_min, value_max = history.min(), history.max()
                if value_max > value_min:
                    normalized_final_value = (history[-1] - value_min) / (value_max - value_min)
                    progress_consistency = 1.0 - abs(normalized_final_value - task_progress)
                else:
                    progress_consistency = 0.5
            else:
                progress_consistency = 0.5

            # Combine value features (weighted average)
            beta_value = 0.4 * trend_score + 0.3 * smoothness_score + 0.3 * progress_consistency
            beta_value = max(0.0, min(beta_value, 1.0))
        else:
            beta_value = 1.0

        # === Combine components based on method ===
        if method == "hybrid":
            self.beta = 0.6 * beta_planning + 0.4 * beta_value
        elif method == "planning_only":
            self.beta = beta_planning
        elif method == "value_only":
            self.beta = beta_value
        else:
            # Fallback to hybrid
            self.beta = 0.6 * beta_planning + 0.4 * beta_value

        # Store component scores for analysis
        self.beta_planning = beta_planning
        self.beta_value = beta_value

    def to_dict(self) -> Dict:
        return {
            "beta": self.beta,
            "value_variance": self.value_variance,
            "value_delta_variance": self.value_delta_variance,
            "n_replan": self.n_replan,
            "p_cbf": self.p_cbf,
            "collision_rate": self.collision_rate,
            "safety_score": self.safety_score,
            "score_variance": self.score_variance,
            "stability_violations": self.stability_violations,
            "stability_violation_rate": self.stability_violation_rate,
            "total_steps": self.total_steps,
            "episode_id": self.episode_id,
            "beta_mode": self.beta_mode,
            "beta_scope": self.beta_scope,
            "beta_neighborhood": self.beta_neighborhood,
            "beta_vv": self.beta_vv,
            "beta_out": self.beta_out,
            "beta_action": self.beta_action,
            "beta_progress": self.beta_progress,
            "beta_success": self.beta_success,
            "clean_ref_valid": self.clean_ref_valid,
            "clean_ref_strategy": self.clean_ref_strategy,
            "n_clean_ref": self.n_clean_ref,
            "beta_neighborhood_valid": self.beta_neighborhood_valid,
            "beta_neighborhood_missing_reason": self.beta_neighborhood_missing_reason,
            "perturbation_type": self.perturbation_type,
            "perturbation_seed": self.perturbation_seed,
            "td_error_mean_abs": self.td_error_mean_abs,
            "td_error_p95_abs": self.td_error_p95_abs,
            "gae_mean_abs": self.gae_mean_abs,
            "gae_p95_abs": self.gae_p95_abs,
            "return_target_variance": self.return_target_variance,
            "oscillation_loss": self.oscillation_loss,
            "oscillation_loss_valid": self.oscillation_loss_valid,
            "oscillation_source": self.oscillation_source,
            "critic_trace_len": self.critic_trace_len,
        }

    def get_summary_for_csv(self) -> Dict[str, float]:
        return {
            "stability_beta": self.beta,
            "stability_beta_planning": getattr(self, 'beta_planning', 0.0),
            "stability_beta_value": getattr(self, 'beta_value', 0.0),
            "stability_value_variance": self.value_variance,
            "stability_value_delta_variance": self.value_delta_variance,
            "stability_td_error_mean_abs": self.td_error_mean_abs,
            "stability_td_error_p95_abs": self.td_error_p95_abs,
            "stability_gae_mean_abs": self.gae_mean_abs,
            "stability_gae_p95_abs": self.gae_p95_abs,
            "stability_return_target_variance": self.return_target_variance,
            "stability_oscillation_loss": self.oscillation_loss,
            "stability_oscillation_loss_valid": float(
                bool(self.oscillation_loss_valid)
            ),
            "stability_oscillation_source": self.oscillation_source,
            "stability_critic_trace_len": float(self.critic_trace_len),
            "stability_n_replan": float(self.n_replan),
            "stability_p_cbf": self.p_cbf,
            "safety_collision_rate": self.collision_rate,
            "safety_score": self.safety_score,
            "stability_score_variance": (
                self.score_variance if self.score_variance is not None else 0.0
            ),
            "stability_violation_rate": (
                self.stability_violation_rate
                if self.stability_violation_rate is not None
                else 0.0
            ),
            "stability_beta_neighborhood": (
                self.beta_neighborhood if self.beta_neighborhood is not None else 0.0
            ),
            "stability_beta_vv": (
                self.beta_vv if self.beta_vv is not None else 0.0
            ),
            "stability_beta_out": (
                self.beta_out if self.beta_out is not None else 0.0
            ),
            "stability_beta_action": (
                self.beta_action if self.beta_action is not None else 0.0
            ),
            "stability_beta_progress": (
                self.beta_progress if self.beta_progress is not None else 0.0
            ),
            "stability_beta_success": (
                self.beta_success if self.beta_success is not None else 0.0
            ),
            "stability_clean_ref_valid": float(bool(self.clean_ref_valid)),
            "stability_clean_ref_strategy": self.clean_ref_strategy,
            "stability_n_clean_ref": float(self.n_clean_ref),
            "stability_beta_neighborhood_valid": float(
                bool(self.beta_neighborhood_valid)
            ),
            "stability_beta_neighborhood_missing_reason": (
                self.beta_neighborhood_missing_reason
            ),
            "stability_beta_scope": self.beta_scope,
            "stability_perturbation_type": self.perturbation_type or "",
            "stability_perturbation_seed": (
                float(self.perturbation_seed) if self.perturbation_seed is not None else 0.0
            ),
        }
