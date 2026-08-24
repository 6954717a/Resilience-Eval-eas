#!/usr/bin/env python3

r"""Single-rollout Stability evidence.

The paper's :math:`\beta_f` is not an episode score: it is a sensitivity over a
clean/perturbed neighborhood.  This object therefore stores only the trajectory
diagnostics available at runtime.  Neighborhood beta lives in
``BetaAggregate`` and is produced by ``StabilityCollector`` post-processing.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StabilityMetrics:
    """Critic and execution diagnostics for one rollout."""

    episode_proxy: Optional[float] = None
    episode_proxy_source: str = "not_estimated"
    value_variance: Optional[float] = None
    value_delta_variance: Optional[float] = None
    n_replan: int = 0
    p_cbf: Optional[float] = None
    collision_rate: Optional[float] = None
    safety_score: Optional[float] = None
    unsafe_rate: Optional[float] = None
    safety_valid: bool = False
    safety_scope: str = "none"
    safety_missing_reason: str = "not_observed"
    distance_constraint_available: Optional[bool] = None
    distance_observation_count: int = 0
    distance_missing_count: int = 0
    distance_constraint_missing_reason: str = ""

    score_variance: Optional[float] = None
    stability_violations: Optional[int] = None
    stability_violation_rate: Optional[float] = None
    total_steps: int = 0
    episode_id: Optional[str] = None

    value_history: List[float] = field(default_factory=list)
    replan_steps: List[int] = field(default_factory=list)

    td_error_mean_abs: Optional[float] = None
    td_error_p95_abs: Optional[float] = None
    gae_mean_abs: Optional[float] = None
    gae_p95_abs: Optional[float] = None
    return_target_variance: Optional[float] = None
    oscillation_loss: Optional[float] = None
    oscillation_loss_valid: bool = False
    oscillation_source: str = ""
    critic_trace_len: int = 0

    def finalize_episode_proxy(self) -> None:
        """Select one clearly labelled single-run diagnostic.

        It is useful for trace QA and legacy comparisons but must not be used as
        the paper's beta sensitivity.
        """

        if self.oscillation_loss_valid and self.oscillation_loss is not None:
            self.episode_proxy = float(self.oscillation_loss)
            self.episode_proxy_source = self.oscillation_source or "critic_trajectory"
        elif self.value_delta_variance is not None and len(self.value_history) > 1:
            self.episode_proxy = float(self.value_delta_variance)
            self.episode_proxy_source = "value_delta_variance_proxy"
        else:
            self.episode_proxy = None
            self.episode_proxy_source = "insufficient_critic_trace"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_proxy": self.episode_proxy,
            "episode_proxy_source": self.episode_proxy_source,
            "value_variance": self.value_variance,
            "value_delta_variance": self.value_delta_variance,
            "n_replan": self.n_replan,
            "p_cbf": self.p_cbf,
            "collision_rate": self.collision_rate,
            "safety_score": self.safety_score,
            "unsafe_rate": self.unsafe_rate,
            "safety_valid": self.safety_valid,
            "safety_scope": self.safety_scope,
            "safety_missing_reason": self.safety_missing_reason,
            "distance_constraint_available": self.distance_constraint_available,
            "distance_observation_count": self.distance_observation_count,
            "distance_missing_count": self.distance_missing_count,
            "distance_constraint_missing_reason": self.distance_constraint_missing_reason,
            "score_variance": self.score_variance,
            "stability_violations": self.stability_violations,
            "stability_violation_rate": self.stability_violation_rate,
            "total_steps": self.total_steps,
            "episode_id": self.episode_id,
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

    def get_summary_for_csv(self) -> Dict[str, Any]:
        return {
            "stability_episode_proxy": self.episode_proxy,
            "stability_episode_proxy_source": self.episode_proxy_source,
            "stability_value_variance": self.value_variance,
            "stability_value_delta_variance": self.value_delta_variance,
            "stability_td_error_mean_abs": self.td_error_mean_abs,
            "stability_td_error_p95_abs": self.td_error_p95_abs,
            "stability_gae_mean_abs": self.gae_mean_abs,
            "stability_gae_p95_abs": self.gae_p95_abs,
            "stability_return_target_variance": self.return_target_variance,
            "stability_oscillation_loss": self.oscillation_loss,
            "stability_oscillation_loss_valid": float(self.oscillation_loss_valid),
            "stability_oscillation_source": self.oscillation_source,
            "stability_critic_trace_len": float(self.critic_trace_len),
            "stability_n_replan": float(self.n_replan),
            "stability_p_cbf": self.p_cbf,
            "safety_collision_rate": self.collision_rate,
            "safety_score": self.safety_score,
            "unsafe_rate": self.unsafe_rate,
            "safety_valid": int(self.safety_valid),
            "safety_scope": self.safety_scope,
            "safety_missing_reason": self.safety_missing_reason,
            "safety_distance_constraint_available": self.distance_constraint_available,
            "safety_distance_observation_count": self.distance_observation_count,
            "safety_distance_missing_count": self.distance_missing_count,
            "safety_distance_constraint_missing_reason": (
                self.distance_constraint_missing_reason
            ),
            "stability_score_variance": self.score_variance,
            "stability_violation_rate": self.stability_violation_rate,
        }
