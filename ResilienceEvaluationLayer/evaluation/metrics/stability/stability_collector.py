#!/usr/bin/env python3

"""
Stability Collector

Collects policy stability metrics from StateEncoder, SafetyMonitor, and planner info.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from habitat_llm.evaluation.metrics.collection import EpisodeCollectionContext
from habitat_llm.evaluation.reporting.tabular import (
    read_csv_rows,
    write_csv_rows_exact,
)
from habitat_llm.evaluation.perturbation.bank import (
    DEFAULT_TEXT_DOSE_TOLERANCE as FORMAL_TEXT_DOSE_TOLERANCE,
    STRESS_FAMILY,
    STRESS_GRID as FORMAL_STRESS_GRID,
)

from .stability_metrics import StabilityMetrics

logger = logging.getLogger(__name__)

_LAMBDA_TOLERANCE = 1.0e-9


def _optional_finite_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    if len(arr) == 1:
        return arr[0]
    q = min(max(q, 0.0), 1.0)
    idx = q * (len(arr) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return arr[lo]
    frac = idx - lo
    return arr[lo] * (1 - frac) + arr[hi] * frac


def _normalized_levenshtein(a: Sequence[str], b: Sequence[str]) -> float:
    """Normalized Levenshtein distance on action token sequences (∈ [0,1])."""
    m, n = len(a), len(b)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return 1.0
    # Use space-efficient DP
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[n] / max(m, n)


@dataclass
class _RawRun:
    """One perturbed rollout's summary row used by the β aggregator."""

    anchor_id: str
    perturbation_type: str
    perturbation_seed: int
    stress_lambda: Optional[float] = None
    stress_family: str = "legacy"
    stress_variant: str = ""
    stress_bank_path: str = ""
    stress_realized: Optional[float] = None
    stress_text_dose: Optional[float] = None
    stress_alignment_valid: bool = False
    strict_stress_contract: bool = False
    value_delta_variance: float = 0.0
    oscillation_loss: float = 0.0
    oscillation_loss_valid: bool = False
    trajectory_loss: Optional[float] = None
    trajectory_loss_valid: bool = False
    perturbation_norm: float = 1.0
    perturbation_realization_valid: bool = True
    perturbation_realization_reason: str = ""
    action_sequence: List[str] = field(default_factory=list)
    task_state_success: float = 0.0
    task_progress: float = 0.0
    c_rec: float = 0.0
    outcome_valid: bool = False
    c_rec_valid: bool = False
    outcome_missing_reason: str = "missing_rollout_outcome_evidence"
    task_family: str = "other"
    is_clean: bool = False


@dataclass
class BetaAggregate:
    """Aggregated β for a single anchor across a perturbation neighborhood."""

    anchor_id: str
    task_family: str
    beta_trajectory: Optional[float] = None
    beta_vv: Optional[float] = None
    beta_out: Optional[float] = None
    beta_hat: Optional[float] = None
    beta_oscillation: Optional[float] = None
    beta_action: Optional[float] = None
    beta_progress: Optional[float] = None
    beta_success: Optional[float] = None
    n_clean: int = 0
    n_perturbed: int = 0
    clean_ref_valid: bool = True
    clean_ref_strategy: str = "medoid"
    beta_neighborhood_valid: bool = True
    missing_reason: str = ""
    beta_scope: str = "neighborhood"
    perturbation_type: str = ""
    stress_lambda: Optional[float] = None
    stress_family: str = "legacy"
    stress_realized: Optional[float] = None
    stress_text_dose: Optional[float] = None
    stress_alignment_valid: bool = False
    n_paired_seeds: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "task_family": self.task_family,
            "beta_trajectory": self.beta_trajectory,
            "beta_vv": self.beta_vv,
            "beta_out": self.beta_out,
            "beta_hat": self.beta_hat,
            "beta_oscillation": self.beta_oscillation,
            "beta_action": self.beta_action,
            "beta_progress": self.beta_progress,
            "beta_success": self.beta_success,
            "n_clean": int(self.n_clean),
            "n_perturbed": int(self.n_perturbed),
            "clean_ref_valid": bool(self.clean_ref_valid),
            "clean_ref_strategy": self.clean_ref_strategy,
            "beta_neighborhood_valid": bool(self.beta_neighborhood_valid),
            "missing_reason": self.missing_reason,
            "beta_scope": self.beta_scope,
            "perturbation_type": self.perturbation_type,
            "stress_lambda": self.stress_lambda,
            "stress_family": self.stress_family,
            "stress_realized": self.stress_realized,
            "stress_text_dose": self.stress_text_dose,
            "stress_alignment_valid": int(self.stress_alignment_valid),
            "stability_n_paired_seeds": int(self.n_paired_seeds),
        }


@dataclass
class BetaFamilyAggregate:
    """Paper Eq. (6) supremum over configured episodes in a task family."""

    task_family: str
    beta_family: Optional[float] = None
    beta_family_valid: bool = False
    n_anchors: int = 0
    n_valid_anchors: int = 0
    missing_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_family": self.task_family,
            "stability_beta_family": self.beta_family,
            "stability_beta_family_valid": int(self.beta_family_valid),
            "stability_beta_family_n_anchors": int(self.n_anchors),
            "stability_beta_family_n_valid_anchors": int(
                self.n_valid_anchors
            ),
            "stability_beta_family_missing_reason": self.missing_reason,
        }


class StabilityCollector:
    """
    Collects Stability metrics from Critic's StateEncoder, SafetyMonitor, and planner.
    """

    dimension = "stability"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.use_saycan_scores = self.config.get("use_saycan_scores", False)
        self.value_variance_window = self.config.get("value_variance_window", 10)

    def collect(
        self,
        critic: Optional[Any],
        planner_infos: List[Dict[str, Any]],
        safety_monitor: Optional[Any],
        planner: Optional[Any] = None,
        safety_summary: Optional[Mapping[str, Any]] = None,
    ) -> StabilityMetrics:
        """
        Collect Stability metrics from multiple sources.

        Args:
            critic: A2CCritic instance with StateEncoder
            planner_infos: List of planner info dicts from episode
            safety_monitor: SafetyMonitor instance
            planner: Optional planner for SayCan scores

        Returns:
            StabilityMetrics instance
        """
        metrics = StabilityMetrics()

        # 1. Extract value variance from Critic's StateEncoder
        if critic is not None:
            try:
                if hasattr(critic, "get_recent_value_statistics"):
                    value_stats = critic.get_recent_value_statistics(
                        window=self.value_variance_window
                    )
                    value_history = list(value_stats.get("value_history", []))
                    metrics.value_history = value_history
                    if value_history:
                        metrics.value_variance = float(
                            value_stats.get("value_variance", 0.0)
                        )
                    if len(value_history) > 1:
                        metrics.value_delta_variance = float(
                            value_stats.get("value_delta_variance", 0.0)
                        )
                else:
                    metrics.value_variance = critic.compute_value_variance(
                        window=self.value_variance_window
                    )
                    if hasattr(critic, "compute_value_delta_variance"):
                        metrics.value_delta_variance = critic.compute_value_delta_variance(
                            window=self.value_variance_window
                        )
                logger.debug(
                    "Extracted stability value stats: variance=%s delta_variance=%s",
                    metrics.value_variance,
                    metrics.value_delta_variance,
                )
            except Exception as exc:
                logger.warning("Failed to extract value_variance from Critic: %s", exc)
                metrics.value_variance = None
                metrics.value_delta_variance = None

            try:
                if hasattr(critic, "get_stability_trace_summary"):
                    trace_summary = critic.get_stability_trace_summary()
                    metrics.oscillation_loss_valid = bool(
                        trace_summary.get("oscillation_loss_valid", False)
                    )
                    if metrics.oscillation_loss_valid:
                        metrics.td_error_mean_abs = float(
                            trace_summary.get("td_error_mean_abs", 0.0)
                        )
                        metrics.td_error_p95_abs = float(
                            trace_summary.get("td_error_p95_abs", 0.0)
                        )
                        metrics.gae_mean_abs = float(
                            trace_summary.get("gae_mean_abs", 0.0)
                        )
                        metrics.gae_p95_abs = float(
                            trace_summary.get("gae_p95_abs", 0.0)
                        )
                        metrics.return_target_variance = float(
                            trace_summary.get("return_target_variance", 0.0)
                        )
                        metrics.oscillation_loss = float(
                            trace_summary.get("oscillation_loss", 0.0)
                        )
                    metrics.oscillation_source = str(
                        trace_summary.get("oscillation_source", "")
                    )
                    metrics.critic_trace_len = int(
                        trace_summary.get("critic_trace_len", 0) or 0
                    )
            except Exception as exc:
                logger.warning(
                    "Failed to extract critic stability trace summary: %s", exc
                )

        # 2. Extract N_replan from planner_infos
        if planner_infos:
            try:
                last_info = planner_infos[-1]
                replanning_count = last_info.get("replanning_count", {})

                # Handle both dict and int formats
                if isinstance(replanning_count, Mapping):
                    # Max replanning count across agents
                    metrics.n_replan = max(replanning_count.values()) if replanning_count else 0
                else:
                    metrics.n_replan = int(replanning_count) if replanning_count else 0

                # Track replan steps
                previous_count = None
                for info in planner_infos:
                    step = info.get(
                        "total_step_count",
                        info.get("sim_step_count", info.get("step", 0)),
                    )
                    replanned = info.get("replanned")
                    if isinstance(replanned, Mapping):
                        if any(bool(value) for value in replanned.values()):
                            metrics.replan_steps.append(step)
                        continue
                    if replanned is not None:
                        if bool(replanned):
                            metrics.replan_steps.append(step)
                        continue
                    replan_count = info.get("replanning_count", {})
                    if isinstance(replan_count, Mapping):
                        current_count = max(replan_count.values()) if replan_count else 0
                        if previous_count is not None and current_count > previous_count:
                            metrics.replan_steps.append(step)
                        previous_count = current_count
                    else:
                        current_count = int(replan_count or 0)
                        if previous_count is not None and current_count > previous_count:
                            metrics.replan_steps.append(step)
                        previous_count = current_count

                logger.debug("Extracted n_replan: %d", metrics.n_replan)
            except Exception as exc:
                logger.warning("Failed to extract n_replan from planner_infos: %s", exc)
                metrics.n_replan = 0

        # 3. Extract P_cbf from SafetyMonitor
        if safety_summary is not None or safety_monitor is not None:
            try:
                if safety_summary is not None:
                    safety_stats = dict(safety_summary)
                elif hasattr(safety_monitor, "get_statistics"):
                    safety_stats = safety_monitor.get_statistics()
                elif hasattr(safety_monitor, "get_summary"):
                    safety_stats = safety_monitor.get_summary()
                else:
                    raise AttributeError(
                        "SafetyMonitor exposes neither get_statistics() nor get_summary()."
                    )
                metrics.p_cbf = _optional_finite_float(
                    safety_stats.get("total_cbf_penalty")
                )
                metrics.collision_rate = _optional_finite_float(
                    safety_stats.get("collision_rate")
                )
                metrics.safety_score = _optional_finite_float(
                    safety_stats.get("safety_score")
                )
                metrics.unsafe_rate = _optional_finite_float(
                    safety_stats.get("unsafe_rate")
                )
                raw_safety_valid = safety_stats.get("safety_valid")
                metrics.safety_valid = (
                    raw_safety_valid is True
                    or str(raw_safety_valid).strip().lower()
                    in {"1", "1.0", "true", "yes"}
                )
                metrics.safety_scope = str(
                    safety_stats.get("safety_scope") or "none"
                )
                metrics.safety_missing_reason = str(
                    safety_stats.get("safety_missing_reason")
                    or ("" if metrics.safety_valid else "missing_safety_validity")
                )
                raw_distance_available = safety_stats.get(
                    "distance_constraint_available"
                )
                metrics.distance_constraint_available = (
                    None
                    if raw_distance_available is None
                    else bool(raw_distance_available)
                )
                metrics.distance_observation_count = int(
                    safety_stats.get("distance_observation_count", 0) or 0
                )
                metrics.distance_missing_count = int(
                    safety_stats.get("distance_missing_count", 0) or 0
                )
                metrics.distance_constraint_missing_reason = str(
                    safety_stats.get("distance_constraint_missing_reason", "")
                    or ""
                )
                logger.debug("Extracted p_cbf: %s", metrics.p_cbf)
            except Exception as exc:
                logger.warning("Failed to extract p_cbf from SafetyMonitor: %s", exc)
                metrics.p_cbf = None
                metrics.collision_rate = None
                metrics.safety_score = None
                metrics.unsafe_rate = None
                metrics.safety_valid = False
                metrics.safety_scope = "none"
                metrics.safety_missing_reason = (
                    f"safety_summary_error:{type(exc).__name__}"
                )

        # 4. Optional: Extract SayCan scores if enabled
        if self.use_saycan_scores and planner is not None:
            self._extract_saycan_scores(planner, metrics)

        # 5. Finalize the explicitly non-beta single-run diagnostic.
        if planner_infos:
            metrics.total_steps = int(
                planner_infos[-1].get("total_step_count", len(planner_infos))
            )
        else:
            metrics.total_steps = 0

        metrics.finalize_episode_proxy()

        return metrics

    def collect_episode(
        self,
        context: EpisodeCollectionContext,
    ) -> StabilityMetrics:
        """Collect through the shared immutable episode context."""

        return self.collect(
            critic=context.critic,
            planner_infos=list(context.planner_infos),
            safety_monitor=None,
            planner=context.planner,
            safety_summary=context.monitor_summary("safety"),
        )

    def _extract_saycan_scores(self, planner: Any, metrics: StabilityMetrics) -> None:
        """
        Extract SayCan-specific stability metrics if available.

        Args:
            planner: Planner instance
            metrics: StabilityMetrics to populate
        """
        if not hasattr(planner, "saycan_analyzer"):
            return

        try:
            saycan_analyzer = planner.saycan_analyzer
            if saycan_analyzer is None:
                return

            # Analyze episode to get stability metrics
            saycan_data = saycan_analyzer.analyze_episode()
            if not saycan_data:
                return

            summary = saycan_data.get("summary", {})
            metrics.score_variance = summary.get("score_variance", None)
            metrics.stability_violations = summary.get("stability_violations", None)
            metrics.stability_violation_rate = summary.get("stability_violation_rate", None)

            logger.info(
                "Extracted SayCan stability: violations=%d, rate=%.3f",
                metrics.stability_violations or 0,
                metrics.stability_violation_rate or 0.0,
            )

        except Exception as exc:
            logger.warning("Failed to extract SayCan stability metrics: %s", exc)

    def collect_from_info(self, info: Dict[str, Any]) -> StabilityMetrics:
        """
        Collect Stability metrics from episode info dict (for backward compatibility).

        Args:
            info: Episode info dictionary

        Returns:
            StabilityMetrics instance
        """
        metrics = StabilityMetrics()

        # Extract only single-rollout evidence. Formal beta fields are produced
        # by aggregate_beta_over_anchors and do not belong in this object.
        metrics.episode_proxy = info.get("stability_episode_proxy")
        metrics.episode_proxy_source = str(
            info.get("stability_episode_proxy_source", "legacy_row")
        )
        metrics.value_variance = info.get("stability_value_variance")
        metrics.value_delta_variance = info.get("stability_value_delta_variance")
        metrics.n_replan = int(info.get("stability_n_replan", 0))
        metrics.p_cbf = _optional_finite_float(info.get("stability_p_cbf"))
        metrics.collision_rate = _optional_finite_float(
            info.get("safety_collision_rate", info.get("collision_rate"))
        )
        metrics.safety_score = _optional_finite_float(info.get("safety_score"))
        metrics.unsafe_rate = _optional_finite_float(info.get("unsafe_rate"))
        raw_safety_valid = info.get("safety_valid")
        metrics.safety_valid = (
            raw_safety_valid is True
            or str(raw_safety_valid).strip().lower()
            in {"1", "1.0", "true", "yes"}
        )
        metrics.safety_scope = str(info.get("safety_scope") or "none")
        metrics.safety_missing_reason = str(
            info.get("safety_missing_reason")
            or ("" if metrics.safety_valid else "missing_safety_validity")
        )
        raw_distance_available = info.get(
            "safety_distance_constraint_available"
        )
        metrics.distance_constraint_available = (
            None
            if raw_distance_available in (None, "")
            else str(raw_distance_available).strip().lower()
            in {"1", "1.0", "true", "yes"}
        )
        metrics.distance_observation_count = int(
            info.get("safety_distance_observation_count", 0) or 0
        )
        metrics.distance_missing_count = int(
            info.get("safety_distance_missing_count", 0) or 0
        )
        metrics.distance_constraint_missing_reason = str(
            info.get("safety_distance_constraint_missing_reason", "") or ""
        )
        metrics.score_variance = info.get("stability_score_variance", None)
        metrics.stability_violation_rate = info.get("stability_violation_rate", None)
        metrics.total_steps = int(info.get("total_step_count", 0))
        metrics.td_error_mean_abs = info.get("stability_td_error_mean_abs")
        metrics.td_error_p95_abs = info.get("stability_td_error_p95_abs")
        metrics.gae_mean_abs = info.get("stability_gae_mean_abs")
        metrics.gae_p95_abs = info.get("stability_gae_p95_abs")
        metrics.return_target_variance = info.get("stability_return_target_variance")
        metrics.oscillation_loss = info.get("stability_oscillation_loss")
        metrics.oscillation_loss_valid = str(
            info.get("stability_oscillation_loss_valid", 0.0)
        ).strip().lower() in {"1", "1.0", "true", "yes"}
        metrics.oscillation_source = str(
            info.get("stability_oscillation_source", "")
        )
        metrics.critic_trace_len = int(info.get("stability_critic_trace_len", 0) or 0)

        return metrics

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    @staticmethod
    def _row_perturbation_norm(run: _RawRun, epsilon: float) -> float:
        # Aggregation admits only rows with a positive, observed realized dose.
        # The epsilon is numerical regularization, not a substitute for missing
        # realization evidence.
        return abs(float(run.perturbation_norm)) + max(float(epsilon), 0.0)

    def _compute_beta_components(
        self,
        clean: Sequence[_RawRun],
        perturbed: Sequence[_RawRun],
        *,
        alpha: float,
        include_rebound: bool,
    ) -> Dict[str, Optional[float]]:
        if not clean or not perturbed:
            return {
                "beta_trajectory": None,
                "beta_vv": None,
                "beta_out": None,
                "beta_hat": None,
                "beta_oscillation": None,
                "beta_action": None,
                "beta_progress": None,
                "beta_success": None,
            }

        epsilon = float(self.config.get("perturbation_norm_epsilon", 1e-6))
        ref_vdv = self._mean([r.value_delta_variance for r in clean])
        ref_loss_values = [
            r.oscillation_loss for r in clean if r.oscillation_loss_valid
        ]
        ref_oscillation_loss = (
            self._mean(ref_loss_values) if ref_loss_values else ref_vdv
        )
        ref_action = self._clean_action_medoid(clean)
        ref_sr = self._mean([r.task_state_success for r in clean])
        ref_progress = self._mean([r.task_progress for r in clean])
        ref_crec = self._mean([r.c_rec for r in clean])
        clean_trajectory_losses = [
            float(run.trajectory_loss)
            for run in clean
            if run.trajectory_loss_valid and run.trajectory_loss is not None
        ]
        ref_trajectory_loss = (
            self._mean(clean_trajectory_losses)
            if clean_trajectory_losses
            else None
        )

        per_run_trajectory: List[float] = []
        per_run_osc: List[float] = []
        per_run_out: List[float] = []
        per_run_action: List[float] = []
        per_run_success: List[float] = []
        per_run_progress: List[float] = []
        for run in perturbed:
            norm = self._row_perturbation_norm(run, epsilon)
            if (
                ref_trajectory_loss is not None
                and run.trajectory_loss_valid
                and run.trajectory_loss is not None
            ):
                per_run_trajectory.append(
                    abs(float(run.trajectory_loss) - ref_trajectory_loss) / norm
                )
            if run.oscillation_loss_valid or ref_loss_values:
                loss_delta = abs(float(run.oscillation_loss) - ref_oscillation_loss)
            else:
                loss_delta = max(0.0, run.value_delta_variance - ref_vdv)
            beta_osc = loss_delta / norm
            lev = _normalized_levenshtein(run.action_sequence, ref_action)
            delta_sr = abs(run.task_state_success - ref_sr)
            delta_progress = abs(run.task_progress - ref_progress)
            per_run_osc.append(beta_osc)
            per_run_action.append(lev / norm)
            per_run_success.append(delta_sr / norm)
            per_run_progress.append(delta_progress / norm)
            if include_rebound:
                delta_crec = abs(run.c_rec - ref_crec)
                per_run_out.append(
                    (
                        alpha * lev
                        + ((1.0 - alpha) / 3.0) * delta_sr
                        + ((1.0 - alpha) / 3.0) * delta_progress
                        + ((1.0 - alpha) / 3.0) * delta_crec
                    )
                    / norm
                )
            else:
                per_run_out.append(
                    (
                        alpha * lev
                        + ((1.0 - alpha) / 2.0) * delta_sr
                        + ((1.0 - alpha) / 2.0) * delta_progress
                    )
                    / norm
                )

        beta_trajectory = max(per_run_trajectory) if per_run_trajectory else None
        beta_oscillation = _percentile(per_run_osc, 0.95)
        beta_out = _percentile(per_run_out, 0.5)
        return {
            # ``beta_vv`` is retained as a CSV compatibility alias. The formal
            # Eq. (6) value is the supremum trajectory-loss sensitivity.
            "beta_trajectory": beta_trajectory,
            "beta_vv": beta_trajectory,
            "beta_out": float(beta_out),
            "beta_hat": beta_trajectory,
            "beta_oscillation": float(beta_oscillation),
            "beta_action": float(_percentile(per_run_action, 0.5)),
            "beta_progress": float(_percentile(per_run_progress, 0.5)),
            "beta_success": float(_percentile(per_run_success, 0.5)),
        }

    # ==================================================================
    # Neighborhood β̂ aggregation (§2 of the resilience design doc)
    # ==================================================================
    def aggregate_beta_over_anchors(
        self,
        raw_runs: Iterable[Mapping[str, Any]],
        alpha: float = 0.5,
        perturbation_filter: Optional[str] = None,
        include_rebound: Optional[bool] = None,
    ) -> Tuple[Dict[str, BetaAggregate], List[Dict[str, Any]]]:
        r"""Aggregate multi-run rollouts into per-anchor :math:`\hat\beta`.

        Parameters
        ----------
        raw_runs:
            Iterable of dict rows (e.g. from reading ``resilience_raw.csv``).
            Each row must carry: ``anchor_id``, ``perturbation_type``,
            ``perturbation_seed``, ``value_delta_variance``,
            ``task_state_success``, ``task_percent_complete``, ``task_family``, and
            ``action_sequence`` (a pipe-delimited string or list of action
            names). Clean rollouts use ``perturbation_type ∈ {"", "clean"}``.
        alpha:
            Weight for the normalized Levenshtein term in ``beta_out``. The
            remaining mass splits between success and progress deltas by default.
        Returns
        -------
        per_anchor:
            Mapping ``anchor_id -> BetaAggregate``.
        raw_rows_for_stats:
            A flattened list of rows suitable for KS / Mann-Whitney tests.
            Each row carries ``anchor_id``, ``task_family``, ``beta_vv``,
            ``beta_out``, and is grouped in downstream scripts.
        """
        runs = self._normalize_raw_runs(raw_runs, perturbation_filter=perturbation_filter)
        if not runs:
            return {}, []
        include_rebound = (
            bool(self.config.get("include_rebound_in_neighborhood", False))
            if include_rebound is None
            else bool(include_rebound)
        )

        by_anchor: Dict[str, List[_RawRun]] = {}
        for run in runs:
            by_anchor.setdefault(run.anchor_id, []).append(run)

        aggregates: Dict[str, BetaAggregate] = {}
        stats_rows: List[Dict[str, Any]] = []
        for anchor_id, anchor_runs in by_anchor.items():
            strict_runs = [
                run for run in anchor_runs if run.strict_stress_contract
            ]
            if strict_runs:
                stress_aggregates, _ = (
                    self._aggregate_formal_beta_by_stress_for_anchor(
                        anchor_id=anchor_id,
                        runs=strict_runs,
                        alpha=alpha,
                        include_rebound=include_rebound,
                    )
                )
                stress_families = {
                    key[1] for key in stress_aggregates.keys()
                }
                task_family = strict_runs[0].task_family
                if len(stress_families) != 1:
                    headline = BetaAggregate(
                        anchor_id=anchor_id,
                        task_family=task_family,
                        clean_ref_valid=False,
                        clean_ref_strategy="episode_seed_exact",
                        beta_neighborhood_valid=False,
                        missing_reason="mixed_formal_stress_family",
                        beta_scope="invalid_formal_paired_neighborhood",
                    )
                else:
                    stress_family = next(iter(stress_families))
                    clean_aggregate = stress_aggregates.get(
                        (anchor_id, stress_family, 0.0)
                    )
                    positive = [
                        stress_aggregates.get(
                            (anchor_id, stress_family, float(stress_lambda))
                        )
                        for stress_lambda in FORMAL_STRESS_GRID[1:]
                    ]
                    positive_complete = bool(
                        clean_aggregate is not None
                        and clean_aggregate.beta_neighborhood_valid
                        and all(
                            member is not None
                            and member.beta_neighborhood_valid
                            and member.beta_trajectory is not None
                            for member in positive
                        )
                    )
                    valid_positive = [
                        member for member in positive if member is not None
                    ]
                    if positive_complete:
                        trajectory_values = [
                            float(member.beta_trajectory)
                            for member in valid_positive
                            if member.beta_trajectory is not None
                        ]

                        def _component_values(name: str) -> List[float]:
                            return [
                                float(value)
                                for member in valid_positive
                                for value in [getattr(member, name)]
                                if value is not None
                            ]

                        headline = BetaAggregate(
                            anchor_id=anchor_id,
                            task_family=task_family,
                            beta_trajectory=max(trajectory_values),
                            beta_vv=max(trajectory_values),
                            beta_hat=max(trajectory_values),
                            beta_out=_percentile(
                                _component_values("beta_out"), 0.5
                            ),
                            beta_oscillation=_percentile(
                                _component_values("beta_oscillation"), 0.95
                            ),
                            beta_action=_percentile(
                                _component_values("beta_action"), 0.5
                            ),
                            beta_progress=_percentile(
                                _component_values("beta_progress"), 0.5
                            ),
                            beta_success=_percentile(
                                _component_values("beta_success"), 0.5
                            ),
                            n_clean=int(clean_aggregate.n_clean),
                            n_perturbed=sum(
                                int(member.n_perturbed)
                                for member in valid_positive
                            ),
                            clean_ref_valid=True,
                            clean_ref_strategy="episode_seed_exact",
                            beta_neighborhood_valid=True,
                            beta_scope=(
                                "paper_eq6_formal_paired_"
                                "positive_lambda_sup"
                            ),
                            perturbation_type=stress_family,
                            stress_family=stress_family,
                            stress_alignment_valid=True,
                            n_paired_seeds=int(
                                clean_aggregate.n_paired_seeds
                            ),
                        )
                    else:
                        reasons = sorted(
                            {
                                member.missing_reason
                                for member in (
                                    [clean_aggregate] + valid_positive
                                )
                                if member is not None
                                and member.missing_reason
                            }
                        )
                        headline = BetaAggregate(
                            anchor_id=anchor_id,
                            task_family=task_family,
                            n_clean=(
                                int(clean_aggregate.n_clean)
                                if clean_aggregate is not None
                                else 0
                            ),
                            n_perturbed=sum(
                                int(member.n_perturbed)
                                for member in valid_positive
                            ),
                            clean_ref_valid=bool(
                                clean_aggregate
                                and clean_aggregate.clean_ref_valid
                            ),
                            clean_ref_strategy="episode_seed_exact",
                            beta_neighborhood_valid=False,
                            missing_reason=(
                                ";".join(reasons)
                                or "incomplete_episode_seed_stress_grid"
                            ),
                            beta_scope="invalid_formal_paired_neighborhood",
                            perturbation_type=stress_family,
                            stress_family=stress_family,
                            stress_alignment_valid=False,
                            n_paired_seeds=(
                                int(clean_aggregate.n_paired_seeds)
                                if clean_aggregate is not None
                                else 0
                            ),
                        )
                aggregates[anchor_id] = headline
                stats_rows.append(headline.to_dict())
                # Frozen-family rows must never fall back to the legacy clean
                # mean/neighborhood implementation.
                continue
            clean = [r for r in anchor_runs if r.is_clean]
            perturbed = [r for r in anchor_runs if not r.is_clean]
            if not perturbed:
                task_family = clean[0].task_family if clean else "other"
                aggregates[anchor_id] = BetaAggregate(
                    anchor_id=anchor_id,
                    task_family=task_family,
                    n_clean=len(clean),
                    n_perturbed=0,
                    clean_ref_valid=bool(clean)
                    and all(run.trajectory_loss_valid for run in clean),
                    beta_neighborhood_valid=False,
                    missing_reason="missing_perturbed_neighborhood",
                    beta_scope="invalid_missing_perturbed_neighborhood",
                )
                stats_rows.append(aggregates[anchor_id].to_dict())
                continue
            if not clean:
                logger.warning(
                    "Anchor %s has no clean baseline run; marking neighborhood beta invalid.",
                    anchor_id,
                )
                task_family = perturbed[0].task_family if perturbed else "other"
                aggregates[anchor_id] = BetaAggregate(
                    anchor_id=anchor_id,
                    task_family=task_family,
                    n_clean=0,
                    n_perturbed=len(perturbed),
                    clean_ref_valid=False,
                    beta_neighborhood_valid=False,
                    missing_reason="missing_clean_reference",
                    beta_scope=(
                        f"invalid_missing_clean_M={len(perturbed)}_Pi={len({r.perturbation_type for r in perturbed})}"
                    ),
                )
                stats_rows.append(aggregates[anchor_id].to_dict())
                continue
            perturbation_realization_complete = all(
                run.perturbation_realization_valid for run in perturbed
            )
            outcome_evidence_complete = all(
                run.outcome_valid for run in clean + perturbed
            )
            rebound_evidence_complete = (
                all(run.c_rec_valid for run in clean + perturbed)
                if include_rebound
                else True
            )
            components = (
                self._compute_beta_components(
                    clean,
                    perturbed,
                    alpha=alpha,
                    include_rebound=include_rebound,
                )
                if perturbation_realization_complete
                and outcome_evidence_complete
                and rebound_evidence_complete
                else self._compute_beta_components(
                    clean,
                    [],
                    alpha=alpha,
                    include_rebound=include_rebound,
                )
            )
            clean_trace_complete = all(run.trajectory_loss_valid for run in clean)
            perturbed_trace_complete = all(
                run.trajectory_loss_valid for run in perturbed
            )
            beta_valid = bool(
                components["beta_hat"] is not None
                and clean_trace_complete
                and perturbed_trace_complete
                and perturbation_realization_complete
                and outcome_evidence_complete
                and rebound_evidence_complete
            )
            if not perturbation_realization_complete:
                missing_reason = "perturbation_realization_invalid"
            elif not outcome_evidence_complete:
                missing_reason = "rollout_outcome_evidence_invalid"
            elif not rebound_evidence_complete:
                missing_reason = "rebound_evidence_invalid"
            elif not any(run.trajectory_loss_valid for run in clean):
                missing_reason = "clean_stage_trajectory_loss_missing"
            elif not clean_trace_complete:
                missing_reason = "clean_stage_trajectory_loss_incomplete"
            elif not any(run.trajectory_loss_valid for run in perturbed):
                missing_reason = "perturbed_stage_trajectory_loss_missing"
            elif not perturbed_trace_complete:
                missing_reason = "perturbed_stage_trajectory_loss_incomplete"
            else:
                missing_reason = ""

            task_family = perturbed[0].task_family if perturbed else "other"
            aggregates[anchor_id] = BetaAggregate(
                anchor_id=anchor_id,
                task_family=task_family,
                beta_trajectory=(
                    components["beta_trajectory"] if beta_valid else None
                ),
                beta_vv=components["beta_vv"] if beta_valid else None,
                beta_out=components["beta_out"],
                beta_hat=components["beta_hat"] if beta_valid else None,
                beta_oscillation=components["beta_oscillation"],
                beta_action=components["beta_action"],
                beta_progress=components["beta_progress"],
                beta_success=components["beta_success"],
                n_clean=len(clean),
                n_perturbed=len(perturbed),
                clean_ref_valid=clean_trace_complete,
                clean_ref_strategy="mean_stage_trajectory_loss",
                beta_neighborhood_valid=beta_valid,
                missing_reason=missing_reason,
                beta_scope=(
                    f"paper_eq6_stage_mahalanobis_sup_M={len(perturbed)}_Pi={len({r.perturbation_type for r in perturbed})}"
                ),
            )
            stats_rows.append(aggregates[anchor_id].to_dict())

        return aggregates, stats_rows

    def _aggregate_beta_subset(
        self,
        *,
        anchor_id: str,
        clean: Sequence[_RawRun],
        perturbed: Sequence[_RawRun],
        alpha: float,
        include_rebound: bool,
        beta_scope: str,
        perturbation_type: str = "",
        stress_lambda: Optional[float] = None,
    ) -> BetaAggregate:
        """Build one auditable beta from a clean reference and one subset."""

        task_family = (
            perturbed[0].task_family
            if perturbed
            else (clean[0].task_family if clean else "other")
        )
        if not clean:
            return BetaAggregate(
                anchor_id=anchor_id,
                task_family=task_family,
                n_clean=0,
                n_perturbed=len(perturbed),
                clean_ref_valid=False,
                beta_neighborhood_valid=False,
                missing_reason="missing_clean_reference",
                beta_scope=f"invalid_{beta_scope}",
                perturbation_type=perturbation_type,
                stress_lambda=stress_lambda,
            )
        if not perturbed:
            return BetaAggregate(
                anchor_id=anchor_id,
                task_family=task_family,
                n_clean=len(clean),
                n_perturbed=0,
                clean_ref_valid=all(run.trajectory_loss_valid for run in clean),
                beta_neighborhood_valid=False,
                missing_reason="missing_perturbed_neighborhood",
                beta_scope=f"invalid_{beta_scope}",
                perturbation_type=perturbation_type,
                stress_lambda=stress_lambda,
            )

        realization_complete = all(
            run.perturbation_realization_valid for run in perturbed
        )
        outcome_complete = all(run.outcome_valid for run in clean + perturbed)
        rebound_complete = (
            all(run.c_rec_valid for run in clean + perturbed)
            if include_rebound
            else True
        )
        clean_trace_complete = all(run.trajectory_loss_valid for run in clean)
        perturbed_trace_complete = all(
            run.trajectory_loss_valid for run in perturbed
        )
        inputs_complete = bool(
            realization_complete
            and outcome_complete
            and rebound_complete
            and clean_trace_complete
            and perturbed_trace_complete
        )
        components = self._compute_beta_components(
            clean,
            perturbed if inputs_complete else (),
            alpha=alpha,
            include_rebound=include_rebound,
        )
        beta_valid = bool(
            inputs_complete and components["beta_hat"] is not None
        )
        if not realization_complete:
            missing_reason = "perturbation_realization_invalid"
        elif not outcome_complete:
            missing_reason = "rollout_outcome_evidence_invalid"
        elif not rebound_complete:
            missing_reason = "rebound_evidence_invalid"
        elif not any(run.trajectory_loss_valid for run in clean):
            missing_reason = "clean_stage_trajectory_loss_missing"
        elif not clean_trace_complete:
            missing_reason = "clean_stage_trajectory_loss_incomplete"
        elif not any(run.trajectory_loss_valid for run in perturbed):
            missing_reason = "perturbed_stage_trajectory_loss_missing"
        elif not perturbed_trace_complete:
            missing_reason = "perturbed_stage_trajectory_loss_incomplete"
        else:
            missing_reason = ""

        return BetaAggregate(
            anchor_id=anchor_id,
            task_family=task_family,
            beta_trajectory=(
                components["beta_trajectory"] if beta_valid else None
            ),
            beta_vv=components["beta_vv"] if beta_valid else None,
            beta_out=components["beta_out"],
            beta_hat=components["beta_hat"] if beta_valid else None,
            beta_oscillation=components["beta_oscillation"],
            beta_action=components["beta_action"],
            beta_progress=components["beta_progress"],
            beta_success=components["beta_success"],
            n_clean=len(clean),
            n_perturbed=len(perturbed),
            clean_ref_valid=clean_trace_complete,
            clean_ref_strategy="mean_stage_trajectory_loss",
            beta_neighborhood_valid=beta_valid,
            missing_reason=missing_reason,
            beta_scope=beta_scope,
            perturbation_type=perturbation_type,
            stress_lambda=stress_lambda,
        )

    def _aggregate_formal_beta_pairs(
        self,
        *,
        anchor_id: str,
        pairs: Sequence[Tuple[_RawRun, _RawRun]],
        alpha: float,
        include_rebound: bool,
        stress_family: str,
        stress_lambda: float,
    ) -> BetaAggregate:
        """Aggregate exact episode×seed clean/stress comparisons.

        Each stressed rollout is first compared with the clean rollout from
        the same seed. Only then are seed-local sensitivities summarized, so a
        clean mean from another seed cannot change a row's denominator.
        """

        task_family = pairs[0][1].task_family if pairs else "other"
        if not pairs:
            return BetaAggregate(
                anchor_id=anchor_id,
                task_family=task_family,
                n_clean=0,
                n_perturbed=0,
                clean_ref_valid=False,
                beta_neighborhood_valid=False,
                missing_reason="incomplete_episode_seed_stress_grid",
                beta_scope="invalid_paired_episode_seed_stress_grid",
                stress_family=stress_family,
                stress_lambda=float(stress_lambda),
                stress_realized=float(stress_lambda),
                stress_alignment_valid=False,
            )

        per_pair = [
            self._aggregate_beta_subset(
                anchor_id=anchor_id,
                clean=[clean],
                perturbed=[perturbed],
                alpha=alpha,
                include_rebound=include_rebound,
                beta_scope=(
                    f"paired_episode_seed_lambda={stress_lambda:g}"
                ),
                perturbation_type=perturbed.perturbation_type,
                stress_lambda=float(stress_lambda),
            )
            for clean, perturbed in pairs
        ]
        invalid = [item for item in per_pair if not item.beta_neighborhood_valid]
        if invalid:
            reasons = sorted({item.missing_reason for item in invalid})
            return BetaAggregate(
                anchor_id=anchor_id,
                task_family=task_family,
                n_clean=len(pairs),
                n_perturbed=len(pairs),
                clean_ref_valid=all(item.clean_ref_valid for item in per_pair),
                clean_ref_strategy="episode_seed_exact",
                beta_neighborhood_valid=False,
                missing_reason=";".join(reasons),
                beta_scope=(
                    f"invalid_paired_episode_seed_lambda={stress_lambda:g}"
                ),
                stress_family=stress_family,
                stress_lambda=float(stress_lambda),
                stress_realized=float(stress_lambda),
                stress_text_dose=_percentile(
                    [
                        float(perturbed.stress_text_dose)
                        for _, perturbed in pairs
                        if perturbed.stress_text_dose is not None
                    ],
                    0.5,
                ),
                stress_alignment_valid=False,
                n_paired_seeds=len(pairs),
            )

        def _values(field_name: str) -> List[float]:
            return [
                float(value)
                for item in per_pair
                for value in [getattr(item, field_name)]
                if value is not None
            ]

        trajectory_values = _values("beta_trajectory")
        return BetaAggregate(
            anchor_id=anchor_id,
            task_family=task_family,
            beta_trajectory=(
                max(trajectory_values) if trajectory_values else None
            ),
            beta_vv=max(trajectory_values) if trajectory_values else None,
            beta_out=_percentile(_values("beta_out"), 0.5),
            beta_hat=max(trajectory_values) if trajectory_values else None,
            beta_oscillation=_percentile(
                _values("beta_oscillation"), 0.95
            ),
            beta_action=_percentile(_values("beta_action"), 0.5),
            beta_progress=_percentile(_values("beta_progress"), 0.5),
            beta_success=_percentile(_values("beta_success"), 0.5),
            n_clean=len(pairs),
            n_perturbed=len(pairs),
            clean_ref_valid=True,
            clean_ref_strategy="episode_seed_exact",
            beta_neighborhood_valid=bool(trajectory_values),
            missing_reason="" if trajectory_values else "trajectory_loss_missing",
            beta_scope=(
                f"paper_beta_at_lambda={stress_lambda:g}_"
                f"paired_episode_seed_n={len(pairs)}"
            ),
            perturbation_type=stress_family,
            stress_family=stress_family,
            stress_lambda=float(stress_lambda),
            stress_realized=float(stress_lambda),
            stress_text_dose=_percentile(
                [
                    float(perturbed.stress_text_dose)
                    for _, perturbed in pairs
                    if perturbed.stress_text_dose is not None
                ],
                0.5,
            ),
            stress_alignment_valid=all(
                perturbed.stress_alignment_valid for _, perturbed in pairs
            ),
            n_paired_seeds=len(pairs),
        )

    def _aggregate_formal_beta_by_stress_for_anchor(
        self,
        *,
        anchor_id: str,
        runs: Sequence[_RawRun],
        alpha: float,
        include_rebound: bool,
    ) -> Tuple[Dict[Tuple[str, str, float], BetaAggregate], List[Dict[str, Any]]]:
        aggregates: Dict[Tuple[str, str, float], BetaAggregate] = {}
        rows: List[Dict[str, Any]] = []
        by_family: Dict[str, List[_RawRun]] = defaultdict(list)
        for run in runs:
            by_family[run.stress_family].append(run)

        expected = {round(value, 12) for value in FORMAL_STRESS_GRID}
        for stress_family, family_runs in by_family.items():
            task_family = family_runs[0].task_family if family_runs else "other"
            bank_paths = {
                run.stress_bank_path for run in family_runs if run.stress_bank_path
            }
            if len(bank_paths) != 1 or any(
                not run.stress_bank_path for run in family_runs
            ):
                for stress_lambda in FORMAL_STRESS_GRID:
                    aggregate = BetaAggregate(
                        anchor_id=anchor_id,
                        task_family=task_family,
                        clean_ref_valid=False,
                        clean_ref_strategy="episode_seed_exact",
                        beta_neighborhood_valid=False,
                        missing_reason="mixed_or_missing_perturbation_bank_path",
                        beta_scope="invalid_paired_episode_seed_stress_grid",
                        stress_family=stress_family,
                        stress_lambda=float(stress_lambda),
                        stress_realized=float(stress_lambda),
                        stress_alignment_valid=False,
                    )
                    aggregates[(anchor_id, stress_family, float(stress_lambda))] = aggregate
                    rows.append(aggregate.to_dict())
                continue
            by_seed: Dict[int, List[_RawRun]] = defaultdict(list)
            for run in family_runs:
                by_seed[int(run.perturbation_seed)].append(run)
            complete: Dict[int, Dict[float, _RawRun]] = {}
            for seed, seed_runs in by_seed.items():
                by_lambda: Dict[float, List[_RawRun]] = defaultdict(list)
                for run in seed_runs:
                    if run.stress_lambda is not None:
                        by_lambda[round(float(run.stress_lambda), 12)].append(run)
                if (
                    set(by_lambda) == expected
                    and all(len(items) == 1 for items in by_lambda.values())
                    and len(
                        {
                            (
                                items[0].stress_variant,
                                items[0].stress_bank_path,
                            )
                            for items in by_lambda.values()
                        }
                    )
                    == 1
                    and all(
                        items[0].stress_variant and items[0].stress_bank_path
                        for items in by_lambda.values()
                    )
                    and all(
                        items[0].perturbation_realization_valid
                        and items[0].stress_alignment_valid
                        for items in by_lambda.values()
                    )
                ):
                    complete[seed] = {
                        stress_lambda: items[0]
                        for stress_lambda, items in by_lambda.items()
                    }

            if not complete:
                for stress_lambda in FORMAL_STRESS_GRID:
                    aggregate = BetaAggregate(
                        anchor_id=anchor_id,
                        task_family=task_family,
                        n_clean=0,
                        n_perturbed=0,
                        clean_ref_valid=False,
                        clean_ref_strategy="episode_seed_exact",
                        beta_neighborhood_valid=False,
                        missing_reason="incomplete_episode_seed_stress_grid",
                        beta_scope="invalid_paired_episode_seed_stress_grid",
                        stress_family=stress_family,
                        stress_lambda=float(stress_lambda),
                        stress_realized=float(stress_lambda),
                        stress_alignment_valid=False,
                    )
                    aggregates[(anchor_id, stress_family, float(stress_lambda))] = aggregate
                    rows.append(aggregate.to_dict())
                continue

            clean_runs = [levels[0.0] for levels in complete.values()]
            clean_trace_complete = all(
                run.trajectory_loss_valid for run in clean_runs
            )
            clean_outcome_complete = all(run.outcome_valid for run in clean_runs)
            clean_rebound_complete = (
                all(run.c_rec_valid for run in clean_runs)
                if include_rebound
                else True
            )
            clean_valid = bool(
                clean_trace_complete
                and clean_outcome_complete
                and clean_rebound_complete
            )
            if not clean_outcome_complete:
                clean_reason = "rollout_outcome_evidence_invalid"
            elif not clean_rebound_complete:
                clean_reason = "rebound_evidence_invalid"
            elif not clean_trace_complete:
                clean_reason = "clean_stage_trajectory_loss_incomplete"
            else:
                clean_reason = ""
            clean_aggregate = BetaAggregate(
                anchor_id=anchor_id,
                task_family=task_family,
                beta_trajectory=0.0 if clean_valid else None,
                beta_vv=0.0 if clean_valid else None,
                beta_out=0.0 if clean_valid else None,
                beta_hat=0.0 if clean_valid else None,
                beta_oscillation=0.0 if clean_valid else None,
                beta_action=0.0 if clean_valid else None,
                beta_progress=0.0 if clean_valid else None,
                beta_success=0.0 if clean_valid else None,
                n_clean=len(clean_runs),
                n_perturbed=0,
                clean_ref_valid=clean_trace_complete,
                clean_ref_strategy="episode_seed_exact",
                beta_neighborhood_valid=clean_valid,
                missing_reason=clean_reason,
                beta_scope=(
                    f"clean_reference_lambda=0_paired_seed_n={len(clean_runs)}"
                ),
                stress_family=stress_family,
                stress_lambda=0.0,
                stress_realized=0.0,
                stress_text_dose=0.0,
                stress_alignment_valid=True,
                n_paired_seeds=len(clean_runs),
            )
            aggregates[(anchor_id, stress_family, 0.0)] = clean_aggregate
            rows.append(clean_aggregate.to_dict())

            for stress_lambda in FORMAL_STRESS_GRID[1:]:
                pairs = [
                    (levels[0.0], levels[round(float(stress_lambda), 12)])
                    for levels in complete.values()
                ]
                aggregate = self._aggregate_formal_beta_pairs(
                    anchor_id=anchor_id,
                    pairs=pairs,
                    alpha=alpha,
                    include_rebound=include_rebound,
                    stress_family=stress_family,
                    stress_lambda=float(stress_lambda),
                )
                aggregates[(anchor_id, stress_family, float(stress_lambda))] = aggregate
                rows.append(aggregate.to_dict())
        return aggregates, rows

    def aggregate_beta_by_stress(
        self,
        raw_runs: Iterable[Mapping[str, Any]],
        alpha: float = 0.5,
        include_rebound: Optional[bool] = None,
    ) -> Tuple[
        Dict[Union[Tuple[str, float], Tuple[str, str, float]], BetaAggregate],
        List[Dict[str, Any]],
    ]:
        r"""Compute :math:`\beta_f(\lambda)` without cross-lambda leakage.

        Every non-zero stress cell compares only rollouts requested at that
        ``lambda`` with the same anchor's clean reference.  The explicit
        zero-stress reference is defined as beta=0 when its clean trajectory
        and required outcome evidence are complete.
        """

        runs = self._normalize_raw_runs(raw_runs)
        if not runs:
            return {}, []
        include_rebound = (
            bool(self.config.get("include_rebound_in_neighborhood", False))
            if include_rebound is None
            else bool(include_rebound)
        )
        by_anchor: Dict[str, List[_RawRun]] = {}
        for run in runs:
            by_anchor.setdefault(run.anchor_id, []).append(run)

        aggregates: Dict[
            Union[Tuple[str, float], Tuple[str, str, float]], BetaAggregate
        ] = {}
        rows: List[Dict[str, Any]] = []
        for anchor_id, anchor_runs in by_anchor.items():
            strict_runs = [
                run for run in anchor_runs if run.strict_stress_contract
            ]
            if strict_runs:
                strict_aggregates, strict_rows = (
                    self._aggregate_formal_beta_by_stress_for_anchor(
                        anchor_id=anchor_id,
                        runs=strict_runs,
                        alpha=alpha,
                        include_rebound=include_rebound,
                    )
                )
                aggregates.update(strict_aggregates)
                rows.extend(strict_rows)
                # Presence of the frozen contract is intentional. Do not fall
                # back to legacy averaging if it is incomplete or invalid.
                continue
            clean = [run for run in anchor_runs if run.is_clean]
            task_family = clean[0].task_family if clean else (
                anchor_runs[0].task_family if anchor_runs else "other"
            )
            clean_trace_complete = bool(clean) and all(
                run.trajectory_loss_valid for run in clean
            )
            clean_outcome_complete = bool(clean) and all(
                run.outcome_valid for run in clean
            )
            clean_rebound_complete = (
                bool(clean) and all(run.c_rec_valid for run in clean)
                if include_rebound
                else True
            )
            clean_valid = bool(
                clean_trace_complete
                and clean_outcome_complete
                and clean_rebound_complete
            )
            if not clean:
                clean_reason = "missing_clean_reference"
            elif not clean_outcome_complete:
                clean_reason = "rollout_outcome_evidence_invalid"
            elif not clean_rebound_complete:
                clean_reason = "rebound_evidence_invalid"
            elif not clean_trace_complete:
                clean_reason = "clean_stage_trajectory_loss_incomplete"
            else:
                clean_reason = ""
            clean_aggregate = BetaAggregate(
                anchor_id=anchor_id,
                task_family=task_family,
                beta_trajectory=0.0 if clean_valid else None,
                beta_vv=0.0 if clean_valid else None,
                beta_out=0.0 if clean_valid else None,
                beta_hat=0.0 if clean_valid else None,
                beta_oscillation=0.0 if clean_valid else None,
                beta_action=0.0 if clean_valid else None,
                beta_progress=0.0 if clean_valid else None,
                beta_success=0.0 if clean_valid else None,
                n_clean=len(clean),
                n_perturbed=0,
                clean_ref_valid=clean_trace_complete,
                clean_ref_strategy="mean_stage_trajectory_loss",
                beta_neighborhood_valid=clean_valid,
                missing_reason=clean_reason,
                beta_scope="clean_reference_lambda=0",
                stress_lambda=0.0,
            )
            aggregates[(anchor_id, 0.0)] = clean_aggregate
            rows.append(clean_aggregate.to_dict())

            stress_levels = sorted(
                {
                    round(float(run.stress_lambda), 12)
                    for run in anchor_runs
                    if not run.is_clean and run.stress_lambda is not None
                }
            )
            for stress_lambda in stress_levels:
                perturbed = [
                    run
                    for run in anchor_runs
                    if not run.is_clean
                    and run.stress_lambda is not None
                    and round(float(run.stress_lambda), 12) == stress_lambda
                ]
                aggregate = self._aggregate_beta_subset(
                    anchor_id=anchor_id,
                    clean=clean,
                    perturbed=perturbed,
                    alpha=alpha,
                    include_rebound=include_rebound,
                    beta_scope=(
                        f"paper_beta_at_lambda={stress_lambda:g}_"
                        f"M={len(perturbed)}_Pi="
                        f"{len({run.perturbation_type for run in perturbed})}"
                    ),
                    stress_lambda=float(stress_lambda),
                )
                aggregates[(anchor_id, float(stress_lambda))] = aggregate
                rows.append(aggregate.to_dict())
        return aggregates, rows

    def aggregate_beta_by_stress_seed(
        self,
        raw_runs: Iterable[Mapping[str, Any]],
        alpha: float = 0.5,
        include_rebound: Optional[bool] = None,
    ) -> Dict[Tuple[str, str, float, int], BetaAggregate]:
        """Return formal lambda-local beta without cross-seed leakage.

        GE forms its final support set from complete episode x seed paths after
        checking Rebound, safety, lineage, and outcome evidence.  A beta that
        was first pooled over every available seed would therefore retain
        information from seeds that GE later excludes.  Materialize one
        aggregate per ``(episode, stress family, lambda, seed)`` instead; the
        Boundary collector can then narrow the paired set without changing
        the beta carried by any surviving row.
        """

        runs = self._normalize_raw_runs(raw_runs)
        if not runs:
            return {}
        include_rebound = (
            bool(self.config.get("include_rebound_in_neighborhood", False))
            if include_rebound is None
            else bool(include_rebound)
        )
        grouped: Dict[Tuple[str, str, int], List[_RawRun]] = defaultdict(list)
        for run in runs:
            if not run.strict_stress_contract:
                continue
            grouped[
                (
                    run.anchor_id,
                    run.stress_family,
                    int(run.perturbation_seed),
                )
            ].append(run)

        aggregates: Dict[Tuple[str, str, float, int], BetaAggregate] = {}
        for (anchor_id, stress_family, seed), seed_runs in grouped.items():
            seed_aggregates, _ = self._aggregate_formal_beta_by_stress_for_anchor(
                anchor_id=anchor_id,
                runs=seed_runs,
                alpha=alpha,
                include_rebound=include_rebound,
            )
            for key, aggregate in seed_aggregates.items():
                key_anchor, key_family, stress_lambda = key
                if key_family != stress_family:
                    continue
                aggregates[
                    (
                        key_anchor,
                        key_family,
                        round(float(stress_lambda), 12),
                        int(seed),
                    )
                ] = aggregate
        return aggregates

    @staticmethod
    def aggregate_beta_by_family(
        aggregates: Mapping[str, BetaAggregate],
    ) -> Dict[str, BetaFamilyAggregate]:
        """Collapse anchor sensitivities into Eq. (6)'s family supremum.

        The configured episode set is the claimed evaluation set. Dropping an
        invalid anchor could bias a supremum downward, so the family result is
        valid only when every anchor neighborhood in that family is valid.
        """

        grouped: Dict[str, List[BetaAggregate]] = {}
        for aggregate in aggregates.values():
            grouped.setdefault(aggregate.task_family, []).append(aggregate)

        results: Dict[str, BetaFamilyAggregate] = {}
        for family, members in grouped.items():
            valid_members = [
                member
                for member in members
                if member.beta_neighborhood_valid
                and member.beta_hat is not None
            ]
            complete = bool(members) and len(valid_members) == len(members)
            beta_family = (
                max(float(member.beta_hat) for member in valid_members)
                if complete
                else None
            )
            results[family] = BetaFamilyAggregate(
                task_family=family,
                beta_family=beta_family,
                beta_family_valid=complete,
                n_anchors=len(members),
                n_valid_anchors=len(valid_members),
                missing_reason=(
                    "" if complete else "incomplete_anchor_neighborhoods"
                ),
            )
        return results

    def aggregate_beta_by_perturbation(
        self,
        raw_runs: Iterable[Mapping[str, Any]],
        alpha: float = 0.5,
        include_rebound: Optional[bool] = None,
    ) -> Tuple[Dict[Tuple[str, str], BetaAggregate], List[Dict[str, Any]]]:
        """Aggregate beta for each ``(anchor_id, perturbation_type)`` pair.

        This is the correct table for comparing surface rewrite vs filler
        injection. The broader neighborhood aggregate intentionally mixes all
        perturbations for a given anchor and must not be grouped by perturbation
        type in downstream plots.
        """
        runs = self._normalize_raw_runs(raw_runs)
        if not runs:
            return {}, []
        include_rebound = (
            bool(self.config.get("include_rebound_in_neighborhood", False))
            if include_rebound is None
            else bool(include_rebound)
        )

        by_anchor: Dict[str, List[_RawRun]] = {}
        for run in runs:
            by_anchor.setdefault(run.anchor_id, []).append(run)

        aggregates: Dict[Tuple[str, str], BetaAggregate] = {}
        rows: List[Dict[str, Any]] = []
        for anchor_id, anchor_runs in by_anchor.items():
            clean = [r for r in anchor_runs if r.is_clean]
            perturbed_types = sorted(
                {r.perturbation_type for r in anchor_runs if not r.is_clean}
            )
            for perturbation_type in perturbed_types:
                perturbed = [
                    r
                    for r in anchor_runs
                    if (not r.is_clean and r.perturbation_type == perturbation_type)
                ]
                task_family = (
                    perturbed[0].task_family
                    if perturbed
                    else (clean[0].task_family if clean else "other")
                )
                if not clean:
                    aggregate = BetaAggregate(
                        anchor_id=anchor_id,
                        task_family=task_family,
                        n_clean=0,
                        n_perturbed=len(perturbed),
                        clean_ref_valid=False,
                        beta_neighborhood_valid=False,
                        missing_reason="missing_clean_reference",
                        beta_scope=(
                            f"invalid_missing_clean_perturbation={perturbation_type}"
                        ),
                        perturbation_type=perturbation_type,
                    )
                else:
                    perturbation_realization_complete = all(
                        run.perturbation_realization_valid for run in perturbed
                    )
                    components = (
                        self._compute_beta_components(
                            clean,
                            perturbed,
                            alpha=alpha,
                            include_rebound=include_rebound,
                        )
                        if perturbation_realization_complete
                        else self._compute_beta_components(
                            clean,
                            [],
                            alpha=alpha,
                            include_rebound=include_rebound,
                        )
                    )
                    clean_trace_complete = all(
                        run.trajectory_loss_valid for run in clean
                    )
                    perturbed_trace_complete = all(
                        run.trajectory_loss_valid for run in perturbed
                    )
                    beta_valid = bool(
                        components["beta_hat"] is not None
                        and clean_trace_complete
                        and perturbed_trace_complete
                        and perturbation_realization_complete
                    )
                    if not perturbation_realization_complete:
                        missing_reason = "perturbation_realization_invalid"
                    elif not any(run.trajectory_loss_valid for run in clean):
                        missing_reason = "clean_stage_trajectory_loss_missing"
                    elif not clean_trace_complete:
                        missing_reason = "clean_stage_trajectory_loss_incomplete"
                    elif not any(run.trajectory_loss_valid for run in perturbed):
                        missing_reason = "perturbed_stage_trajectory_loss_missing"
                    elif not perturbed_trace_complete:
                        missing_reason = "perturbed_stage_trajectory_loss_incomplete"
                    else:
                        missing_reason = ""
                    aggregate = BetaAggregate(
                        anchor_id=anchor_id,
                        task_family=task_family,
                        beta_trajectory=(
                            components["beta_trajectory"] if beta_valid else None
                        ),
                        beta_vv=components["beta_vv"] if beta_valid else None,
                        beta_out=components["beta_out"],
                        beta_hat=components["beta_hat"] if beta_valid else None,
                        beta_oscillation=components["beta_oscillation"],
                        beta_action=components["beta_action"],
                        beta_progress=components["beta_progress"],
                        beta_success=components["beta_success"],
                        n_clean=len(clean),
                        n_perturbed=len(perturbed),
                        clean_ref_valid=clean_trace_complete,
                        clean_ref_strategy="mean_stage_trajectory_loss",
                        beta_neighborhood_valid=beta_valid,
                        missing_reason=missing_reason,
                        beta_scope=(
                            f"paper_eq6_stage_mahalanobis_sup_perturbation={perturbation_type}_M={len(perturbed)}"
                        ),
                        perturbation_type=perturbation_type,
                    )
                aggregates[(anchor_id, perturbation_type)] = aggregate
                rows.append(aggregate.to_dict())
        return aggregates, rows

    @staticmethod
    def _clean_action_medoid(clean_runs: Sequence[_RawRun]) -> Sequence[str]:
        """Return the clean action sequence closest to all other clean runs."""
        if not clean_runs:
            return []
        if len(clean_runs) == 1:
            return clean_runs[0].action_sequence
        best_idx = 0
        best_score = float("inf")
        for idx, candidate in enumerate(clean_runs):
            score = 0.0
            for other in clean_runs:
                score += _normalized_levenshtein(
                    candidate.action_sequence,
                    other.action_sequence,
                )
            score /= max(1, len(clean_runs) - 1)
            if score < best_score:
                best_idx = idx
                best_score = score
        return clean_runs[best_idx].action_sequence

    # ------------------------------------------------------------------
    # Row normalization + IO helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_raw_runs(
        raw_runs: Iterable[Mapping[str, Any]],
        perturbation_filter: Optional[str] = None,
    ) -> List[_RawRun]:
        out: List[_RawRun] = []
        for row in raw_runs:
            if not isinstance(row, Mapping):
                continue
            ptype = str(row.get("perturbation_type", "") or "").strip().lower()
            if perturbation_filter and ptype != perturbation_filter.lower():
                continue
            is_clean = ptype in ("", "clean", "baseline")
            actions_raw = row.get("action_sequence", "")
            if isinstance(actions_raw, str):
                actions: List[str] = [
                    tok for tok in actions_raw.split("|") if tok
                ]
            elif isinstance(actions_raw, (list, tuple)):
                actions = [str(t) for t in actions_raw]
            else:
                actions = []
            try:
                vdv = float(
                    row.get(
                        "value_delta_variance",
                        row.get("stability_value_delta_variance", 0.0),
                    )
                    or 0.0
                )
            except (TypeError, ValueError):
                vdv = 0.0
            try:
                oscillation_loss = float(
                    row.get("stability_oscillation_loss", 0.0) or 0.0
                )
            except (TypeError, ValueError):
                oscillation_loss = 0.0
            oscillation_valid_raw = row.get(
                "stability_oscillation_loss_valid",
                row.get("oscillation_loss_valid", 0.0),
            )
            oscillation_valid = str(oscillation_valid_raw).strip().lower() in {
                "1",
                "1.0",
                "true",
                "yes",
            }
            trajectory_loss: Optional[float]
            try:
                raw_trajectory_loss = row.get("stability_trajectory_loss")
                trajectory_loss = (
                    None
                    if raw_trajectory_loss in (None, "")
                    else float(raw_trajectory_loss)
                )
            except (TypeError, ValueError):
                trajectory_loss = None
            trajectory_valid_raw = row.get(
                "stability_trajectory_loss_valid", 0
            )
            trajectory_valid = (
                trajectory_loss is not None
                and str(trajectory_valid_raw).strip().lower()
                in {"1", "1.0", "true", "yes"}
            )
            stress_family_keys = (
                "perturbation_stress_family",
                "stress_family",
            )
            stress_realized_keys = (
                "perturbation_stress_realized",
                "stress_realized",
            )
            stress_text_dose_keys = (
                "perturbation_text_dose",
                "stress_text_dose",
                "text_dose",
            )
            stress_alignment_keys = (
                "perturbation_alignment_valid",
                "stress_alignment_valid",
                "alignment_valid",
            )
            declared_stress_family = str(
                row.get("perturbation_stress_family")
                or row.get("stress_family")
                or ""
            ).strip().lower()
            strict_stress_contract = bool(
                declared_stress_family == STRESS_FAMILY
                or row.get("perturbation_bank_path") not in (None, "")
            )

            def _first_value(keys: Sequence[str]) -> Any:
                for key in keys:
                    if row.get(key) not in (None, ""):
                        return row.get(key)
                return None

            raw_stress_family = _first_value(stress_family_keys)
            stress_family = str(
                raw_stress_family
                or (
                    "unresolved_stress_family"
                    if strict_stress_contract
                    else "legacy"
                )
            )
            stress_realized = _optional_finite_float(
                _first_value(stress_realized_keys)
            )
            stress_text_dose = _optional_finite_float(
                _first_value(stress_text_dose_keys)
            )
            alignment_raw = _first_value(stress_alignment_keys)
            alignment_declared = str(alignment_raw).strip().lower() in {
                "1",
                "1.0",
                "true",
                "yes",
            }
            perturbation_norm = 1.0
            perturbation_realization_valid = True
            perturbation_realization_reason = ""
            if not is_clean:
                audit_valid_raw = row.get("perturbation_valid")
                if (
                    strict_stress_contract
                    and audit_valid_raw in (None, "")
                    and alignment_raw not in (None, "")
                ):
                    audit_valid_raw = alignment_raw
                audit_valid = str(audit_valid_raw).strip().lower() in {
                    "1",
                    "1.0",
                    "true",
                    "yes",
                }
                realized_raw = (
                    stress_realized
                    if strict_stress_contract
                    else row.get("perturbation_realized")
                )
                if audit_valid_raw in (None, ""):
                    perturbation_realization_valid = False
                    perturbation_realization_reason = (
                        "missing_runtime_perturbation_audit"
                    )
                elif not audit_valid:
                    perturbation_realization_valid = False
                    perturbation_realization_reason = str(
                        row.get("perturbation_reason")
                        or "runtime_perturbation_audit_invalid"
                    )
                elif realized_raw in (None, ""):
                    perturbation_realization_valid = False
                    perturbation_realization_reason = (
                        "missing_realized_perturbation_dose"
                    )
                else:
                    try:
                        perturbation_norm = float(realized_raw)
                    except (TypeError, ValueError):
                        perturbation_realization_valid = False
                        perturbation_realization_reason = (
                            "invalid_realized_perturbation_dose"
                        )
                    else:
                        if (
                            not math.isfinite(perturbation_norm)
                            or perturbation_norm > 1.0
                        ):
                            perturbation_realization_valid = False
                            perturbation_realization_reason = (
                                "out_of_range_realized_perturbation_dose"
                            )
                        elif perturbation_norm <= 0.0:
                            perturbation_realization_valid = False
                            perturbation_realization_reason = (
                                "nonpositive_realized_perturbation_dose"
                            )
            try:
                seed = int(row.get("perturbation_seed", 0) or 0)
            except (TypeError, ValueError):
                seed = 0
            stress_lambda = _optional_finite_float(
                row.get(
                    "stress_lambda",
                    row.get(
                        "ge_stress_lambda",
                        row.get("perturbation_intensity", 0.0),
                    ),
                )
            )
            if is_clean:
                stress_lambda = 0.0
            stress_alignment_valid = False
            if strict_stress_contract:
                target = stress_lambda
                expected = {round(value, 12) for value in FORMAL_STRESS_GRID}
                if raw_stress_family in (None, ""):
                    perturbation_realization_valid = False
                    perturbation_realization_reason = "missing_stress_family"
                elif target is None or round(float(target), 12) not in expected:
                    perturbation_realization_valid = False
                    perturbation_realization_reason = (
                        "unexpected_formal_stress_lambda"
                    )
                elif stress_realized is None:
                    perturbation_realization_valid = False
                    perturbation_realization_reason = "missing_stress_realized"
                elif abs(float(stress_realized) - float(target)) > _LAMBDA_TOLERANCE:
                    perturbation_realization_valid = False
                    perturbation_realization_reason = (
                        "stress_realized_target_mismatch"
                    )
                elif stress_text_dose is None:
                    perturbation_realization_valid = False
                    perturbation_realization_reason = "missing_stress_text_dose"
                elif (
                    abs(float(stress_text_dose) - float(target))
                    > FORMAL_TEXT_DOSE_TOLERANCE
                ):
                    perturbation_realization_valid = False
                    perturbation_realization_reason = (
                        "stress_text_dose_alignment_failed"
                    )
                elif not alignment_declared:
                    perturbation_realization_valid = False
                    perturbation_realization_reason = "stress_alignment_invalid"
                else:
                    stress_alignment_valid = True
                if not is_clean and stress_realized is not None:
                    perturbation_norm = float(stress_realized)
            success_value = _optional_finite_float(row.get("task_state_success"))
            progress_value = _optional_finite_float(
                row.get("task_percent_complete")
            )
            crec_value = _optional_finite_float(
                row.get("c_rec", row.get("rebound_c_rec"))
            )
            outcome_valid_flag = row.get("task_outcome_valid")
            outcome_valid = (
                success_value is not None
                and progress_value is not None
                and (
                    outcome_valid_flag in (None, "")
                    or str(outcome_valid_flag).strip().lower()
                    in {"1", "1.0", "true", "yes"}
                )
            )
            c_rec_valid_flag = row.get("rebound_c_rec_valid")
            c_rec_valid = (
                crec_value is not None
                and str(c_rec_valid_flag).strip().lower()
                in {"1", "1.0", "true", "yes"}
            )
            missing_outcomes = []
            if success_value is None:
                missing_outcomes.append("task_state_success")
            if progress_value is None:
                missing_outcomes.append("task_percent_complete")
            if outcome_valid_flag not in (None, "") and not outcome_valid:
                missing_outcomes.append("task_outcome_invalid")
            success = 0.0 if success_value is None else success_value
            progress = 0.0 if progress_value is None else progress_value
            crec = 0.0 if crec_value is None else crec_value
            out.append(
                _RawRun(
                    anchor_id=str(row.get("anchor_id", row.get("episode_id", "unknown"))),
                    perturbation_type=ptype or "clean",
                    perturbation_seed=seed,
                    stress_lambda=(
                        None
                        if stress_lambda is None
                        else float(stress_lambda)
                    ),
                    stress_family=stress_family,
                    stress_variant=str(
                        row.get("perturbation_variant") or ""
                    ),
                    stress_bank_path=str(
                        row.get("perturbation_bank_path") or ""
                    ),
                    stress_realized=stress_realized,
                    stress_text_dose=stress_text_dose,
                    stress_alignment_valid=stress_alignment_valid,
                    strict_stress_contract=strict_stress_contract,
                    value_delta_variance=vdv,
                    oscillation_loss=oscillation_loss,
                    oscillation_loss_valid=oscillation_valid,
                    trajectory_loss=trajectory_loss,
                    trajectory_loss_valid=trajectory_valid,
                    perturbation_norm=perturbation_norm,
                    perturbation_realization_valid=(
                        perturbation_realization_valid
                    ),
                    perturbation_realization_reason=(
                        perturbation_realization_reason
                    ),
                    action_sequence=actions,
                    task_state_success=success,
                    task_progress=progress,
                    c_rec=crec,
                    outcome_valid=outcome_valid,
                    c_rec_valid=c_rec_valid,
                    outcome_missing_reason=(
                        ""
                        if not missing_outcomes
                        else "missing:" + ",".join(missing_outcomes)
                    ),
                    task_family=str(row.get("task_family", "other")),
                    is_clean=is_clean,
                )
            )
        return out

    @staticmethod
    def read_resilience_raw_csv(path: Union[str, Path]) -> List[Dict[str, Any]]:
        return list(read_csv_rows(path))

    @staticmethod
    def write_beta_neighborhood_csv(
        aggregates: Mapping[str, BetaAggregate],
        output_path: Union[str, Path],
    ) -> Path:
        fieldnames = [
            "anchor_id",
            "task_family",
            "beta_trajectory",
            "beta_vv",
            "beta_out",
            "beta_hat",
            "beta_oscillation",
            "beta_action",
            "beta_progress",
            "beta_success",
            "n_clean",
            "n_perturbed",
            "clean_ref_valid",
            "clean_ref_strategy",
            "beta_neighborhood_valid",
            "missing_reason",
            "beta_scope",
            "perturbation_type",
            "stress_lambda",
            "stress_family",
            "stress_realized",
            "stress_text_dose",
            "stress_alignment_valid",
            "stability_n_paired_seeds",
        ]
        return write_csv_rows_exact(
            output_path,
            [aggregate.to_dict() for aggregate in aggregates.values()],
            fieldnames=fieldnames,
            overwrite_policy="replace",
            empty_policy="write_header",
        )

    @staticmethod
    def write_beta_family_csv(
        aggregates: Mapping[str, BetaFamilyAggregate],
        output_path: Union[str, Path],
    ) -> Path:
        rows = [aggregate.to_dict() for aggregate in aggregates.values()]
        return write_csv_rows_exact(
            output_path,
            rows,
            fieldnames=None if rows else ["task_family"],
            overwrite_policy="replace",
            empty_policy="write_header",
        )

    @staticmethod
    def pairwise_family_tests(
        stats_rows: Sequence[Mapping[str, Any]],
        metric_key: str = "beta_hat",
    ) -> List[Dict[str, Any]]:
        """Run KS + Mann-Whitney pairwise tests between task families.

        Returns a list of records ``{"family_a", "family_b", "ks_stat",
        "ks_pvalue", "mw_stat", "mw_pvalue", "n_a", "n_b"}``.
        Safely degrades when ``scipy`` is not installed — in that case the
        p-values are ``None`` and the test statistics use a rank-based
        fallback (KS via two-sample CDF difference; MW via midrank sum).
        """
        by_family: Dict[str, List[float]] = {}
        for row in stats_rows:
            if row.get("beta_neighborhood_valid") is False:
                continue
            family = str(row.get("task_family", "other"))
            try:
                val = float(row.get(metric_key, 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            by_family.setdefault(family, []).append(val)

        families = sorted(by_family.keys())
        results: List[Dict[str, Any]] = []
        try:
            from scipy.stats import ks_2samp, mannwhitneyu  # type: ignore

            has_scipy = True
        except ImportError:  # pragma: no cover
            has_scipy = False

        for i, fa in enumerate(families):
            for fb in families[i + 1 :]:
                a_vals = by_family[fa]
                b_vals = by_family[fb]
                if len(a_vals) < 2 or len(b_vals) < 2:
                    continue
                if has_scipy:
                    ks_stat, ks_p = ks_2samp(a_vals, b_vals)
                    try:
                        mw_stat, mw_p = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
                    except ValueError:
                        mw_stat, mw_p = 0.0, None
                else:
                    ks_stat = StabilityCollector._ks_stat_fallback(a_vals, b_vals)
                    ks_p = None
                    mw_stat = StabilityCollector._rank_sum_fallback(a_vals, b_vals)
                    mw_p = None
                results.append(
                    {
                        "family_a": fa,
                        "family_b": fb,
                        "ks_stat": float(ks_stat),
                        "ks_pvalue": float(ks_p) if ks_p is not None else None,
                        "mw_stat": float(mw_stat),
                        "mw_pvalue": float(mw_p) if mw_p is not None else None,
                        "n_a": len(a_vals),
                        "n_b": len(b_vals),
                        "metric_key": metric_key,
                    }
                )
        return results

    @staticmethod
    def _ks_stat_fallback(a: Sequence[float], b: Sequence[float]) -> float:
        """Two-sample KS test statistic using empirical CDF difference."""
        combined = sorted(set(a) | set(b))
        if not combined:
            return 0.0
        n_a = len(a)
        n_b = len(b)
        if n_a == 0 or n_b == 0:
            return 0.0
        a_sorted = sorted(a)
        b_sorted = sorted(b)

        def _cdf(sorted_values: Sequence[float], x: float) -> float:
            # Count of sorted_values <= x divided by n
            import bisect

            return bisect.bisect_right(sorted_values, x) / len(sorted_values)

        return max(abs(_cdf(a_sorted, x) - _cdf(b_sorted, x)) for x in combined)

    @staticmethod
    def _rank_sum_fallback(a: Sequence[float], b: Sequence[float]) -> float:
        """Mann-Whitney U fallback using midrank sum (no p-value)."""
        combined = [(v, 0) for v in a] + [(v, 1) for v in b]
        combined.sort(key=lambda pair: pair[0])
        ranks = [0.0] * len(combined)
        i = 0
        while i < len(combined):
            j = i
            while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
                j += 1
            midrank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[k] = midrank
            i = j + 1
        rank_sum_a = sum(ranks[k] for k, (_v, grp) in enumerate(combined) if grp == 0)
        n_a = len(a)
        n_b = len(b)
        return rank_sum_a - n_a * (n_a + 1) / 2.0 if n_b > 0 else 0.0
