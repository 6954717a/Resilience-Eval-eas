#!/usr/bin/env python3

"""
Stability Collector

Collects policy stability metrics from StateEncoder, SafetyMonitor, and planner info.
"""

import csv
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .stability_metrics import StabilityMetrics

logger = logging.getLogger(__name__)


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
    value_delta_variance: float = 0.0
    oscillation_loss: float = 0.0
    oscillation_loss_valid: bool = False
    perturbation_norm: float = 1.0
    action_sequence: List[str] = field(default_factory=list)
    task_state_success: float = 0.0
    task_progress: float = 0.0
    c_rec: float = 0.0
    task_family: str = "other"
    is_clean: bool = False


@dataclass
class BetaAggregate:
    """Aggregated β for a single anchor across a perturbation neighborhood."""

    anchor_id: str
    task_family: str
    beta_vv: float = 0.0
    beta_out: float = 0.0
    beta_hat: float = 0.0
    beta_oscillation: float = 0.0
    beta_action: float = 0.0
    beta_progress: float = 0.0
    beta_success: float = 0.0
    n_clean: int = 0
    n_perturbed: int = 0
    clean_ref_valid: bool = True
    clean_ref_strategy: str = "medoid"
    beta_neighborhood_valid: bool = True
    missing_reason: str = ""
    beta_scope: str = "neighborhood"
    perturbation_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "task_family": self.task_family,
            "beta_vv": float(self.beta_vv),
            "beta_out": float(self.beta_out),
            "beta_hat": float(self.beta_hat),
            "beta_oscillation": float(self.beta_oscillation),
            "beta_action": float(self.beta_action),
            "beta_progress": float(self.beta_progress),
            "beta_success": float(self.beta_success),
            "n_clean": int(self.n_clean),
            "n_perturbed": int(self.n_perturbed),
            "clean_ref_valid": bool(self.clean_ref_valid),
            "clean_ref_strategy": self.clean_ref_strategy,
            "beta_neighborhood_valid": bool(self.beta_neighborhood_valid),
            "missing_reason": self.missing_reason,
            "beta_scope": self.beta_scope,
            "perturbation_type": self.perturbation_type,
        }


class StabilityCollector:
    """
    Collects Stability metrics from Critic's StateEncoder, SafetyMonitor, and planner.
    """

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
                    metrics.value_history = list(value_stats.get("value_history", []))
                    metrics.value_variance = float(
                        value_stats.get("value_variance", 0.0)
                    )
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
                    "Extracted stability value stats: variance=%.4f delta_variance=%.4f",
                    metrics.value_variance,
                    metrics.value_delta_variance,
                )
            except Exception as exc:
                logger.warning("Failed to extract value_variance from Critic: %s", exc)
                metrics.value_variance = 0.0
                metrics.value_delta_variance = 0.0

            try:
                if hasattr(critic, "get_stability_trace_summary"):
                    trace_summary = critic.get_stability_trace_summary()
                    metrics.td_error_mean_abs = float(
                        trace_summary.get("td_error_mean_abs", 0.0) or 0.0
                    )
                    metrics.td_error_p95_abs = float(
                        trace_summary.get("td_error_p95_abs", 0.0) or 0.0
                    )
                    metrics.gae_mean_abs = float(
                        trace_summary.get("gae_mean_abs", 0.0) or 0.0
                    )
                    metrics.gae_p95_abs = float(
                        trace_summary.get("gae_p95_abs", 0.0) or 0.0
                    )
                    metrics.return_target_variance = float(
                        trace_summary.get("return_target_variance", 0.0) or 0.0
                    )
                    metrics.oscillation_loss = float(
                        trace_summary.get("oscillation_loss", 0.0) or 0.0
                    )
                    metrics.oscillation_loss_valid = bool(
                        trace_summary.get("oscillation_loss_valid", False)
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
                if isinstance(replanning_count, dict):
                    # Max replanning count across agents
                    metrics.n_replan = max(replanning_count.values()) if replanning_count else 0
                else:
                    metrics.n_replan = int(replanning_count) if replanning_count else 0

                # Track replan steps
                for info in planner_infos:
                    step = info.get(
                        "total_step_count",
                        info.get("sim_step_count", info.get("step", 0)),
                    )
                    replan_count = info.get("replanning_count", {})
                    if isinstance(replan_count, dict):
                        if any(count > 0 for count in replan_count.values()):
                            metrics.replan_steps.append(step)
                    elif replan_count > 0:
                        metrics.replan_steps.append(step)

                logger.debug("Extracted n_replan: %d", metrics.n_replan)
            except Exception as exc:
                logger.warning("Failed to extract n_replan from planner_infos: %s", exc)
                metrics.n_replan = 0

        # 3. Extract P_cbf from SafetyMonitor
        if safety_monitor is not None:
            try:
                if hasattr(safety_monitor, "get_statistics"):
                    safety_stats = safety_monitor.get_statistics()
                elif hasattr(safety_monitor, "get_summary"):
                    safety_stats = safety_monitor.get_summary()
                else:
                    raise AttributeError(
                        "SafetyMonitor exposes neither get_statistics() nor get_summary()."
                    )
                metrics.p_cbf = safety_stats.get("total_cbf_penalty", 0.0)
                metrics.collision_rate = float(
                    safety_stats.get("collision_rate", 0.0) or 0.0
                )
                metrics.safety_score = float(
                    safety_stats.get("safety_score", 1.0) or 1.0
                )
                logger.debug("Extracted p_cbf: %.4f", metrics.p_cbf)
            except Exception as exc:
                logger.warning("Failed to extract p_cbf from SafetyMonitor: %s", exc)
                metrics.p_cbf = 0.0
                metrics.collision_rate = 0.0
                metrics.safety_score = 1.0

        # 4. Optional: Extract SayCan scores if enabled
        if self.use_saycan_scores and planner is not None:
            self._extract_saycan_scores(planner, metrics)

        # 5. Compute derived metrics (beta)
        if planner_infos:
            metrics.total_steps = int(
                planner_infos[-1].get("total_step_count", len(planner_infos))
            )
            # Extract task progress and success status
            last_info = planner_infos[-1]
            task_progress = float(last_info.get("task_percent_complete", 0.0))
            task_success = bool(last_info.get("task_state_success", False))
        else:
            metrics.total_steps = 0
            task_progress = 0.0
            task_success = False

        # Call corrected computation method with hybrid approach
        metrics.compute_derived_metrics(
            task_progress=task_progress,
            task_success=task_success,
            total_steps=metrics.total_steps,
            method="hybrid"
        )

        return metrics

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

        # Extract from legacy keys
        metrics.beta = info.get("stability_beta", 1.0)
        metrics.value_variance = info.get("stability_value_variance", 0.0)
        metrics.value_delta_variance = info.get("stability_value_delta_variance", 0.0)
        metrics.n_replan = int(info.get("stability_n_replan", 0))
        metrics.p_cbf = info.get("stability_p_cbf", 0.0)
        metrics.collision_rate = info.get(
            "safety_collision_rate",
            info.get("collision_rate", 0.0),
        )
        metrics.safety_score = info.get("safety_score", 1.0)
        metrics.score_variance = info.get("stability_score_variance", None)
        metrics.stability_violation_rate = info.get("stability_violation_rate", None)
        metrics.total_steps = int(info.get("total_step_count", 0))
        metrics.td_error_mean_abs = float(
            info.get("stability_td_error_mean_abs", 0.0) or 0.0
        )
        metrics.td_error_p95_abs = float(
            info.get("stability_td_error_p95_abs", 0.0) or 0.0
        )
        metrics.gae_mean_abs = float(info.get("stability_gae_mean_abs", 0.0) or 0.0)
        metrics.gae_p95_abs = float(info.get("stability_gae_p95_abs", 0.0) or 0.0)
        metrics.return_target_variance = float(
            info.get("stability_return_target_variance", 0.0) or 0.0
        )
        metrics.oscillation_loss = float(
            info.get("stability_oscillation_loss", 0.0) or 0.0
        )
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
        # Treat missing norms as unit perturbations so legacy rows remain
        # usable. Clean rows are not used as perturbed rows here.
        return max(float(run.perturbation_norm or 1.0), float(epsilon))

    def _compute_beta_components(
        self,
        clean: Sequence[_RawRun],
        perturbed: Sequence[_RawRun],
        *,
        alpha: float,
        weight_vv: float,
        weight_out: float,
        include_rebound: bool,
    ) -> Dict[str, float]:
        if not clean or not perturbed:
            return {
                "beta_vv": 0.0,
                "beta_out": 0.0,
                "beta_hat": 0.0,
                "beta_oscillation": 0.0,
                "beta_action": 0.0,
                "beta_progress": 0.0,
                "beta_success": 0.0,
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

        per_run_osc: List[float] = []
        per_run_out: List[float] = []
        per_run_action: List[float] = []
        per_run_success: List[float] = []
        per_run_progress: List[float] = []
        for run in perturbed:
            norm = self._row_perturbation_norm(run, epsilon)
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

        beta_oscillation = _percentile(per_run_osc, 0.95)
        beta_out = _percentile(per_run_out, 0.5)
        return {
            "beta_vv": float(beta_oscillation),
            "beta_out": float(beta_out),
            "beta_hat": float(weight_vv * beta_oscillation + weight_out * beta_out),
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
        weight_vv: float = 0.5,
        weight_out: float = 0.5,
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
        weight_vv, weight_out:
            Convex-combination weights for :math:`\hat\beta`. Default ``0.5/0.5``.

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
            clean = [r for r in anchor_runs if r.is_clean]
            perturbed = [r for r in anchor_runs if not r.is_clean]
            if not perturbed:
                logger.debug("Anchor %s has no perturbed runs; skipping.", anchor_id)
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
            else:
                ref_vv = float(
                    sum(r.value_delta_variance for r in clean) / len(clean)
                )
                ref_action = self._clean_action_medoid(clean)
                ref_sr = float(sum(r.task_state_success for r in clean) / len(clean))
                ref_progress = float(sum(r.task_progress for r in clean) / len(clean))
                ref_crec = float(sum(r.c_rec for r in clean) / len(clean))

            # β_vv: 95th percentile of positive variance excess
            vv_excess = [
                max(0.0, r.value_delta_variance - ref_vv) for r in perturbed
            ]
            beta_vv = _percentile(vv_excess, 0.95)

            # beta_out: median action distance plus success/progress deltas.
            per_run_out: List[float] = []
            per_run_action: List[float] = []
            per_run_success: List[float] = []
            per_run_progress: List[float] = []
            for r in perturbed:
                lev = _normalized_levenshtein(r.action_sequence, ref_action)
                delta_sr = abs(r.task_state_success - ref_sr)
                delta_progress = abs(r.task_progress - ref_progress)
                per_run_action.append(lev)
                per_run_success.append(delta_sr)
                per_run_progress.append(delta_progress)
                if include_rebound:
                    delta_crec = abs(r.c_rec - ref_crec)
                    per_run_out.append(
                        alpha * lev
                        + ((1.0 - alpha) / 3.0) * delta_sr
                        + ((1.0 - alpha) / 3.0) * delta_progress
                        + ((1.0 - alpha) / 3.0) * delta_crec
                    )
                else:
                    per_run_out.append(
                        alpha * lev
                        + ((1.0 - alpha) / 2.0) * delta_sr
                        + ((1.0 - alpha) / 2.0) * delta_progress
                    )
            beta_out = _percentile(per_run_out, 0.5)
            beta_action = _percentile(per_run_action, 0.5)
            beta_success = _percentile(per_run_success, 0.5)
            beta_progress = _percentile(per_run_progress, 0.5)

            beta_hat = weight_vv * beta_vv + weight_out * beta_out
            components = self._compute_beta_components(
                clean,
                perturbed,
                alpha=alpha,
                weight_vv=weight_vv,
                weight_out=weight_out,
                include_rebound=include_rebound,
            )
            beta_vv = components["beta_vv"]
            beta_out = components["beta_out"]
            beta_hat = components["beta_hat"]
            beta_oscillation = components["beta_oscillation"]
            beta_action = components["beta_action"]
            beta_progress = components["beta_progress"]
            beta_success = components["beta_success"]

            task_family = perturbed[0].task_family if perturbed else "other"
            aggregates[anchor_id] = BetaAggregate(
                anchor_id=anchor_id,
                task_family=task_family,
                beta_vv=float(beta_vv),
                beta_out=float(beta_out),
                beta_hat=float(beta_hat),
                beta_oscillation=float(beta_oscillation),
                beta_action=float(beta_action),
                beta_progress=float(beta_progress),
                beta_success=float(beta_success),
                n_clean=len(clean),
                n_perturbed=len(perturbed),
                clean_ref_valid=True,
                clean_ref_strategy="medoid",
                beta_neighborhood_valid=True,
                missing_reason="",
                beta_scope=(
                    f"neighborhood_M={len(perturbed)}_Pi={len({r.perturbation_type for r in perturbed})}"
                ),
            )
            stats_rows.append(aggregates[anchor_id].to_dict())

        return aggregates, stats_rows

    def aggregate_beta_by_perturbation(
        self,
        raw_runs: Iterable[Mapping[str, Any]],
        alpha: float = 0.5,
        weight_vv: float = 0.5,
        weight_out: float = 0.5,
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
                    components = self._compute_beta_components(
                        clean,
                        perturbed,
                        alpha=alpha,
                        weight_vv=weight_vv,
                        weight_out=weight_out,
                        include_rebound=include_rebound,
                    )
                    aggregate = BetaAggregate(
                        anchor_id=anchor_id,
                        task_family=task_family,
                        beta_vv=float(components["beta_vv"]),
                        beta_out=float(components["beta_out"]),
                        beta_hat=float(components["beta_hat"]),
                        beta_oscillation=float(components["beta_oscillation"]),
                        beta_action=float(components["beta_action"]),
                        beta_progress=float(components["beta_progress"]),
                        beta_success=float(components["beta_success"]),
                        n_clean=len(clean),
                        n_perturbed=len(perturbed),
                        clean_ref_valid=True,
                        clean_ref_strategy="medoid",
                        beta_neighborhood_valid=True,
                        missing_reason="",
                        beta_scope=(
                            f"perturbation={perturbation_type}_M={len(perturbed)}"
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
            try:
                perturbation_norm = float(
                    row.get(
                        "perturbation_intensity",
                        row.get("stress_lambda", row.get("ge_stress_lambda", 1.0)),
                    )
                    or 1.0
                )
            except (TypeError, ValueError):
                perturbation_norm = 1.0
            try:
                seed = int(row.get("perturbation_seed", 0) or 0)
            except (TypeError, ValueError):
                seed = 0
            try:
                success = float(row.get("task_state_success", 0.0) or 0.0)
            except (TypeError, ValueError):
                success = 0.0
            try:
                progress = float(row.get("task_percent_complete", 0.0) or 0.0)
            except (TypeError, ValueError):
                progress = 0.0
            try:
                crec = float(row.get("c_rec", row.get("rebound_c_rec", 0.0)) or 0.0)
            except (TypeError, ValueError):
                crec = 0.0
            out.append(
                _RawRun(
                    anchor_id=str(row.get("anchor_id", row.get("episode_id", "unknown"))),
                    perturbation_type=ptype or "clean",
                    perturbation_seed=seed,
                    value_delta_variance=vdv,
                    oscillation_loss=oscillation_loss,
                    oscillation_loss_valid=oscillation_valid,
                    perturbation_norm=perturbation_norm,
                    action_sequence=actions,
                    task_state_success=success,
                    task_progress=progress,
                    c_rec=crec,
                    task_family=str(row.get("task_family", "other")),
                    is_clean=is_clean,
                )
            )
        return out

    @staticmethod
    def read_resilience_raw_csv(path: Union[str, Path]) -> List[Dict[str, Any]]:
        path = Path(path)
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]

    @staticmethod
    def write_beta_neighborhood_csv(
        aggregates: Mapping[str, BetaAggregate],
        output_path: Union[str, Path],
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "anchor_id",
            "task_family",
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
        ]
        with open(output_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for agg in aggregates.values():
                writer.writerow(agg.to_dict())
        return output_path

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
