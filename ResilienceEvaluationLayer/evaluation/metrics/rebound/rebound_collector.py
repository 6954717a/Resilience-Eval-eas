#!/usr/bin/env python3

"""
Rebound Collector

Collects recovery metrics from ReboundManager and CoPAL RecoveryMetrics.
Consolidates data from multiple sources into unified ReboundMetrics.
"""

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from habitat_llm.evaluation.metrics.collection import EpisodeCollectionContext
from habitat_llm.evaluation.stage_baseline import (
    RECOVERY_POOL_MODALITY,
    StageAnchor,
    StageBaseline,
    StageBaselineDim,
    StageResolver,
)

from .cog_load_tracker import CHANNELS as _COG_CHANNELS
from .cog_load_tracker import CogLoadTracker
from .formal_cost import (
    bounded_recovery_cost_index,
    recovery_intensity,
    regularized_mahalanobis,
)
from .recovery_features import (
    REBOUND_FORMULA_VERSION,
    build_recovery_feature_groups,
)
from .rebound_metrics import FailureEvent, ReboundMetrics, StageRecoveryRecord
from .target_graph import parse_action_entry

logger = logging.getLogger(__name__)


# Default hyper-parameters (override via ``config`` or ``resilience_config.yaml``)
_DEFAULT_CREC_CONFIG = {
    "epsilon": 1e-6,
    "huber_k": 3.0,
    "allow_family_baseline_fallback": False,
    "allow_terminal_segment_fallback": True,
    "allow_anchor_segment_fallback": False,
    "mahalanobis_regularizer": 1.0e-3,
}


def _huber(x: float, k: float) -> float:
    """Huber-like ψ: identity below k, logarithmic above to avoid domination."""
    x = max(0.0, float(x))
    if x <= k:
        return x
    return k + math.log1p(x - k)


def _channel_mahalanobis(
    values: Sequence[float],
    baseline: StageBaseline,
    group: str,
    regularizer: float,
) -> Optional[float]:
    """Eq. (2) channel magnitude, or ``None`` for an old/incomplete baseline."""

    return regularized_mahalanobis(
        values,
        baseline.recovery_covariance.get(group, []),
        sample_count=int(baseline.recovery_covariance_n_samples or 0),
        regularizer=regularizer,
        mean=baseline.recovery_mean.get(group, []),
    )


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


def _baseline_support_provenance(
    baseline: Optional[StageBaseline],
) -> Dict[str, Any]:
    """Return compact baseline provenance for diagnostics."""

    if baseline is None:
        return {}
    return {
        "formal_valid": bool(getattr(baseline, "formal_valid", False)),
        "formal_support_scope": str(
            getattr(baseline, "formal_support_scope", "")
        ),
        "formal_component_support": dict(
            getattr(baseline, "formal_component_support", {}) or {}
        ),
        "recovery_covariance_scope": dict(
            getattr(baseline, "recovery_covariance_scope", {}) or {}
        ),
        "recovery_covariance_source": dict(
            getattr(baseline, "recovery_covariance_source", {}) or {}
        ),
    }


def _has_recovery_statistics(baseline: Optional[StageBaseline]) -> bool:
    if baseline is None:
        return False
    if int(getattr(baseline, "recovery_covariance_n_samples", 0) or 0) < 1:
        return False
    means = getattr(baseline, "recovery_mean", {}) or {}
    covariances = getattr(baseline, "recovery_covariance", {}) or {}
    for group in ("cog", "phy", "debt"):
        mean = list(means.get(group, []) or [])
        covariance = list(covariances.get(group, []) or [])
        if len(mean) != 3 or len(covariance) != 3:
            return False
        if any(len(list(row or [])) != 3 for row in covariance):
            return False
        values = mean + [value for row in covariance for value in row]
        try:
            if not all(math.isfinite(float(value)) for value in values):
                return False
        except (TypeError, ValueError):
            return False
    return True


def _selection_missing_reason(level: str, *, segment: bool = False) -> str:
    return (
        "stage_baseline_segment_missing"
        if segment
        else "stage_baseline_key_missing"
    )


def _select_baseline_with_level(
    baselines: Mapping[Tuple[str, int, str], StageBaseline],
    anchor: StageAnchor,
) -> Tuple[Optional[StageBaseline], str]:
    """Resolve exact statistics first, then the same family-stage pool."""
    if not baselines:
        return None, "none"
    key = anchor.key()
    exact = baselines.get(key)
    if _has_recovery_statistics(exact):
        return exact, "exact"
    if int(anchor.stage) >= 0:
        pool_key = (
            anchor.family,
            int(anchor.stage),
            RECOVERY_POOL_MODALITY,
        )
        pool = baselines.get(pool_key)
        if _has_recovery_statistics(pool):
            return pool, "family_stage_recovery_pool"
    if exact is not None:
        return exact, "exact_statistics_incomplete"
    return None, "missing"


def _select_baseline(
    baselines: Mapping[Tuple[str, int, str], StageBaseline],
    anchor: StageAnchor,
    *,
    allow_family_fallback: bool = False,
) -> Optional[StageBaseline]:
    """Lookup a baseline without universal fallback.

    Family-level fallbacks are intentionally opt-in because a mismatched
    baseline changes the formal meaning of C_rec. The old "first baseline"
    fallback is deliberately removed.
    """
    baseline, _level = _select_baseline_with_level(baselines, anchor)
    if baseline is not None or not allow_family_fallback:
        return baseline
    for candidate_key in (
        (anchor.family, -1, anchor.modality),
        (anchor.family, -1, "planning"),
    ):
        candidate, _candidate_level = _select_baseline_with_level(
            baselines,
            StageAnchor(
                family=candidate_key[0],
                stage=candidate_key[1],
                modality=candidate_key[2],
            ),
        )
        if candidate is not None:
            return candidate
    return None


def _select_baseline_with_policy(
    baselines: Mapping[Tuple[str, int, str], StageBaseline],
    anchor: StageAnchor,
    *,
    allow_family_fallback: bool = False,
) -> Tuple[Optional[StageBaseline], str]:
    baseline, level = _select_baseline_with_level(baselines, anchor)
    if baseline is not None or not allow_family_fallback:
        return baseline, level
    for candidate_key in (
        (anchor.family, -1, anchor.modality),
        (anchor.family, -1, "planning"),
    ):
        candidate, _candidate_level = _select_baseline_with_level(
            baselines,
            StageAnchor(
                family=candidate_key[0],
                stage=candidate_key[1],
                modality=candidate_key[2],
            ),
        )
        if candidate is not None:
            return candidate, "family"
    return None, "missing"


def _baseline_w_rem_star(
    baseline: StageBaseline,
    *,
    episode_id: Optional[str],
    stage: int,
) -> Optional[float]:
    """Return the episode×stage clean-continuation denominator.

    Estimator-produced V3 baselines carry explicit trajectory provenance and
    must resolve the current episode exactly.  The scalar fallback is retained
    only for provenance-free in-memory baselines used by lightweight callers.
    """

    resolver = getattr(baseline, "w_rem_star_for_episode_stage", None)
    if callable(resolver) and episode_id not in (None, ""):
        value = resolver(str(episode_id), int(stage))
        if value is not None:
            try:
                return max(1.0, float(value))
            except (TypeError, ValueError):
                return None
        if getattr(baseline, "W_rem_star_by_episode_stage", {}):
            return None
    value = getattr(baseline, "W_rem_star_swe", None)
    if value is None:
        value = getattr(baseline, "W_rem_seg_bar", 1.0)
    try:
        return max(1.0, float(value))
    except (TypeError, ValueError):
        return None


def _select_segment_baseline_with_policy(
    baselines: Mapping[Tuple[str, int, str], StageBaseline],
    anchor: StageAnchor,
    *,
    fallback_baseline: StageBaseline,
    allow_family_fallback: bool = False,
    allow_terminal_fallback: bool = True,
    allow_anchor_fallback: bool = False,
) -> Tuple[Optional[StageBaseline], str, str]:
    """Lookup a segment baseline with explicit, auditable fallbacks.

    The disturbance anchor baseline is a scientifically meaningful fallback for
    terminal/unknown tail segments (stage < 0): those segments do not denote a
    new clean continuation stage, so skipping the whole recovery window makes a
    detected rebound look like zero-cost recovery.  Non-terminal segment fallback
    stays opt-in because it changes the formal C_rec reference.
    """
    baseline, level = _select_baseline_with_policy(
        baselines,
        anchor,
        allow_family_fallback=allow_family_fallback,
    )
    if baseline is not None:
        return baseline, level, ""

    if allow_terminal_fallback and int(anchor.stage) < 0:
        return (
            fallback_baseline,
            "anchor_terminal_fallback",
            "terminal_segment_anchor_fallback",
        )

    if allow_anchor_fallback:
        return (
            fallback_baseline,
            "anchor_segment_fallback",
            "segment_anchor_fallback",
        )

    return None, level, _selection_missing_reason(level, segment=True)


class ReboundCollector:
    """
    Collects Rebound metrics from planner's ReboundManager and optional CoPAL components.
    """

    dimension = "rebound"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.gamma = self.config.get("gamma", 2.0)  # Backtrack weight for B_epi
        crec_cfg = dict(_DEFAULT_CREC_CONFIG)
        crec_cfg.update(self.config.get("c_rec", {}) or {})
        self.crec_config: Dict[str, Any] = crec_cfg
        self._resolver: Optional[StageResolver] = None

    # ----------------------------------------------------------------------
    # Lazy resolver so non-C_rec callers don't pay the import cost.
    # ----------------------------------------------------------------------
    @property
    def resolver(self) -> StageResolver:
        if self._resolver is None:
            self._resolver = StageResolver()
        return self._resolver

    def collect_from_planner(
        self,
        planner: Any,
        total_steps: int,
        t_rec: Optional[float] = None,
    ) -> ReboundMetrics:
        """
        Collect Rebound metrics from planner's rebound_managers.

        Args:
            planner: LLMPlanner instance with rebound_managers attribute
            total_steps: Total planning steps in episode
            t_rec: Optional externally observed recovery duration

        Returns:
            ReboundMetrics instance with consolidated data
        """
        metrics = ReboundMetrics(total_steps=total_steps)

        # Check if planner has rebound_managers
        if not hasattr(planner, "rebound_managers"):
            logger.warning("Planner does not have rebound_managers attribute")
            return metrics

        rebound_managers = planner.rebound_managers
        if not rebound_managers:
            logger.info("No rebound managers found in planner")
            return metrics

        # Aggregate metrics from all agents' ReboundManagers
        all_failure_events = []
        total_retry = 0
        total_backtrack = 0
        total_replan = 0
        total_context_summarize = 0
        fault_distribution = {}

        for agent_uid, manager in rebound_managers.items():
            # Get metrics from ReboundManager
            manager_metrics = manager.get_metrics(total_steps)

            # Extract failure history
            strategy_retry = 0
            strategy_backtrack = 0
            for start_step, end_step, strategy in manager.failure_history:
                event = FailureEvent(
                    start_step=start_step,
                    end_step=end_step,
                    strategy=strategy,
                    fault_type="unknown",  # ReboundManager doesn't track fault types
                    success=True,  # Resolved failures are successful
                )
                all_failure_events.append(event)

                # Count strategies
                if strategy == "retry":
                    strategy_retry += 1
                elif strategy in {"rollback", "backtrack"}:
                    strategy_backtrack += 1
                elif strategy == "replan":
                    total_replan += 1
                elif strategy == "context_summarize":
                    total_context_summarize += 1

            if manager.active_failure_start is not None:
                unresolved_end_step = max(int(total_steps), int(manager.active_failure_start) + 1)
                all_failure_events.append(
                    FailureEvent(
                        start_step=int(manager.active_failure_start),
                        end_step=unresolved_end_step,
                        strategy="active_failure_unresolved",
                        fault_type="active_failure",
                        success=False,
                    )
                )

            total_retry += int(manager_metrics.get("retry_count", strategy_retry))
            total_backtrack += int(
                manager_metrics.get("backtrack_count", strategy_backtrack)
            )

            # Aggregate fault distribution
            fault_dist = manager_metrics.get("fault_distribution", {})
            for fault_type, count in fault_dist.items():
                fault_distribution[fault_type] = fault_distribution.get(fault_type, 0) + count

        # Populate ReboundMetrics
        metrics.failure_events = all_failure_events
        metrics.retry_count = total_retry
        metrics.backtrack_count = total_backtrack
        metrics.replan_count = total_replan
        metrics.context_summarize_count = total_context_summarize
        metrics.total_failures = len(all_failure_events)
        metrics.successful_recoveries = sum(1 for event in all_failure_events if event.success)
        metrics.failed_recoveries = sum(1 for event in all_failure_events if not event.success)
        metrics.fault_distribution = fault_distribution

        if t_rec is not None:
            metrics.t_rec = float(t_rec)

        # Compute derived metrics (MTTR, MTBF, B_epi, RR)
        metrics.compute_derived_metrics(gamma=self.gamma)

        # Optional: Extract CoPAL metrics if available
        self._extract_copal_metrics(planner, metrics)

        return metrics

    def collect_episode(
        self,
        context: EpisodeCollectionContext,
    ) -> ReboundMetrics:
        """Collect through the shared end-of-episode collector boundary."""

        raw_t_rec = context.info.get("rebound_t_rec", 0.0)
        try:
            t_rec = float(raw_t_rec or 0.0)
        except (TypeError, ValueError):
            t_rec = 0.0
        return self.collect_from_planner(
            planner=context.planner,
            total_steps=context.total_steps,
            t_rec=t_rec,
        )

    def _extract_copal_metrics(self, planner: Any, metrics: ReboundMetrics) -> None:
        """
        Extract CoPAL-specific metrics if available.

        Args:
            planner: LLMPlanner instance
            metrics: ReboundMetrics to populate
        """
        # Check if planner has CoPAL recovery metrics
        if not hasattr(planner, "copal_recovery_metrics"):
            return

        try:
            copal_metrics = planner.copal_recovery_metrics
            if copal_metrics is None:
                return

            # Extract RSR, RTR, RSTC
            copal_stats = copal_metrics.get_metrics()
            metrics.rsr = copal_stats.get("copal_rsr", None)
            metrics.rtr = copal_stats.get("copal_rtr", None)
            metrics.rstc = copal_stats.get("copal_rstc", None)

            logger.info("Extracted CoPAL metrics: RSR=%.3f, RTR=%.3f", metrics.rsr or 0, metrics.rtr or 0)

        except Exception as exc:
            logger.warning("Failed to extract CoPAL metrics: %s", exc)

    def collect_from_info(self, info: Dict[str, Any]) -> ReboundMetrics:
        """
        Collect Rebound metrics from episode info dict (for backward compatibility).

        Args:
            info: Episode info dictionary

        Returns:
            ReboundMetrics instance
        """
        metrics = ReboundMetrics()

        # Extract from legacy keys
        metrics.b_epi = info.get("rebound_b_epi", 0.0)
        metrics.mttr = info.get("rebound_mttr", 0.0)
        metrics.mtbf = info.get("rebound_mtbf", 0.0)
        metrics.recovery_ratio = info.get("rebound_recovery_ratio", 1.0)
        metrics.t_rec = info.get("rebound_t_rec", 0.0)
        metrics.retry_count = int(info.get("rebound_retry_count", 0))
        metrics.backtrack_count = int(info.get("rebound_backtrack_count", 0))
        metrics.total_failures = int(info.get("rebound_total_failures", 0))
        metrics.total_steps = int(info.get("total_step_count", 0))
        raw_c_rec_valid = info.get("rebound_c_rec_valid", False)
        metrics.c_rec_valid = (
            raw_c_rec_valid
            if isinstance(raw_c_rec_valid, bool)
            else str(raw_c_rec_valid).strip().lower() in {"1", "true", "yes"}
        )
        metrics.c_rec_missing_reason = str(
            info.get("rebound_c_rec_missing_reason", "")
        )
        try:
            metrics.c_rec = float(info.get("rebound_c_rec", 0.0) or 0.0)
        except (TypeError, ValueError):
            metrics.c_rec = 0.0
        raw_index = info.get("rebound_c_rec_index_100")
        try:
            metrics.c_rec_index_100 = (
                float(raw_index) if raw_index not in (None, "") else None
            )
        except (TypeError, ValueError):
            metrics.c_rec_index_100 = None
        raw_lower_bound = info.get("rebound_c_rec_observed_lower_bound")
        try:
            metrics.c_rec_observed_lower_bound = (
                float(raw_lower_bound)
                if raw_lower_bound not in (None, "")
                else None
            )
        except (TypeError, ValueError):
            metrics.c_rec_observed_lower_bound = None
        metrics.c_rec_formula_version = str(
            info.get("rebound_c_rec_formula_version")
            or REBOUND_FORMULA_VERSION
        )
        metrics.raw_window_count = int(info.get("rebound_raw_window_count", 0) or 0)
        metrics.formal_window_count = int(
            info.get(
                "rebound_formal_window_count",
                info.get("rebound_num_recovery_windows", 0),
            )
            or 0
        )
        metrics.skipped_window_count = int(
            info.get("rebound_skipped_window_count", 0) or 0
        )

        # CoPAL metrics
        metrics.rsr = info.get("copal_rsr", None)
        metrics.rtr = info.get("copal_rtr", None)

        return metrics

    # ==================================================================
    # Multi-dim C_rec pipeline (plan/sim/runtime) — driven by
    # :class:`OnlineReboundTracker`.
    # ==================================================================
    def compute_stage_crec_multidim(
        self,
        transitions: Sequence[Mapping[str, Any]],
        baselines: Mapping[Tuple[str, int, str], StageBaseline],
        tracker_summary: Mapping[str, Any],
        metrics: Optional[ReboundMetrics] = None,
    ) -> ReboundMetrics:
        r"""Compute dimension-aware :math:`C_\text{rec}` from a tracker summary.

        Parameters
        ----------
        transitions:
            Canonical executed-action/outcome transition records.  Legacy
            per-step info or critic records remain accepted for old artifacts.
        baselines:
            Mapping ``(f, ℓ, m) → StageBaseline`` produced by
            :class:`StageBaselineEstimator`.
        tracker_summary:
            Output of :meth:`OnlineReboundTracker.finalize` (``windows``,
            ``anchors``, ``progress_deltas``, ``q_deltas``, …).

        Implementation sketch
        ---------------------
        For each closed window ``w = (t_d, t_r)``:

        1. Walk per-step inside the window and accumulate per-dimension
           excess work ``W_t^{cog}`` (volume × invalidity from the cog
           tracker) and ``W_t^{phy,plan}`` / ``W_t^{phy,sim}``
           (sim-step / plan-step overheads above nominal).
        2. Form cognitive, physical, and state-debt excess vectors relative
           to the matching clean StageBaseline.
        3. Convert each vector into the covariance-aware channel norm in
           Eq. (2), then combine the three channel norms by root mean square.
        4. Weight each recovery step by :math:`\Delta\tau_t/\tau^*(a_t)` and
           accumulate it over the closed window.
        5. Divide by the remaining clean work at disturbance entry as Eq. (3).

        Legacy per-channel and plan/sim fields remain detailed diagnostics;
        ``c_rec`` is the covariance-aware formal result.
        """
        metrics = metrics or ReboundMetrics()

        raw_windows = list(tracker_summary.get("windows") or ())
        anchor_dicts: Sequence[Mapping[str, Any]] = (
            tracker_summary.get("anchors") or ()
        )
        cog_steps = list(tracker_summary.get("cognitive_steps") or ())
        template_distribution = dict(tracker_summary.get("template_distribution") or {})

        # Stash event-level info on metrics (small payloads, JSON-friendly)
        metrics.template_distribution = template_distribution
        metrics.num_template_events = int(tracker_summary.get("total_events") or 0)
        metrics.raw_window_count = len(raw_windows)
        metrics.tracker_window_count = len(raw_windows)
        metrics.formal_window_count = 0
        metrics.skipped_window_count = 0
        metrics.c_rec_valid = True
        metrics.c_rec_missing_reason = ""
        metrics.c_rec_index_100 = None
        metrics.c_rec_observed_lower_bound = None
        metrics.g_rec_observed_total = None
        metrics.observed_censored_window_count = 0
        metrics.c_rec_formula_version = REBOUND_FORMULA_VERSION
        metrics.baseline_match_level = "none"
        metrics.w_rem_star = 0.0
        metrics.g_rec_total = 0.0
        metrics.recovery_window_evidence = []

        if not raw_windows:
            metrics.c_rec = 0.0
            metrics.c_rec_plan = 0.0
            metrics.c_rec_sim = 0.0
            metrics.c_rec_index_100 = 0.0
            metrics.num_recovery_windows = 0
            metrics.formal_window_count = 0
            metrics.skipped_window_count = 0
            metrics.baseline_match_level = "none"
            metrics.c_rec_valid = True
            metrics.c_rec_missing_reason = ""
            return metrics

        if not baselines:
            metrics.c_rec = 0.0
            metrics.c_rec_plan = 0.0
            metrics.c_rec_sim = 0.0
            metrics.c_rec_index_100 = None
            metrics.num_recovery_windows = 0
            metrics.formal_window_count = 0
            metrics.skipped_window_count = len(raw_windows)
            metrics.c_rec_valid = False
            metrics.c_rec_missing_reason = "stage_baseline_missing"
            metrics.baseline_match_level = "none"
            metrics.recovery_window_evidence = [
                self._build_window_evidence(
                    window_index=i,
                    win=win,
                    valid=False,
                    missing_reason="stage_baseline_missing",
                    baseline=None,
                    baseline_match_level="none",
                )
                for i, win in enumerate(raw_windows)
            ]
            return metrics

        cfg = self.crec_config
        eps = float(cfg["epsilon"])
        huber_k = float(cfg["huber_k"])
        mahalanobis_regularizer = float(cfg["mahalanobis_regularizer"])
        allow_family_fallback = bool(cfg.get("allow_family_baseline_fallback", False))
        allow_terminal_segment_fallback = bool(
            cfg.get("allow_terminal_segment_fallback", True)
        )
        allow_anchor_segment_fallback = bool(
            cfg.get("allow_anchor_segment_fallback", False)
        )

        # --- reconstruct StageAnchor objects for fast baseline lookup ---
        anchors: List[StageAnchor] = []
        for a in anchor_dicts:
            anchors.append(
                StageAnchor(
                    family=str(a.get("family", "other")),
                    stage=int(a.get("stage", -1)),
                    modality=str(a.get("modality", "planning")),
                    progress=float(a.get("progress", 0.0)),
                )
            )
        metrics.stage_switches = StageResolver.stage_switches(anchors)

        # cog volumes at each step  (for W_cog computation)
        cog_vol_by_step: Dict[int, Dict[str, float]] = {}
        cog_validity_by_step: Dict[int, Dict[str, float]] = {}
        for entry in cog_steps:
            step_idx = int(entry.get("step", -1))
            channels = entry.get("channels") or {}
            if not isinstance(channels, Mapping):
                continue
            vols: Dict[str, float] = {}
            vals: Dict[str, float] = {}
            for ch, sub in channels.items():
                if not isinstance(sub, Mapping):
                    continue
                vols[str(ch)] = float(sub.get("volume", 0.0) or 0.0)
                v = sub.get("validity")
                vals[str(ch)] = 1.0 if v is None else float(v)
            cog_vol_by_step[step_idx] = vols
            cog_validity_by_step[step_idx] = vals

        records: List[StageRecoveryRecord] = []
        evidence_records: List[Dict[str, Any]] = []
        baseline_match_levels: List[str] = []
        observed_censored_costs: List[float] = []
        observed_censored_numerators: List[float] = []
        n_trans = len(transitions)
        episode_id_raw = tracker_summary.get("episode_id")
        if episode_id_raw in (None, "") and transitions:
            first_transition = transitions[0]
            if isinstance(first_transition, Mapping):
                episode_id_raw = first_transition.get("episode_id")
                if episode_id_raw in (None, ""):
                    snapshot = first_transition.get("info_snapshot") or {}
                    if isinstance(snapshot, Mapping):
                        episode_id_raw = snapshot.get("episode_id")
        episode_id = (
            str(episode_id_raw)
            if episode_id_raw not in (None, "")
            else None
        )
        canonical_transition_indexing = any(
            isinstance(record, Mapping) and "transition_step" in record
            for record in transitions
        )

        for window_index, win in enumerate(raw_windows):
            window_recovered = win.get("recovered") is not False
            t_d = int(win.get("t_d", 0))
            t_r = int(win.get("t_r", t_d))
            anchor_t_d_payload = win.get("anchor_t_d") or {}
            anchor_t_d = StageAnchor(
                family=str(anchor_t_d_payload.get("family", "other")),
                stage=int(anchor_t_d_payload.get("stage", -1)),
                modality=str(anchor_t_d_payload.get("modality", "planning")),
                progress=float(anchor_t_d_payload.get("progress", 0.0)),
            )
            # A terminal/unknown disturbance anchor has no independent clean
            # continuation denominator.  The online tracker is responsible for
            # backtracking t_d to the last real stage; if it cannot, fail closed.
            if anchor_t_d.stage < 0:
                evidence_records.append(
                    self._build_window_evidence(
                        window_index=window_index,
                        win=win,
                        valid=False,
                        missing_reason="invalid_td_anchor",
                        baseline=None,
                        baseline_match_level="none",
                    )
                )
                continue
            baseline_td, baseline_match_level = _select_baseline_with_policy(
                baselines,
                anchor_t_d,
                allow_family_fallback=allow_family_fallback,
            )
            if baseline_td is None:
                evidence_records.append(
                    self._build_window_evidence(
                        window_index=window_index,
                        win=win,
                        valid=False,
                        missing_reason=_selection_missing_reason(
                            baseline_match_level
                        ),
                        baseline=None,
                        baseline_match_level=baseline_match_level,
                    )
                )
                continue
            W_rem_star = _baseline_w_rem_star(
                baseline_td,
                episode_id=episode_id,
                stage=int(anchor_t_d.stage),
            )
            if W_rem_star is None:
                evidence_records.append(
                    self._build_window_evidence(
                        window_index=window_index,
                        win=win,
                        valid=False,
                        missing_reason="episode_stage_remaining_work_missing",
                        baseline=baseline_td,
                        baseline_match_level=baseline_match_level,
                    )
                )
                continue
            if not window_recovered and n_trans == 0:
                evidence_records.append(
                    self._build_window_evidence(
                        window_index=window_index,
                        win=win,
                        valid=False,
                        missing_reason=str(
                            win.get("closure_reason")
                            or "recovery_window_censored"
                        ),
                        baseline=baseline_td,
                        baseline_match_level=baseline_match_level,
                    )
                )
                continue

            step_trace = win.get("step_trace") or []
            # ------------------------------------------------------------------
            # Segment the window by contiguous stage runs to compute φ_k.
            # ------------------------------------------------------------------
            segments: List[Dict[str, Any]] = []
            if step_trace:
                cur: Optional[Dict[str, Any]] = None
                for entry in step_trace:
                    step_idx = int(entry.get("step", 0))
                    fam = str(entry.get("family", anchor_t_d.family))
                    stage_i = int(entry.get("stage", anchor_t_d.stage))
                    mod = str(entry.get("modality", anchor_t_d.modality))
                    q = float(entry.get("q", 0.0))
                    if (
                        cur is None
                        or cur["family"] != fam
                        or cur["stage"] != stage_i
                        or cur["modality"] != mod
                    ):
                        cur = {
                            "family": fam,
                            "stage": stage_i,
                            "modality": mod,
                            "start": step_idx,
                            "end": step_idx,
                            "q_start": q,
                            "q_end": q,
                        }
                        segments.append(cur)
                    else:
                        cur["end"] = step_idx
                        cur["q_end"] = q
            if not segments:
                segments = [
                    {
                        "family": anchor_t_d.family,
                        "stage": anchor_t_d.stage,
                        "modality": anchor_t_d.modality,
                        "start": t_d,
                        "end": t_r,
                        "q_start": 0.0,
                        "q_end": 0.0,
                    }
                ]

            # ------------------------------------------------------------------
            # Per-step g_t accumulation (plan / sim dimensions + cognitive channel)
            # ------------------------------------------------------------------
            g_rec_total = 0.0
            g_rec_plan = 0.0
            g_rec_sim = 0.0
            g_cog_total = 0.0
            g_phy_total = 0.0
            g_debt_total = 0.0
            cog_swe = 0.0
            cog_swe_pln = 0.0
            cog_swe_per = 0.0
            cog_swe_coord = 0.0
            phy_swe = 0.0
            phy_swe_plan = 0.0
            phy_swe_sim = 0.0
            phy_swe_gap = 0.0
            phy_swe_redo = 0.0
            state_debt_swe = 0.0
            state_debt_prog = 0.0
            state_debt_goal = 0.0
            state_debt_risk = 0.0
            runtime_overhead = 0.0
            runtime_baseline = 0.0
            omega_vals: List[float] = []
            gap_steps = 0
            redo_steps = 0
            template_codes = tuple(win.get("template_codes") or ())
            window_template_set = {str(code) for code in template_codes}
            channel_covariance_missing = False

            for step_idx in range(t_d, t_r + 1):
                if step_idx <= 0:
                    continue
                # Canonical tracker transitions use one-based transition_step
                # with no synthetic initial record.  Historical critic/planner
                # artifacts include an initial state at index zero.
                record_index = (
                    step_idx - 1
                    if canonical_transition_indexing
                    else step_idx
                )
                if record_index < 0 or record_index >= n_trans:
                    continue
                prev_rec = (
                    transitions[record_index - 1]
                    if record_index > 0
                    else {}
                )
                curr_rec = transitions[record_index]
                if record_index < len(anchors):
                    anchor_step = anchors[record_index]
                else:
                    anchor_step = anchor_t_d
                baseline_step = (
                    _select_baseline(
                        baselines,
                        anchor_step,
                        allow_family_fallback=allow_family_fallback,
                    )
                    or baseline_td
                )

                feats = self._derive_multidim_step_features(
                    prev_rec, curr_rec, anchor_step, baseline_step,
                )

                # Cognitive work: volume × invalidity over baseline channels
                vols = cog_vol_by_step.get(step_idx, {})
                vals = cog_validity_by_step.get(step_idx, {})

                # Physical work: plan overhead + sim overhead above baselines
                plan_base = max(1.0, baseline_step.dim("plan").tau_bar)
                sim_base = max(1.0, baseline_step.dim("sim").tau_bar)
                W_phy_plan = max(0.0, (feats["d_plan"] - plan_base)) / plan_base
                W_phy_sim = max(0.0, (feats["d_sim"] - sim_base)) / sim_base
                # Add redo penalty when observed (retries / backtracks).
                if feats["redo_event"]:
                    W_phy_sim += 1.0
                phy_swe += W_phy_sim + W_phy_plan

                # State debt & Ω_t
                D_prog = feats["dp_neg"] / (baseline_step.delta_p_bar + 1e-6)
                D_goal = feats["dq_neg"] / (baseline_step.delta_q_bar + 1e-6)
                if baseline_step.r_bar > 0:
                    D_risk = max(
                        0.0,
                        (feats["risk"] - baseline_step.r_bar)
                        / (baseline_step.r_bar + 1e-6),
                    )
                else:
                    D_risk = max(0.0, feats["risk"])
                omega_t = 1.0 + (
                    _huber(D_prog, huber_k)
                    + _huber(D_goal, huber_k)
                    + _huber(D_risk, huber_k)
                ) / 3.0
                omega_vals.append(omega_t)

                step_template_set = set(feats.get("step_template_codes") or ())
                active_template_set = window_template_set | step_template_set
                search_gap = (
                    feats["has_nav"]
                    and feats["d_sim"] > 0.0
                    and feats["no_forward_progress"]
                    and bool(active_template_set & {"T3", "R1"})
                )
                manip_gap = (
                    feats["has_manip"]
                    and feats["d_sim"] > 0.0
                    and feats["no_forward_progress"]
                    and bool(active_template_set & {"T1", "T2", "T4"})
                )
                refined_W_phy_gap = 0.0
                if search_gap:
                    refined_W_phy_gap += feats["d_sim"] / sim_base
                if manip_gap:
                    refined_W_phy_gap += 0.5 * feats["d_sim"] / sim_base
                if feats["has_wait"] and feats["no_forward_progress"]:
                    # Wait is idle physical work, not a coordination event.
                    # Use the larger observed plan/sim share so a zero-sim
                    # Wait still contributes one nominal planning-step gap.
                    refined_W_phy_gap += max(
                        feats["d_plan"] / plan_base,
                        feats["d_sim"] / sim_base,
                    )

                refined_W_phy_redo_plan = 1.0 if feats["redo_event"] else 0.0
                refined_W_phy_redo_sim = (
                    feats["d_sim"] / sim_base
                    if feats["redo_event"] and feats["d_sim"] > 0.0
                    else 0.0
                )
                refined_W_phy_plan = max(
                    0.0, (feats["d_plan"] - plan_base)
                ) / plan_base + refined_W_phy_redo_plan
                refined_W_phy_sim = max(
                    0.0, (feats["d_sim"] - sim_base)
                ) / sim_base + refined_W_phy_redo_sim
                old_phy_total = W_phy_plan + W_phy_sim
                refined_phy_total = (
                    refined_W_phy_plan + refined_W_phy_sim + refined_W_phy_gap
                )
                phy_swe += refined_phy_total - old_phy_total
                phy_swe_plan += refined_W_phy_plan
                phy_swe_sim += refined_W_phy_sim
                phy_swe_gap += refined_W_phy_gap
                phy_swe_redo += refined_W_phy_redo_plan + refined_W_phy_redo_sim
                if refined_W_phy_gap > 0.0:
                    gap_steps += 1
                if feats["redo_event"]:
                    redo_steps += 1
                debt_prog = _huber(D_prog, huber_k)
                debt_goal = _huber(D_goal, huber_k)
                debt_risk = _huber(D_risk, huber_k)
                state_step = (debt_prog + debt_goal + debt_risk) / 3.0
                state_debt_swe += state_step
                state_debt_prog += debt_prog
                state_debt_goal += debt_goal
                state_debt_risk += debt_risk
                omega_t = 1.0 + state_step
                omega_vals[-1] = omega_t

                # Eq. (2): covariance-aware cognitive, physical and debt
                # channel deviations and their RMS recovery intensity.
                # Eq. (3) applies time/remaining-work normalization below.
                feature_groups = build_recovery_feature_groups(
                    volumes=vols,
                    baseline_volumes=baseline_step.v_bar_channels,
                    validities=vals,
                    plan_work=feats["d_plan"],
                    sim_work=feats["d_sim"],
                    plan_baseline=plan_base,
                    sim_baseline=sim_base,
                    idle_gap=refined_W_phy_gap,
                    redo_event=bool(feats["redo_event"]),
                    progress_debt=feats["dp_neg"],
                    goal_debt=feats["dq_neg"],
                    risk=feats["risk"],
                    progress_baseline=baseline_step.delta_p_bar,
                    goal_baseline=baseline_step.delta_q_bar,
                    risk_baseline=baseline_step.r_bar,
                )
                cog_channels = dict(zip(_COG_CHANNELS, feature_groups["cog"]))
                cog_step = float(sum(cog_channels.values()))
                cog_swe += cog_step
                cog_swe_pln += cog_channels.get("pln", 0.0)
                cog_swe_per += cog_channels.get("per", 0.0)
                cog_swe_coord += cog_channels.get("coord", 0.0)
                g_cog = _channel_mahalanobis(
                    feature_groups["cog"],
                    baseline_step,
                    "cog",
                    mahalanobis_regularizer,
                )
                g_phy = _channel_mahalanobis(
                    feature_groups["phy"],
                    baseline_step,
                    "phy",
                    mahalanobis_regularizer,
                )
                g_debt = _channel_mahalanobis(
                    feature_groups["debt"],
                    baseline_step,
                    "debt",
                    mahalanobis_regularizer,
                )
                if g_cog is None or g_phy is None or g_debt is None:
                    channel_covariance_missing = True
                    break
                g_rec_step = recovery_intensity((g_cog, g_phy, g_debt))
                time_weight = max(float(feats["d_sim"]), 1.0) / max(
                    float(baseline_step.tau_star), eps
                )
                g_rec_total += time_weight * g_rec_step
                g_cog_total += time_weight * g_cog
                g_phy_total += time_weight * g_phy
                g_debt_total += time_weight * g_debt
                # Kept only inside the detailed per-window artifact.
                g_rec_plan += math.sqrt(
                    (g_cog * g_cog + g_phy * g_phy) / 2.0
                )
                g_rec_sim += g_rec_step

                # Runtime (monitor only)
                runtime_overhead += feats.get("dt_runtime", 0.0)
                runtime_baseline += max(1e-6, baseline_step.dim("runtime").tau_bar)

            if channel_covariance_missing:
                evidence_records.append(
                    self._build_window_evidence(
                        window_index=window_index,
                        win=win,
                        valid=False,
                        missing_reason="stage_baseline_recovery_covariance_missing",
                        baseline=baseline_td,
                        baseline_match_level=baseline_match_level,
                    )
                )
                continue

            # ------------------------------------------------------------------
            # Cross-stage W_rem^x  — φ_k × W_rem^{*,seg,x}(a^k) per dimension.
            # ------------------------------------------------------------------
            W_rem_plan = 0.0
            W_rem_sim = 0.0
            W_rem_runtime = 0.0
            total_span = max(1, sum(seg["end"] - seg["start"] + 1 for seg in segments))
            stages_spanned: List[int] = []
            segment_missing_reason = ""
            segment_fallbacks: List[Dict[str, Any]] = []
            segment_baselines: List[Dict[str, Any]] = []
            window_match_levels = {baseline_match_level}
            for seg in segments:
                seg_anchor = StageAnchor(
                    family=seg["family"],
                    stage=seg["stage"],
                    modality=seg["modality"],
                )
                seg_baseline, seg_match_level, seg_fallback_reason = _select_segment_baseline_with_policy(
                    baselines,
                    seg_anchor,
                    fallback_baseline=baseline_td,
                    allow_family_fallback=allow_family_fallback,
                    allow_terminal_fallback=allow_terminal_segment_fallback,
                    allow_anchor_fallback=allow_anchor_segment_fallback,
                )
                if seg_baseline is None:
                    segment_missing_reason = (
                        seg_fallback_reason or "stage_baseline_segment_missing"
                    )
                    break
                if int(seg_anchor.stage) >= 0:
                    seg_w_rem_star = _baseline_w_rem_star(
                        seg_baseline,
                        episode_id=episode_id,
                        stage=int(seg_anchor.stage),
                    )
                    if seg_w_rem_star is None:
                        segment_missing_reason = (
                            "episode_stage_remaining_work_missing"
                        )
                        break
                window_match_levels.add(seg_match_level)
                segment_baselines.append(
                    {
                        "segment": {
                            "family": seg_anchor.family,
                            "stage": int(seg_anchor.stage),
                            "modality": seg_anchor.modality,
                            "start": int(seg["start"]),
                            "end": int(seg["end"]),
                        },
                        "baseline_key": {
                            "family": seg_baseline.family,
                            "stage": int(seg_baseline.stage),
                            "modality": seg_baseline.modality,
                        },
                        "match_level": seg_match_level,
                        "formal_support": _baseline_support_provenance(
                            seg_baseline
                        ),
                    }
                )
                if seg_fallback_reason:
                    segment_fallbacks.append(
                        {
                            "family": seg_anchor.family,
                            "stage": int(seg_anchor.stage),
                            "modality": seg_anchor.modality,
                            "start": int(seg["start"]),
                            "end": int(seg["end"]),
                            "match_level": seg_match_level,
                            "reason": seg_fallback_reason,
                            "fallback_to": {
                                "family": baseline_td.family,
                                "stage": int(baseline_td.stage),
                                "modality": baseline_td.modality,
                            },
                        }
                    )
                seg_len = seg["end"] - seg["start"] + 1
                q_delta = max(0.0, float(seg.get("q_end", 0.0)) - float(seg.get("q_start", 0.0)))
                # φ_k: fractional contribution of this stage to the window.
                # Dimension-specific, but we only have q-based progress; use
                # q_delta as a proxy, fall back to step-length share if Δq=0.
                phi_k_q = min(1.0, q_delta)
                phi_k_share = seg_len / total_span
                phi_k = max(phi_k_q, phi_k_share)
                W_rem_plan += phi_k * seg_baseline.dim("plan").W_rem_seg_bar
                W_rem_sim += phi_k * seg_baseline.dim("sim").W_rem_seg_bar
                W_rem_runtime += phi_k * seg_baseline.dim("runtime").W_rem_seg_bar
                stages_spanned.append(int(seg["stage"]))

            window_match_level = (
                sorted(window_match_levels)[0]
                if len(window_match_levels) == 1
                else "mixed:" + "+".join(sorted(window_match_levels))
            )

            if segment_missing_reason:
                evidence_records.append(
                    self._build_window_evidence(
                        window_index=window_index,
                        win=win,
                        valid=False,
                        missing_reason=segment_missing_reason,
                        baseline=baseline_td,
                        baseline_match_level=window_match_level,
                        segments=segments,
                        segment_fallbacks=segment_fallbacks,
                        segment_baselines=segment_baselines,
                    )
                )
                continue
            baseline_match_levels.append(window_match_level)

            W_rem_plan = max(W_rem_plan, 1.0)
            W_rem_sim = max(W_rem_sim, 1.0)
            W_rem_runtime = max(W_rem_runtime, 0.0)
            c_rec_plan = g_rec_plan / (W_rem_plan + eps)
            c_rec_sim = g_rec_sim / (W_rem_sim + eps)
            c_rec_cog = g_cog_total / (W_rem_star + eps)
            c_rec_cog_pln = cog_swe_pln / (W_rem_star + eps)
            c_rec_cog_per = cog_swe_per / (W_rem_star + eps)
            c_rec_cog_coord = cog_swe_coord / (W_rem_star + eps)
            c_rec_phy_plan = phy_swe_plan / (W_rem_star + eps)
            c_rec_phy_sim = phy_swe_sim / (W_rem_star + eps)
            c_rec_phy_gap = phy_swe_gap / (W_rem_star + eps)
            c_rec_phy_redo = phy_swe_redo / (W_rem_star + eps)
            c_rec_phy = g_phy_total / (W_rem_star + eps)
            c_rec_state_prog = state_debt_prog / (W_rem_star + eps)
            c_rec_state_goal = state_debt_goal / (W_rem_star + eps)
            c_rec_state_risk = state_debt_risk / (W_rem_star + eps)
            c_rec_state_debt = g_debt_total / (W_rem_star + eps)
            c_rec = g_rec_total / (W_rem_star + eps)
            omega_mean = sum(omega_vals) / len(omega_vals) if omega_vals else 1.0
            runtime_overhead_ratio = (
                runtime_overhead / (runtime_baseline + eps)
                if runtime_baseline > 0
                else 0.0
            )

            record = StageRecoveryRecord(
                    stage_family=anchor_t_d.family,
                    stage_index=int(anchor_t_d.stage),
                    stage_modality=anchor_t_d.modality,
                    t_d=int(t_d),
                    t_r=int(t_r),
                    duration=max(0, int(t_r) - int(t_d) + 1),
                    cog_swe=float(cog_swe),
                    cog_swe_pln=float(cog_swe_pln),
                    cog_swe_per=float(cog_swe_per),
                    cog_swe_coord=float(cog_swe_coord),
                    phy_swe=float(phy_swe),
                    phy_swe_plan=float(phy_swe_plan),
                    phy_swe_sim=float(phy_swe_sim),
                    phy_swe_gap=float(phy_swe_gap),
                    phy_swe_redo=float(phy_swe_redo),
                    state_debt_swe=float(state_debt_swe),
                    state_debt_prog=float(state_debt_prog),
                    state_debt_goal=float(state_debt_goal),
                    state_debt_risk=float(state_debt_risk),
                    omega_mean=float(omega_mean),
                    g_rec_sum_total=float(g_rec_total),
                    g_rec_sum_plan=float(g_rec_plan),
                    g_rec_sum_sim=float(g_rec_sim),
                    W_rem_star=float(W_rem_star),
                    W_rem_plan=float(W_rem_plan),
                    W_rem_sim=float(W_rem_sim),
                    W_rem_runtime=float(W_rem_runtime),
                    runtime_overhead=float(runtime_overhead),
                    c_rec_plan=float(c_rec_plan),
                    c_rec_sim=float(c_rec_sim),
                    c_rec_cog=float(c_rec_cog),
                    c_rec_cog_pln=float(c_rec_cog_pln),
                    c_rec_cog_per=float(c_rec_cog_per),
                    c_rec_cog_coord=float(c_rec_cog_coord),
                    c_rec_phy=float(c_rec_phy),
                    c_rec_phy_plan=float(c_rec_phy_plan),
                    c_rec_phy_sim=float(c_rec_phy_sim),
                    c_rec_phy_gap=float(c_rec_phy_gap),
                    c_rec_phy_redo=float(c_rec_phy_redo),
                    c_rec_state_debt=float(c_rec_state_debt),
                    c_rec_state_prog=float(c_rec_state_prog),
                    c_rec_state_goal=float(c_rec_state_goal),
                    c_rec_state_risk=float(c_rec_state_risk),
                    c_rec=float(c_rec),
                    runtime_overhead_ratio=float(runtime_overhead_ratio),
                    template_codes=list(template_codes),
                    stages_spanned=stages_spanned,
                    gap_steps=int(gap_steps),
                    redo_steps=int(redo_steps),
                    # Backwards-compat: expose the sim-dim values in the
                    # legacy g_rec_sum / W_rem_seg fields.
                    g_rec_sum=float(g_rec_total),
                    W_rem_seg=float(W_rem_star),
                    closed_by_stage_switch=False,
                )
            if window_recovered:
                records.append(record)
            else:
                observed_censored_costs.append(float(c_rec))
                observed_censored_numerators.append(float(g_rec_total))
            evidence_records.append(
                self._build_window_evidence(
                    window_index=window_index,
                    win=win,
                    valid=window_recovered,
                    missing_reason=(
                        ""
                        if window_recovered
                        else str(
                            win.get("closure_reason")
                            or "recovery_window_censored"
                        )
                    ),
                    baseline=baseline_td,
                    baseline_match_level=window_match_level,
                    w_rem_star=W_rem_star,
                    g_rec_total=g_rec_total,
                    cog_swe=cog_swe,
                    phy_swe=phy_swe,
                    state_debt_swe=state_debt_swe,
                    c_rec=c_rec,
                    segments=segments,
                    segment_fallbacks=segment_fallbacks,
                    segment_baselines=segment_baselines,
                    observed_lower_bound=not window_recovered,
                )
            )

        metrics.stage_recovery_records = records
        metrics.recovery_window_evidence = evidence_records
        metrics.num_recovery_windows = len(records)
        metrics.formal_window_count = len(records)
        metrics.skipped_window_count = max(0, len(raw_windows) - len(records))
        metrics.c_rec_valid = metrics.skipped_window_count == 0
        if metrics.skipped_window_count > 0:
            missing_reasons = [
                str(ev.get("missing_reason", "recovery_window_unavailable"))
                for ev in evidence_records
                if not bool(ev.get("valid", False))
            ]
            metrics.c_rec_missing_reason = (
                "partial_recovery_windows:"
                + ",".join(sorted(set(missing_reasons)))
                if records
                else (
                    missing_reasons[0]
                    if missing_reasons
                    else "recovery_window_unavailable"
                )
            )
        else:
            metrics.c_rec_missing_reason = ""
        if baseline_match_levels:
            unique_levels = sorted(set(baseline_match_levels))
            metrics.baseline_match_level = (
                unique_levels[0] if len(unique_levels) == 1 else "mixed"
            )
        elif raw_windows:
            metrics.baseline_match_level = "none"
        metrics.c_rec_plan = float(sum(r.c_rec_plan for r in records))
        metrics.c_rec_sim = float(sum(r.c_rec_sim for r in records))
        metrics.c_rec = float(sum(r.c_rec for r in records))
        metrics.c_rec_index_100 = (
            bounded_recovery_cost_index(metrics.c_rec)
            if metrics.c_rec_valid
            else None
        )
        metrics.g_rec_total = float(sum(r.g_rec_sum_total for r in records))
        metrics.observed_censored_window_count = len(observed_censored_costs)
        if observed_censored_costs:
            metrics.c_rec_observed_lower_bound = float(
                metrics.c_rec + sum(observed_censored_costs)
            )
            metrics.g_rec_observed_total = float(
                metrics.g_rec_total + sum(observed_censored_numerators)
            )
        else:
            metrics.c_rec_observed_lower_bound = None
            metrics.g_rec_observed_total = None
        metrics.w_rem_star = float(
            sum(r.W_rem_star for r in records) / len(records)
        ) if records else 0.0
        metrics.c_rec_cog = float(sum(r.c_rec_cog for r in records))
        metrics.c_rec_cog_pln = float(sum(r.c_rec_cog_pln for r in records))
        metrics.c_rec_cog_per = float(sum(r.c_rec_cog_per for r in records))
        metrics.c_rec_cog_coord = float(sum(r.c_rec_cog_coord for r in records))
        metrics.c_rec_phy = float(sum(r.c_rec_phy for r in records))
        metrics.c_rec_phy_plan = float(sum(r.c_rec_phy_plan for r in records))
        metrics.c_rec_phy_sim = float(sum(r.c_rec_phy_sim for r in records))
        metrics.c_rec_phy_gap = float(sum(r.c_rec_phy_gap for r in records))
        metrics.c_rec_phy_redo = float(sum(r.c_rec_phy_redo for r in records))
        metrics.c_rec_state_debt = float(sum(r.c_rec_state_debt for r in records))
        metrics.c_rec_state_prog = float(sum(r.c_rec_state_prog for r in records))
        metrics.c_rec_state_goal = float(sum(r.c_rec_state_goal for r in records))
        metrics.c_rec_state_risk = float(sum(r.c_rec_state_risk for r in records))
        if records:
            metrics.c_rec_state_modulation = float(
                sum(r.omega_mean for r in records) / len(records)
            )
            metrics.c_rec_p90 = _percentile([r.c_rec for r in records], 0.9)
            metrics.runtime_overhead_ratio = float(
                sum(r.runtime_overhead_ratio for r in records) / len(records)
            )
            metrics.stages_spanned_mean = float(
                sum(len(r.stages_spanned) for r in records) / len(records)
            )
        else:
            metrics.c_rec_state_modulation = 1.0
            metrics.c_rec_p90 = 0.0
            metrics.runtime_overhead_ratio = 0.0
            metrics.stages_spanned_mean = 0.0

        # Expose a flat view of events for CSV/JSON logging
        metrics.rebound_events = []
        for win in raw_windows:
            common = {
                "templates": list(win.get("template_codes") or []),
                "severity": float(win.get("severity_sum", 0.0)),
            }
            metrics.rebound_events.append(
                {
                    **common,
                    "kind": "t_d",
                    "step": int(win.get("t_d", 0)),
                    "anchor": win.get("anchor_t_d"),
                }
            )
            if win.get("recovered") is not False:
                metrics.rebound_events.append(
                    {
                        **common,
                        "kind": "t_r",
                        "step": int(win.get("t_r", 0)),
                        "anchor": win.get("anchor_t_r"),
                    }
                )
            else:
                metrics.rebound_events.append(
                    {
                        **common,
                        "kind": "censored",
                        "step": int(win.get("t_r", win.get("t_d", 0))),
                        "anchor": win.get("anchor_t_r"),
                        "reason": str(
                            win.get("closure_reason")
                            or "recovery_window_censored"
                        ),
                    }
                )

        return metrics

    # ------------------------------------------------------------------
    # Window evidence records keep C_rec auditable instead of opaque.
    # ------------------------------------------------------------------
    def _build_window_evidence(
        self,
        *,
        window_index: int,
        win: Mapping[str, Any],
        valid: bool,
        missing_reason: str,
        baseline: Optional[StageBaseline],
        baseline_match_level: str,
        w_rem_star: float = 0.0,
        g_rec_total: float = 0.0,
        cog_swe: float = 0.0,
        phy_swe: float = 0.0,
        state_debt_swe: float = 0.0,
        c_rec: float = 0.0,
        segments: Optional[Sequence[Mapping[str, Any]]] = None,
        segment_fallbacks: Optional[Sequence[Mapping[str, Any]]] = None,
        segment_baselines: Optional[Sequence[Mapping[str, Any]]] = None,
        observed_lower_bound: bool = False,
    ) -> Dict[str, Any]:
        anchor_payload = win.get("anchor_t_d") or {}
        baseline_key = None
        if baseline is not None:
            baseline_key = {
                "family": baseline.family,
                "stage": int(baseline.stage),
                "modality": baseline.modality,
            }
        return {
            "window_index": int(window_index),
            "valid": bool(valid),
            "observed_lower_bound": bool(observed_lower_bound),
            "missing_reason": str(missing_reason or ""),
            "c_rec_formula_version": REBOUND_FORMULA_VERSION,
            "t_d": int(win.get("t_d", 0)),
            "t_r": int(win.get("t_r", win.get("t_d", 0))),
            "t_end": int(win.get("t_end", win.get("t_r", win.get("t_d", 0)))),
            "trigger": str(win.get("trigger", "")),
            "template_codes": list(win.get("template_codes") or []),
            "severity": float(win.get("severity_sum", 0.0) or 0.0),
            "anchor_t_d": dict(anchor_payload) if isinstance(anchor_payload, Mapping) else {},
            "baseline_key": baseline_key,
            "baseline_match_level": str(baseline_match_level or "none"),
            "baseline_formal_support": _baseline_support_provenance(baseline),
            "segment_count": len(segments or ()),
            "segments": [dict(seg) for seg in (segments or ())],
            "segment_fallbacks": [
                dict(fallback) for fallback in (segment_fallbacks or ())
            ],
            "segment_baselines": [
                dict(segment) for segment in (segment_baselines or ())
            ],
            "features": {
                "cognitive": {
                    "source": "online_tracker.cognitive_steps",
                    "raw": float(cog_swe),
                    "normalized": (
                        float(cog_swe / max(w_rem_star, 1.0))
                        if valid or observed_lower_bound
                        else 0.0
                    ),
                },
                "physical": {
                    "source": "planner_info.step_deltas+action_trace",
                    "raw": float(phy_swe),
                    "normalized": (
                        float(phy_swe / max(w_rem_star, 1.0))
                        if valid or observed_lower_bound
                        else 0.0
                    ),
                },
                "state_debt": {
                    "source": "progress/q/risk deltas",
                    "raw": float(state_debt_swe),
                    "normalized": (
                        float(state_debt_swe / max(w_rem_star, 1.0))
                        if valid or observed_lower_bound
                        else 0.0
                    ),
                },
                "g_rec_total": {
                    "source": "unified_cog_phy_state_density",
                    "raw": float(g_rec_total),
                    "normalized": float(c_rec),
                },
                "W_rem_star": {
                    "source": (
                        "stage_baseline.W_rem_star_by_episode_stage"
                    ),
                    "raw": float(w_rem_star),
                    "normalized": float(w_rem_star),
                },
            },
        }

    # ------------------------------------------------------------------
    # Shared helper: per-step multidim features
    # ------------------------------------------------------------------
    def _derive_multidim_step_features(
        self,
        prev: Mapping[str, Any],
        curr: Mapping[str, Any],
        anchor: StageAnchor,
        baseline: StageBaseline,
    ) -> Dict[str, Any]:
        prev_info = prev.get("info_snapshot", prev) if isinstance(prev, Mapping) else {}
        curr_info = curr.get("info_snapshot", curr) if isinstance(curr, Mapping) else {}
        if not isinstance(prev_info, Mapping):
            prev_info = prev if isinstance(prev, Mapping) else {}
        if not isinstance(curr_info, Mapping):
            curr_info = curr if isinstance(curr, Mapping) else {}

        def _get(name: str, default: float = 0.0) -> float:
            value: Any = None
            if isinstance(curr_info, Mapping) and name in curr_info:
                value = curr_info[name]
            elif isinstance(curr, Mapping) and name in curr:
                value = curr[name]
            if value is None:
                return float(default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        def _get_prev(name: str, default: float = 0.0) -> float:
            value: Any = None
            if isinstance(prev_info, Mapping) and name in prev_info:
                value = prev_info[name]
            elif isinstance(prev, Mapping) and name in prev:
                value = prev[name]
            if value is None:
                return float(default)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float(default)

        def _has(name: str) -> bool:
            return (
                isinstance(curr_info, Mapping) and name in curr_info
            ) or (isinstance(curr, Mapping) and name in curr)

        d_plan = (
            max(0.0, _get("delta_plan"))
            if _has("delta_plan")
            else max(
                0.0,
                _get("planning_step_count", 0.0)
                - _get_prev("planning_step_count", 0.0),
            )
        )
        if d_plan == 0.0 and not _has("delta_plan"):
            d_plan = 1.0
        d_sim = (
            max(0.0, _get("delta_sim"))
            if _has("delta_sim")
            else max(
                0.0,
                _get("sim_step_count", 0.0)
                - _get_prev("sim_step_count", 0.0),
            )
        )
        dt_runtime = (
            max(0.0, _get("delta_runtime"))
            if _has("delta_runtime")
            else max(
                0.0,
                _get("runtime_elapsed", 0.0)
                - _get_prev("runtime_elapsed", 0.0),
            )
        )

        dp = (
            _get("delta_progress")
            if _has("delta_progress")
            else _get("task_percent_complete", 0.0)
            - _get_prev("task_percent_complete", 0.0)
        )
        dq = (
            _get("delta_q")
            if _has("delta_q")
            else _get("proposition_satisfied_fraction", 0.0)
            - _get_prev("proposition_satisfied_fraction", 0.0)
        )
        dSR = (
            _get("delta_task_state_success")
            if _has("delta_task_state_success")
            else _get("task_state_success", 0.0)
            - _get_prev("task_state_success", 0.0)
        )
        risk = _get(
            "cbf_penalty_step",
            _get("cbf_penalty", _get("safety_penalty", 0.0)),
        )

        tags = curr_info.get("analysis_tags") or curr.get("analysis_tags") or []
        if isinstance(tags, str):
            tags = [tags]
        tag_set = {str(t) for t in tags}

        raw_template_codes = (
            curr_info.get("_rebound_template_codes")
            or curr.get("_rebound_template_codes")
            or curr_info.get("rebound_template_codes")
            or curr.get("rebound_template_codes")
            or []
        )
        if isinstance(raw_template_codes, str):
            raw_template_codes = [raw_template_codes]
        step_template_codes = {str(code) for code in raw_template_codes}

        frames = []
        high_level_actions = (
            curr_info.get("high_level_actions")
            or curr.get("high_level_actions")
            or {}
        )
        if isinstance(high_level_actions, Mapping):
            step_id = int(
                _get("planning_step_count", _get("loop_step_count", 0.0)) or 0.0
            )
            for agent_uid, action_entry in high_level_actions.items():
                try:
                    parsed_uid = int(agent_uid)
                except (TypeError, ValueError):
                    parsed_uid = None
                frame = parse_action_entry(
                    action_entry,
                    step=step_id,
                    agent_uid=parsed_uid,
                )
                if frame is not None:
                    frames.append(frame)

        has_nav = any(frame.is_nav for frame in frames)
        has_manip = any(frame.is_manipulation for frame in frames)
        has_pick = any(frame.is_pick for frame in frames)
        has_place = any(frame.is_place for frame in frames)
        coord_event = any(
            frame.name in {"handover", "coordinate"} for frame in frames
        )
        has_wait = any(frame.name == "wait" for frame in frames)
        redo_event = bool(tag_set & {"retry_event", "backtrack_event"})
        no_forward_progress = dp <= 1e-6 and dq <= 1e-6 and dSR <= 1e-6

        return {
            "d_plan": d_plan,
            "d_sim": d_sim,
            "dt_runtime": dt_runtime,
            "dp": dp,
            "dq": dq,
            "dSR": dSR,
            "dp_neg": max(0.0, -dp),
            "dq_neg": max(0.0, -dq),
            "risk": risk,
            "redo_event": redo_event,
            "replan_event": bool(tag_set & {"replan_anchor", "planning_transition"}),
            "response_event": "response_event" in tag_set,
            "coord_event": coord_event,
            "has_nav": has_nav,
            "has_manip": has_manip,
            "has_pick": has_pick,
            "has_place": has_place,
            "has_wait": has_wait,
            "no_forward_progress": no_forward_progress,
            "step_template_codes": sorted(step_template_codes),
        }
