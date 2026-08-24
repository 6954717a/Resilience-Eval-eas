#!/usr/bin/env python3

"""
Rebound Collector

Collects recovery metrics from ReboundManager and CoPAL RecoveryMetrics.
Consolidates data from multiple sources into unified ReboundMetrics.
"""

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .cog_load_tracker import CHANNELS as _COG_CHANNELS
from .cog_load_tracker import CogLoadTracker
from .rebound_metrics import FailureEvent, ReboundMetrics, StageRecoveryRecord
from .stage_anchor import StageAnchor, StageResolver
from .stage_baseline import StageBaseline, StageBaselineDim
from .target_graph import parse_action_entry

logger = logging.getLogger(__name__)


# Default hyper-parameters (override via ``config`` or ``resilience_config.yaml``)
_DEFAULT_CREC_CONFIG = {
    "epsilon": 1e-6,
    "huber_k": 3.0,
    "confirm_steps": 2,          # consecutive steps required to confirm t_d
    "recovery_hold_steps": None, # if None, derived from τ^* of the anchor
    "tau_pln": 0.1,
    "tau_per": 0.1,
    "tau_coord": 0.1,
    "allow_family_baseline_fallback": False,
    "allow_terminal_segment_fallback": True,
    "allow_anchor_segment_fallback": False,
}


def _huber(x: float, k: float) -> float:
    """Huber-like ψ: identity below k, logarithmic above to avoid domination."""
    x = max(0.0, float(x))
    if x <= k:
        return x
    return k + math.log1p(x - k)


def _soft_compress(x: float) -> float:
    """φ(x) = log(1 + x) soft compressor used in g_t^rec."""
    return math.log1p(max(0.0, float(x)))


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


def _select_baseline_with_level(
    baselines: Mapping[Tuple[str, int, str], StageBaseline],
    anchor: StageAnchor,
) -> Tuple[Optional[StageBaseline], str]:
    """Strict lookup of the baseline for an anchor with explicit provenance."""
    if not baselines:
        return None, "none"
    key = anchor.key()
    if key in baselines:
        return baselines[key], "exact"
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
    for candidate in (
        (anchor.family, -1, anchor.modality),
        (anchor.family, -1, "planning"),
    ):
        if candidate in baselines:
            return baselines[candidate]
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
    for candidate in (
        (anchor.family, -1, anchor.modality),
        (anchor.family, -1, "planning"),
    ):
        if candidate in baselines:
            return baselines[candidate], "family"
    return None, "missing"


def _baseline_w_rem_star(baseline: StageBaseline) -> float:
    value = getattr(baseline, "W_rem_star_swe", None)
    if value is None:
        value = getattr(baseline, "W_rem_seg_bar", 1.0)
    try:
        return max(1.0, float(value))
    except (TypeError, ValueError):
        return 1.0


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

    return None, "missing", "stage_baseline_segment_missing"


class ReboundCollector:
    """
    Collects Rebound metrics from planner's ReboundManager and optional CoPAL components.
    """

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
            degradation_monitor: Optional DegradationMonitor for T_rec

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
    # C_rec pipeline (§1.3 of the resilience design doc)
    # ==================================================================
    def compute_stage_crec(
        self,
        transitions: Sequence[Mapping[str, Any]],
        baselines: Mapping[Tuple[str, int, str], StageBaseline],
        episode: Any = None,
        family_override: Optional[str] = None,
        metrics: Optional[ReboundMetrics] = None,
        failure_history: Optional[Sequence[Tuple[int, int, str]]] = None,
    ) -> ReboundMetrics:
        """Compute stage-based recovery cost :math:`C_{rec}^{(i)}`.

        Parameters
        ----------
        transitions:
            Ordered sequence of per-step ``info`` / record dicts. Each entry
            must carry at minimum ``task_percent_complete``,
            ``proposition_satisfied_fraction``, ``sim_step_count`` and
            ``auto_eval_proposition_tracker``. Critic export records work as-is.
        baselines:
            Mapping from ``(f, ℓ, m)`` to :class:`StageBaseline`. Produced by
            :class:`StageBaselineEstimator` on clean runs.
        episode:
            Optional episode object used to resolve ``f`` when the trajectory
            does not carry ``task_family`` directly.
        metrics:
            Optional pre-existing :class:`ReboundMetrics` to populate (e.g.
            one produced by :meth:`collect_from_planner`). When ``None``, a
            fresh instance is created.
        failure_history:
            Optional list of ``(t_start, t_end, strategy)`` tuples (same
            format as ``ReboundManager.failure_history``). Used to resolve
            ``redo_event`` step markers for ``T_t^redo``.
        """
        metrics = metrics or ReboundMetrics()
        if len(transitions) < 2 or not baselines:
            metrics.c_rec = 0.0
            metrics.num_recovery_windows = 0
            metrics.formal_window_count = 0
            metrics.c_rec_valid = False
            metrics.c_rec_missing_reason = (
                "insufficient_transitions" if len(transitions) < 2 else "stage_baseline_missing"
            )
            return metrics

        cfg = self.crec_config
        eps = float(cfg["epsilon"])
        huber_k = float(cfg["huber_k"])
        confirm_steps = int(cfg["confirm_steps"])
        tau_pln = float(cfg["tau_pln"])
        tau_per = float(cfg["tau_per"])
        tau_coord = float(cfg["tau_coord"])

        # Precompute redo step set from failure_history if provided.
        redo_steps: set = set()
        if failure_history:
            for start, end, strategy in failure_history:
                if strategy in {"retry", "rollback", "backtrack"}:
                    for step in range(int(start), int(end) + 1):
                        redo_steps.add(step)

        # Resolve anchors along the trajectory.
        resolver = self.resolver
        anchors: List[StageAnchor] = [
            resolver.resolve(info, episode=episode, family_override=family_override)
            for info in transitions
        ]
        metrics.stage_switches = resolver.stage_switches(anchors)

        # Stream through transitions, opening/closing recovery windows and
        # accumulating per-step g_t^rec.
        records: List[StageRecoveryRecord] = []
        in_window = False
        degr_streak = 0
        current: Optional[_WindowAccumulator] = None
        recovery_hold_target = float(cfg["recovery_hold_steps"] or 0)

        for idx in range(1, len(transitions)):
            prev = transitions[idx - 1]
            curr = transitions[idx]
            anchor = anchors[idx]
            baseline = _select_baseline(baselines, anchor) or StageBaseline(
                family=anchor.family, stage=anchor.stage, modality=anchor.modality
            )

            step_feats = self._derive_step_features(
                prev=prev,
                curr=curr,
                anchor=anchor,
                baseline=baseline,
                tau_pln=tau_pln,
                tau_per=tau_per,
                tau_coord=tau_coord,
                redo_steps=redo_steps,
            )

            # ----------------------------------------------------------
            # Open / close recovery windows based on the contract triggers
            # ----------------------------------------------------------
            degr_now = (
                step_feats["dSR"] < 0
                or step_feats["dp_neg"] > (baseline.delta_p_bar + eps)
                or step_feats["dq"] < 0
                or step_feats["unsafe"]
            )
            healthy_now = (
                step_feats["dp"] >= 0.0
                and step_feats["dq"] >= 0.0
                and not step_feats["unsafe"]
            )

            if not in_window:
                degr_streak = degr_streak + 1 if degr_now else 0
                if degr_streak >= confirm_steps:
                    # Open a window anchored at the first degraded step.
                    t_d = idx - (degr_streak - 1)
                    current = _WindowAccumulator(
                        anchor=anchor,
                        baseline=baseline,
                        t_d=t_d,
                    )
                    current.add_step(step_feats, huber_k=huber_k)
                    # Hold length derived from τ^* (in steps) if not provided.
                    if cfg["recovery_hold_steps"] is None:
                        recovery_hold_target = max(1.0, baseline.tau_star)
                    in_window = True
                    degr_streak = 0
            else:
                # Already in a window; either continue or close it.
                stage_changed = (
                    anchor.stage != current.anchor.stage
                    or anchor.family != current.anchor.family
                    or anchor.modality != current.anchor.modality
                )
                if stage_changed:
                    current.closed_by_stage_switch = True
                    current.t_r = idx - 1
                    records.append(current.finalize(eps))
                    in_window = False
                    current = None
                    # Re-evaluate: new anchor may also be degraded.
                    if degr_now:
                        degr_streak = 1
                    else:
                        degr_streak = 0
                    continue

                current.add_step(step_feats, huber_k=huber_k)
                if healthy_now:
                    current.healthy_streak += 1
                else:
                    current.healthy_streak = 0
                if current.healthy_streak >= math.ceil(recovery_hold_target):
                    current.t_r = idx
                    records.append(current.finalize(eps))
                    in_window = False
                    current = None
                    degr_streak = 0

        # Flush any window still open at trajectory end.
        if in_window and current is not None:
            current.t_r = len(transitions) - 1
            records.append(current.finalize(eps))

        metrics.stage_recovery_records = records
        metrics.num_recovery_windows = len(records)
        metrics.formal_window_count = len(records)
        metrics.raw_window_count = len(records)
        metrics.skipped_window_count = 0
        metrics.c_rec_valid = True
        metrics.c_rec_missing_reason = ""
        metrics.c_rec = float(sum(r.c_rec for r in records))
        metrics.c_rec_cog = float(sum(r.cog_swe for r in records))
        metrics.c_rec_phy = float(sum(r.phy_swe for r in records))
        if records:
            metrics.c_rec_state_modulation = float(
                sum(r.omega_mean for r in records) / len(records)
            )
            metrics.c_rec_p90 = _percentile([r.c_rec for r in records], 0.9)
        else:
            metrics.c_rec_state_modulation = 1.0
            metrics.c_rec_p90 = 0.0

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
            Per-step ``info`` or critic-record dicts (same format as
            :meth:`compute_stage_crec`).  Used only for per-step sim/runtime
            quantities that the tracker does not duplicate.
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
        2. Build :math:`\Omega_t` from state-debt ratios over the
           anchor baselines.
        3. Aggregate :math:`g_t^\text{rec}` per-dimension:
           :math:`g_t^{rec,x} = \sqrt{(\phi(\widetilde W^{cog})^2 + \phi(\widetilde W^{phy,x})^2)/2} \cdot \Omega_t`.
        4. Compute cross-stage :math:`W_\text{rem}` per-dimension as
           :math:`W_\text{rem}^x = \sum_k \varphi_k^x \cdot W_\text{rem}^{*,seg,x}(a^{(k)})`
           where :math:`\varphi_k^x \in [0,1]` is the fractional completion
           the window achieved inside stage ``k`` in dimension ``x``.
        5. :math:`C_\text{rec}^{(h),x} = \sum_t g_t^{rec,x} / W_\text{rem}^x`.
        """
        metrics = metrics or ReboundMetrics()

        raw_windows = list(tracker_summary.get("windows") or ())
        anchor_dicts: Sequence[Mapping[str, Any]] = (
            tracker_summary.get("anchors") or ()
        )
        progress_deltas = list(tracker_summary.get("progress_deltas") or ())
        q_deltas = list(tracker_summary.get("q_deltas") or ())
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
        metrics.baseline_match_level = "none"
        metrics.w_rem_star = 0.0
        metrics.g_rec_total = 0.0
        metrics.recovery_window_evidence = []

        if not raw_windows:
            metrics.c_rec = 0.0
            metrics.c_rec_plan = 0.0
            metrics.c_rec_sim = 0.0
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
        n_trans = len(transitions)

        for window_index, win in enumerate(raw_windows):
            t_d = int(win.get("t_d", 0))
            t_r = int(win.get("t_r", t_d))
            anchor_t_d_payload = win.get("anchor_t_d") or {}
            anchor_t_d = StageAnchor(
                family=str(anchor_t_d_payload.get("family", "other")),
                stage=int(anchor_t_d_payload.get("stage", -1)),
                modality=str(anchor_t_d_payload.get("modality", "planning")),
                progress=float(anchor_t_d_payload.get("progress", 0.0)),
            )
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
                        missing_reason="stage_baseline_key_missing",
                        baseline=None,
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

            for step_idx in range(t_d, t_r + 1):
                if step_idx >= n_trans or step_idx <= 0:
                    continue
                prev_rec = transitions[step_idx - 1]
                curr_rec = transitions[step_idx]
                if step_idx < len(anchors):
                    anchor_step = anchors[step_idx]
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
                cog_step = 0.0
                vols = cog_vol_by_step.get(step_idx, {})
                vals = cog_validity_by_step.get(step_idx, {})
                for channel in _COG_CHANNELS:
                    v = float(vols.get(channel, 0.0))
                    v_bar = float(baseline_step.v_bar_channels.get(channel, 0.0))
                    q = float(vals.get(channel, 1.0))
                    cog_step += max(0.0, v - v_bar) * max(0.0, 1.0 - q)
                cog_swe += cog_step
                phi_cog = _soft_compress(cog_step)

                # Physical work: plan overhead + sim overhead above baselines
                plan_base = max(1.0, baseline_step.dim("plan").tau_bar)
                sim_base = max(1.0, baseline_step.dim("sim").tau_bar)
                W_phy_plan = max(0.0, (feats["d_plan"] - plan_base)) / plan_base
                W_phy_sim = max(0.0, (feats["d_sim"] - sim_base)) / sim_base
                # Add redo penalty when observed (retries / backtracks).
                if feats["redo_event"]:
                    W_phy_sim += 1.0
                phy_swe += W_phy_sim + W_phy_plan
                phi_phy_plan = _soft_compress(W_phy_plan)
                phi_phy_sim = _soft_compress(W_phy_sim)

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

                # Refine the coarse accumulation above into explicit cognitive /
                # physical / state-debt sub-buckets without breaking the legacy
                # control flow that the rest of the file relies on.
                cog_channels: Dict[str, float] = {}
                for channel in _COG_CHANNELS:
                    v = float(vols.get(channel, 0.0))
                    v_bar = float(baseline_step.v_bar_channels.get(channel, 0.0))
                    q = float(vals.get(channel, 1.0))
                    cog_channels[channel] = max(0.0, v - v_bar) * max(0.0, 1.0 - q)
                refined_cog_step = float(sum(cog_channels.values()))
                cog_swe += refined_cog_step - cog_step
                cog_swe_pln += cog_channels.get("pln", 0.0)
                cog_swe_per += cog_channels.get("per", 0.0)
                cog_swe_coord += cog_channels.get("coord", 0.0)
                cog_step = refined_cog_step
                phi_cog = _soft_compress(cog_step)

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
                phi_phy_plan = _soft_compress(refined_W_phy_plan)
                phi_phy_sim = _soft_compress(refined_W_phy_sim + refined_W_phy_gap)

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

                # g_t^{rec, x}
                phi_phy_total = max(phi_phy_plan, phi_phy_sim)
                phi_state = _soft_compress(state_step)
                g_rec_total += math.sqrt(
                    (
                        phi_cog * phi_cog
                        + phi_phy_total * phi_phy_total
                        + phi_state * phi_state
                    )
                    / 3.0
                )
                g_rec_plan += math.sqrt((phi_cog * phi_cog + phi_phy_plan * phi_phy_plan) / 2.0) * omega_t
                g_rec_sim += math.sqrt((phi_cog * phi_cog + phi_phy_sim * phi_phy_sim) / 2.0) * omega_t

                # Runtime (monitor only)
                runtime_overhead += feats.get("dt_runtime", 0.0)
                runtime_baseline += max(1e-6, baseline_step.dim("runtime").tau_bar)

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
                window_match_levels.add(seg_match_level)
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
                    )
                )
                continue
            baseline_match_levels.append(window_match_level)

            W_rem_plan = max(W_rem_plan, 1.0)
            W_rem_sim = max(W_rem_sim, 1.0)
            W_rem_runtime = max(W_rem_runtime, 0.0)
            W_rem_star = _baseline_w_rem_star(baseline_td)

            c_rec_plan = g_rec_plan / (W_rem_plan + eps)
            c_rec_sim = g_rec_sim / (W_rem_sim + eps)
            c_rec_cog = cog_swe / (W_rem_star + eps)
            c_rec_cog_pln = cog_swe_pln / (W_rem_star + eps)
            c_rec_cog_per = cog_swe_per / (W_rem_star + eps)
            c_rec_cog_coord = cog_swe_coord / (W_rem_star + eps)
            c_rec_phy_plan = phy_swe_plan / (W_rem_star + eps)
            c_rec_phy_sim = phy_swe_sim / (W_rem_star + eps)
            c_rec_phy_gap = phy_swe_gap / (W_rem_star + eps)
            c_rec_phy_redo = phy_swe_redo / (W_rem_star + eps)
            c_rec_phy = phy_swe / (W_rem_star + eps)
            c_rec_state_prog = state_debt_prog / (W_rem_star + eps)
            c_rec_state_goal = state_debt_goal / (W_rem_star + eps)
            c_rec_state_risk = state_debt_risk / (W_rem_star + eps)
            c_rec_state_debt = state_debt_swe / (W_rem_star + eps)
            c_rec = g_rec_total / (W_rem_star + eps)
            omega_mean = sum(omega_vals) / len(omega_vals) if omega_vals else 1.0
            runtime_overhead_ratio = (
                runtime_overhead / (runtime_baseline + eps)
                if runtime_baseline > 0
                else 0.0
            )

            records.append(
                StageRecoveryRecord(
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
            )
            evidence_records.append(
                self._build_window_evidence(
                    window_index=window_index,
                    win=win,
                    valid=True,
                    missing_reason="",
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
                str(ev.get("missing_reason", "formal_window_unavailable"))
                for ev in evidence_records
                if not bool(ev.get("valid", False))
            ]
            metrics.c_rec_missing_reason = (
                "partial_formal_windows:"
                + ",".join(sorted(set(missing_reasons)))
                if records
                else (missing_reasons[0] if missing_reasons else "formal_window_unavailable")
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
        metrics.g_rec_total = float(sum(r.g_rec_sum_total for r in records))
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
        metrics.rebound_events = [
            ev
            for win in raw_windows
            for ev in [
                {
                    "kind": "t_d",
                    "step": int(win.get("t_d", 0)),
                    "templates": list(win.get("template_codes") or []),
                    "severity": float(win.get("severity_sum", 0.0)),
                    "anchor": win.get("anchor_t_d"),
                },
                {
                    "kind": "t_r",
                    "step": int(win.get("t_r", 0)),
                    "templates": list(win.get("template_codes") or []),
                    "severity": float(win.get("severity_sum", 0.0)),
                    "anchor": win.get("anchor_t_r"),
                },
            ]
        ]

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
            "missing_reason": str(missing_reason or ""),
            "t_d": int(win.get("t_d", 0)),
            "t_r": int(win.get("t_r", win.get("t_d", 0))),
            "t_end": int(win.get("t_end", win.get("t_r", win.get("t_d", 0)))),
            "trigger": str(win.get("trigger", "")),
            "template_codes": list(win.get("template_codes") or []),
            "severity": float(win.get("severity_sum", 0.0) or 0.0),
            "anchor_t_d": dict(anchor_payload) if isinstance(anchor_payload, Mapping) else {},
            "baseline_key": baseline_key,
            "baseline_match_level": str(baseline_match_level or "none"),
            "segment_count": len(segments or ()),
            "segments": [dict(seg) for seg in (segments or ())],
            "segment_fallbacks": [
                dict(fallback) for fallback in (segment_fallbacks or ())
            ],
            "features": {
                "cognitive": {
                    "source": "online_tracker.cognitive_steps",
                    "raw": float(cog_swe),
                    "normalized": float(cog_swe / max(w_rem_star, 1.0)) if valid else 0.0,
                },
                "physical": {
                    "source": "planner_info.step_deltas+action_trace",
                    "raw": float(phy_swe),
                    "normalized": float(phy_swe / max(w_rem_star, 1.0)) if valid else 0.0,
                },
                "state_debt": {
                    "source": "progress/q/risk deltas",
                    "raw": float(state_debt_swe),
                    "normalized": (
                        float(state_debt_swe / max(w_rem_star, 1.0)) if valid else 0.0
                    ),
                },
                "g_rec_total": {
                    "source": "unified_cog_phy_state_density",
                    "raw": float(g_rec_total),
                    "normalized": float(c_rec),
                },
                "W_rem_star": {
                    "source": "stage_baseline.W_rem_star_swe",
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

        d_plan = max(0.0, _get("planning_step_count", 0.0) - _get_prev("planning_step_count", 0.0))
        if d_plan == 0.0:
            d_plan = 1.0  # one planning step always elapsed by construction
        d_sim = max(0.0, _get("sim_step_count", 0.0) - _get_prev("sim_step_count", 0.0))
        dt_runtime = max(
            0.0,
            _get("runtime_elapsed", 0.0) - _get_prev("runtime_elapsed", 0.0),
        )

        dp = _get("task_percent_complete", 0.0) - _get_prev("task_percent_complete", 0.0)
        dq = _get("proposition_satisfied_fraction", 0.0) - _get_prev(
            "proposition_satisfied_fraction", 0.0
        )
        dSR = _get("task_state_success", 0.0) - _get_prev("task_state_success", 0.0)
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
            frame.name in {"handover", "coordinate", "wait"} for frame in frames
        ) or str(anchor.modality).lower() == "coordination"
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
            "no_forward_progress": no_forward_progress,
            "step_template_codes": sorted(step_template_codes),
        }

    # ------------------------------------------------------------------
    # Per-step feature extraction (used only by compute_stage_crec)
    # ------------------------------------------------------------------
    def _derive_step_features(
        self,
        prev: Mapping[str, Any],
        curr: Mapping[str, Any],
        anchor: StageAnchor,
        baseline: StageBaseline,
        tau_pln: float,
        tau_per: float,
        tau_coord: float,
        redo_steps: set,
    ) -> Dict[str, float]:
        prev_info = prev.get("info_snapshot", prev) if isinstance(prev, Mapping) else {}
        curr_info = curr.get("info_snapshot", curr) if isinstance(curr, Mapping) else {}
        if not isinstance(prev_info, Mapping):
            prev_info = prev if isinstance(prev, Mapping) else {}
        if not isinstance(curr_info, Mapping):
            curr_info = curr if isinstance(curr, Mapping) else {}

        # Merge flat fields that are not inside info_snapshot
        def _field(name: str, default: float = 0.0) -> float:
            source = curr_info if name in curr_info else curr
            if isinstance(source, Mapping):
                v = source.get(name, default)
                try:
                    return float(v) if v is not None else float(default)
                except (TypeError, ValueError):
                    return float(default)
            return float(default)

        def _prev_field(name: str, default: float = 0.0) -> float:
            source = prev_info if name in prev_info else prev
            if isinstance(source, Mapping):
                v = source.get(name, default)
                try:
                    return float(v) if v is not None else float(default)
                except (TypeError, ValueError):
                    return float(default)
            return float(default)

        prev_sim = _prev_field("sim_step_count", 0.0)
        curr_sim = _field("sim_step_count", prev_sim)
        d_sim = max(0.0, curr_sim - prev_sim)
        tau_star = float(baseline.tau_star) if baseline.tau_star > 0 else 1.0
        dt_sim = d_sim if d_sim > 0 else tau_star

        dp = _field("task_percent_complete", 0.0) - _prev_field("task_percent_complete", 0.0)
        dq = _field("proposition_satisfied_fraction", 0.0) - _prev_field(
            "proposition_satisfied_fraction", 0.0
        )
        dSR = _field("task_state_success", 0.0) - _prev_field("task_state_success", 0.0)

        risk = _field(
            "cbf_penalty_step",
            _field("cbf_penalty", _field("safety_penalty", 0.0)),
        )
        unsafe = risk > 0.0

        # Events from analysis_tags
        tags = curr_info.get("analysis_tags") or curr.get("analysis_tags") or []
        if isinstance(tags, str):
            tags = [tags]
        tag_set = {str(t) for t in tags}
        replan_event = bool(tag_set & {"replan_anchor", "planning_transition"})
        response_event = "response_event" in tag_set

        step_idx = int(curr_info.get("planning_step_count", curr_info.get("loop_step_count", 0)) or 0)
        redo_event = step_idx in redo_steps or bool(tag_set & {"retry_event", "backtrack_event"})

        # Cognitive work times
        T_pln = (1.0 if replan_event else 0.0) * tau_pln
        T_per = (1.0 if response_event else 0.0) * tau_per
        T_coord = (1.0 if anchor.modality == "coordination" else 0.0) * tau_coord

        # Physical work times (attribute dt_sim to active modality)
        T_nav = dt_sim if anchor.modality == "navigation" else 0.0
        T_man = dt_sim if anchor.modality == "manipulation" else 0.0
        T_redo = dt_sim if redo_event else 0.0

        # Excess work over nominal (SWE units)
        def _excess(actual: float, nominal_key: str) -> float:
            nominal = float(baseline.T_bar.get(nominal_key, 0.0))
            return max(0.0, (actual - nominal) / tau_star)

        W_cog = _excess(T_pln, "pln") + _excess(T_per, "per") + _excess(T_coord, "coord")
        W_phy = _excess(T_nav, "nav") + _excess(T_man, "man") + _excess(T_redo, "redo")

        # State-debt ratios
        dp_neg = max(0.0, -dp)
        dq_neg = max(0.0, -dq)
        D_prog = dp_neg / (baseline.delta_p_bar + 1e-6)
        D_goal = dq_neg / (baseline.delta_q_bar + 1e-6)
        D_risk = max(0.0, (risk - baseline.r_bar) / (baseline.r_bar + 1e-6)) if baseline.r_bar > 0 else max(0.0, risk)

        return {
            "dp": dp,
            "dq": dq,
            "dSR": dSR,
            "dp_neg": dp_neg,
            "dq_neg": dq_neg,
            "risk": risk,
            "unsafe": unsafe,
            "W_cog": W_cog,
            "W_phy": W_phy,
            "D_prog": D_prog,
            "D_goal": D_goal,
            "D_risk": D_risk,
            "dt_sim": dt_sim,
        }


class _WindowAccumulator:
    """Running aggregator for a single ``(t_d, t_r)`` recovery window."""

    __slots__ = (
        "anchor",
        "baseline",
        "t_d",
        "t_r",
        "cog_swe",
        "phy_swe",
        "omega_values",
        "g_rec_sum",
        "healthy_streak",
        "closed_by_stage_switch",
    )

    def __init__(self, anchor: StageAnchor, baseline: StageBaseline, t_d: int) -> None:
        self.anchor = anchor
        self.baseline = baseline
        self.t_d = int(t_d)
        self.t_r = int(t_d)
        self.cog_swe = 0.0
        self.phy_swe = 0.0
        self.omega_values: List[float] = []
        self.g_rec_sum = 0.0
        self.healthy_streak = 0
        self.closed_by_stage_switch = False

    def add_step(self, feats: Mapping[str, float], huber_k: float) -> None:
        phi_cog = _soft_compress(feats["W_cog"])
        phi_phy = _soft_compress(feats["W_phy"])
        omega = 1.0 + (
            _huber(feats["D_prog"], huber_k)
            + _huber(feats["D_goal"], huber_k)
            + _huber(feats["D_risk"], huber_k)
        ) / 3.0
        g_rec = math.sqrt((phi_cog * phi_cog + phi_phy * phi_phy) / 2.0) * omega
        self.cog_swe += phi_cog
        self.phy_swe += phi_phy
        self.omega_values.append(omega)
        self.g_rec_sum += g_rec

    def finalize(self, eps: float) -> StageRecoveryRecord:
        denom = max(self.baseline.W_rem_seg_bar, 1e-6) + eps
        c_rec = self.g_rec_sum / denom
        omega_mean = (
            sum(self.omega_values) / len(self.omega_values) if self.omega_values else 1.0
        )
        return StageRecoveryRecord(
            stage_family=self.anchor.family,
            stage_index=int(self.anchor.stage),
            stage_modality=self.anchor.modality,
            t_d=self.t_d,
            t_r=self.t_r,
            duration=max(0, self.t_r - self.t_d + 1),
            cog_swe=float(self.cog_swe),
            phy_swe=float(self.phy_swe),
            omega_mean=float(omega_mean),
            g_rec_sum=float(self.g_rec_sum),
            W_rem_seg=float(self.baseline.W_rem_seg_bar),
            c_rec=float(c_rec),
            closed_by_stage_switch=bool(self.closed_by_stage_switch),
        )
