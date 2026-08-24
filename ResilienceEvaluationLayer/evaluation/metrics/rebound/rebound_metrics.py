#!/usr/bin/env python3

"""
Rebound Metrics Data Classes.

Defines the unified data structures for recovery metrics from failures.
Consolidates metrics from ReboundManager
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .formal_cost import bounded_recovery_cost_index
from .recovery_features import REBOUND_FORMULA_VERSION


@dataclass
class FailureEvent:
    """Represents a single failure and recovery event."""

    start_step: int           # Planning step when failure detected
    end_step: int             # Planning step when recovery completed
    strategy: str             # Recovery strategy used (retry, backtrack, replan, etc.)
    fault_type: str           # Type of fault (context_overflow, tool_failure, etc.)
    success: bool             # Whether recovery was successful

    @property
    def duration(self) -> int:
        """Recovery duration in planning steps."""
        return self.end_step - self.start_step


@dataclass
class StageRecoveryRecord:
    """A single recovery window ``(t_d^{(h)}, t_r^{(h)})`` within a stage anchor.

    Encodes the aggregate quantities used by the :math:`C_{rec}^{(h)}` formula
    in §1.3 of the resilience design doc.  Recovery windows may span
    multiple stages; the cross-stage :math:`W_\text{rem}` is computed via
    :math:`\varphi_k`-weighted sum over all stages the window visited (see
    :meth:`ReboundCollector.compute_stage_crec_multidim`).

    Dimensions
    ----------
    ``c_rec`` is the formal covariance-aware, time-weighted recovery cost.
    ``c_rec_plan`` and ``c_rec_sim`` retain decomposition diagnostics, while
    ``runtime_overhead_ratio`` is monitor-only and is excluded from the
    paper-facing scalar to avoid hardware bias.
    """

    stage_family: str = "other"
    stage_index: int = -1
    stage_modality: str = "planning"

    t_d: int = 0
    t_r: int = 0
    duration: int = 0

    # Raw, pre-normalization burdens accumulated inside the window.
    cog_swe: float = 0.0
    cog_swe_pln: float = 0.0
    cog_swe_per: float = 0.0
    cog_swe_coord: float = 0.0

    phy_swe: float = 0.0
    phy_swe_plan: float = 0.0
    phy_swe_sim: float = 0.0
    phy_swe_gap: float = 0.0
    phy_swe_redo: float = 0.0

    state_debt_swe: float = 0.0
    state_debt_prog: float = 0.0
    state_debt_goal: float = 0.0
    state_debt_risk: float = 0.0

    omega_mean: float = 1.0
    g_rec_sum_total: float = 0.0
    g_rec_sum_plan: float = 0.0
    g_rec_sum_sim: float = 0.0
    W_rem_star: float = 1.0
    W_rem_plan: float = 1.0
    W_rem_sim: float = 1.0
    W_rem_runtime: float = 0.0
    runtime_overhead: float = 0.0

    # Normalized window-level diagnostics.
    c_rec_plan: float = 0.0
    c_rec_sim: float = 0.0
    c_rec_cog: float = 0.0
    c_rec_cog_pln: float = 0.0
    c_rec_cog_per: float = 0.0
    c_rec_cog_coord: float = 0.0
    c_rec_phy: float = 0.0
    c_rec_phy_plan: float = 0.0
    c_rec_phy_sim: float = 0.0
    c_rec_phy_gap: float = 0.0
    c_rec_phy_redo: float = 0.0
    c_rec_state_debt: float = 0.0
    c_rec_state_prog: float = 0.0
    c_rec_state_goal: float = 0.0
    c_rec_state_risk: float = 0.0
    c_rec: float = 0.0
    runtime_overhead_ratio: float = 0.0

    template_codes: List[str] = field(default_factory=list)
    stages_spanned: List[int] = field(default_factory=list)
    gap_steps: int = 0
    redo_steps: int = 0

    # Backwards-compatible single-dimension aliases.
    g_rec_sum: float = 0.0
    W_rem_seg: float = 1.0
    closed_by_stage_switch: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage_family": self.stage_family,
            "stage_index": int(self.stage_index),
            "stage_modality": self.stage_modality,
            "t_d": int(self.t_d),
            "t_r": int(self.t_r),
            "duration": int(self.duration),
            "cog_swe": float(self.cog_swe),
            "cog_swe_pln": float(self.cog_swe_pln),
            "cog_swe_per": float(self.cog_swe_per),
            "cog_swe_coord": float(self.cog_swe_coord),
            "phy_swe": float(self.phy_swe),
            "phy_swe_plan": float(self.phy_swe_plan),
            "phy_swe_sim": float(self.phy_swe_sim),
            "phy_swe_gap": float(self.phy_swe_gap),
            "phy_swe_redo": float(self.phy_swe_redo),
            "state_debt_swe": float(self.state_debt_swe),
            "state_debt_prog": float(self.state_debt_prog),
            "state_debt_goal": float(self.state_debt_goal),
            "state_debt_risk": float(self.state_debt_risk),
            "omega_mean": float(self.omega_mean),
            "g_rec_sum_total": float(self.g_rec_sum_total),
            "g_rec_sum_plan": float(self.g_rec_sum_plan),
            "g_rec_sum_sim": float(self.g_rec_sum_sim),
            "W_rem_star": float(self.W_rem_star),
            "W_rem_plan": float(self.W_rem_plan),
            "W_rem_sim": float(self.W_rem_sim),
            "W_rem_runtime": float(self.W_rem_runtime),
            "runtime_overhead": float(self.runtime_overhead),
            "c_rec_plan": float(self.c_rec_plan),
            "c_rec_sim": float(self.c_rec_sim),
            "c_rec_cog": float(self.c_rec_cog),
            "c_rec_cog_pln": float(self.c_rec_cog_pln),
            "c_rec_cog_per": float(self.c_rec_cog_per),
            "c_rec_cog_coord": float(self.c_rec_cog_coord),
            "c_rec_phy": float(self.c_rec_phy),
            "c_rec_phy_plan": float(self.c_rec_phy_plan),
            "c_rec_phy_sim": float(self.c_rec_phy_sim),
            "c_rec_phy_gap": float(self.c_rec_phy_gap),
            "c_rec_phy_redo": float(self.c_rec_phy_redo),
            "c_rec_state_debt": float(self.c_rec_state_debt),
            "c_rec_state_prog": float(self.c_rec_state_prog),
            "c_rec_state_goal": float(self.c_rec_state_goal),
            "c_rec_state_risk": float(self.c_rec_state_risk),
            "c_rec": float(self.c_rec),
            "runtime_overhead_ratio": float(self.runtime_overhead_ratio),
            "template_codes": list(self.template_codes),
            "stages_spanned": [int(s) for s in self.stages_spanned],
            "gap_steps": int(self.gap_steps),
            "redo_steps": int(self.redo_steps),
            "g_rec_sum": float(self.g_rec_sum),
            "W_rem_seg": float(self.W_rem_seg),
            "closed_by_stage_switch": bool(self.closed_by_stage_switch),
        }


@dataclass
class ReboundMetrics:
    """
    Unified recovery metrics from failures.

    Legacy failure counters stay available for compatibility, while the main
    analysis path now uses the multi-dimensional C_rec family.
    """

    # Legacy episode-level counters.
    b_epi: float = 0.0
    mttr: float = 0.0
    mtbf: float = 0.0
    recovery_ratio: float = 1.0
    t_rec: float = 0.0

    retry_count: int = 0
    backtrack_count: int = 0
    replan_count: int = 0
    context_summarize_count: int = 0

    total_failures: int = 0
    successful_recoveries: int = 0
    failed_recoveries: int = 0

    failure_events: List[FailureEvent] = field(default_factory=list)
    fault_distribution: Dict[str, int] = field(default_factory=dict)

    rsr: Optional[float] = None
    rtr: Optional[float] = None
    rstc: Optional[Dict[str, int]] = None

    total_steps: int = 0
    episode_id: Optional[str] = None

    # Main formal metrics.
    c_rec: float = 0.0
    c_rec_index_100: Optional[float] = None
    c_rec_plan: float = 0.0
    c_rec_sim: float = 0.0

    c_rec_cog: float = 0.0
    c_rec_cog_pln: float = 0.0
    c_rec_cog_per: float = 0.0
    c_rec_cog_coord: float = 0.0

    c_rec_phy: float = 0.0
    c_rec_phy_plan: float = 0.0
    c_rec_phy_sim: float = 0.0
    c_rec_phy_gap: float = 0.0
    c_rec_phy_redo: float = 0.0

    c_rec_state_debt: float = 0.0
    c_rec_state_prog: float = 0.0
    c_rec_state_goal: float = 0.0
    c_rec_state_risk: float = 0.0
    c_rec_state_modulation: float = 1.0

    c_rec_p90: float = 0.0
    g_rec_total: float = 0.0
    w_rem_star: float = 0.0
    runtime_overhead_ratio: float = 0.0
    num_recovery_windows: int = 0
    raw_window_count: int = 0
    formal_window_count: int = 0
    skipped_window_count: int = 0
    c_rec_valid: bool = False
    c_rec_missing_reason: str = ""
    c_rec_observed_lower_bound: Optional[float] = None
    g_rec_observed_total: Optional[float] = None
    observed_censored_window_count: int = 0
    c_rec_formula_version: str = REBOUND_FORMULA_VERSION
    baseline_match_level: str = "none"
    stage_switches: int = 0
    stages_spanned_mean: float = 0.0
    num_template_events: int = 0
    baseline_loaded: bool = False
    tracker_window_count: int = 0
    tracker_window_open_final: bool = False
    template_distribution: Dict[str, int] = field(default_factory=dict)
    stage_recovery_records: List[StageRecoveryRecord] = field(default_factory=list)
    recovery_window_evidence: List[Dict[str, Any]] = field(default_factory=list)
    rebound_events: List[Dict[str, Any]] = field(default_factory=list)

    def compute_derived_metrics(self, gamma: float = 2.0) -> None:
        """Compute legacy derived metrics from raw counts."""
        self.b_epi = self.retry_count + gamma * self.backtrack_count

        if self.failure_events:
            total_recovery_time = sum(event.duration for event in self.failure_events)
            self.mttr = total_recovery_time / len(self.failure_events)
        else:
            self.mttr = 0.0

        if self.total_failures > 0:
            self.mtbf = self.total_steps / self.total_failures
            self.recovery_ratio = self.successful_recoveries / self.total_failures
        else:
            self.mtbf = float(self.total_steps) if self.total_steps > 0 else 0.0
            self.recovery_ratio = 1.0

        self.rsr = self.recovery_ratio

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-friendly dictionary."""
        def formal_value(value: float) -> Optional[float]:
            return float(value) if self.c_rec_valid else None
        return {
            "b_epi": self.b_epi,
            "mttr": self.mttr,
            "mtbf": self.mtbf,
            "recovery_ratio": self.recovery_ratio,
            "t_rec": self.t_rec,
            "retry_count": self.retry_count,
            "backtrack_count": self.backtrack_count,
            "replan_count": self.replan_count,
            "context_summarize_count": self.context_summarize_count,
            "total_failures": self.total_failures,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "fault_distribution": self.fault_distribution,
            "rsr": self.rsr,
            "rtr": self.rtr,
            "rstc": self.rstc,
            "total_steps": self.total_steps,
            "episode_id": self.episode_id,
            "c_rec": formal_value(self.c_rec),
            "c_rec_index_100": (
                (
                    self.c_rec_index_100
                    if self.c_rec_index_100 is not None
                    else bounded_recovery_cost_index(self.c_rec)
                )
                if self.c_rec_valid
                else None
            ),
            "c_rec_plan": formal_value(self.c_rec_plan),
            "c_rec_sim": formal_value(self.c_rec_sim),
            "c_rec_cog": formal_value(self.c_rec_cog),
            "c_rec_cog_pln": formal_value(self.c_rec_cog_pln),
            "c_rec_cog_per": formal_value(self.c_rec_cog_per),
            "c_rec_cog_coord": formal_value(self.c_rec_cog_coord),
            "c_rec_phy": formal_value(self.c_rec_phy),
            "c_rec_phy_plan": formal_value(self.c_rec_phy_plan),
            "c_rec_phy_sim": formal_value(self.c_rec_phy_sim),
            "c_rec_phy_gap": formal_value(self.c_rec_phy_gap),
            "c_rec_phy_redo": formal_value(self.c_rec_phy_redo),
            "c_rec_state_debt": formal_value(self.c_rec_state_debt),
            "c_rec_state_prog": formal_value(self.c_rec_state_prog),
            "c_rec_state_goal": formal_value(self.c_rec_state_goal),
            "c_rec_state_risk": formal_value(self.c_rec_state_risk),
            "c_rec_state_modulation": formal_value(
                self.c_rec_state_modulation
            ),
            "c_rec_p90": formal_value(self.c_rec_p90),
            "g_rec_total": formal_value(self.g_rec_total),
            "w_rem_star": formal_value(self.w_rem_star),
            "runtime_overhead_ratio": self.runtime_overhead_ratio,
            "num_recovery_windows": int(self.num_recovery_windows),
            "raw_window_count": int(self.raw_window_count),
            "formal_window_count": int(self.formal_window_count),
            "skipped_window_count": int(self.skipped_window_count),
            "c_rec_valid": bool(self.c_rec_valid),
            "c_rec_missing_reason": self.c_rec_missing_reason,
            "c_rec_observed_lower_bound": self.c_rec_observed_lower_bound,
            "g_rec_observed_total": self.g_rec_observed_total,
            "observed_censored_window_count": int(
                self.observed_censored_window_count
            ),
            "c_rec_formula_version": self.c_rec_formula_version,
            "baseline_match_level": self.baseline_match_level,
            "stage_switches": int(self.stage_switches),
            "stages_spanned_mean": float(self.stages_spanned_mean),
            "num_template_events": int(self.num_template_events),
            "baseline_loaded": bool(self.baseline_loaded),
            "tracker_window_count": int(self.tracker_window_count),
            "tracker_window_open_final": bool(self.tracker_window_open_final),
            "template_distribution": dict(self.template_distribution),
            "stage_recovery_records": [r.to_dict() for r in self.stage_recovery_records],
            "recovery_window_evidence": list(self.recovery_window_evidence),
            "rebound_events": list(self.rebound_events),
        }

    def get_summary_for_csv(self) -> Dict[str, Any]:
        """
        Get flattened CSV metrics for public episode summaries.

        The recovery object still keeps richer support fields internally
        (plan/sim splits, per-window subcomponents, legacy counters), but the
        default CSV surface is intentionally compact: one total recovery cost,
        three explanatory dimensions, and a minimal diagnosis bundle that tells
        us whether formal windows were actually available.
        """
        def formal_value(value: float) -> Optional[float]:
            return float(value) if self.c_rec_valid else None

        index_100 = (
            (
                self.c_rec_index_100
                if self.c_rec_index_100 is not None
                else bounded_recovery_cost_index(self.c_rec)
            )
            if self.c_rec_valid
            else None
        )
        return {
            "rebound_c_rec": formal_value(self.c_rec),
            "rebound_c_rec_index_100": (
                float(index_100) if index_100 is not None else None
            ),
            "rebound_c_rec_cog": formal_value(self.c_rec_cog),
            "rebound_c_rec_phy": formal_value(self.c_rec_phy),
            "rebound_c_rec_state_debt": formal_value(self.c_rec_state_debt),
            "rebound_num_recovery_windows": float(self.num_recovery_windows),
            "rebound_raw_window_count": float(self.raw_window_count),
            "rebound_formal_window_count": float(self.formal_window_count),
            "rebound_skipped_window_count": float(self.skipped_window_count),
            "rebound_c_rec_valid": float(bool(self.c_rec_valid)),
            "rebound_c_rec_missing_reason": self.c_rec_missing_reason,
            "rebound_c_rec_observed_lower_bound": (
                float(self.c_rec_observed_lower_bound)
                if self.c_rec_observed_lower_bound is not None
                else None
            ),
            "rebound_g_rec_observed_total": (
                float(self.g_rec_observed_total)
                if self.g_rec_observed_total is not None
                else None
            ),
            "rebound_observed_censored_window_count": float(
                self.observed_censored_window_count
            ),
            "rebound_c_rec_formula_version": self.c_rec_formula_version,
            "rebound_baseline_match_level": self.baseline_match_level,
            "rebound_w_rem_star": formal_value(self.w_rem_star),
            "rebound_g_rec_total": formal_value(self.g_rec_total),
            "rebound_baseline_loaded": float(bool(self.baseline_loaded)),
            "rebound_tracker_window_count": float(self.tracker_window_count),
            "rebound_tracker_window_open_final": float(
                bool(self.tracker_window_open_final)
            ),
        }
