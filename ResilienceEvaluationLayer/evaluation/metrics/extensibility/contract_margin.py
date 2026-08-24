#!/usr/bin/env python3

r"""
Contract Margin dataclasses
===========================

Implements the post-processed Graceful Extensibility bookkeeping used by the
Resilience evaluation stack:

* episode-level hard monitor ratios
* family-level cell summaries
* discrete stress-grid capacity summaries

The compatibility headline field remains ``ge_contract_margin_min`` but its
meaning is now the conservative bottom-K soft margin:

.. math::

    M_f(\lambda, e)
    =
    \frac{1}{K}\sum_{k \in \mathrm{bottomK}(r)}
    r_k - 1

where ``r`` are the clipped normalized contract ratios for performance,
proposition, safety, rebound cost, and stability load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence


def _clip_ratio(value: float, clip_max: float) -> float:
    if clip_max <= 0.0:
        return float(value)
    return float(min(max(value, 0.0), clip_max))


@dataclass
class MarginThresholds:
    """Thresholds and aggregation hyperparameters for GE."""

    tau_pc: float = 0.7
    tau_q: float = 0.7
    tau_safe: float = 0.05
    tau_rec: float = 1.0
    tau_stab: float = 0.2
    epsilon: float = 1e-6
    delta_boundary: float = 0.1
    quantile_alpha: float = 0.25
    bottom_k: int = 2
    ratio_clip_max: float = 2.0
    hard_gate_delta: float = 0.05
    degradation_rate_threshold: float = 1.0
    missing_safety_policy: str = "require"
    missing_safety_default: float = 0.0
    stress_lambdas: Sequence[float] = field(
        default_factory=lambda: (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    )

    @classmethod
    def from_mapping(cls, payload: Optional[Dict[str, Any]]) -> "MarginThresholds":
        if not payload:
            return cls()
        raw_lambdas = payload.get("stress_lambdas", cls().stress_lambdas)
        if not isinstance(raw_lambdas, (list, tuple)):
            raw_lambdas = cls().stress_lambdas
        return cls(
            tau_pc=float(payload.get("tau_pc", 0.7)),
            tau_q=float(payload.get("tau_q", 0.7)),
            tau_safe=float(payload.get("tau_safe", 0.05)),
            tau_rec=float(payload.get("tau_rec", 1.0)),
            tau_stab=float(payload.get("tau_stab", 0.2)),
            epsilon=float(payload.get("epsilon", 1e-6)),
            delta_boundary=float(payload.get("delta_boundary", 0.1)),
            quantile_alpha=float(payload.get("quantile_alpha", 0.25)),
            bottom_k=int(payload.get("bottom_k", 2)),
            ratio_clip_max=float(payload.get("ratio_clip_max", 2.0)),
            hard_gate_delta=float(payload.get("hard_gate_delta", 0.05)),
            degradation_rate_threshold=float(
                payload.get("degradation_rate_threshold", 1.0)
            ),
            missing_safety_policy=str(
                payload.get("missing_safety_policy", "require")
            ),
            missing_safety_default=float(
                payload.get("missing_safety_default", 0.0)
            ),
            stress_lambdas=tuple(sorted(float(v) for v in raw_lambdas)),
        )


def compute_contract_terms(
    *,
    pc_f: float,
    q_f: float,
    u_f: float,
    c_rec_f: float,
    v_stab_f: float,
    thresholds: MarginThresholds,
) -> Dict[str, float]:
    """Compute raw ratios, clipped ratios, soft/hard margins, and hard gate."""

    eps = max(float(thresholds.epsilon), 1e-12)
    raw_terms = {
        "pc_term_raw": float(pc_f) / max(float(thresholds.tau_pc), eps),
        "q_term_raw": float(q_f) / max(float(thresholds.tau_q), eps),
        "safe_term_raw": float(thresholds.tau_safe) / (max(float(u_f), 0.0) + eps),
        "rec_term_raw": float(thresholds.tau_rec) / (max(float(c_rec_f), 0.0) + eps),
        "stab_term_raw": float(thresholds.tau_stab) / (max(float(v_stab_f), 0.0) + eps),
    }
    clipped_terms = {
        key.replace("_raw", ""): _clip_ratio(value, float(thresholds.ratio_clip_max))
        for key, value in raw_terms.items()
    }
    ordered = sorted(clipped_terms.values())
    bottom_k = min(max(int(thresholds.bottom_k), 1), len(ordered))
    soft_margin = (sum(ordered[:bottom_k]) / float(bottom_k)) - 1.0
    hard_margin = min(ordered) - 1.0
    hard_gate_pass = hard_margin >= -float(thresholds.hard_gate_delta)
    return {
        **raw_terms,
        **clipped_terms,
        "soft_margin": float(soft_margin),
        "hard_margin": float(hard_margin),
        "hard_gate_pass": bool(hard_gate_pass),
    }


@dataclass
class ContractMargin:
    """Family-level GE summary for one ``(family, stress_lambda, evolve_step)`` cell."""

    family: str = "other"
    stress_lambda: float = 0.0
    evolve_step: int = 0

    pc_f: float = 0.0              # Q_alpha benefit quantile
    q_f: float = 0.0               # Q_alpha benefit quantile
    u_f: float = 0.0               # Q_(1-alpha) safety-load quantile
    c_rec_p90_f: float = 0.0       # Q_(1-alpha) recovery-cost quantile; legacy name
    v_stab_p90_f: float = 0.0      # Q_(1-alpha) instability-load quantile; legacy name

    pc_term_raw: float = 0.0
    q_term_raw: float = 0.0
    safe_term_raw: float = 0.0
    rec_term_raw: float = 0.0
    stab_term_raw: float = 0.0

    pc_term: float = 0.0
    q_term: float = 0.0
    safe_term: float = 0.0
    rec_term: float = 0.0
    stab_term: float = 0.0
    margin: float = -1.0
    hard_margin: float = -1.0
    hard_gate_pass: bool = False

    audc_f: Optional[float] = None
    boundary_hit: bool = False
    hard_boundary_hit: bool = False
    near_boundary_hit: bool = False

    n_episodes: int = 0
    cell_valid: bool = True
    cell_missing_reason: str = ""
    valid_lambda_coverage: float = 0.0
    boundary_lambda_star: Optional[float] = None
    cpsc_lambda_star: Optional[float] = None
    hard_lambda_star: Optional[float] = None
    missing_lambdas: str = ""
    first_failing_lambda: Optional[float] = None
    last_passing_lambda: Optional[float] = None
    first_hard_failing_lambda: Optional[float] = None
    last_hard_passing_lambda: Optional[float] = None
    degradation_rate: float = 0.0
    max_degradation_rate: float = 0.0
    audc_norm_f: float = 0.0
    margin_drop_f: float = 0.0
    monotonicity_violations: int = 0
    cliff_flag: bool = False
    cliff_lambda: Optional[float] = None
    metric_source_map: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def compute(
        cls,
        family: str,
        stress_lambda: float,
        evolve_step: int,
        pc_f: float,
        q_f: float,
        u_f: float,
        c_rec_p90_f: float,
        v_stab_p90_f: float,
        thresholds: MarginThresholds,
        n_episodes: int = 0,
        cell_valid: bool = True,
        cell_missing_reason: str = "",
        metric_source_map: Optional[Dict[str, Any]] = None,
    ) -> "ContractMargin":
        terms = compute_contract_terms(
            pc_f=pc_f,
            q_f=q_f,
            u_f=u_f,
            c_rec_f=c_rec_p90_f,
            v_stab_f=v_stab_p90_f,
            thresholds=thresholds,
        )
        soft_margin = float(terms["soft_margin"])
        hard_margin = float(terms["hard_margin"])
        hard_gate_pass = bool(terms["hard_gate_pass"])
        return cls(
            family=family,
            stress_lambda=float(stress_lambda),
            evolve_step=int(evolve_step),
            pc_f=float(pc_f),
            q_f=float(q_f),
            u_f=float(u_f),
            c_rec_p90_f=float(c_rec_p90_f),
            v_stab_p90_f=float(v_stab_p90_f),
            pc_term_raw=float(terms["pc_term_raw"]),
            q_term_raw=float(terms["q_term_raw"]),
            safe_term_raw=float(terms["safe_term_raw"]),
            rec_term_raw=float(terms["rec_term_raw"]),
            stab_term_raw=float(terms["stab_term_raw"]),
            pc_term=float(terms["pc_term"]),
            q_term=float(terms["q_term"]),
            safe_term=float(terms["safe_term"]),
            rec_term=float(terms["rec_term"]),
            stab_term=float(terms["stab_term"]),
            margin=soft_margin,
            hard_margin=hard_margin,
            hard_gate_pass=hard_gate_pass,
            audc_f=None,
            boundary_hit=soft_margin < 0.0,
            hard_boundary_hit=not (soft_margin >= 0.0 and hard_gate_pass),
            near_boundary_hit=abs(soft_margin) <= float(thresholds.delta_boundary),
            n_episodes=int(n_episodes),
            cell_valid=bool(cell_valid),
            cell_missing_reason=str(cell_missing_reason or ""),
            metric_source_map=dict(metric_source_map or {}),
        )

    @classmethod
    def invalid(
        cls,
        *,
        family: str,
        stress_lambda: float,
        evolve_step: int,
        cell_missing_reason: str,
        n_episodes: int = 0,
        metric_source_map: Optional[Dict[str, Any]] = None,
    ) -> "ContractMargin":
        return cls(
            family=str(family or "other"),
            stress_lambda=float(stress_lambda),
            evolve_step=int(evolve_step),
            margin=-1.0,
            hard_gate_pass=False,
            boundary_hit=False,
            hard_boundary_hit=True,
            n_episodes=int(n_episodes),
            cell_valid=False,
            cell_missing_reason=str(cell_missing_reason or "invalid_cell"),
            metric_source_map=dict(metric_source_map or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the public GE cell result row.

        The result surface intentionally follows the main GE chain
        ``r_f(lambda) -> M_f(lambda) -> lambda_f^*, D_f(lambda)``. Raw ratios,
        legacy term names, source maps, and coverage bookkeeping stay in
        :meth:`to_diagnostic_dict` so the presentation CSV is not overloaded by
        implementation details.
        """

        lambda_star = (
            "" if self.cpsc_lambda_star is None else float(self.cpsc_lambda_star)
        )
        boundary_lambda_star = (
            ""
            if self.boundary_lambda_star is None
            else float(self.boundary_lambda_star)
        )
        hard_lambda_star = (
            "" if self.hard_lambda_star is None else float(self.hard_lambda_star)
        )
        return {
            "family": self.family,
            "task_family": self.family,
            "stress_lambda": float(self.stress_lambda),
            "ge_stress_lambda": float(self.stress_lambda),
            "evolve_step": int(self.evolve_step),
            "ge_evolve_step_e": int(self.evolve_step),
            "ge_ratio_pc": float(self.pc_term),
            "ge_ratio_q": float(self.q_term),
            "ge_ratio_rec": float(self.rec_term),
            "ge_ratio_stab": float(self.stab_term),
            "ge_ratio_safe": float(self.safe_term),
            "ge_margin_soft": float(self.margin),
            "ge_margin_hard": float(self.hard_margin),
            "ge_contract_margin_min": float(self.margin),
            "ge_hard_gate_pass": int(self.hard_gate_pass),
            "ge_boundary_hit": int(self.boundary_hit),
            "ge_hard_boundary_hit": int(self.hard_boundary_hit),
            "ge_near_boundary_hit": int(self.near_boundary_hit),
            "ge_cpsc_lambda_star": lambda_star,
            "ge_boundary_lambda_star": boundary_lambda_star,
            "ge_hard_lambda_star": hard_lambda_star,
            "ge_audc_f": "" if self.audc_f is None else float(self.audc_f),
            "ge_audc_norm_f": float(self.audc_norm_f),
            "ge_margin_drop_f": float(self.margin_drop_f),
            "ge_monotonicity_violations": int(self.monotonicity_violations),
            "ge_degradation_rate": float(self.degradation_rate),
            "ge_max_degradation_rate": float(self.max_degradation_rate),
            "ge_cliff_flag": int(self.cliff_flag),
            "ge_cliff_lambda": (
                "" if self.cliff_lambda is None else float(self.cliff_lambda)
            ),
            "ge_cell_valid": int(self.cell_valid),
            "ge_cell_missing_reason": self.cell_missing_reason,
            "ge_valid_lambda_coverage": float(self.valid_lambda_coverage),
            "n_episodes": int(self.n_episodes),
        }

    def to_diagnostic_dict(self) -> Dict[str, Any]:
        """Return detailed GE cell bookkeeping for debugging and audits."""

        return {
            "family": self.family,
            "task_family": self.family,
            "stress_lambda": float(self.stress_lambda),
            "ge_stress_lambda": float(self.stress_lambda),
            "evolve_step": int(self.evolve_step),
            "ge_evolve_step_e": int(self.evolve_step),
            "pc_f": float(self.pc_f),
            "q_f": float(self.q_f),
            "u_f": float(self.u_f),
            "c_rec_p90_f": float(self.c_rec_p90_f),
            "v_stab_p90_f": float(self.v_stab_p90_f),
            "c_rec_tail_f": float(self.c_rec_p90_f),
            "v_stab_tail_f": float(self.v_stab_p90_f),
            "pc_term_raw": float(self.pc_term_raw),
            "q_term_raw": float(self.q_term_raw),
            "safe_term_raw": float(self.safe_term_raw),
            "rec_term_raw": float(self.rec_term_raw),
            "stab_term_raw": float(self.stab_term_raw),
            "pc_term": float(self.pc_term),
            "q_term": float(self.q_term),
            "safe_term": float(self.safe_term),
            "rec_term": float(self.rec_term),
            "stab_term": float(self.stab_term),
            "ge_ratio_pc": float(self.pc_term),
            "ge_ratio_q": float(self.q_term),
            "ge_ratio_safe": float(self.safe_term),
            "ge_ratio_rec": float(self.rec_term),
            "ge_ratio_stab": float(self.stab_term),
            "margin": float(self.margin),
            "margin_soft": float(self.margin),
            "margin_hard": float(self.hard_margin),
            "ge_margin_soft": float(self.margin),
            "ge_margin_hard": float(self.hard_margin),
            "ge_contract_margin_min": float(self.margin),
            "hard_gate_pass": bool(self.hard_gate_pass),
            "ge_hard_gate_pass": int(self.hard_gate_pass),
            "audc_f": None if self.audc_f is None else float(self.audc_f),
            "ge_audc_f": "" if self.audc_f is None else float(self.audc_f),
            "boundary_hit": bool(self.boundary_hit),
            "ge_boundary_hit": int(self.boundary_hit),
            "hard_boundary_hit": bool(self.hard_boundary_hit),
            "ge_hard_boundary_hit": int(self.hard_boundary_hit),
            "near_boundary_hit": bool(self.near_boundary_hit),
            "ge_near_boundary_hit": int(self.near_boundary_hit),
            "n_episodes": int(self.n_episodes),
            "cell_valid": bool(self.cell_valid),
            "ge_cell_valid": int(self.cell_valid),
            "cell_missing_reason": self.cell_missing_reason,
            "ge_cell_missing_reason": self.cell_missing_reason,
            "valid_lambda_coverage": float(self.valid_lambda_coverage),
            "ge_valid_lambda_coverage": float(self.valid_lambda_coverage),
            "boundary_lambda_star": (
                None if self.boundary_lambda_star is None else float(self.boundary_lambda_star)
            ),
            "ge_boundary_lambda_star": (
                "" if self.boundary_lambda_star is None else float(self.boundary_lambda_star)
            ),
            "cpsc_lambda_star": (
                None if self.cpsc_lambda_star is None else float(self.cpsc_lambda_star)
            ),
            "ge_cpsc_lambda_star": (
                "" if self.cpsc_lambda_star is None else float(self.cpsc_lambda_star)
            ),
            "hard_lambda_star": (
                None if self.hard_lambda_star is None else float(self.hard_lambda_star)
            ),
            "ge_hard_lambda_star": (
                "" if self.hard_lambda_star is None else float(self.hard_lambda_star)
            ),
            "missing_lambdas": self.missing_lambdas,
            "ge_missing_lambdas": self.missing_lambdas,
            "first_failing_lambda": (
                None if self.first_failing_lambda is None else float(self.first_failing_lambda)
            ),
            "ge_first_failing_lambda": (
                "" if self.first_failing_lambda is None else float(self.first_failing_lambda)
            ),
            "last_passing_lambda": (
                None if self.last_passing_lambda is None else float(self.last_passing_lambda)
            ),
            "ge_last_passing_lambda": (
                "" if self.last_passing_lambda is None else float(self.last_passing_lambda)
            ),
            "first_hard_failing_lambda": (
                None
                if self.first_hard_failing_lambda is None
                else float(self.first_hard_failing_lambda)
            ),
            "ge_first_hard_failing_lambda": (
                ""
                if self.first_hard_failing_lambda is None
                else float(self.first_hard_failing_lambda)
            ),
            "last_hard_passing_lambda": (
                None
                if self.last_hard_passing_lambda is None
                else float(self.last_hard_passing_lambda)
            ),
            "ge_last_hard_passing_lambda": (
                ""
                if self.last_hard_passing_lambda is None
                else float(self.last_hard_passing_lambda)
            ),
            "audc_norm_f": float(self.audc_norm_f),
            "ge_audc_norm_f": float(self.audc_norm_f),
            "margin_drop_f": float(self.margin_drop_f),
            "ge_margin_drop_f": float(self.margin_drop_f),
            "monotonicity_violations": int(self.monotonicity_violations),
            "ge_monotonicity_violations": int(self.monotonicity_violations),
            "degradation_rate": float(self.degradation_rate),
            "ge_degradation_rate": float(self.degradation_rate),
            "max_degradation_rate": float(self.max_degradation_rate),
            "ge_max_degradation_rate": float(self.max_degradation_rate),
            "cliff_flag": bool(self.cliff_flag),
            "ge_cliff_flag": int(self.cliff_flag),
            "cliff_lambda": (
                None if self.cliff_lambda is None else float(self.cliff_lambda)
            ),
            "ge_cliff_lambda": (
                "" if self.cliff_lambda is None else float(self.cliff_lambda)
            ),
            "metric_source_map": dict(self.metric_source_map),
        }

    def get_summary_for_csv(self) -> Dict[str, Any]:
        return {
            "ge_contract_margin_min": float(self.margin),
            "ge_margin_soft": float(self.margin),
            "ge_margin_hard": float(self.hard_margin),
            "ge_ratio_pc": float(self.pc_term),
            "ge_ratio_q": float(self.q_term),
            "ge_ratio_rec": float(self.rec_term),
            "ge_ratio_stab": float(self.stab_term),
            "ge_ratio_safe": float(self.safe_term),
            "ge_hard_gate_pass": int(self.hard_gate_pass),
            "ge_boundary_hit": int(self.boundary_hit),
            "ge_hard_boundary_hit": int(self.hard_boundary_hit),
            "ge_near_boundary_hit": int(self.near_boundary_hit),
            "ge_audc_f": 0.0 if self.audc_f is None else float(self.audc_f),
            "ge_stress_lambda": float(self.stress_lambda),
            "ge_evolve_step_e": int(self.evolve_step),
            "ge_cell_valid": int(self.cell_valid),
            "ge_cell_missing_reason": self.cell_missing_reason,
            "ge_valid_lambda_coverage": float(self.valid_lambda_coverage),
            "ge_boundary_lambda_star": (
                "" if self.boundary_lambda_star is None else float(self.boundary_lambda_star)
            ),
            "ge_cpsc_lambda_star": (
                "" if self.cpsc_lambda_star is None else float(self.cpsc_lambda_star)
            ),
            "ge_hard_lambda_star": (
                "" if self.hard_lambda_star is None else float(self.hard_lambda_star)
            ),
            "ge_audc_norm_f": float(self.audc_norm_f),
            "ge_margin_drop_f": float(self.margin_drop_f),
            "ge_monotonicity_violations": int(self.monotonicity_violations),
            "ge_degradation_rate": float(self.degradation_rate),
            "ge_max_degradation_rate": float(self.max_degradation_rate),
            "ge_cliff_flag": int(self.cliff_flag),
            "ge_cliff_lambda": (
                "" if self.cliff_lambda is None else float(self.cliff_lambda)
            ),
        }
