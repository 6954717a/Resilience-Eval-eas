#!/usr/bin/env python3

"""
Graceful Extensibility post-processing collector.

Transforms per-episode rollout rows into:

1. episode-level contract evidence
2. family/stress/evolve cells
3. discrete stress-grid capacity summaries
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .contract_margin import (
    ContractMargin,
    MarginThresholds,
    compute_contract_terms,
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


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> Optional[bool]:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return None


def _required_float(
    row: Mapping[str, Any],
    keys: Sequence[str],
    label: str,
    missing: List[str],
    sources: Dict[str, str],
) -> Optional[float]:
    for key in keys:
        value = _coerce_float(row.get(key))
        if value is not None:
            sources[label] = str(key)
            return value
    missing.append(f"missing_{label}")
    return None


def _format_lambda_list(values: Sequence[float]) -> str:
    return "|".join(f"{float(v):.6g}" for v in values)


def _unique_missing_reason(reasons: Sequence[str]) -> str:
    ordered: List[str] = []
    seen = set()
    for item in reasons:
        if not item:
            continue
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ";".join(ordered)


@dataclass
class EpisodeContractEvidence:
    episode_id: str
    anchor_id: str
    perturbation_type: str
    perturbation_seed: int
    phase_label: str
    family: str
    stress_lambda: float
    evolve_step: int
    task_percent_complete: Optional[float] = None
    proposition_satisfied_fraction: Optional[float] = None
    safety_load: Optional[float] = None
    c_rec: Optional[float] = None
    stability_load: Optional[float] = None
    pc_ratio_raw: Optional[float] = None
    q_ratio_raw: Optional[float] = None
    safe_ratio_raw: Optional[float] = None
    rec_ratio_raw: Optional[float] = None
    stab_ratio_raw: Optional[float] = None
    pc_ratio: Optional[float] = None
    q_ratio: Optional[float] = None
    safe_ratio: Optional[float] = None
    rec_ratio: Optional[float] = None
    stab_ratio: Optional[float] = None
    episode_margin: Optional[float] = None
    episode_soft_margin: Optional[float] = None
    episode_hard_margin: Optional[float] = None
    hard_gate_pass: bool = False
    valid: bool = False
    missing_reasons: List[str] = field(default_factory=list)
    metric_sources: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return the public episode-level hard monitor row ``h_i(lambda)``."""

        return {
            "episode_id": self.episode_id,
            "anchor_id": self.anchor_id,
            "perturbation_type": self.perturbation_type,
            "perturbation_seed": int(self.perturbation_seed),
            "phase_label": self.phase_label,
            "task_family": self.family,
            "stress_lambda": float(self.stress_lambda),
            "ge_stress_lambda": float(self.stress_lambda),
            "evolve_step": int(self.evolve_step),
            "ge_evolve_step_e": int(self.evolve_step),
            "ge_episode_margin": self.episode_margin,
            "ge_episode_margin_hard": self.episode_hard_margin,
            "ge_episode_margin_soft": self.episode_soft_margin,
            "ge_episode_hard_gate_pass": int(self.hard_gate_pass),
            "ge_episode_hard_boundary_hit": int(not self.hard_gate_pass),
            "valid": int(self.valid),
            "missing_reasons": _unique_missing_reason(self.missing_reasons),
        }

    def to_diagnostic_dict(self) -> Dict[str, Any]:
        """Return detailed episode primitives and ratios for audit/debug use."""

        return {
            "episode_id": self.episode_id,
            "anchor_id": self.anchor_id,
            "perturbation_type": self.perturbation_type,
            "perturbation_seed": int(self.perturbation_seed),
            "phase_label": self.phase_label,
            "task_family": self.family,
            "stress_lambda": float(self.stress_lambda),
            "ge_stress_lambda": float(self.stress_lambda),
            "evolve_step": int(self.evolve_step),
            "ge_evolve_step_e": int(self.evolve_step),
            "task_percent_complete": self.task_percent_complete,
            "proposition_satisfied_fraction": self.proposition_satisfied_fraction,
            "safety_load": self.safety_load,
            "rebound_c_rec": self.c_rec,
            "stability_load": self.stability_load,
            "pc_ratio_raw": self.pc_ratio_raw,
            "q_ratio_raw": self.q_ratio_raw,
            "safe_ratio_raw": self.safe_ratio_raw,
            "rec_ratio_raw": self.rec_ratio_raw,
            "stab_ratio_raw": self.stab_ratio_raw,
            "pc_ratio": self.pc_ratio,
            "q_ratio": self.q_ratio,
            "safe_ratio": self.safe_ratio,
            "rec_ratio": self.rec_ratio,
            "stab_ratio": self.stab_ratio,
            "episode_margin": self.episode_margin,
            "ge_episode_margin": self.episode_margin,
            "episode_margin_hard": self.episode_hard_margin,
            "episode_margin_soft": self.episode_soft_margin,
            "ge_episode_margin_hard": self.episode_hard_margin,
            "ge_episode_margin_soft": self.episode_soft_margin,
            "hard_gate_pass": int(self.hard_gate_pass),
            "ge_episode_hard_gate_pass": int(self.hard_gate_pass),
            "ge_episode_hard_boundary_hit": int(not self.hard_gate_pass),
            "valid": int(self.valid),
            "missing_reasons": _unique_missing_reason(self.missing_reasons),
            "metric_source_map": dict(self.metric_sources),
        }


class MarginCollector:
    """Aggregate rollout rows into GE evidence, cells, and capacities."""

    def __init__(self, thresholds: Optional[MarginThresholds] = None) -> None:
        self.thresholds = thresholds or MarginThresholds()
        self.episode_evidence: List[EpisodeContractEvidence] = []
        self.invalid_rows: List[Dict[str, Any]] = []
        self.capacity_rows: List[Dict[str, Any]] = []
        self.capacity_diagnostic_rows: List[Dict[str, Any]] = []

    def aggregate(
        self,
        rollouts: Iterable[Mapping[str, Any]],
    ) -> Dict[Tuple[str, float, int], ContractMargin]:
        valid_buckets: Dict[Tuple[str, float, int], List[EpisodeContractEvidence]] = defaultdict(list)
        invalid_bucket_reasons: Dict[Tuple[str, float, int], List[str]] = defaultdict(list)
        invalid_bucket_sources: Dict[Tuple[str, float, int], List[Dict[str, str]]] = defaultdict(list)
        seen_keys: set[Tuple[str, float, int]] = set()

        self.episode_evidence = []
        self.invalid_rows = []
        self.capacity_rows = []
        self.capacity_diagnostic_rows = []

        for row in rollouts:
            if not isinstance(row, Mapping):
                continue
            evidence = self._parse_row(row)
            self.episode_evidence.append(evidence)
            key = (evidence.family, evidence.stress_lambda, evidence.evolve_step)
            seen_keys.add(key)
            if evidence.valid:
                valid_buckets[key].append(evidence)
            else:
                invalid_bucket_reasons[key].extend(evidence.missing_reasons)
                invalid_bucket_sources[key].append(dict(evidence.metric_sources))
                self.invalid_rows.append(evidence.to_dict())

        margins: Dict[Tuple[str, float, int], ContractMargin] = {}
        alpha = float(self.thresholds.quantile_alpha)
        upper_q = 1.0 - alpha
        for key in sorted(seen_keys, key=lambda item: (str(item[0]), float(item[1]), int(item[2]))):
            family, stress_lambda, evolve_step = key
            rows = self._select_rows_for_cell(valid_buckets.get(key, []), stress_lambda)
            if not rows:
                margins[key] = ContractMargin.invalid(
                    family=family,
                    stress_lambda=stress_lambda,
                    evolve_step=evolve_step,
                    cell_missing_reason=_unique_missing_reason(invalid_bucket_reasons.get(key, ["invalid_margin_cell"])),
                    n_episodes=0,
                    metric_source_map=self._merge_metric_sources(invalid_bucket_sources.get(key, [])),
                )
                continue
            pc = _percentile([r.task_percent_complete for r in rows if r.task_percent_complete is not None], alpha)
            q = _percentile([r.proposition_satisfied_fraction for r in rows if r.proposition_satisfied_fraction is not None], alpha)
            u = _percentile([r.safety_load for r in rows if r.safety_load is not None], upper_q)
            c_rec = _percentile([r.c_rec for r in rows if r.c_rec is not None], upper_q)
            v_stab = _percentile([r.stability_load for r in rows if r.stability_load is not None], upper_q)
            margins[key] = ContractMargin.compute(
                family=family,
                stress_lambda=stress_lambda,
                evolve_step=evolve_step,
                pc_f=pc,
                q_f=q,
                u_f=u,
                c_rec_p90_f=c_rec,
                v_stab_p90_f=v_stab,
                thresholds=self.thresholds,
                n_episodes=len(rows),
                cell_valid=True,
                cell_missing_reason="",
                metric_source_map=self._merge_metric_sources([r.metric_sources for r in rows]),
            )
        return margins

    @staticmethod
    def _select_rows_for_cell(
        rows: Sequence[EpisodeContractEvidence],
        stress_lambda: float,
    ) -> List[EpisodeContractEvidence]:
        """Select rows used to estimate one family/lambda cell.

        Exp-Q1 GE treats lambda=0 as the zero-stress reference.  Some runners
        write both an explicit clean reference row and a no-op stress row at
        lambda=0. Mixing the two silently changes the reference cell and makes
        lambda curves hard to interpret. Prefer the explicit clean reference
        when it is present; otherwise use all valid rows in the cell.
        """

        rows = list(rows or [])
        if abs(float(stress_lambda)) > 1e-12:
            return rows
        clean_rows = [
            row
            for row in rows
            if row.perturbation_type.strip().lower() in {"clean", "baseline"}
            or row.phase_label.strip().lower() == "clean_reference"
        ]
        return clean_rows or rows

    def finalize_audc(
        self,
        margins: Mapping[Tuple[str, float, int], ContractMargin],
    ) -> Dict[Tuple[str, int], float]:
        by_family_evolve: Dict[Tuple[str, int], List[ContractMargin]] = defaultdict(list)
        for margin in margins.values():
            by_family_evolve[(margin.family, margin.evolve_step)].append(margin)

        audc_map: Dict[Tuple[str, int], float] = {}
        capacity_rows: List[Dict[str, Any]] = []
        capacity_diagnostic_rows: List[Dict[str, Any]] = []
        expected_grid = tuple(sorted(float(v) for v in self.thresholds.stress_lambdas))

        for (family, evolve_step), ordered in by_family_evolve.items():
            ordered.sort(key=lambda item: item.stress_lambda)
            valid_ordered = [m for m in ordered if m.cell_valid]
            observed_valid = sorted(float(m.stress_lambda) for m in valid_ordered)
            expected = list(expected_grid) if expected_grid else observed_valid
            expected_set = {float(v) for v in expected}
            observed_set = {float(v) for v in observed_valid}
            if expected:
                coverage = float(len(observed_set & expected_set) / len(expected_set))
                missing_lambdas = sorted(expected_set - observed_set)
            else:
                coverage = 0.0 if not observed_valid else 1.0
                missing_lambdas = []

            xs = [m.stress_lambda for m in valid_ordered]
            ys = [max(m.margin + 1.0, 0.0) for m in valid_ordered]
            audc = 0.0
            for idx in range(1, len(xs)):
                dx = xs[idx] - xs[idx - 1]
                audc += 0.5 * (ys[idx] + ys[idx - 1]) * dx
            lambda_span = max(xs) - min(xs) if len(xs) >= 2 else 0.0
            audc_norm = audc / lambda_span if lambda_span > 0.0 else audc

            passing = [
                m.stress_lambda
                for m in valid_ordered
                if m.margin >= 0.0
            ]
            failing = [
                m.stress_lambda
                for m in valid_ordered
                if m.margin < 0.0
            ]
            hard_passing = [
                m.stress_lambda
                for m in valid_ordered
                if m.margin >= 0.0 and m.hard_gate_pass
            ]
            hard_failing = [
                m.stress_lambda
                for m in valid_ordered
                if not (m.margin >= 0.0 and m.hard_gate_pass)
            ]
            lambda_star = max(passing) if passing else None
            first_failing = min(failing) if failing else None
            last_passing = max(passing) if passing else None
            hard_lambda_star = max(hard_passing) if hard_passing else None
            first_hard_failing = min(hard_failing) if hard_failing else None
            last_hard_passing = max(hard_passing) if hard_passing else None
            rates_by_lambda: Dict[float, float] = {}
            monotonicity_violations = 0
            previous = None
            for margin in valid_ordered:
                rate = 0.0
                if previous is not None:
                    delta_lambda = margin.stress_lambda - previous.stress_lambda
                    if delta_lambda > 0.0:
                        rate = max(0.0, previous.margin - margin.margin) / delta_lambda
                        if margin.margin > previous.margin + 1e-12:
                            monotonicity_violations += 1
                rates_by_lambda[float(margin.stress_lambda)] = float(rate)
                previous = margin
            max_degradation_rate = max(rates_by_lambda.values()) if rates_by_lambda else 0.0
            margin_drop = 0.0
            if valid_ordered:
                margin_drop = float(valid_ordered[0].margin - valid_ordered[-1].margin)
            cliff_lambdas = [
                lam
                for lam, rate in sorted(rates_by_lambda.items())
                if rate >= float(self.thresholds.degradation_rate_threshold)
            ]
            cliff_lambda = cliff_lambdas[0] if cliff_lambdas else None
            cliff_flag = cliff_lambda is not None

            audc_map[(family, evolve_step)] = float(audc)
            capacity_row = {
                "task_family": family,
                "family": family,
                "evolve_step": int(evolve_step),
                "ge_cpsc_lambda_star": (
                    "" if lambda_star is None else float(lambda_star)
                ),
                "ge_boundary_lambda_star": (
                    "" if lambda_star is None else float(lambda_star)
                ),
                "ge_hard_lambda_star": (
                    "" if hard_lambda_star is None else float(hard_lambda_star)
                ),
                "ge_audc_f": float(audc),
                "ge_audc_norm_f": float(audc_norm),
                "ge_valid_lambda_coverage": float(coverage),
                "ge_valid_cell_count": int(len(valid_ordered)),
                "ge_max_degradation_rate": float(max_degradation_rate),
                "ge_margin_drop_f": float(margin_drop),
                "ge_monotonicity_violations": int(monotonicity_violations),
                "ge_cliff_flag": int(cliff_flag),
                "ge_cliff_lambda": (
                    "" if cliff_lambda is None else float(cliff_lambda)
                ),
                "ge_coverage_warning": int(coverage < 1.0),
            }
            capacity_rows.append(capacity_row)
            capacity_diagnostic_row = dict(capacity_row)
            capacity_diagnostic_row.update(
                {
                    "ge_missing_lambdas": _format_lambda_list(missing_lambdas),
                    "ge_first_failing_lambda": (
                        "" if first_failing is None else float(first_failing)
                    ),
                    "ge_last_passing_lambda": (
                        "" if last_passing is None else float(last_passing)
                    ),
                    "ge_first_hard_failing_lambda": (
                        "" if first_hard_failing is None else float(first_hard_failing)
                    ),
                    "ge_last_hard_passing_lambda": (
                        "" if last_hard_passing is None else float(last_hard_passing)
                    ),
                    "ge_expected_lambda_count": int(len(expected)),
                    "ge_valid_lambda_count": int(len(observed_valid)),
                    "ge_cell_count": int(len(ordered)),
                }
            )
            capacity_diagnostic_rows.append(capacity_diagnostic_row)
            for margin in ordered:
                margin.audc_f = float(audc)
                margin.audc_norm_f = float(audc_norm)
                margin.valid_lambda_coverage = float(coverage)
                margin.boundary_lambda_star = lambda_star
                margin.cpsc_lambda_star = lambda_star
                margin.hard_lambda_star = hard_lambda_star
                margin.missing_lambdas = _format_lambda_list(missing_lambdas)
                margin.first_failing_lambda = first_failing
                margin.last_passing_lambda = last_passing
                margin.first_hard_failing_lambda = first_hard_failing
                margin.last_hard_passing_lambda = last_hard_passing
                margin.degradation_rate = rates_by_lambda.get(
                    float(margin.stress_lambda),
                    0.0,
                )
                margin.max_degradation_rate = float(max_degradation_rate)
                margin.margin_drop_f = float(margin_drop)
                margin.monotonicity_violations = int(monotonicity_violations)
                margin.cliff_flag = bool(cliff_flag)
                margin.cliff_lambda = cliff_lambda
        self.capacity_rows = capacity_rows
        self.capacity_diagnostic_rows = capacity_diagnostic_rows
        return audc_map

    def _parse_row(self, row: Mapping[str, Any]) -> EpisodeContractEvidence:
        missing: List[str] = []
        sources: Dict[str, str] = {}

        episode_id = str(row.get("episode_id") or row.get("anchor_id") or "unknown")
        anchor_id = str(row.get("anchor_id") or episode_id)
        family = str(row.get("task_family", "other"))
        perturbation_type = str(row.get("perturbation_type", "clean"))
        perturbation_seed = _coerce_int(row.get("perturbation_seed")) or 0
        phase_label = str(row.get("phase_label") or "")
        stress_lambda = _coerce_float(row.get("stress_lambda", row.get("ge_stress_lambda")))
        evolve_step = _coerce_int(row.get("evolve_step", row.get("ge_evolve_step_e")))

        task_percent_complete = _required_float(
            row,
            ("task_percent_complete",),
            "task_percent_complete",
            missing,
            sources,
        )
        proposition_satisfied_fraction = _required_float(
            row,
            ("proposition_satisfied_fraction",),
            "proposition_satisfied_fraction",
            missing,
            sources,
        )
        safety_load = self._parse_safety_load(row, missing, sources)
        c_rec = self._parse_c_rec(row, missing, sources)
        stability_load = self._parse_stability_load(row, missing, sources)

        if stress_lambda is None:
            missing.append("missing_stress_lambda")
            stress_lambda = 0.0
        if evolve_step is None:
            missing.append("missing_evolve_step")
            evolve_step = 0

        evidence = EpisodeContractEvidence(
            episode_id=episode_id,
            anchor_id=anchor_id,
            perturbation_type=perturbation_type,
            perturbation_seed=int(perturbation_seed),
            phase_label=phase_label,
            family=family,
            stress_lambda=float(stress_lambda),
            evolve_step=int(evolve_step),
            task_percent_complete=task_percent_complete,
            proposition_satisfied_fraction=proposition_satisfied_fraction,
            safety_load=safety_load,
            c_rec=c_rec,
            stability_load=stability_load,
            valid=not missing,
            missing_reasons=list(missing),
            metric_sources=dict(sources),
        )
        if not evidence.valid:
            return evidence

        terms = compute_contract_terms(
            pc_f=float(task_percent_complete),
            q_f=float(proposition_satisfied_fraction),
            u_f=float(safety_load),
            c_rec_f=float(c_rec),
            v_stab_f=float(stability_load),
            thresholds=self.thresholds,
        )
        evidence.pc_ratio_raw = float(terms["pc_term_raw"])
        evidence.q_ratio_raw = float(terms["q_term_raw"])
        evidence.safe_ratio_raw = float(terms["safe_term_raw"])
        evidence.rec_ratio_raw = float(terms["rec_term_raw"])
        evidence.stab_ratio_raw = float(terms["stab_term_raw"])
        evidence.pc_ratio = float(terms["pc_term"])
        evidence.q_ratio = float(terms["q_term"])
        evidence.safe_ratio = float(terms["safe_term"])
        evidence.rec_ratio = float(terms["rec_term"])
        evidence.stab_ratio = float(terms["stab_term"])
        evidence.episode_soft_margin = float(terms["soft_margin"])
        evidence.episode_hard_margin = float(terms["hard_margin"])
        # The episode-level GE monitor in the paper is the hard min-ratio
        # margin. Keep ge_episode_margin as that canonical value and expose
        # ge_episode_margin_soft separately for diagnosis.
        evidence.episode_margin = evidence.episode_hard_margin
        evidence.hard_gate_pass = bool(terms["hard_gate_pass"])
        return evidence

    @staticmethod
    def _parse_c_rec(
        row: Mapping[str, Any],
        missing: List[str],
        sources: Dict[str, str],
    ) -> Optional[float]:
        valid_flag = _truthy(row.get("rebound_c_rec_valid"))
        if valid_flag is False:
            reason = str(row.get("rebound_c_rec_missing_reason") or "invalid_rebound_c_rec")
            missing.append(reason)
            return None
        for key in ("rebound_c_rec", "c_rec", "rebound_c_rec_p90"):
            value = _coerce_float(row.get(key))
            if value is not None:
                sources["c_rec"] = key if valid_flag is not None else f"{key}:legacy_unvalidated"
                return value
        missing.append("missing_rebound_c_rec")
        return None

    @staticmethod
    def _parse_stability_load(
        row: Mapping[str, Any],
        missing: List[str],
        sources: Dict[str, str],
    ) -> Optional[float]:
        valid_flag = _truthy(row.get("stability_beta_neighborhood_valid"))
        beta = _coerce_float(row.get("stability_beta_neighborhood"))
        if beta is not None and valid_flag is not False:
            sources["stability_load"] = (
                "stability_beta_neighborhood"
                if valid_flag is not None
                else "stability_beta_neighborhood:legacy_unvalidated"
            )
            return beta
        for key in ("stability_value_delta_variance", "value_delta_variance"):
            value = _coerce_float(row.get(key))
            if value is not None:
                sources["stability_load"] = f"{key}:proxy"
                return value
        missing.append("missing_stability_load")
        return None

    def _parse_safety_load(
        self,
        row: Mapping[str, Any],
        missing: List[str],
        sources: Dict[str, str],
    ) -> Optional[float]:
        explicit = _coerce_float(row.get("unsafe_rate"))
        if explicit is not None:
            sources["unsafe_rate"] = "unsafe_rate"
            return explicit

        components: List[float] = []
        source_parts: List[str] = []
        for key in ("constraint_violation_rate", "safety_constraint_violation_rate"):
            value = _coerce_float(row.get(key))
            if value is not None:
                components.append(value)
                source_parts.append(key)
                break

        collision = _coerce_float(row.get("safety_collision_rate", row.get("collision_rate")))
        if collision is not None:
            components.append(collision)
            source_parts.append(
                "safety_collision_rate" if row.get("safety_collision_rate") not in (None, "") else "collision_rate"
            )

        penalty_rate = _coerce_float(row.get("normalized_cbf_penalty_rate"))
        if penalty_rate is not None:
            components.append(penalty_rate)
            source_parts.append("normalized_cbf_penalty_rate")
        else:
            penalty = _coerce_float(row.get("stability_p_cbf"))
            total_steps = _coerce_float(row.get("total_step_count", row.get("sim_step_count")))
            if penalty is not None and total_steps is not None and total_steps > 0:
                components.append(penalty / total_steps)
                source_parts.append("stability_p_cbf/total_step_count")

        if components:
            sources["unsafe_rate"] = "+".join(source_parts)
            return float(sum(components))
        policy = str(getattr(self.thresholds, "missing_safety_policy", "require"))
        if policy.strip().lower() in {"assume_zero", "zero", "safe_default"}:
            sources["unsafe_rate"] = "missing_safety_policy:assume_zero"
            return float(getattr(self.thresholds, "missing_safety_default", 0.0))
        missing.append("missing_unsafe_rate")
        return None

    @staticmethod
    def _merge_metric_sources(source_items: Sequence[Mapping[str, str]]) -> Dict[str, str]:
        merged: Dict[str, set[str]] = defaultdict(set)
        for row_sources in source_items:
            for key, value in row_sources.items():
                merged[key].add(str(value))
        return {key: "|".join(sorted(values)) for key, values in merged.items()}

    @staticmethod
    def write_csv(
        margins: Mapping[Tuple[str, float, int], ContractMargin],
        output_path: Union[str, Path],
    ) -> Path:
        rows = [margin.to_dict() for margin in margins.values()]
        return MarginCollector.write_rows(rows, output_path)

    @staticmethod
    def write_diagnostics_csv(
        margins: Mapping[Tuple[str, float, int], ContractMargin],
        output_path: Union[str, Path],
    ) -> Path:
        rows = [margin.to_diagnostic_dict() for margin in margins.values()]
        return MarginCollector.write_rows(rows, output_path)

    @staticmethod
    def write_rows(
        rows: Sequence[Mapping[str, Any]],
        output_path: Union[str, Path],
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            return output_path
        fieldnames = sorted({str(key) for row in rows for key in row.keys()})
        with open(output_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
        return output_path

    @staticmethod
    def read_rollout_csv(path: Union[str, Path]) -> List[Dict[str, Any]]:
        path = Path(path)
        if not path.exists():
            return []
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return [dict(row) for row in reader]
