#!/usr/bin/env python3

"""
Graceful Extensibility post-processing collector.

Transforms per-episode rollout rows into:

1. episode-level boundary evidence
2. family/stress/evolve cells
3. discrete stress-grid capacity summaries
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from habitat_llm.evaluation.metrics.statistical_baseline import (
    MetricReference,
    StatisticalBaseline,
    empirical_quantile,
)
from habitat_llm.evaluation.reporting.tabular import (
    read_csv_rows,
    write_csv_rows_exact,
)
from habitat_llm.evaluation.perturbation.bank import (
    DEFAULT_TEXT_DOSE_TOLERANCE as FORMAL_TEXT_DOSE_TOLERANCE,
    STRESS_FAMILY,
    STRESS_GRID as FORMAL_STRESS_GRID,
)
from habitat_llm.evaluation.stage_baseline import normalize_judge_model

from .boundary_margin import (
    BoundaryMargin,
    BoundaryThresholds,
    compute_boundary_terms,
)


_LAMBDA_TOLERANCE = 1.0e-9

# Legacy callers consume ``(family, lambda, evolve_step)``.  Frozen stress
# rows carry enough provenance to use the strict five-dimensional key without
# silently merging Judges or stress curricula.
LegacyCellKey = Tuple[str, float, int]
FormalCellKey = Tuple[str, str, str, float, int]
CellKey = Union[LegacyCellKey, FormalCellKey]
LegacyCurveKey = Tuple[str, int]
FormalCurveKey = Tuple[str, str, str, int]
CurveKey = Union[LegacyCurveKey, FormalCurveKey]


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
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
class EpisodeBoundaryEvidence:
    episode_id: str
    anchor_id: str
    perturbation_type: str
    perturbation_seed: int
    phase_label: str
    family: str
    stress_lambda: float
    evolve_step: int
    stress_family: str = "legacy"
    perturbation_variant: str = ""
    perturbation_bank_path: str = ""
    stress_realized: Optional[float] = None
    stress_text_dose: Optional[float] = None
    stress_alignment_valid: bool = False
    strict_stress_contract: bool = False
    paired_grid_eligible: bool = True
    paired_grid_missing_reason: str = ""
    judge_model: str = "unconfigured-judge"
    judge_base_url: str = ""
    stage_baseline_id: str = ""
    stage_baseline_path: str = ""
    baseline_contract_id: str = ""
    state_encoder_id: str = ""
    value_checkpoint_id: str = ""
    stage_baseline_episode_id: str = ""
    stage_baseline_episode_match: bool = False
    reward_shaper_valid: bool = False
    critic_lifecycle_valid: bool = False
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
    boundary_reference_valid: bool = False
    boundary_reference_n: int = 0
    boundary_reference_missing_reason: str = "not_estimated"

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
            "stress_family": self.stress_family,
            "perturbation_stress_family": self.stress_family,
            "perturbation_variant": self.perturbation_variant,
            "perturbation_bank_path": self.perturbation_bank_path,
            "stress_realized": self.stress_realized,
            "perturbation_stress_realized": self.stress_realized,
            "stress_text_dose": self.stress_text_dose,
            "perturbation_text_dose": self.stress_text_dose,
            "stress_alignment_valid": int(self.stress_alignment_valid),
            "perturbation_alignment_valid": int(self.stress_alignment_valid),
            "stress_contract_formal": int(self.strict_stress_contract),
            "ge_paired_grid_eligible": int(self.paired_grid_eligible),
            "evolve_step": int(self.evolve_step),
            "ge_evolve_step_e": int(self.evolve_step),
            "judge_model": self.judge_model,
            "stage_baseline_id": self.stage_baseline_id,
            "stage_baseline_path": self.stage_baseline_path,
            "baseline_contract_id": self.baseline_contract_id,
            "state_encoder_id": self.state_encoder_id,
            "value_checkpoint_id": self.value_checkpoint_id,
            "stage_baseline_episode_id": self.stage_baseline_episode_id,
            "stage_baseline_episode_match": int(self.stage_baseline_episode_match),
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
            "stress_family": self.stress_family,
            "perturbation_stress_family": self.stress_family,
            "perturbation_variant": self.perturbation_variant,
            "perturbation_bank_path": self.perturbation_bank_path,
            "stress_realized": self.stress_realized,
            "perturbation_stress_realized": self.stress_realized,
            "stress_text_dose": self.stress_text_dose,
            "perturbation_text_dose": self.stress_text_dose,
            "stress_alignment_valid": int(self.stress_alignment_valid),
            "perturbation_alignment_valid": int(self.stress_alignment_valid),
            "stress_contract_formal": int(self.strict_stress_contract),
            "ge_paired_grid_eligible": int(self.paired_grid_eligible),
            "ge_paired_grid_missing_reason": self.paired_grid_missing_reason,
            "evolve_step": int(self.evolve_step),
            "ge_evolve_step_e": int(self.evolve_step),
            "judge_model": self.judge_model,
            "judge_base_url": self.judge_base_url,
            "stage_baseline_id": self.stage_baseline_id,
            "stage_baseline_path": self.stage_baseline_path,
            "baseline_contract_id": self.baseline_contract_id,
            "state_encoder_id": self.state_encoder_id,
            "value_checkpoint_id": self.value_checkpoint_id,
            "stage_baseline_episode_id": self.stage_baseline_episode_id,
            "stage_baseline_episode_match": int(self.stage_baseline_episode_match),
            "reward_shaper_valid": int(self.reward_shaper_valid),
            "critic_lifecycle_valid": int(self.critic_lifecycle_valid),
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
            "ge_boundary_reference_valid": int(self.boundary_reference_valid),
            "ge_boundary_reference_n": int(self.boundary_reference_n),
            "ge_boundary_reference_missing_reason": self.boundary_reference_missing_reason,
        }


class BoundaryCollector:
    """Aggregate rollout rows into GE evidence, cells, and capacities."""

    def __init__(self, thresholds: Optional[BoundaryThresholds] = None) -> None:
        self.thresholds = thresholds or BoundaryThresholds()
        self.episode_evidence: List[EpisodeBoundaryEvidence] = []
        self.invalid_rows: List[Dict[str, Any]] = []
        self.capacity_rows: List[Dict[str, Any]] = []
        self.capacity_diagnostic_rows: List[Dict[str, Any]] = []
        self.statistical_baselines: Dict[Tuple[str, int], StatisticalBaseline] = {}
        self.formal_pairing_stats: Dict[FormalCurveKey, Dict[str, int]] = {}

    @staticmethod
    def _curve_key(evidence: EpisodeBoundaryEvidence) -> CurveKey:
        if evidence.strict_stress_contract:
            return (
                normalize_judge_model(evidence.judge_model),
                evidence.family,
                evidence.stress_family,
                int(evidence.evolve_step),
            )
        return (evidence.family, int(evidence.evolve_step))

    @classmethod
    def _cell_key(cls, evidence: EpisodeBoundaryEvidence) -> CellKey:
        curve = cls._curve_key(evidence)
        if evidence.strict_stress_contract:
            judge, family, stress_family, evolve_step = curve
            return (
                str(judge),
                str(family),
                str(stress_family),
                float(evidence.stress_lambda),
                int(evolve_step),
            )
        family, evolve_step = curve
        return (str(family), float(evidence.stress_lambda), int(evolve_step))

    @staticmethod
    def _unpack_cell_key(
        key: CellKey,
    ) -> Tuple[str, str, str, float, int, bool]:
        if len(key) == 5:
            judge, family, stress_family, stress_lambda, evolve_step = key
            return (
                str(judge),
                str(family),
                str(stress_family),
                float(stress_lambda),
                int(evolve_step),
                True,
            )
        family, stress_lambda, evolve_step = key
        return (
            "unconfigured-judge",
            str(family),
            "legacy",
            float(stress_lambda),
            int(evolve_step),
            False,
        )

    @staticmethod
    def _baseline_scope_key(
        *,
        family: str,
        judge_model: str = "",
        stress_family: str = "legacy",
        formal: bool = False,
    ) -> str:
        if not formal:
            return str(family)
        return "\x1f".join(
            (
                normalize_judge_model(judge_model),
                str(family),
                str(stress_family),
            )
        )

    def _prepare_formal_pairing(
        self,
        rows: Sequence[EpisodeBoundaryEvidence],
    ) -> None:
        """Keep one common episode×seed support set across all six levels."""

        grouped: Dict[
            FormalCurveKey,
            Dict[Tuple[str, int], List[EpisodeBoundaryEvidence]],
        ] = defaultdict(lambda: defaultdict(list))
        for evidence in rows:
            if not evidence.strict_stress_contract:
                evidence.paired_grid_eligible = True
                continue
            curve = self._curve_key(evidence)
            assert len(curve) == 4
            grouped[curve][
                (evidence.episode_id, int(evidence.perturbation_seed))
            ].append(evidence)

        expected = {round(value, 12) for value in FORMAL_STRESS_GRID}
        self.formal_pairing_stats = {}
        for curve, pair_rows in grouped.items():
            complete = 0
            for pair, members in pair_rows.items():
                by_lambda: Dict[float, List[EpisodeBoundaryEvidence]] = defaultdict(list)
                for evidence in members:
                    evidence.paired_grid_eligible = False
                    by_lambda[round(float(evidence.stress_lambda), 12)].append(
                        evidence
                    )
                observed = set(by_lambda)
                duplicated = any(len(items) != 1 for items in by_lambda.values())
                rows_valid = all(evidence.valid for evidence in members)
                path_witnesses = {
                    (
                        evidence.perturbation_variant,
                        evidence.perturbation_bank_path,
                    )
                    for evidence in members
                }
                path_complete = bool(
                    len(path_witnesses) == 1
                    and all(next(iter(path_witnesses)))
                )
                eligible = bool(
                    observed == expected
                    and not duplicated
                    and rows_valid
                    and path_complete
                )
                if eligible:
                    complete += 1
                    for evidence in members:
                        evidence.paired_grid_eligible = True
                    continue
                if duplicated:
                    reason = "duplicate_episode_seed_stress_level"
                elif observed != expected:
                    reason = "incomplete_episode_seed_stress_grid"
                elif not path_complete:
                    reason = "mixed_or_missing_frozen_perturbation_path"
                else:
                    reason = "invalid_episode_seed_stress_grid"
                for evidence in members:
                    evidence.valid = False
                    evidence.paired_grid_missing_reason = reason
                    if reason not in evidence.missing_reasons:
                        evidence.missing_reasons.append(reason)
                    evidence.metric_sources["paired_stress_grid"] = (
                        f"episode={pair[0]},seed={pair[1]}"
                    )
            curve_bank_paths = {
                evidence.perturbation_bank_path
                for members in pair_rows.values()
                for evidence in members
                if evidence.paired_grid_eligible
            }
            if len(curve_bank_paths) > 1:
                complete = 0
                for members in pair_rows.values():
                    for evidence in members:
                        if not evidence.paired_grid_eligible:
                            continue
                        evidence.valid = False
                        evidence.paired_grid_eligible = False
                        evidence.paired_grid_missing_reason = (
                            "mixed_perturbation_bank_path"
                        )
                        evidence.missing_reasons.append(
                            "mixed_perturbation_bank_path"
                        )
            self.formal_pairing_stats[curve] = {
                "total_pairs": int(len(pair_rows)),
                "complete_pairs": int(complete),
            }

    @staticmethod
    def _attach_margin_scope(
        margin: BoundaryMargin,
        *,
        judge_model: str,
        stress_family: str,
        formal: bool,
        pairing_stats: Optional[Mapping[str, int]] = None,
    ) -> BoundaryMargin:
        # BoundaryMargin predates Judge/stress-family curves and is intentionally
        # kept source compatible. Collector serializers expose these attributes.
        margin.ge_judge_model = str(judge_model)
        margin.ge_stress_family = str(stress_family)
        margin.ge_formal_stress_contract = bool(formal)
        margin.ge_total_episode_seed_pairs = int(
            (pairing_stats or {}).get("total_pairs", 0)
        )
        margin.ge_complete_episode_seed_pairs = int(
            (pairing_stats or {}).get("complete_pairs", 0)
        )
        return margin

    def aggregate(
        self,
        rollouts: Iterable[Mapping[str, Any]],
    ) -> Dict[CellKey, BoundaryMargin]:
        valid_buckets: Dict[CellKey, List[EpisodeBoundaryEvidence]] = defaultdict(list)
        invalid_buckets: Dict[CellKey, List[EpisodeBoundaryEvidence]] = defaultdict(list)
        seen_keys: set[CellKey] = set()

        self.episode_evidence = []
        self.invalid_rows = []
        self.capacity_rows = []
        self.capacity_diagnostic_rows = []
        self.statistical_baselines = {}

        for row in rollouts:
            if not isinstance(row, Mapping):
                continue
            evidence = self._parse_row(row)
            self.episode_evidence.append(evidence)

        self._prepare_formal_pairing(self.episode_evidence)

        lineage_by_curve: Dict[CurveKey, set[Tuple[str, ...]]] = defaultdict(set)
        for evidence in self.episode_evidence:
            if evidence.strict_stress_contract and not evidence.paired_grid_eligible:
                continue
            lineage = (
                (
                    evidence.stage_baseline_path
                    if evidence.strict_stress_contract
                    else evidence.stage_baseline_path
                    or evidence.stage_baseline_id
                ),
                normalize_judge_model(evidence.judge_model),
            )
            if all(lineage):
                lineage_by_curve[self._curve_key(evidence)].add(lineage)
        mixed_lineage_curves = {
            key for key, values in lineage_by_curve.items() if len(values) > 1
        }

        for evidence in self.episode_evidence:
            curve = self._curve_key(evidence)
            if curve in mixed_lineage_curves:
                evidence.valid = False
                evidence.missing_reasons.append("mixed_baseline_lineage")
            key = self._cell_key(evidence)
            seen_keys.add(key)
            if evidence.valid:
                valid_buckets[key].append(evidence)
            else:
                self.invalid_rows.append(evidence.to_dict())
                # Incomplete formal episode×seed paths are coverage exclusions,
                # not evidence that poisons otherwise complete paired paths.
                if not (
                    evidence.strict_stress_contract
                    and not evidence.paired_grid_eligible
                ):
                    invalid_buckets[key].append(evidence)

        # A formal curve always exposes all six configured cells, including
        # cells absent because every candidate path was incomplete.
        for curve in self.formal_pairing_stats:
            judge, family, stress_family, evolve_step = curve
            for stress_lambda in FORMAL_STRESS_GRID:
                seen_keys.add(
                    (
                        judge,
                        family,
                        stress_family,
                        float(stress_lambda),
                        int(evolve_step),
                    )
                )

        self.statistical_baselines = self._estimate_statistical_baselines(
            self.episode_evidence
        )
        self._apply_episode_boundaries(self.episode_evidence)

        margins: Dict[CellKey, BoundaryMargin] = {}
        alpha = float(self.thresholds.quantile_alpha)
        upper_q = 1.0 - alpha
        for key in sorted(seen_keys, key=lambda item: tuple(str(v) for v in item)):
            (
                judge_model,
                family,
                stress_family,
                stress_lambda,
                evolve_step,
                formal,
            ) = self._unpack_cell_key(key)
            curve: CurveKey = (
                (judge_model, family, stress_family, evolve_step)
                if formal
                else (family, evolve_step)
            )
            pairing_stats = (
                self.formal_pairing_stats.get(curve, {}) if formal else {}
            )
            valid_candidates = list(valid_buckets.get(key, []))
            invalid_candidates = list(invalid_buckets.get(key, []))
            all_candidates = valid_candidates + invalid_candidates
            has_explicit_clean = bool(
                abs(float(stress_lambda)) <= 1e-12
                and any(self._is_explicit_clean(row) for row in all_candidates)
            )
            if abs(float(stress_lambda)) <= 1e-12 and not has_explicit_clean:
                margins[key] = self._attach_margin_scope(
                    BoundaryMargin.invalid(
                        family=family,
                        stress_lambda=stress_lambda,
                        evolve_step=evolve_step,
                        cell_missing_reason=(
                            "no_complete_episode_seed_stress_grid"
                            if formal and not pairing_stats.get("complete_pairs")
                            else "clean_evaluation_reference_missing"
                        ),
                        n_episodes=len(all_candidates),
                        metric_source_map=self._merge_metric_sources(
                            [row.metric_sources for row in all_candidates]
                        ),
                    ),
                    judge_model=judge_model,
                    stress_family=stress_family,
                    formal=formal,
                    pairing_stats=pairing_stats,
                )
                continue
            if has_explicit_clean:
                rows = [
                    row for row in valid_candidates if self._is_explicit_clean(row)
                ]
                selected_invalid = [
                    row for row in invalid_candidates if self._is_explicit_clean(row)
                ]
            else:
                rows = self._select_rows_for_cell(
                    valid_candidates, stress_lambda
                )
                selected_invalid = invalid_candidates
            if not rows or selected_invalid:
                selected_reasons = [
                    reason
                    for row in selected_invalid
                    for reason in row.missing_reasons
                ]
                selected_sources = [
                    row.metric_sources for row in rows + selected_invalid
                ]
                margins[key] = self._attach_margin_scope(
                    BoundaryMargin.invalid(
                        family=family,
                        stress_lambda=stress_lambda,
                        evolve_step=evolve_step,
                        cell_missing_reason=_unique_missing_reason(
                            selected_reasons
                            or [
                                "no_complete_episode_seed_stress_grid"
                                if formal
                                and not pairing_stats.get("complete_pairs")
                                else "invalid_margin_cell"
                            ]
                        ),
                        n_episodes=len(rows) + len(selected_invalid),
                        metric_source_map=self._merge_metric_sources(
                            selected_sources
                        ),
                    ),
                    judge_model=judge_model,
                    stress_family=stress_family,
                    formal=formal,
                    pairing_stats=pairing_stats,
                )
                continue
            baseline_scope = self._baseline_scope_key(
                family=family,
                judge_model=judge_model,
                stress_family=stress_family,
                formal=formal,
            )
            baseline = self.statistical_baselines.get((baseline_scope, 0))
            if formal and not bool(baseline and baseline.valid):
                margins[key] = self._attach_margin_scope(
                    BoundaryMargin.invalid(
                        family=family,
                        stress_lambda=stress_lambda,
                        evolve_step=evolve_step,
                        cell_missing_reason=(
                            baseline.missing_reason
                            if baseline and baseline.missing_reason
                            else "insufficient_clean_boundary_reference"
                        ),
                        n_episodes=len(rows),
                        metric_source_map=self._merge_metric_sources(
                            [row.metric_sources for row in rows]
                        ),
                    ),
                    judge_model=judge_model,
                    stress_family=stress_family,
                    formal=True,
                    pairing_stats=pairing_stats,
                )
                continue
            pc = empirical_quantile([r.task_percent_complete for r in rows if r.task_percent_complete is not None], alpha)
            q = empirical_quantile([r.proposition_satisfied_fraction for r in rows if r.proposition_satisfied_fraction is not None], alpha)
            u = empirical_quantile([r.safety_load for r in rows if r.safety_load is not None], upper_q)
            c_rec = empirical_quantile([r.c_rec for r in rows if r.c_rec is not None], upper_q)
            v_stab = empirical_quantile([r.stability_load for r in rows if r.stability_load is not None], upper_q)
            cell_thresholds = self._resolved_thresholds(
                family,
                evolve_step,
                judge_model=judge_model,
                stress_family=stress_family,
                formal=formal,
            )
            source_map = self._merge_metric_sources(
                [r.metric_sources for r in rows]
            )
            source_map.update(
                {
                    "boundary_reference": (
                        str(baseline.metadata.get("reference_source", ""))
                        if baseline and baseline.valid
                        else "configured_fallback"
                    ),
                    "boundary_reference_n": str(
                        baseline.n_samples if baseline else 0
                    ),
                    "boundary_reference_valid": str(
                        int(bool(baseline and baseline.valid))
                    ),
                }
            )
            margins[key] = self._attach_margin_scope(
                BoundaryMargin.compute(
                    family=family,
                    stress_lambda=stress_lambda,
                    evolve_step=evolve_step,
                    pc_f=pc,
                    q_f=q,
                    u_f=u,
                    c_rec_p90_f=c_rec,
                    v_stab_p90_f=v_stab,
                    thresholds=cell_thresholds,
                    n_episodes=len(rows),
                    cell_valid=True,
                    cell_missing_reason="",
                    metric_source_map=source_map,
                ),
                judge_model=judge_model,
                stress_family=stress_family,
                formal=formal,
                pairing_stats=pairing_stats,
            )
        return margins

    @staticmethod
    def _is_explicit_clean(row: EpisodeBoundaryEvidence) -> bool:
        return (
            row.perturbation_type.strip().lower() in {"clean", "baseline"}
            or row.phase_label.strip().lower() == "clean_reference"
        )

    def _estimate_statistical_baselines(
        self,
        rows: Sequence[EpisodeBoundaryEvidence],
    ) -> Dict[Tuple[str, int], StatisticalBaseline]:
        """Estimate Judge-scoped clean references for each GE family."""

        grouped: Dict[str, List[EpisodeBoundaryEvidence]] = defaultdict(list)
        for row in rows:
            if row.valid and (
                not row.strict_stress_contract or row.paired_grid_eligible
            ):
                scope_key = self._baseline_scope_key(
                    family=row.family,
                    judge_model=row.judge_model,
                    stress_family=row.stress_family,
                    formal=row.strict_stress_contract,
                )
                grouped[scope_key].append(row)

        baselines: Dict[Tuple[str, int], StatisticalBaseline] = {}
        alpha = float(self.thresholds.quantile_alpha)
        for scope_key, candidates in grouped.items():
            representative = candidates[0]
            family = representative.family
            formal = representative.strict_stress_contract
            stress_family = representative.stress_family
            clean = [
                row
                for row in candidates
                if self._is_explicit_clean(row) and row.evolve_step == 0
            ]
            reference_source = "explicit_clean_evolve_zero"
            if not clean:
                clean = [row for row in candidates if self._is_explicit_clean(row)]
                reference_source = "explicit_clean_any_evolve"

            judge_models = sorted({row.judge_model for row in clean if row.judge_model})
            judge_urls = sorted({row.judge_base_url for row in clean if row.judge_base_url})
            metric_specs = {
                "task_percent_complete": ("higher_is_better", "task_percent_complete"),
                "proposition_satisfied_fraction": (
                    "higher_is_better",
                    "proposition_satisfied_fraction",
                ),
                "safety_load": ("lower_is_better", "safety_load"),
                "rebound_c_rec": ("lower_is_better", "c_rec"),
                "stability_load": ("lower_is_better", "stability_load"),
            }
            references: Dict[str, MetricReference] = {}
            for metric, (direction, attribute) in metric_specs.items():
                values = [getattr(row, attribute) for row in clean]
                references[metric] = MetricReference.estimate(
                    metric,
                    direction,
                    (value for value in values if value is not None),
                    alpha,
                )
            baselines[(scope_key, 0)] = StatisticalBaseline(
                scope=(
                    "judge_task_family_stress_family"
                    if formal
                    else "task_family"
                ),
                key={
                    "task_family": family,
                    "stress_family": stress_family,
                    "reference_evolve_step": 0,
                    "scope_key": scope_key,
                },
                references=references,
                judge_model="|".join(judge_models) or "unconfigured-judge",
                judge_base_url="|".join(judge_urls),
                min_samples=int(self.thresholds.min_clean_samples),
                metadata={
                    "reference_source": reference_source,
                    "reference_mode": self.thresholds.reference_mode,
                    "benefit_retention_ratio": float(
                        self.thresholds.benefit_retention_ratio
                    ),
                    "cost_tolerance_ratio": float(
                        self.thresholds.cost_tolerance_ratio
                    ),
                },
            )
        return baselines

    def _resolved_thresholds(
        self,
        family: str,
        evolve_step: int,
        *,
        judge_model: str = "",
        stress_family: str = "legacy",
        formal: bool = False,
    ) -> BoundaryThresholds:
        """Resolve configured fallbacks against the clean reference distribution."""

        scope_key = self._baseline_scope_key(
            family=family,
            judge_model=judge_model,
            stress_family=stress_family,
            formal=formal,
        )
        baseline = self.statistical_baselines.get((scope_key, 0))
        if (
            str(self.thresholds.reference_mode).strip().lower() != "clean_quantile"
            or baseline is None
            or not baseline.valid
        ):
            return self.thresholds

        refs = baseline.references
        benefit_retention = max(float(self.thresholds.benefit_retention_ratio), 0.0)
        cost_tolerance = max(float(self.thresholds.cost_tolerance_ratio), 0.0)
        return replace(
            self.thresholds,
            tau_pc=(
                refs["task_percent_complete"].lower_quantile * benefit_retention
            ),
            tau_q=(
                refs["proposition_satisfied_fraction"].lower_quantile
                * benefit_retention
            ),
            tau_safe=refs["safety_load"].upper_quantile * cost_tolerance,
            tau_rec=refs["rebound_c_rec"].upper_quantile * cost_tolerance,
            tau_stab=refs["stability_load"].upper_quantile * cost_tolerance,
        )

    def _apply_episode_boundaries(
        self,
        rows: Sequence[EpisodeBoundaryEvidence],
    ) -> None:
        """Attach episode monitor ratios after clean references are available."""

        for evidence in rows:
            if not evidence.valid:
                continue
            scope_key = self._baseline_scope_key(
                family=evidence.family,
                judge_model=evidence.judge_model,
                stress_family=evidence.stress_family,
                formal=evidence.strict_stress_contract,
            )
            baseline = self.statistical_baselines.get(
                (scope_key, 0)
            )
            evidence.boundary_reference_valid = bool(baseline and baseline.valid)
            evidence.boundary_reference_n = baseline.n_samples if baseline else 0
            evidence.boundary_reference_missing_reason = (
                "" if baseline and baseline.valid else (
                    baseline.missing_reason if baseline else "clean_reference_missing"
                )
            )
            thresholds = self._resolved_thresholds(
                evidence.family,
                evidence.evolve_step,
                judge_model=evidence.judge_model,
                stress_family=evidence.stress_family,
                formal=evidence.strict_stress_contract,
            )
            terms = compute_boundary_terms(
                pc_f=float(evidence.task_percent_complete),
                q_f=float(evidence.proposition_satisfied_fraction),
                u_f=float(evidence.safety_load),
                c_rec_f=float(evidence.c_rec),
                v_stab_f=float(evidence.stability_load),
                thresholds=thresholds,
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
            evidence.episode_margin = evidence.episode_hard_margin
            evidence.hard_gate_pass = bool(terms["hard_gate_pass"])

    def statistical_baseline_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for baseline in self.statistical_baselines.values():
            for metric, reference in baseline.references.items():
                row = {
                    "task_family": baseline.key.get("task_family", "other"),
                    "stress_family": baseline.key.get("stress_family", "legacy"),
                    "reference_evolve_step": baseline.key.get(
                        "reference_evolve_step", 0
                    ),
                    "judge_model": baseline.judge_model,
                    "judge_base_url": baseline.judge_base_url,
                    "scope": baseline.scope,
                    "reference_valid": int(baseline.valid),
                    "reference_missing_reason": baseline.missing_reason,
                    "min_samples": int(baseline.min_samples),
                }
                row.update(reference.to_dict())
                rows.append(row)
        return rows

    @staticmethod
    def _select_rows_for_cell(
        rows: Sequence[EpisodeBoundaryEvidence],
        stress_lambda: float,
    ) -> List[EpisodeBoundaryEvidence]:
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
        margins: Mapping[CellKey, BoundaryMargin],
    ) -> Dict[CurveKey, float]:
        by_curve: Dict[CurveKey, List[BoundaryMargin]] = defaultdict(list)
        for margin in margins.values():
            formal = bool(getattr(margin, "ge_formal_stress_contract", False))
            if formal:
                curve: CurveKey = (
                    str(getattr(margin, "ge_judge_model", "unconfigured-judge")),
                    margin.family,
                    str(getattr(margin, "ge_stress_family", "unresolved_stress_family")),
                    margin.evolve_step,
                )
            else:
                curve = (margin.family, margin.evolve_step)
            by_curve[curve].append(margin)

        audc_map: Dict[CurveKey, float] = {}
        capacity_rows: List[Dict[str, Any]] = []
        capacity_diagnostic_rows: List[Dict[str, Any]] = []

        for curve, ordered in by_curve.items():
            formal = len(curve) == 4
            if formal:
                judge_model, family, stress_family, evolve_step = curve
                expected_grid = FORMAL_STRESS_GRID
            else:
                family, evolve_step = curve
                judge_model = "unconfigured-judge"
                stress_family = "legacy"
                expected_grid = tuple(
                    sorted(float(v) for v in self.thresholds.stress_lambdas)
                )
            ordered.sort(key=lambda item: item.stress_lambda)
            configured_set = {
                round(float(value), 12) for value in expected_grid
            }
            valid_ordered = [
                margin
                for margin in ordered
                if margin.cell_valid
                and (
                    not configured_set
                    or round(float(margin.stress_lambda), 12)
                    in configured_set
                )
            ]
            observed_valid = sorted(
                round(float(m.stress_lambda), 12) for m in valid_ordered
            )
            expected = list(expected_grid) if expected_grid else observed_valid
            expected_set = {round(float(v), 12) for v in expected}
            observed_set = set(observed_valid)
            missing_lambdas = sorted(expected_set - observed_set)
            total_pairs = max(
                (
                    int(getattr(margin, "ge_total_episode_seed_pairs", 0))
                    for margin in ordered
                ),
                default=0,
            )
            complete_pairs = max(
                (
                    int(getattr(margin, "ge_complete_episode_seed_pairs", 0))
                    for margin in ordered
                ),
                default=0,
            )
            capacity_valid = bool(
                len(observed_set) >= 2
                and (not expected_grid or not missing_lambdas)
                and (not formal or complete_pairs > 0)
            )
            if formal and complete_pairs <= 0:
                capacity_missing_reason = "no_complete_episode_seed_stress_grid"
            elif not observed_set:
                capacity_missing_reason = "no_valid_stress_cells"
            elif len(observed_set) < 2:
                capacity_missing_reason = "requires_at_least_two_valid_stress_levels"
            elif expected_grid and missing_lambdas:
                capacity_missing_reason = "incomplete_configured_stress_grid"
            else:
                capacity_missing_reason = ""
            if expected:
                coverage = float(len(observed_set & expected_set) / len(expected_set))
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
            # Capacity is the longest continuously passing prefix. A recovery
            # after the first failed level is diagnostic non-monotonicity and
            # cannot enlarge the reported boundary.
            prefix_passing: List[float] = []
            prefix_open = True
            hard_prefix_passing: List[float] = []
            hard_prefix_open = True
            pass_after_failure = False
            hard_pass_after_failure = False
            for margin in valid_ordered:
                soft_pass = margin.margin >= 0.0
                hard_pass = soft_pass and margin.hard_gate_pass
                if prefix_open and soft_pass:
                    prefix_passing.append(float(margin.stress_lambda))
                else:
                    if not prefix_open and soft_pass:
                        pass_after_failure = True
                    prefix_open = False
                if hard_prefix_open and hard_pass:
                    hard_prefix_passing.append(float(margin.stress_lambda))
                else:
                    if not hard_prefix_open and hard_pass:
                        hard_pass_after_failure = True
                    hard_prefix_open = False
            lambda_star = prefix_passing[-1] if prefix_passing else None
            first_failing = min(failing) if failing else None
            last_passing = max(passing) if passing else None
            hard_lambda_star = (
                hard_prefix_passing[-1] if hard_prefix_passing else None
            )
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
            monotonicity_violation = bool(
                monotonicity_violations > 0
                or pass_after_failure
                or hard_pass_after_failure
            )
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

            if not capacity_valid:
                lambda_star = None
                hard_lambda_star = None
                audc = 0.0
                audc_norm = 0.0
                margin_drop = 0.0
                max_degradation_rate = 0.0
                cliff_flag = False
                cliff_lambda = None
            else:
                audc_map[curve] = float(audc)
            capacity_row = {
                "judge_model": str(judge_model),
                "task_family": family,
                "family": family,
                "stress_family": str(stress_family),
                "stress_contract_formal": int(formal),
                "evolve_step": int(evolve_step),
                "ge_boundary_lambda_star": (
                    "" if lambda_star is None else float(lambda_star)
                ),
                "ge_hard_lambda_star": (
                    "" if hard_lambda_star is None else float(hard_lambda_star)
                ),
                "ge_audc_f": float(audc) if capacity_valid else "",
                "ge_audc_norm_f": float(audc_norm) if capacity_valid else "",
                "ge_valid_lambda_coverage": float(coverage),
                "ge_valid_cell_count": int(len(valid_ordered)),
                "ge_capacity_valid": int(capacity_valid),
                "ge_capacity_missing_reason": capacity_missing_reason,
                "ge_max_degradation_rate": (
                    float(max_degradation_rate) if capacity_valid else ""
                ),
                "ge_margin_drop_f": float(margin_drop) if capacity_valid else "",
                "ge_monotonicity_violations": (
                    int(monotonicity_violations) if capacity_valid else ""
                ),
                "ge_monotonicity_violation": (
                    int(monotonicity_violation) if capacity_valid else ""
                ),
                "ge_cliff_flag": int(cliff_flag) if capacity_valid else "",
                "ge_cliff_lambda": (
                    "" if cliff_lambda is None else float(cliff_lambda)
                ),
                "ge_coverage_warning": int(coverage < 1.0),
                "ge_total_episode_seed_pairs": int(total_pairs),
                "ge_complete_episode_seed_pairs": int(complete_pairs),
                "ge_episode_seed_pair_coverage": (
                    float(complete_pairs / total_pairs) if total_pairs else 0.0
                ),
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
                margin.capacity_valid = bool(capacity_valid)
                margin.capacity_missing_reason = capacity_missing_reason
                margin.audc_f = float(audc) if capacity_valid else None
                margin.audc_norm_f = float(audc_norm)
                margin.valid_lambda_coverage = float(coverage)
                margin.boundary_lambda_star = lambda_star
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
                margin.ge_monotonicity_violation = bool(monotonicity_violation)
                margin.cliff_flag = bool(cliff_flag)
                margin.cliff_lambda = cliff_lambda
        self.capacity_rows = capacity_rows
        self.capacity_diagnostic_rows = capacity_diagnostic_rows
        return audc_map

    def _parse_row(self, row: Mapping[str, Any]) -> EpisodeBoundaryEvidence:
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
        # Generic normalization may synthesize ``stress_realized`` for legacy
        # artifacts. Only the frozen-bank fields (or its explicit family name)
        # opt a row into the strict six-level contract.
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
            or ("unresolved_stress_family" if strict_stress_contract else "legacy")
        )
        perturbation_variant = str(row.get("perturbation_variant") or "")
        perturbation_bank_path = str(
            row.get("perturbation_bank_path") or ""
        )
        stress_realized = _coerce_float(_first_value(stress_realized_keys))
        stress_text_dose = _coerce_float(_first_value(stress_text_dose_keys))
        alignment_declared = _truthy(_first_value(stress_alignment_keys))

        is_clean = perturbation_type.strip().lower() in {"", "clean", "baseline"}
        stress_valid_raw = row.get("stress_valid")
        if stress_valid_raw in (None, ""):
            stress_valid_raw = row.get("perturbation_valid")
        if (
            strict_stress_contract
            and stress_valid_raw in (None, "")
            and alignment_declared is not None
        ):
            stress_valid_raw = alignment_declared
        if not is_clean and stress_valid_raw in (None, ""):
            missing.append("invalid_perturbation_realization")
            sources["perturbation_realization"] = (
                "missing_runtime_perturbation_audit"
            )
        elif (
            not is_clean
            and str(stress_valid_raw).strip().lower()
            not in {"1", "1.0", "true", "yes"}
        ):
            missing.append("invalid_perturbation_realization")
            sources["perturbation_realization"] = str(
                row.get("stress_invalid_reason")
                or row.get("perturbation_reason")
                or "runtime_perturbation_audit_invalid"
            )

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

        lineage_present = any(
            row.get(key) not in (None, "")
            for key in (
                "stage_baseline_path",
                "stage_baseline_id",
                "baseline_contract_id",
            )
        )
        if strict_stress_contract or lineage_present:
            baseline_locator = (
                row.get("stage_baseline_path")
                if strict_stress_contract
                else row.get("stage_baseline_path")
                or row.get("stage_baseline_id")
            )
            if not str(baseline_locator or ""):
                missing.append("missing_stage_baseline_path")
            if row.get("judge_model") in (None, ""):
                missing.append("missing_judge_model")
            episode_alignment = _truthy(row.get("stage_baseline_episode_match"))
            aligned_episode_id = str(
                row.get("stage_baseline_episode_id") or ""
            )
            if episode_alignment is None and aligned_episode_id:
                episode_alignment = aligned_episode_id == episode_id
            if episode_alignment is not True or aligned_episode_id != episode_id:
                missing.append("stage_baseline_episode_mismatch")
        if stress_lambda is None:
            missing.append("missing_stress_lambda")
            stress_lambda = 0.0
        if evolve_step is None:
            missing.append("missing_evolve_step")
            evolve_step = 0

        stress_alignment_valid = False
        if strict_stress_contract:
            target = float(stress_lambda)
            expected_levels = {
                round(value, 12) for value in FORMAL_STRESS_GRID
            }
            if raw_stress_family in (None, ""):
                missing.append("missing_stress_family")
            if normalize_judge_model(
                str(row.get("judge_model") or "unconfigured-judge")
            ) == normalize_judge_model("unconfigured-judge"):
                missing.append("missing_judge_model")
            if round(target, 12) not in expected_levels:
                missing.append("unexpected_formal_stress_lambda")
            if stress_realized is None:
                missing.append("missing_stress_realized")
            elif abs(float(stress_realized) - target) > _LAMBDA_TOLERANCE:
                missing.append("stress_realized_target_mismatch")
            if stress_text_dose is None:
                missing.append("missing_stress_text_dose")
            elif abs(float(stress_text_dose) - target) > FORMAL_TEXT_DOSE_TOLERANCE:
                missing.append("stress_text_dose_alignment_failed")
            if alignment_declared is not True:
                missing.append("stress_alignment_invalid")
            stress_alignment_valid = bool(
                alignment_declared is True
                and stress_realized is not None
                and stress_text_dose is not None
                and abs(float(stress_realized) - target) <= _LAMBDA_TOLERANCE
                and abs(float(stress_text_dose) - target)
                <= FORMAL_TEXT_DOSE_TOLERANCE
                and round(target, 12) in expected_levels
            )

        parsed_stage_baseline_episode_id = str(
            row.get("stage_baseline_episode_id") or ""
        )
        parsed_stage_baseline_episode_match = _truthy(
            row.get("stage_baseline_episode_match")
        )
        if parsed_stage_baseline_episode_match is None:
            parsed_stage_baseline_episode_match = bool(
                parsed_stage_baseline_episode_id
                and parsed_stage_baseline_episode_id == episode_id
            )
        else:
            parsed_stage_baseline_episode_match = bool(
                parsed_stage_baseline_episode_match
                and parsed_stage_baseline_episode_id == episode_id
            )

        evidence = EpisodeBoundaryEvidence(
            episode_id=episode_id,
            anchor_id=anchor_id,
            perturbation_type=perturbation_type,
            perturbation_seed=int(perturbation_seed),
            phase_label=phase_label,
            family=family,
            stress_lambda=float(stress_lambda),
            evolve_step=int(evolve_step),
            stress_family=stress_family,
            perturbation_variant=perturbation_variant,
            perturbation_bank_path=perturbation_bank_path,
            stress_realized=stress_realized,
            stress_text_dose=stress_text_dose,
            stress_alignment_valid=stress_alignment_valid,
            strict_stress_contract=strict_stress_contract,
            judge_model=str(row.get("judge_model") or "unconfigured-judge"),
            judge_base_url=str(row.get("judge_base_url") or ""),
            stage_baseline_id=str(row.get("stage_baseline_id") or ""),
            stage_baseline_path=str(row.get("stage_baseline_path") or ""),
            baseline_contract_id=str(row.get("baseline_contract_id") or ""),
            state_encoder_id=str(row.get("state_encoder_id") or ""),
            value_checkpoint_id=str(row.get("value_checkpoint_id") or ""),
            stage_baseline_episode_id=parsed_stage_baseline_episode_id,
            stage_baseline_episode_match=parsed_stage_baseline_episode_match,
            reward_shaper_valid=_truthy(row.get("reward_shaper_valid")) is True,
            critic_lifecycle_valid=(
                _truthy(row.get("critic_lifecycle_valid")) is True
            ),
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
        valid_flag = _truthy(row.get("stability_beta_at_lambda_valid"))
        beta = _coerce_float(row.get("stability_beta_at_lambda"))
        if beta is not None and valid_flag is True:
            sources["stability_load"] = "stability_beta_at_lambda"
            return beta
        reason = str(
            row.get("stability_beta_at_lambda_missing_reason")
            or "missing_formal_stability_beta_at_lambda"
        )
        missing.append(reason)
        return None

    def _parse_safety_load(
        self,
        row: Mapping[str, Any],
        missing: List[str],
        sources: Dict[str, str],
    ) -> Optional[float]:
        required_channels = tuple(
            str(channel).strip().lower()
            for channel in getattr(
                self.thresholds, "required_safety_channels", ()
            )
            if str(channel).strip()
        )
        if required_channels:
            valid_flag = _truthy(row.get("safety_valid"))
            if valid_flag is not True:
                reason = str(
                    row.get("safety_missing_reason")
                    or (
                        "missing_safety_validity"
                        if valid_flag is None
                        else "invalid_safety_evidence"
                    )
                )
                missing.append(reason)
                sources["safety_validity"] = reason
                return None
            raw_scope = str(row.get("safety_scope") or "").strip().lower()
            normalized_scope = raw_scope.replace("_only", "").replace("+", ",")
            observed_channels = {
                token.strip()
                for token in normalized_scope.split(",")
                if token.strip()
            }
            missing_channels = [
                channel
                for channel in required_channels
                if channel not in observed_channels
            ]
            if missing_channels:
                reason = "missing_required_safety_channels:" + ",".join(
                    missing_channels
                )
                missing.append(reason)
                sources["safety_validity"] = reason
                return None
            sources["safety_validity"] = (
                f"safety_valid:{raw_scope or 'unspecified'}"
            )

        explicit = _coerce_float(row.get("unsafe_rate"))
        if explicit is not None:
            sources["unsafe_rate"] = "unsafe_rate"
            return explicit

        # Aggregated violation and CBF rates are each authoritative safety
        # loads.  In particular, SafetyMonitor's CBF penalty already contains
        # collision penalties, so adding collision_rate would double count.
        for key in ("constraint_violation_rate", "safety_constraint_violation_rate"):
            value = _coerce_float(row.get(key))
            if value is not None:
                sources["unsafe_rate"] = key
                return value

        penalty_rate = _coerce_float(row.get("normalized_cbf_penalty_rate"))
        if penalty_rate is not None:
            sources["unsafe_rate"] = "normalized_cbf_penalty_rate"
            return penalty_rate

        penalty = _coerce_float(row.get("stability_p_cbf"))
        total_steps = _coerce_float(
            row.get("total_step_count", row.get("sim_step_count"))
        )
        if penalty is not None and total_steps is not None and total_steps > 0:
            sources["unsafe_rate"] = "stability_p_cbf/total_step_count"
            return penalty / total_steps

        collision_key = (
            "safety_collision_rate"
            if row.get("safety_collision_rate") not in (None, "")
            else "collision_rate"
        )
        collision = _coerce_float(row.get(collision_key))
        if collision is not None:
            sources["unsafe_rate"] = collision_key
            return collision
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
    def margin_to_dict(margin: BoundaryMargin) -> Dict[str, Any]:
        row = margin.to_dict()
        row.update(
            {
                "judge_model": str(
                    getattr(margin, "ge_judge_model", "unconfigured-judge")
                ),
                "stress_family": str(
                    getattr(margin, "ge_stress_family", "legacy")
                ),
                "stress_contract_formal": int(
                    bool(getattr(margin, "ge_formal_stress_contract", False))
                ),
                "ge_total_episode_seed_pairs": int(
                    getattr(margin, "ge_total_episode_seed_pairs", 0)
                ),
                "ge_complete_episode_seed_pairs": int(
                    getattr(margin, "ge_complete_episode_seed_pairs", 0)
                ),
                "ge_monotonicity_violation": (
                    int(bool(getattr(margin, "ge_monotonicity_violation", False)))
                    if margin.capacity_valid
                    else ""
                ),
            }
        )
        return row

    @staticmethod
    def margin_to_diagnostic_dict(margin: BoundaryMargin) -> Dict[str, Any]:
        row = margin.to_diagnostic_dict()
        row.update(BoundaryCollector.margin_to_dict(margin))
        return row

    @staticmethod
    def write_csv(
        margins: Mapping[CellKey, BoundaryMargin],
        output_path: Union[str, Path],
    ) -> Path:
        rows = [
            BoundaryCollector.margin_to_dict(margin)
            for margin in margins.values()
        ]
        return BoundaryCollector.write_rows(rows, output_path)

    @staticmethod
    def write_diagnostics_csv(
        margins: Mapping[CellKey, BoundaryMargin],
        output_path: Union[str, Path],
    ) -> Path:
        rows = [
            BoundaryCollector.margin_to_diagnostic_dict(margin)
            for margin in margins.values()
        ]
        return BoundaryCollector.write_rows(rows, output_path)

    @staticmethod
    def write_rows(
        rows: Sequence[Mapping[str, Any]],
        output_path: Union[str, Path],
    ) -> Path:
        fieldnames = (
            sorted({str(key) for row in rows for key in row.keys()})
            if rows
            else None
        )
        return write_csv_rows_exact(
            output_path,
            rows,
            fieldnames=fieldnames,
            overwrite_policy="replace",
            # A generic empty Boundary table has no authoritative schema. Fail
            # closed instead of truncating/deleting evidence or retaining stale
            # rows as if they belonged to the current run.
            empty_policy="error",
        )

    @staticmethod
    def read_rollout_csv(path: Union[str, Path]) -> List[Dict[str, Any]]:
        return list(read_csv_rows(path))
