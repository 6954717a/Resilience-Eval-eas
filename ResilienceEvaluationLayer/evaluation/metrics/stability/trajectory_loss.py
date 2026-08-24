#!/usr/bin/env python3

r"""Stage-normalized trajectory loss used by paper Eq. (5).

The feature order follows Appendix A.2 exactly::

    x_t = [Delta V_t, delta_t, A_t, Delta p_t, Delta q_t, e_plan,t]

The clean StageBaseline provides ``mu(a_t)`` and ``Sigma(a_t)``.  A rollout
loss is the mean regularized Mahalanobis distance over usable transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from habitat_llm.evaluation.critic_export_contract import (
    load_full_stability_export,
)
from habitat_llm.evaluation.stage_baseline import (
    STABILITY_POOL_MODALITY as STABILITY_STAGE_POOL_MODALITY,
    StageResolver,
)
from habitat_llm.evaluation.stage_baseline.stability_features import (
    STABILITY_FEATURE_NAMES,
    extract_stability_vector,
    record_info,
)


def load_critic_records(path: Path) -> List[Dict[str, Any]]:
    """Return records only when the export proves full-trajectory integrity."""

    export = load_full_stability_export(Path(path))
    return list(export.records) if export.valid else []


@dataclass(frozen=True)
class TrajectoryLossResult:
    loss: Optional[float]
    valid: bool
    used_transitions: int
    candidate_transitions: int
    missing_reason: str = ""
    minimum_coverage: float = 0.0

    @property
    def coverage(self) -> float:
        if self.candidate_transitions <= 0:
            return 0.0
        return float(self.used_transitions) / float(self.candidate_transitions)

    def to_row(self) -> Dict[str, Any]:
        return {
            "stability_trajectory_loss": self.loss,
            "stability_trajectory_loss_valid": int(self.valid),
            "stability_trajectory_loss_steps": int(self.used_transitions),
            "stability_trajectory_loss_candidate_steps": int(
                self.candidate_transitions
            ),
            "stability_trajectory_loss_coverage": self.coverage,
            "stability_trajectory_loss_minimum_coverage": float(
                self.minimum_coverage
            ),
            "stability_trajectory_loss_missing_reason": self.missing_reason,
        }


def _baseline_for_anchor(
    baselines: Mapping[Tuple[str, int, str], Any],
    family: str,
    stage: int,
    modality: str,
) -> Optional[Any]:
    """Resolve an exact or family-stage Stability reference."""

    exact = baselines.get((family, stage, modality))
    if _has_stability_statistics(exact):
        return exact

    pooled = baselines.get((family, stage, STABILITY_STAGE_POOL_MODALITY))
    if _has_stability_statistics(pooled):
        return pooled

    # Preserve the diagnostic reason when neither formal source exists.
    return exact if exact is not None else pooled


def _has_stability_statistics(baseline: Optional[Any]) -> bool:
    if baseline is None:
        return False
    mean = np.asarray(getattr(baseline, "stability_mean", []), dtype=np.float64)
    covariance = np.asarray(
        getattr(baseline, "stability_covariance", []), dtype=np.float64
    )
    sample_count = int(getattr(baseline, "stability_n_samples", 0) or 0)
    return bool(
        sample_count >= 1
        and mean.ndim == 1
        and mean.size > 0
        and covariance.shape == (mean.size, mean.size)
        and np.all(np.isfinite(mean))
        and np.all(np.isfinite(covariance))
    )


def _resolve_anchor_with_terminal_backtrack(
    resolver: StageResolver,
    current: Mapping[str, Any],
    last_nonterminal: Optional[Any],
) -> Tuple[Optional[Any], Optional[Any]]:
    """Resolve an anchor without treating terminal/unknown as a new stage.

    Terminal critic records frequently contain the final reward/TD signal but
    naturally resolve to ``stage=-1``.  Attribute those records to the most
    recent valid non-terminal anchor.  A leading unknown record with no such
    history is ineligible rather than being charged against trajectory
    coverage.
    """

    info = record_info(current)
    anchor = resolver.resolve(
        info,
        family_override=str(
            current.get("task_family") or info.get("task_family") or "other"
        ),
    )
    if int(anchor.stage) >= 0:
        return anchor, anchor
    if last_nonterminal is None or str(last_nonterminal.family) != str(
        anchor.family
    ):
        return None, last_nonterminal
    return last_nonterminal, last_nonterminal


def compute_stage_trajectory_loss(
    export_path: Path,
    baselines: Mapping[Tuple[str, int, str], Any],
    *,
    regularizer: float = 1.0e-3,
    minimum_coverage: float = 0.0,
    resolver: Optional[StageResolver] = None,
) -> TrajectoryLossResult:
    """Compute the stage-centered episode loss from one critic export."""

    try:
        required_coverage = float(minimum_coverage)
    except (TypeError, ValueError):
        required_coverage = 0.0
    if not math.isfinite(required_coverage):
        required_coverage = 0.0
    required_coverage = min(max(required_coverage, 0.0), 1.0)

    export = load_full_stability_export(Path(export_path))
    potential_candidates = max(0, int(export.total_transitions) - 1)
    if not export.valid:
        return TrajectoryLossResult(
            None,
            False,
            0,
            potential_candidates,
            export.missing_reason,
            required_coverage,
        )
    records = export.records
    if potential_candidates == 0:
        return TrajectoryLossResult(
            None,
            False,
            0,
            potential_candidates,
            "insufficient_critic_records",
            required_coverage,
        )
    if not baselines:
        return TrajectoryLossResult(
            None,
            False,
            0,
            potential_candidates,
            "stage_baseline_missing",
            required_coverage,
        )

    resolver = resolver or StageResolver()
    distances: List[float] = []
    candidates = 0
    first_info = record_info(records[0])
    first_anchor = resolver.resolve(
        first_info,
        family_override=str(
            records[0].get("task_family")
            or first_info.get("task_family")
            or "other"
        ),
    )
    last_nonterminal = first_anchor if int(first_anchor.stage) >= 0 else None
    for previous, current in zip(records[:-1], records[1:]):
        anchor, last_nonterminal = _resolve_anchor_with_terminal_backtrack(
            resolver,
            current,
            last_nonterminal,
        )
        if anchor is None:
            continue
        candidates += 1
        vector = extract_stability_vector(previous, current)
        if vector is None:
            continue
        baseline = _baseline_for_anchor(
            baselines, anchor.family, int(anchor.stage), anchor.modality
        )
        if baseline is None:
            continue
        if not _has_stability_statistics(baseline):
            continue
        mean = np.asarray(
            getattr(baseline, "stability_mean", []), dtype=np.float64
        )
        covariance = np.asarray(
            getattr(baseline, "stability_covariance", []), dtype=np.float64
        )
        sample_count = int(getattr(baseline, "stability_n_samples", 0) or 0)
        if (
            sample_count < 1
            or mean.shape != vector.shape
            or covariance.shape != (vector.size, vector.size)
            or not np.all(np.isfinite(mean))
            or not np.all(np.isfinite(covariance))
        ):
            continue
        centered = vector - mean
        stabilized = covariance + max(float(regularizer), 1.0e-12) * np.eye(
            vector.size, dtype=np.float64
        )
        squared = float(centered.T @ np.linalg.pinv(stabilized) @ centered)
        distances.append(float(np.sqrt(max(squared, 0.0))))

    used = len(distances)
    if candidates == 0:
        return TrajectoryLossResult(
            None,
            False,
            0,
            0,
            "no_eligible_nonterminal_stability_transition",
            required_coverage,
        )
    if used == 0:
        return TrajectoryLossResult(
            None,
            False,
            0,
            candidates,
            "no_transition_with_matching_stability_baseline",
            required_coverage,
        )
    return TrajectoryLossResult(
        float(np.mean(distances)),
        True,
        used,
        candidates,
        "",
        required_coverage,
    )


__all__ = [
    "STABILITY_FEATURE_NAMES",
    "STABILITY_STAGE_POOL_MODALITY",
    "TrajectoryLossResult",
    "compute_stage_trajectory_loss",
    "extract_stability_vector",
    "load_critic_records",
    "record_info",
]
