"""Shared stage-anchor baseline models and estimation utilities.

This singular package contains executable code. Persisted Judge-scoped data
continues to live under ``evaluation/stage_baselines/`` (plural), so moving the
implementation does not alter existing artifact identities or lookup paths.

The package is intentionally imported lazily: stability trajectory features
also depend on :class:`StageResolver`, so eagerly importing the estimator here
would create an order-dependent package import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .stage_anchor import StageAnchor, StageResolver
from .alignment import (
    STAGE_BASELINE_ALIGNMENT_SCOPE,
    baseline_episode_matches,
    baseline_supported_episode_ids,
    judge_models_match,
    normalize_judge_model,
    stage_baseline_alignment_key,
)

_BASELINE_EXPORTS = frozenset(
    {
        "ALL_WORK_MODALITIES",
        "BASELINE_DIMENSIONS",
        "RECOVERY_POOL_MODALITY",
        "STABILITY_POOL_MODALITY",
        "StageBaseline",
        "StageBaselineDim",
        "StageBaselineEstimator",
        "main",
    }
)

_EVIDENCE_EXPORTS = frozenset(
    {
        "STAGE_BASELINE_EVIDENCE_SCHEMA_VERSION",
        "StageBaselineEvidence",
        "StageBaselineEvidenceStore",
    }
)

if TYPE_CHECKING:
    from .stage_baseline import (
        ALL_WORK_MODALITIES,
        BASELINE_DIMENSIONS,
        RECOVERY_POOL_MODALITY,
        STABILITY_POOL_MODALITY,
        StageBaseline,
        StageBaselineDim,
        StageBaselineEstimator,
        main,
    )


def __getattr__(name: str) -> Any:
    if name in _EVIDENCE_EXPORTS:
        from . import evidence_store as implementation

        value = getattr(implementation, name)
        globals()[name] = value
        return value
    if name not in _BASELINE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import stage_baseline as implementation

    value = getattr(implementation, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _BASELINE_EXPORTS | _EVIDENCE_EXPORTS)


__all__ = [
    "StageAnchor",
    "StageResolver",
    "StageBaseline",
    "StageBaselineDim",
    "StageBaselineEstimator",
    "BASELINE_DIMENSIONS",
    "ALL_WORK_MODALITIES",
    "RECOVERY_POOL_MODALITY",
    "STABILITY_POOL_MODALITY",
    "STAGE_BASELINE_EVIDENCE_SCHEMA_VERSION",
    "StageBaselineEvidence",
    "StageBaselineEvidenceStore",
    "STAGE_BASELINE_ALIGNMENT_SCOPE",
    "baseline_episode_matches",
    "baseline_supported_episode_ids",
    "judge_models_match",
    "normalize_judge_model",
    "stage_baseline_alignment_key",
]
