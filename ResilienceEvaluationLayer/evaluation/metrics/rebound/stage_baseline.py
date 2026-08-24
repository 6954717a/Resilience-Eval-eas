"""Compatibility imports for the evaluation-level StageBaseline API."""

from habitat_llm.evaluation.stage_baseline.stage_baseline import (
    ALL_WORK_MODALITIES,
    BASELINE_DIMENSIONS,
    RECOVERY_POOL_MODALITY,
    STABILITY_POOL_MODALITY,
    StageBaseline,
    StageBaselineDim,
    StageBaselineEstimator,
    main,
)

__all__ = [
    "StageBaseline",
    "StageBaselineDim",
    "StageBaselineEstimator",
    "BASELINE_DIMENSIONS",
    "ALL_WORK_MODALITIES",
    "RECOVERY_POOL_MODALITY",
    "STABILITY_POOL_MODALITY",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
