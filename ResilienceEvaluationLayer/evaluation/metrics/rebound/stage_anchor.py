"""Compatibility imports for the evaluation-level stage-anchor API."""

from habitat_llm.evaluation.stage_baseline.stage_anchor import (
    StageAnchor,
    StageResolver,
    _classify_modality,
    _extract_action_string,
)

__all__ = ["StageAnchor", "StageResolver"]
