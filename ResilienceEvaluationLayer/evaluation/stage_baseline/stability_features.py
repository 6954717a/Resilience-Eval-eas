"""Neutral Stability trajectory features shared by baseline and metrics code."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


STABILITY_FEATURE_NAMES: Tuple[str, ...] = (
    "delta_value",
    "td_error",
    "gae_advantage",
    "delta_progress",
    "delta_proposition",
    "planning_event",
)


def _number(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def record_info(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge a compact critic snapshot with required top-level keys."""

    snapshot = record.get("info_snapshot")
    info = dict(snapshot) if isinstance(snapshot, Mapping) else {}
    for key in (
        "episode_id",
        "task_family",
        "task_percent_complete",
        "proposition_satisfied_fraction",
        "task_state_success",
        "sim_step_count",
        "planning_step_count",
        "analysis_tags",
    ):
        if record.get(key) not in (None, ""):
            info[key] = record[key]
    return info


def extract_stability_vector(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    planning_event: Optional[bool] = None,
) -> Optional[np.ndarray]:
    """Extract Appendix A.2's six-dimensional vector from critic records."""

    previous_value = _number(previous.get("predicted_value"))
    current_value = _number(current.get("predicted_value"))
    td_error = _number(current.get("td_error"))
    advantage = _number(current.get("gae_advantage"))
    if None in (previous_value, current_value, td_error, advantage):
        return None

    previous_info = record_info(previous)
    current_info = record_info(current)
    previous_progress = _number(previous_info.get("task_percent_complete"))
    current_progress = _number(current_info.get("task_percent_complete"))
    previous_proposition = _number(
        previous_info.get("proposition_satisfied_fraction")
    )
    current_proposition = _number(
        current_info.get("proposition_satisfied_fraction")
    )
    if None in (
        previous_progress,
        current_progress,
        previous_proposition,
        current_proposition,
    ):
        return None

    if planning_event is None:
        tags = current.get("analysis_tags") or []
        if isinstance(tags, str):
            tags = [tags]
        planning_event = bool(
            current.get("planning_transition")
            or "replan_anchor" in {str(tag) for tag in tags}
        )

    return np.asarray(
        [
            current_value - previous_value,
            td_error,
            advantage,
            current_progress - previous_progress,
            current_proposition - previous_proposition,
            float(bool(planning_event)),
        ],
        dtype=np.float64,
    )


__all__ = [
    "STABILITY_FEATURE_NAMES",
    "extract_stability_vector",
    "record_info",
]
