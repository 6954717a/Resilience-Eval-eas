"""Strict contract for critic exports used as complete trajectories.

Focused critic exports are useful diagnostics, but they are not trajectories:
joining two selected rows can create a transition that never occurred in the
rollout.  Formal Stability and StageBaseline consumers therefore use this
module to accept only exports that prove that every transition was retained.
"""

from __future__ import annotations

import json
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class FullTrajectoryExport:
    """Validated state records from one full critic trajectory export."""

    records: List[Dict[str, Any]]
    valid: bool
    missing_reason: str = ""
    total_transitions: int = 0
    selected_transitions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


def _nonnegative_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _invalid(
    reason: str,
    *,
    total: int = 0,
    selected: int = 0,
) -> FullTrajectoryExport:
    return FullTrajectoryExport(
        records=[],
        valid=False,
        missing_reason=reason,
        total_transitions=max(0, int(total)),
        selected_transitions=max(0, int(selected)),
        metadata={},
    )


def load_full_stability_export(path: Path) -> FullTrajectoryExport:
    """Load and validate a complete, state-only critic export.

    JSON lists and JSONL files intentionally fail closed because they cannot
    carry the export-level manifest and selection accounting needed to prove
    completeness.
    """

    try:
        resolved_path = Path(path)
        if resolved_path.suffix == ".gz":
            with gzip.open(resolved_path, "rt", encoding="utf-8") as handle:
                text = handle.read().strip()
        else:
            text = resolved_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return _invalid("critic_export_unreadable")
    if not text:
        return _invalid("critic_export_empty")
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _invalid("critic_export_invalid_json")
    if not isinstance(payload, Mapping):
        return _invalid("critic_export_payload_not_mapping")

    manifest = payload.get("export_manifest")
    if not isinstance(manifest, Mapping):
        return _invalid("critic_export_manifest_missing")
    if manifest.get("full_stability_trajectory") is not True:
        return _invalid("critic_export_not_full_stability_trajectory")
    if manifest.get("minimal_schema") is not True:
        return _invalid("critic_export_not_minimal_schema")
    if str(manifest.get("record_mode") or "") != "state_only":
        return _invalid("critic_export_record_mode_not_state_only")

    selection = payload.get("selection_summary")
    if not isinstance(selection, Mapping):
        return _invalid("critic_export_selection_summary_missing")
    total = _nonnegative_int(selection.get("total_transitions"))
    selected = _nonnegative_int(selection.get("selected_transitions"))
    if total is None or selected is None:
        return _invalid("critic_export_selection_count_invalid")
    if selected != total:
        return _invalid(
            "critic_export_selection_incomplete",
            total=total,
            selected=selected,
        )

    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        return _invalid(
            "critic_export_records_missing",
            total=total,
            selected=selected,
        )
    if any(not isinstance(record, Mapping) for record in raw_records):
        return _invalid(
            "critic_export_record_invalid",
            total=total,
            selected=selected,
        )

    state_records = [
        dict(record)
        for record in raw_records
        if record.get("state_kind") == "state"
    ]
    if len(state_records) != selected or len(raw_records) != selected:
        return _invalid(
            "critic_export_state_record_count_mismatch",
            total=total,
            selected=selected,
        )

    indices: List[int] = []
    for record in state_records:
        index = _nonnegative_int(record.get("transition_index"))
        if index is None:
            return _invalid(
                "critic_export_transition_index_invalid",
                total=total,
                selected=selected,
            )
        indices.append(index)
    if sorted(indices) != list(range(total)):
        return _invalid(
            "critic_export_transition_index_non_contiguous",
            total=total,
            selected=selected,
        )

    state_records.sort(key=lambda record: int(record["transition_index"]))
    inherited = {
        key: payload.get(key)
        for key in ("episode_id", "episode_filename", "task_family")
        if payload.get(key) is not None
    }
    records = [{**inherited, **record} for record in state_records]
    return FullTrajectoryExport(
        records=records,
        valid=True,
        total_transitions=total,
        selected_transitions=selected,
        metadata={
            key: payload.get(key)
            for key in (
                "critic_phase",
                "judge_model",
                "state_encoder_id",
                "value_checkpoint_id",
                "reward_shaper_valid",
                "critic_lifecycle_valid",
            )
        },
    )


__all__ = ["FullTrajectoryExport", "load_full_stability_export"]
