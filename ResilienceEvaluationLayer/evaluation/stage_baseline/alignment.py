"""Shared Judge/episode alignment rules for StageBaseline artifacts.

Value-network and state-encoder identifiers are provenance only. Runtime
alignment is defined by the configured Judge model and episode represented by
each clean trajectory.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple


STAGE_BASELINE_ALIGNMENT_SCOPE = "judge_model_episode"


def normalize_judge_model(value: Any) -> str:
    """Return the comparison form of a Judge model name."""

    return str(value or "").strip().casefold()


def normalize_episode_id(value: Any) -> str:
    return str(value or "").strip()


def stage_baseline_alignment_key(judge_model: Any, episode_id: Any) -> str:
    """Build the explicit statistical alignment key for one episode."""

    judge = normalize_judge_model(judge_model)
    episode = normalize_episode_id(episode_id)
    return f"{judge}::{episode}" if judge and episode else ""


def judge_models_match(left: Any, right: Any) -> bool:
    left_normalized = normalize_judge_model(left)
    right_normalized = normalize_judge_model(right)
    return bool(left_normalized and left_normalized == right_normalized)


def baseline_supported_episode_ids(metadata: Mapping[str, Any]) -> Tuple[str, ...]:
    """Read the successful episode support recorded by a baseline snapshot.

    New cumulative snapshots expose ``successful_episode_ids``.  The fallbacks
    retain compatibility with older episode-scoped artifacts without treating
    an arbitrary requested episode list as stronger evidence than observed
    calibration rows.
    """

    for field_name in ("available_episode_ids", "successful_episode_ids"):
        values_raw = metadata.get(field_name)
        if isinstance(values_raw, Sequence) and not isinstance(
            values_raw, (str, bytes)
        ):
            values = {normalize_episode_id(value) for value in values_raw}
            if values:
                return tuple(sorted(value for value in values if value))

    calibration = metadata.get("calibration")
    if isinstance(calibration, Mapping):
        observed = calibration.get("observed_seed_map")
        if isinstance(observed, Mapping):
            values = {normalize_episode_id(value) for value in observed}
            if values:
                return tuple(sorted(value for value in values if value))

    legacy_formal = metadata.get("formal_episode_ids")
    if isinstance(legacy_formal, Sequence) and not isinstance(
        legacy_formal, (str, bytes)
    ):
        values = {normalize_episode_id(value) for value in legacy_formal}
        if values:
            return tuple(sorted(value for value in values if value))

    episode_id = normalize_episode_id(metadata.get("episode_id"))
    if episode_id:
        return (episode_id,)

    return ()


def baseline_episode_matches(metadata: Mapping[str, Any], episode_id: Any) -> bool:
    episode = normalize_episode_id(episode_id)
    return bool(episode and episode in set(baseline_supported_episode_ids(metadata)))


__all__ = [
    "STAGE_BASELINE_ALIGNMENT_SCOPE",
    "baseline_episode_matches",
    "baseline_supported_episode_ids",
    "judge_models_match",
    "normalize_episode_id",
    "normalize_judge_model",
    "stage_baseline_alignment_key",
]
