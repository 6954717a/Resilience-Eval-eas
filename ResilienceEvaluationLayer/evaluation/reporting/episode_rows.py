"""Pure helpers for selecting and materializing one episode CSV row."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Set

from habitat_llm.evaluation.proposition_outcome import (
    materialize_proposition_outcome,
)

from .resilience_schema import RUNTIME_METRIC_KEYS


DEFAULT_EPISODE_DYNAMIC_PREFIXES = ("perturbation_",)
INSTRUCTION_LINEAGE_KEYS = (
    "canonical_instruction",
    "policy_instruction",
    "reward_instruction",
    "task_instruction",
)


def collect_episode_stats_keys(
    info: Dict[str, Any],
    *,
    canonical_metric_keys: Sequence[str] = RUNTIME_METRIC_KEYS,
    compatibility_alias_keys: Sequence[str] = (),
    dynamic_prefixes: Sequence[str] = DEFAULT_EPISODE_DYNAMIC_PREFIXES,
) -> Set[str]:
    """Return scalar-extraction keys and materialize per-agent replan counts."""

    materialize_proposition_outcome(info)
    stats_keys = {
        "task_percent_complete",
        "task_state_success",
        "proposition_satisfied_fraction",
        "sim_step_count",
        "replanning_count",
        "runtime",
        *canonical_metric_keys,
        *compatibility_alias_keys,
    }
    replanning_count = info.get("replanning_count")
    if isinstance(replanning_count, dict):
        for agent_id, count in replanning_count.items():
            key = f"replanning_count_{agent_id}"
            stats_keys.add(key)
            info[key] = count

    for key in info:
        if key != "perturbation_audit" and key.startswith(tuple(dynamic_prefixes)):
            stats_keys.add(key)
    return stats_keys


def build_episode_csv_metrics(
    stats_episode: Dict[str, Any],
    info: Dict[str, Any],
    *,
    canonical_metric_keys: Sequence[str] = RUNTIME_METRIC_KEYS,
    compatibility_alias_keys: Sequence[str] = (),
    dynamic_prefixes: Sequence[str] = DEFAULT_EPISODE_DYNAMIC_PREFIXES,
) -> Dict[str, Any]:
    """Merge numeric stats with validity and string-valued audit evidence."""

    materialize_proposition_outcome(info)
    csv_metrics = dict(stats_episode)
    for key in (*canonical_metric_keys, *compatibility_alias_keys):
        value = info.get(key)
        csv_metrics[key] = "" if value is None else value

    for key in (
        "proposition_satisfied_fraction",
        "proposition_satisfied_fraction_source",
        "task_family",
        "action_sequence",
        *INSTRUCTION_LINEAGE_KEYS,
    ):
        value = info.get(key)
        if value is not None:
            csv_metrics[key] = value

    for key, value in info.items():
        if key != "perturbation_audit" and key.startswith(tuple(dynamic_prefixes)):
            csv_metrics[key] = value
    return csv_metrics


__all__ = [
    "DEFAULT_EPISODE_DYNAMIC_PREFIXES",
    "build_episode_csv_metrics",
    "collect_episode_stats_keys",
]
