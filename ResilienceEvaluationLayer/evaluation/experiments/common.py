"""
Shared utilities for StateEncoder and ValueNetwork experiments.
"""

from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np

TASK_FAMILY_PRIORITY = (
    "mixed_temporal",
    "object_state",
    "spatial_relation",
    "room_transfer",
    "placement_only",
    "other",
)

_OBJECT_STATE_FUNCTIONS = {
    "is_clean",
    "is_dirty",
    "is_filled",
    "is_empty",
    "is_powered_on",
    "is_powered_off",
}
_SPATIAL_RELATION_FUNCTIONS = {"is_next_to"}
_ROOM_TRANSFER_FUNCTIONS = {"is_in_room"}
_PLACEMENT_FUNCTIONS = {"is_on_top", "is_inside", "is_on_floor"}

STATE_ENCODER_CATEGORY_NAMES = (
    "instruction",
    "task-object grounding",
    "target/receptacle grounding",
    "room grounding",
    "agent kinematics",
    "object spatial state",
    "visibility/count",
    "holding/manipulation",
    "progress/outcome",
    "object-state affordance",
    "padding/noise",
)

NUMERICAL_FEATURE_CATEGORY_HINTS = {
    "agent_pos_x": "agent kinematics",
    "agent_pos_y": "agent kinematics",
    "agent_pos_z": "agent kinematics",
    "agent_yaw": "agent kinematics",
    "agent2_pos_x": "agent kinematics",
    "agent2_pos_y": "agent kinematics",
    "agent2_pos_z": "agent kinematics",
    "agent2_yaw": "agent kinematics",
    "task_progress": "progress/outcome",
    "proposition_satisfied_fraction": "progress/outcome",
    "min_task_obj_dist_norm": "object spatial state",
    "mean_task_obj_dist_norm": "object spatial state",
    "min_agent_target_dist_norm": "object spatial state",
    "min_obj_target_dist_norm": "object spatial state",
    "mean_obj_target_dist_norm": "object spatial state",
    "task_object_fraction_at_target": "progress/outcome",
    "task_object_count_norm": "task-object grounding",
    "task_target_count_norm": "target/receptacle grounding",
    "task_entity_count_norm": "task-object grounding",
    "task_room_count_norm": "room grounding",
    "task_proposition_count_norm": "progress/outcome",
    "task_object_visible_fraction": "visibility/count",
    "task_target_visible_fraction": "visibility/count",
    "task_object_visible_count_norm": "visibility/count",
    "task_target_visible_count_norm": "visibility/count",
    "any_agent_holding": "holding/manipulation",
    "any_agent_holding_task_object": "holding/manipulation",
    "primary_agent_holding": "holding/manipulation",
    "primary_agent_holding_task_object": "holding/manipulation",
    "held_task_object_fraction": "holding/manipulation",
    "min_agent_target_dist_when_holding_norm": "holding/manipulation",
    "action_history_count_norm": "progress/outcome",
}


def _episode_get(episode: Any, key: str, default: Any = None) -> Any:
    if isinstance(episode, Mapping):
        return episode.get(key, default)
    return getattr(episode, key, default)


def _constraint_to_mapping(constraint: Any) -> Dict[str, Any]:
    if isinstance(constraint, Mapping):
        return dict(constraint)

    result: Dict[str, Any] = {}
    constraint_type = getattr(constraint, "type", None)
    if constraint_type is not None:
        result["type"] = constraint_type
    elif constraint is not None:
        result["type"] = constraint.__class__.__name__

    args = getattr(constraint, "args", None)
    if args is not None:
        result["args"] = dict(args)
    return result


def _proposition_function_name(prop: Any) -> str:
    if isinstance(prop, Mapping):
        return str(prop.get("function_name", ""))
    return str(getattr(prop, "function_name", ""))


def infer_task_family(episode: Any) -> str:
    """
    Infer a canonical task family for an episode.

    Priority:
      mixed_temporal > object_state > spatial_relation > room_transfer > placement_only
    """
    constraints = _episode_get(episode, "evaluation_constraints", []) or []
    propositions = _episode_get(episode, "evaluation_propositions", []) or []

    for constraint in constraints:
        constraint_dict = _constraint_to_mapping(constraint)
        if constraint_dict.get("type") != "TemporalConstraint":
            continue
        dag_edges = constraint_dict.get("args", {}).get("dag_edges", []) or []
        if len(dag_edges) > 0:
            return "mixed_temporal"

    proposition_names = {_proposition_function_name(prop) for prop in propositions}
    if proposition_names & _OBJECT_STATE_FUNCTIONS:
        return "object_state"
    if proposition_names & _SPATIAL_RELATION_FUNCTIONS:
        return "spatial_relation"
    if proposition_names & _ROOM_TRANSFER_FUNCTIONS:
        return "room_transfer"
    if proposition_names and proposition_names.issubset(_PLACEMENT_FUNCTIONS):
        return "placement_only"
    return "other"


def load_dataset(dataset_path: str | Path) -> Dict[str, Any]:
    dataset_path = Path(dataset_path)
    opener = gzip.open if dataset_path.suffix == ".gz" else open
    with opener(dataset_path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, MutableMapping):
        raise ValueError(f"Dataset at `{dataset_path}` is not a mapping.")
    return dict(data)


def load_dataset_episodes(dataset_path: str | Path) -> List[Dict[str, Any]]:
    dataset = load_dataset(dataset_path)
    episodes = dataset.get("episodes", [])
    if not isinstance(episodes, list):
        raise ValueError(f"Dataset at `{dataset_path}` has invalid `episodes` content.")
    return episodes


def load_json(path: str | Path) -> Any:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, Mapping):
                records.append(dict(parsed))
    return records


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            json.dump(dict(record), handle, ensure_ascii=False)
            handle.write("\n")
    return path


def iter_files(root: str | Path, pattern: str) -> List[Path]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(root.rglob(pattern))


def stratified_episode_sample(
    episodes: Sequence[Dict[str, Any]],
    sample_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Sample episodes while preserving task-family proportions as closely as possible.
    """
    if sample_size <= 0:
        return []
    if sample_size >= len(episodes):
        return list(episodes)

    rng = random.Random(seed)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[infer_task_family(episode)].append(episode)

    for bucket in grouped.values():
        rng.shuffle(bucket)

    family_names = [name for name in TASK_FAMILY_PRIORITY if name in grouped]
    counts = {name: len(grouped[name]) for name in family_names}
    total = sum(counts.values())
    if total == 0:
        return []

    raw_targets = {
        name: sample_size * counts[name] / total
        for name in family_names
    }
    allocations = {
        name: min(counts[name], int(raw_targets[name]))
        for name in family_names
    }

    remaining = sample_size - sum(allocations.values())
    if remaining > 0:
        remainders = sorted(
            family_names,
            key=lambda name: (raw_targets[name] - allocations[name], counts[name]),
            reverse=True,
        )
        for name in remainders:
            if remaining <= 0:
                break
            if allocations[name] >= counts[name]:
                continue
            allocations[name] += 1
            remaining -= 1

    sampled: List[Dict[str, Any]] = []
    for name in family_names:
        sampled.extend(grouped[name][: allocations[name]])

    rng.shuffle(sampled)
    return sampled


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def cosine_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    return 1.0 - cosine_similarity(vec_a, vec_b)


def classify_feature_category(feature_name: str) -> str:
    if feature_name.startswith("pad_"):
        return "padding/noise"
    return NUMERICAL_FEATURE_CATEGORY_HINTS.get(feature_name, "padding/noise")


def category_template(default_value: Any = 0) -> Dict[str, Any]:
    return {name: default_value for name in STATE_ENCODER_CATEGORY_NAMES}


def pearson_corr(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) == 0:
        return 0.0
    a = np.asarray(values_a, dtype=np.float32)
    b = np.asarray(values_b, dtype=np.float32)
    a_std = float(a.std())
    b_std = float(b.std())
    if a_std <= 0.0 or b_std <= 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _rank_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.zeros(len(arr), dtype=np.float32)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return ranks


def spearman_corr(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    if len(values_a) != len(values_b) or len(values_a) == 0:
        return 0.0
    ranks_a = _rank_array(values_a)
    ranks_b = _rank_array(values_b)
    return pearson_corr(ranks_a, ranks_b)


def binary_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    if len(labels) != len(scores) or len(labels) == 0:
        return 0.0
    positives = [(score, label) for score, label in zip(scores, labels) if int(label) == 1]
    negatives = [(score, label) for score, label in zip(scores, labels) if int(label) == 0]
    if not positives or not negatives:
        return 0.0
    wins = 0.0
    total = 0.0
    for pos_score, _ in positives:
        for neg_score, _ in negatives:
            total += 1.0
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
    if total <= 0.0:
        return 0.0
    return float(wins / total)


def mean_and_std(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "count": int(arr.size),
    }

