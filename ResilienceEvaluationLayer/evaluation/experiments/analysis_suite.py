"""
Offline analysis suite for StateEncoder and ValueNetwork.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from habitat_llm.evaluation.experiments.common import (
    STATE_ENCODER_CATEGORY_NAMES,
    TASK_FAMILY_PRIORITY,
    binary_auroc,
    cosine_distance,
    ensure_dir,
    iter_files,
    load_dataset_episodes,
    load_json,
    load_jsonl,
    mean_and_std,
    spearman_corr,
    write_json,
)
from habitat_llm.evaluation.experiments.llm_review import (
    build_state_review_packet,
    build_value_review_packet,
    maybe_run_openai_reviews,
    save_review_packets,
)
from habitat_llm.evaluation.experiments.probe_utils import compare_feature_sets
from habitat_llm.evaluation.networks import StateEncoder, ValueNetwork
from habitat_llm.evaluation.utils import compute_gae


def _episode_key(episode: Mapping[str, Any]) -> str:
    return str(episode.get("episode_id"))


def _record_episode_key(record: Mapping[str, Any]) -> str:
    return str(record.get("episode_id"))


def _scene_id(episode: Mapping[str, Any]) -> str:
    return str(episode.get("scene_id", "unknown"))


def _family_key(episode: Mapping[str, Any]) -> str:
    constraints = episode.get("evaluation_constraints", []) or []
    propositions = episode.get("evaluation_propositions", []) or []
    if any((c.get("type") == "TemporalConstraint" and c.get("args", {}).get("dag_edges")) for c in constraints if isinstance(c, Mapping)):
        return "mixed_temporal"
    names = {str(p.get("function_name", "")) for p in propositions if isinstance(p, Mapping)}
    if names & {"is_clean", "is_dirty", "is_filled", "is_empty", "is_powered_on", "is_powered_off"}:
        return "object_state"
    if names & {"is_next_to"}:
        return "spatial_relation"
    if names & {"is_in_room"}:
        return "room_transfer"
    if names and names.issubset({"is_on_top", "is_inside", "is_on_floor"}):
        return "placement_only"
    return "other"


def load_export_payloads(export_roots: Sequence[str | Path]) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for root in export_roots:
        root_path = Path(root)
        if root_path.is_file():
            payloads.append(load_json(root_path))
            continue
        for path in iter_files(root_path, "critic_export-episode_*.json"):
            payloads.append(load_json(path))
    return payloads


def load_state_debug_records(export_roots: Sequence[str | Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for root in export_roots:
        root_path = Path(root)
        if root_path.is_file() and root_path.suffix == ".jsonl":
            records.extend(load_jsonl(root_path))
            continue
        for path in iter_files(root_path, "state_encoder-episode_*.jsonl"):
            records.extend(load_jsonl(path))
    return records


def flatten_state_records(payloads: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for payload in payloads:
        for record in payload.get("records", []) or []:
            if str(record.get("state_kind")) == "state":
                records.append(dict(record))
    return records


def group_state_records_by_episode(
    payloads: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in flatten_state_records(payloads):
        grouped[_record_episode_key(record)].append(record)
    for episode_id in grouped:
        grouped[episode_id] = sorted(
            grouped[episode_id],
            key=lambda record: (
                int(record.get("step_count") or 0),
                int(record.get("transition_index") or 0),
            ),
        )
    return grouped


def pair_transition_records(payload: Mapping[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    record_mode = str(payload.get("analysis_export_record_mode", "state_and_next"))
    if record_mode == "state_only":
        pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for record in payload.get("records", []) or []:
            state_record = dict(record)
            next_record = dict(record)
            next_record["state_kind"] = "next_state"
            if "next_predicted_value" in state_record:
                next_record["predicted_value"] = state_record.get("next_predicted_value")
            if "next_value_features" in state_record:
                next_record["value_features"] = state_record.get("next_value_features")
            pairs.append((state_record, next_record))
        return pairs

    by_index: Dict[int, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for record in payload.get("records", []) or []:
        by_index[int(record.get("transition_index") or 0)][str(record.get("state_kind"))] = dict(record)
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for transition_index in sorted(by_index.keys()):
        state_record = by_index[transition_index].get("state")
        next_record = by_index[transition_index].get("next_state")
        if state_record and next_record:
            pairs.append((state_record, next_record))
    return pairs


def sample_controlled_episodes(
    episodes: Sequence[Mapping[str, Any]],
    per_family: int,
) -> List[Dict[str, Any]]:
    by_family: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for episode in episodes:
        family = _family_key(episode)
        by_family[family][_scene_id(episode)].append(dict(episode))

    selected: List[Dict[str, Any]] = []
    for family in TASK_FAMILY_PRIORITY:
        scene_buckets = by_family.get(family, {})
        if not scene_buckets:
            continue
        scenes = sorted(scene_buckets.keys())
        picks: List[Dict[str, Any]] = []
        scene_cursor = 0
        while len(picks) < per_family and scenes:
            scene = scenes[scene_cursor % len(scenes)]
            bucket = scene_buckets[scene]
            if bucket:
                picks.append(bucket.pop(0))
            else:
                scenes.remove(scene)
                if not scenes:
                    break
                scene_cursor -= 1
            scene_cursor += 1
        selected.extend(picks)
    return selected


def load_checkpoint_bundle(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> Tuple[Dict[str, Any], StateEncoder, ValueNetwork]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {}) or {}
    state_encoder = StateEncoder(
        text_dim=config.get("text_dim", 384),
        numerical_dim=config.get("numerical_dim", 32),
        hidden_dim=config.get("encoder_hidden_dim", 256),
        output_dim=config.get("state_dim", 128),
        text_encoder_model=config.get("text_encoder_model", "all-MiniLM-L6-v2"),
        text_encoder_local_files_only=config.get("text_encoder_local_files_only", False),
        text_encoder_cache_dir=config.get("text_encoder_cache_dir"),
        text_encoder_fallback=config.get("text_encoder_fallback", "hash"),
        device=device,
        debug=False,
        input_variant=config.get("state_encoder_input_variant", "full"),
        context_variant=config.get("state_encoder_context_variant", "full"),
    )
    value_network = ValueNetwork(
        state_dim=config.get("state_dim", 128),
        hidden_dims=config.get("value_hidden_dims", [256, 128, 64]),
        dropout_rate=config.get("dropout_rate", 0.1),
        device=device,
    )
    state_encoder.load_state_dict(checkpoint["state_encoder"])
    value_network.load_state_dict(checkpoint["value_network"])
    return config, state_encoder, value_network


def encode_with_artifacts(
    encoder: StateEncoder,
    world_state: Mapping[str, Any],
    info: Mapping[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    with torch.no_grad():
        state_vector = encoder.encode(copy.deepcopy(dict(world_state)), copy.deepcopy(dict(info)))
    return state_vector.detach().cpu().numpy(), encoder.get_last_forward_record()


def value_forward(
    value_network: ValueNetwork,
    state_vector: np.ndarray,
) -> Tuple[float, np.ndarray]:
    tensor = torch.tensor(state_vector, dtype=torch.float32, device=value_network.device)
    with torch.no_grad():
        value, features = value_network.forward_with_features(tensor)
    return float(value.item()), features.detach().cpu().numpy()


def select_anchor_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    if not records:
        return {}
    sorted_records = sorted(records, key=lambda record: int(record.get("step_count") or 0))
    initial = dict(sorted_records[0])

    def _first(predicate) -> Dict[str, Any]:
        for record in sorted_records:
            if predicate(record):
                return dict(record)
        return dict(sorted_records[-1])

    terminal = dict(sorted_records[-1])
    nonterminal = sorted_records[:-1] if len(sorted_records) > 1 else sorted_records
    high_progress = max(nonterminal, key=lambda record: float(record.get("task_percent_complete", 0.0)))
    initial_progress = float(initial.get("task_percent_complete", 0.0))
    return {
        "initial": initial,
        "visible": _first(lambda r: float(_feature_value(r, "task_object_visible_fraction")) > 0.0),
        "holding": _first(lambda r: float(_feature_value(r, "any_agent_holding_task_object")) > 0.0),
        "progress": _first(lambda r: float(r.get("task_percent_complete", 0.0)) > initial_progress),
        "high_progress": dict(high_progress),
        "terminal": terminal,
    }


def _feature_value(record: Mapping[str, Any], feature_name: str) -> float:
    names = record.get("feature_names", []) or []
    values = record.get("numerical_features", []) or []
    try:
        index = names.index(feature_name)
        return float(values[index])
    except Exception:
        return 0.0


def _record_world_state(record: Mapping[str, Any]) -> Dict[str, Any]:
    world_state = record.get("world_state")
    if isinstance(world_state, Mapping):
        return dict(world_state)
    task_context = record.get("task_context", {}) or {}
    if isinstance(task_context, Mapping):
        return {"task_context": dict(task_context)}
    return {}


def _record_info(record: Mapping[str, Any]) -> Dict[str, Any]:
    info = record.get("info_snapshot")
    if isinstance(info, Mapping):
        return dict(info)
    world_state = _record_world_state(record)
    task_context = world_state.get("task_context", {}) or {}
    return {
        "episode_id": record.get("episode_id"),
        "episode_filename": record.get("episode_filename"),
        "task_instruction": task_context.get("instruction") or "",
        "task_percent_complete": float(record.get("task_percent_complete", 0.0)),
        "task_state_success": float(record.get("task_state_success", 0.0)),
        "step_count": record.get("step_count"),
        "planning_step_count": record.get("planning_step_count", 0),
        "task_family": record.get("task_family"),
        "action_history": [],
        "action_history_count": int(record.get("action_history_count", 0) or 0),
        "action_responses": {},
        "auto_eval_proposition_tracker": {},
    }


def episode_source_summary(episode: Mapping[str, Any]) -> Dict[str, Any]:
    info = episode.get("info", {}) or {}
    extra_info = info.get("extra_info", {}) or {}
    initial_state = extra_info.get("initial_state", []) or []
    source_object_classes = sorted(
        {
            str(object_class)
            for state_element in initial_state
            if isinstance(state_element, Mapping)
            for object_class in (state_element.get("object_classes", []) or [])
            if object_class
        }
    )
    source_furniture_names = sorted(
        {
            str(furniture_name)
            for state_element in initial_state
            if isinstance(state_element, Mapping)
            for furniture_name in (state_element.get("furniture_names", []) or [])
            if furniture_name
        }
    )
    source_allowed_regions = sorted(
        {
            str(region_name)
            for state_element in initial_state
            if isinstance(state_element, Mapping)
            for region_name in (state_element.get("allowed_regions", []) or [])
            if region_name
        }
    )
    return {
        "episode_id": _episode_key(episode),
        "scene_id": _scene_id(episode),
        "instruction": extra_info.get("instruction") or episode.get("instruction"),
        "source_initial_state": initial_state,
        "source_object_classes": source_object_classes,
        "source_furniture_names": source_furniture_names,
        "source_allowed_regions": source_allowed_regions,
        "rigid_obj_count": len(episode.get("rigid_objs", []) or []),
        "name_to_receptacle_count": len(episode.get("name_to_receptacle", {}) or {}),
        "object_labels": info.get("object_labels", {}) or {},
        "episode_object_states": episode.get("object_states", {}) or {},
    }


def run_se1_source_correctness(selected_episodes: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    for episode in selected_episodes:
        source = episode_source_summary(episode)
        initial_state = source["source_initial_state"] or []
        proposition_count = len(episode.get("evaluation_propositions", []) or [])
        clutter_entries = sum(
            1
            for state_element in initial_state
            if isinstance(state_element, Mapping)
            and ("name" in state_element or "template_task_number" in state_element)
        )
        rows.append(
            {
                "episode_id": source["episode_id"],
                "scene_id": source["scene_id"],
                "has_instruction": bool(source["instruction"]),
                "has_initial_state": bool(initial_state),
                "has_task_objects": bool(source["source_object_classes"]),
                "has_targets": bool(source["source_furniture_names"]),
                "has_rooms": bool(source["source_allowed_regions"]),
                "has_parent_receptacles": int(source["name_to_receptacle_count"]) > 0,
                "has_object_states": bool(source["episode_object_states"]),
                "proposition_count": proposition_count,
                "clutter_entries": clutter_entries,
            }
        )
    aggregate = {}
    if rows:
        for key in [
            "has_instruction",
            "has_initial_state",
            "has_task_objects",
            "has_targets",
            "has_rooms",
            "has_parent_receptacles",
            "has_object_states",
        ]:
            aggregate[key] = float(np.mean([float(row[key]) for row in rows]))
        aggregate["mean_proposition_count"] = float(np.mean([row["proposition_count"] for row in rows]))
        aggregate["mean_clutter_entries"] = float(np.mean([row["clutter_entries"] for row in rows]))
    return {"rows": rows, "aggregate": aggregate}


def _get_nested(mapping: Mapping[str, Any], path: str) -> Any:
    current: Any = mapping
    for key in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def build_translation_matrix(
    selected_episodes: Sequence[Mapping[str, Any]],
    episode_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    specs = [
        {
            "source_field": "instruction",
            "source_path": "instruction",
            "intermediate_path": "world_state.task_context.instruction",
            "text_categories": ["instruction"],
            "numerical_categories": [],
            "risk": "low",
            "notes": "Direct task text should enter the text branch.",
        },
        {
            "source_field": "initial_state.object_classes",
            "source_path": "source_object_classes",
            "intermediate_path": "world_state.task_context.source_object_classes",
            "text_categories": ["task-object grounding"],
            "numerical_categories": ["task-object grounding"],
            "risk": "medium",
            "notes": "Source object classes are preserved separately from runtime object handles.",
        },
        {
            "source_field": "initial_state.furniture_names",
            "source_path": "source_furniture_names",
            "intermediate_path": "world_state.task_context.source_furniture_names",
            "text_categories": ["target/receptacle grounding"],
            "numerical_categories": ["target/receptacle grounding"],
            "risk": "medium",
            "notes": "Target furniture names may differ from runtime receptacle instances.",
        },
        {
            "source_field": "initial_state.allowed_regions",
            "source_path": "source_allowed_regions",
            "intermediate_path": "world_state.task_context.source_allowed_regions",
            "text_categories": ["room grounding"],
            "numerical_categories": ["room grounding"],
            "risk": "medium",
            "notes": "Room targets are preserved even when runtime room ids are sparse.",
        },
        {
            "source_field": "rigid_objs",
            "source_path": "rigid_obj_count",
            "intermediate_path": "world_state.object_positions",
            "text_categories": ["object spatial state"],
            "numerical_categories": ["object spatial state"],
            "risk": "low",
            "notes": "Rigid-object transforms should ground runtime spatial context.",
        },
        {
            "source_field": "name_to_receptacle",
            "source_path": "name_to_receptacle_count",
            "intermediate_path": "world_state.object_positions.parent",
            "text_categories": ["object spatial state"],
            "numerical_categories": ["progress/outcome"],
            "risk": "medium",
            "notes": "Parent receptacles affect both text grounding and at-target progress features.",
        },
        {
            "source_field": "object_states",
            "source_path": "episode_object_states",
            "intermediate_path": "world_state.task_context.episode_object_states",
            "text_categories": ["object-state affordance"],
            "numerical_categories": [],
            "risk": "high",
            "notes": "Object-state constraints are currently text-heavy and numerically underrepresented.",
        },
        {
            "source_field": "object_labels",
            "source_path": "object_labels",
            "intermediate_path": "world_state.task_context.object_labels",
            "text_categories": [],
            "numerical_categories": [],
            "risk": "high",
            "notes": "Object labels are carried through but not directly encoded.",
        },
    ]

    rows: List[Dict[str, Any]] = []
    for spec in specs:
        source_present = 0
        intermediate_present = 0
        text_covered = 0
        numerical_covered = 0
        mismatched = 0
        late = 0
        eligible = 0
        for episode in selected_episodes:
            episode_id = _episode_key(episode)
            source = episode_source_summary(episode)
            source_value = source.get(spec["source_path"])
            source_nonempty = bool(source_value)
            if not source_nonempty:
                continue
            eligible += 1
            source_present += 1
            records = list(episode_records.get(episode_id, []))
            if not records:
                continue

            first_record = dict(records[0])
            world_state = _record_world_state(first_record)
            task_context = world_state.get("task_context", {}) or {}
            intermediate_value = None
            if spec["intermediate_path"].startswith("world_state."):
                intermediate_value = _get_nested({"world_state": world_state}, spec["intermediate_path"])
            if intermediate_value:
                intermediate_present += 1

            present_step = None
            for record in records:
                record_world_state = _record_world_state(record)
                record_intermediate = _get_nested({"world_state": record_world_state}, spec["intermediate_path"])
                if record_intermediate:
                    present_step = int(record.get("step_count") or 0)
                    break
            if present_step and present_step > int(records[0].get("step_count") or 0):
                late += 1

            if spec["text_categories"]:
                if any(
                    str(segment.get("category")) in set(spec["text_categories"])
                    for segment in (first_record.get("text_segments", []) or [])
                ):
                    text_covered += 1

            if spec["numerical_categories"]:
                categories = set(first_record.get("feature_categories", []) or [])
                if categories.intersection(set(spec["numerical_categories"])):
                    numerical_covered += 1

            if spec["source_path"] in {"source_object_classes", "source_furniture_names", "source_allowed_regions"}:
                if set(source_value) != set(task_context.get(spec["source_path"], []) or []):
                    mismatched += 1
            elif spec["source_path"] == "instruction":
                task_instruction = task_context.get("instruction") or ""
                if str(source_value).strip() != str(task_instruction).strip():
                    mismatched += 1
            elif spec["source_path"] == "rigid_obj_count":
                if int(source_value) > 0 and not (world_state.get("object_positions", {}) or {}):
                    mismatched += 1
            elif spec["source_path"] == "name_to_receptacle_count":
                parents = [
                    value.get("parent")
                    for value in (world_state.get("object_positions", {}) or {}).values()
                    if isinstance(value, Mapping)
                ]
                if int(source_value) > 0 and not parents:
                    mismatched += 1
            elif spec["source_path"] == "episode_object_states":
                if bool(source_value) != bool(task_context.get("episode_object_states", {}) or {}):
                    mismatched += 1
            elif spec["source_path"] == "object_labels":
                if bool(source_value) != bool(task_context.get("object_labels", {}) or {}):
                    mismatched += 1

        denom = max(eligible, 1)
        rows.append(
            {
                **spec,
                "eligible_episodes": eligible,
                "source_present_rate": source_present / denom,
                "intermediate_present_rate": intermediate_present / denom,
                "text_covered_rate": text_covered / denom,
                "numerical_covered_rate": numerical_covered / denom,
                "mismatch_rate": mismatched / denom,
                "late_rate": late / denom,
                "lost_rate": max(0.0, 1.0 - (intermediate_present / denom)),
            }
        )
    return rows


def _deep_diff_count(value_a: Any, value_b: Any) -> int:
    if type(value_a) is not type(value_b):
        return 1
    if isinstance(value_a, Mapping):
        keys = set(value_a.keys()) | set(value_b.keys())
        return sum(_deep_diff_count(value_a.get(key), value_b.get(key)) for key in keys)
    if isinstance(value_a, list):
        total = abs(len(value_a) - len(value_b))
        for item_a, item_b in zip(value_a, value_b):
            total += _deep_diff_count(item_a, item_b)
        return total
    return 0 if value_a == value_b else 1


def _mutate_world_info(
    record: Mapping[str, Any],
    mutation_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    world_state = copy.deepcopy(_record_world_state(record))
    info = copy.deepcopy(_record_info(record))
    task_context = world_state.get("task_context", {}) or {}
    object_positions = world_state.get("object_positions", {}) or {}
    furniture_positions = world_state.get("furniture_positions", {}) or {}
    task_objects = list(task_context.get("objects", []) or [])
    task_receptacles = list(task_context.get("receptacles", []) or [])
    primary_agent_id = sorted((world_state.get("agent_holdings", {}) or {}).keys())[0] if world_state.get("agent_holdings", {}) else None

    if mutation_name == "correct_target" and task_objects and task_receptacles:
        object_positions.setdefault(task_objects[0], {})["parent"] = task_receptacles[0]
    elif mutation_name == "wrong_target" and task_objects:
        candidates = [name for name in furniture_positions.keys() if name not in task_receptacles]
        object_positions.setdefault(task_objects[0], {})["parent"] = candidates[0] if candidates else "__wrong_target__"
    elif mutation_name == "room_shift":
        task_context["rooms"] = ["counterfactual_room"]
        task_context["source_allowed_regions"] = ["counterfactual_room"]
    elif mutation_name == "holding_toggle" and primary_agent_id is not None and task_objects:
        world_state.setdefault("agent_holdings", {}).setdefault(primary_agent_id, {})["held_object"] = task_objects[0]
    elif mutation_name == "object_state_toggle":
        episode_object_states = task_context.get("episode_object_states", {}) or {}
        toggled = False
        for affordance, handle_map in episode_object_states.items():
            if isinstance(handle_map, Mapping) and handle_map:
                first_key = next(iter(handle_map.keys()))
                episode_object_states[affordance][first_key] = not bool(handle_map[first_key])
                toggled = True
                break
        if not toggled:
            task_context["episode_object_states"] = {"is_clean": {"counterfactual_handle": True}}
    elif mutation_name == "irrelevant_perturbation":
        world_state.setdefault("object_positions", {})["distractor_box"] = {
            "position": [99.0, 99.0, 99.0],
            "parent": "floor",
        }
    else:
        raise ValueError(f"Unknown mutation: {mutation_name}")

    world_state["task_context"] = task_context
    world_state["object_positions"] = object_positions
    info["counterfactual_name"] = mutation_name
    return world_state, info


def apply_category_occlusion(
    record: Mapping[str, Any],
    category: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    world_state = copy.deepcopy(_record_world_state(record))
    info = copy.deepcopy(_record_info(record))
    task_context = world_state.get("task_context", {}) or {}

    if category == "instruction":
        task_context["instruction"] = ""
        info["task_instruction"] = ""
    elif category == "task-object grounding":
        task_context["objects"] = []
        task_context["entities"] = []
        task_context["source_object_classes"] = []
    elif category == "target/receptacle grounding":
        task_context["receptacles"] = []
        task_context["source_furniture_names"] = []
    elif category == "room grounding":
        task_context["rooms"] = []
        task_context["source_allowed_regions"] = []
    elif category == "agent kinematics":
        for pose in (world_state.get("agent_poses", {}) or {}).values():
            pose["position"] = [0.0, 0.0, 0.0]
            pose["yaw"] = 0.0
    elif category == "object spatial state":
        world_state["object_positions"] = {}
    elif category == "visibility/count":
        for obj_name in list(world_state.get("object_positions", {}).keys()):
            if obj_name in set(task_context.get("objects", []) or []):
                del world_state["object_positions"][obj_name]
        for furn_name in list(world_state.get("furniture_positions", {}).keys()):
            if furn_name in set(task_context.get("receptacles", []) or []):
                del world_state["furniture_positions"][furn_name]
    elif category == "holding/manipulation":
        for agent_hold in (world_state.get("agent_holdings", {}) or {}).values():
            agent_hold["held_object"] = None
    elif category == "progress/outcome":
        info["task_percent_complete"] = 0.0
        info["planning_step_count"] = 0
        info["action_history"] = []
        info["action_responses"] = {}
        tracker = info.get("auto_eval_proposition_tracker", {}) or {}
        if "proposition_satisfied_at" in tracker:
            tracker["proposition_satisfied_at"] = [-1 for _ in tracker["proposition_satisfied_at"]]
        info["auto_eval_proposition_tracker"] = tracker
    elif category == "object-state affordance":
        task_context["episode_object_states"] = {}
        task_context["source_initial_state"] = [
            {
                **state_element,
                "object_states": {},
            }
            for state_element in (task_context.get("source_initial_state", []) or [])
            if isinstance(state_element, Mapping)
        ]
        world_state["object_state_snapshot"] = {}
    elif category == "padding/noise":
        pass

    world_state["task_context"] = task_context
    info["occluded_category"] = category
    return world_state, info


def _bin_fraction(value: float) -> int:
    if value <= 0.0:
        return 0
    if value < 0.34:
        return 1
    if value < 0.67:
        return 2
    if value < 1.0:
        return 3
    return 4


def _multilabel_targets(values: Sequence[Sequence[str]]) -> Tuple[np.ndarray, List[str]]:
    vocab = sorted({item for row in values for item in row})
    if not vocab:
        return np.zeros((len(values), 1), dtype=np.float32), []
    index = {token: idx for idx, token in enumerate(vocab)}
    matrix = np.zeros((len(values), len(vocab)), dtype=np.float32)
    for row_idx, row in enumerate(values):
        for token in row:
            matrix[row_idx, index[token]] = 1.0
    return matrix, vocab


def _feature_sets_from_records(
    records: Sequence[Mapping[str, Any]],
    encoder: Optional[StateEncoder] = None,
) -> Dict[str, np.ndarray]:
    def _to_matrix(rows: Sequence[Sequence[float]]) -> np.ndarray:
        max_len = max((len(row) for row in rows), default=0)
        if max_len == 0:
            return np.zeros((len(rows), 0), dtype=np.float32)
        matrix = np.zeros((len(rows), max_len), dtype=np.float32)
        for row_idx, row in enumerate(rows):
            if row:
                matrix[row_idx, : len(row)] = np.asarray(row, dtype=np.float32)
        return matrix

    text_rows = []
    for record in records:
        text_embedding = record.get("text_embedding", []) or []
        if not text_embedding and encoder is not None:
            text_desc = str(record.get("text_desc") or "")
            if text_desc.strip():
                with torch.no_grad():
                    emb = encoder.text_encoder.encode(
                        text_desc,
                        convert_to_tensor=True,
                        device=encoder.device,
                        show_progress_bar=False,
                    )
                text_embedding = emb.detach().cpu().tolist()
        text_rows.append(list(text_embedding))
    text = _to_matrix(text_rows)
    numerical = _to_matrix([record.get("numerical_features", []) or [] for record in records])
    state_vector = _to_matrix([record.get("state_vector", []) or [] for record in records])
    return {
        "text_only": text,
        "numerical_only": numerical,
        "raw_concat": np.concatenate([text, numerical], axis=1) if len(text) else np.zeros((0, 0), dtype=np.float32),
        "state_vector": state_vector,
    }


def _record_source_targets(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    task_object_sets = []
    target_sets = []
    room_sets = []
    holding_state = []
    visibility_bins = []
    at_target_bins = []
    object_state_bins = []
    progress_bins = []
    proposition_bins = []
    terminal_success = []
    return_bins = []
    next_progress_gain = []

    by_episode: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_episode[_record_episode_key(record)].append(record)
    for episode_id in by_episode:
        by_episode[episode_id] = sorted(by_episode[episode_id], key=lambda r: int(r.get("step_count") or 0))

    future_gain_lookup: Dict[Tuple[str, int], float] = {}
    for episode_id, episode_records in by_episode.items():
        progress_values = [float(record.get("task_percent_complete", 0.0)) for record in episode_records]
        for idx, record in enumerate(episode_records):
            future_window = progress_values[idx + 1 : idx + 6]
            future_gain_lookup[(episode_id, idx)] = (
                max(future_window) - progress_values[idx] if future_window else 0.0
            )

    for episode_id, episode_records in by_episode.items():
        episode_terminal_success = float(episode_records[-1].get("task_state_success", 0.0))
        for idx, record in enumerate(episode_records):
            world_state = _record_world_state(record)
            task_context = world_state.get("task_context", {}) or {}
            task_objects = list(task_context.get("objects", []) or task_context.get("source_object_classes", []) or [])
            targets = list(task_context.get("receptacles", []) or task_context.get("source_furniture_names", []) or [])
            rooms = list(task_context.get("rooms", []) or task_context.get("source_allowed_regions", []) or [])
            task_object_sets.append([str(x) for x in task_objects if x])
            target_sets.append([str(x) for x in targets if x])
            room_sets.append([str(x) for x in rooms if x])
            holding_state.append(float(_feature_value(record, "any_agent_holding_task_object")) > 0.0)
            visibility_bins.append(_bin_fraction(_feature_value(record, "task_object_visible_fraction")))
            at_target_bins.append(_bin_fraction(_feature_value(record, "task_object_fraction_at_target")))
            object_state_bins.append(bool(task_context.get("episode_object_states", {}) or _summarize_source_initial_state(task_context)))
            progress_bins.append(_bin_fraction(float(record.get("task_percent_complete", 0.0))))
            proposition_bins.append(_bin_fraction(float(record.get("proposition_satisfied_fraction", 0.0))))
            terminal_success.append(int(episode_terminal_success > 0.5))
            return_bins.append(_bin_fraction(float(record.get("return_target", 0.0))))
            next_progress_gain.append(float(future_gain_lookup[(episode_id, idx)] > 0.0))

    task_objects_y, task_object_vocab = _multilabel_targets(task_object_sets)
    targets_y, target_vocab = _multilabel_targets(target_sets)
    rooms_y, room_vocab = _multilabel_targets(room_sets)

    return {
        "task_object_set": {"task_type": "multilabel", "y": task_objects_y, "vocab": task_object_vocab},
        "target_set": {"task_type": "multilabel", "y": targets_y, "vocab": target_vocab},
        "room_set": {"task_type": "multilabel", "y": rooms_y, "vocab": room_vocab},
        "holding_state": {"task_type": "binary", "y": np.asarray(holding_state, dtype=np.float32)},
        "visibility_bin": {"task_type": "multiclass", "y": np.asarray(visibility_bins, dtype=np.int64)},
        "object_at_target_bin": {"task_type": "multiclass", "y": np.asarray(at_target_bins, dtype=np.int64)},
        "object_state_affordance_bin": {"task_type": "binary", "y": np.asarray(object_state_bins, dtype=np.float32)},
        "task_progress_bin": {"task_type": "multiclass", "y": np.asarray(progress_bins, dtype=np.int64)},
        "proposition_bin": {"task_type": "multiclass", "y": np.asarray(proposition_bins, dtype=np.int64)},
        "next_progress_gain": {"task_type": "binary", "y": np.asarray(next_progress_gain, dtype=np.float32)},
        "terminal_success": {"task_type": "binary", "y": np.asarray(terminal_success, dtype=np.float32)},
        "return_bin": {"task_type": "multiclass", "y": np.asarray(return_bins, dtype=np.int64)},
    }


def _summarize_source_initial_state(task_context: Mapping[str, Any]) -> bool:
    for state_element in (task_context.get("source_initial_state", []) or []):
        if isinstance(state_element, Mapping) and (state_element.get("object_states", {}) or {}):
            return True
    return False


def run_se2_stability(
    selected_records: Sequence[Mapping[str, Any]],
    encoder: Optional[StateEncoder],
) -> Dict[str, Any]:
    if encoder is None:
        return {"skipped": True, "reason": "checkpoint not provided"}

    def _mode_stats(train_mode: bool) -> Dict[str, float]:
        cosine_stds = []
        l2_stds = []
        dim_stds = []
        encoder.train(train_mode)
        for record in selected_records:
            world_state = _record_world_state(record)
            info = _record_info(record)
            if not world_state:
                continue
            vectors = []
            for _ in range(30):
                vec, _ = encode_with_artifacts(encoder, world_state, info)
                vectors.append(vec)
            arr = np.asarray(vectors, dtype=np.float32)
            mean_vec = arr.mean(axis=0)
            distances = [cosine_distance(vec, mean_vec) for vec in arr]
            cosine_stds.append(float(np.std(distances)))
            l2_stds.append(float(np.std(np.linalg.norm(arr - mean_vec, axis=1))))
            dim_stds.append(float(arr.std(axis=0).mean()))
        return {
            "cosine_std_mean": float(np.mean(cosine_stds)) if cosine_stds else 0.0,
            "l2_std_mean": float(np.mean(l2_stds)) if l2_stds else 0.0,
            "dim_std_mean": float(np.mean(dim_stds)) if dim_stds else 0.0,
            "num_states": len(cosine_stds),
        }

    return {"eval": _mode_stats(False), "train": _mode_stats(True)}


def run_se3_counterfactuals(
    anchor_records: Sequence[Mapping[str, Any]],
    encoder: Optional[StateEncoder],
) -> Dict[str, Any]:
    if encoder is None:
        return {"skipped": True, "reason": "checkpoint not provided"}

    grouped: Dict[str, List[float]] = defaultdict(list)
    detailed_rows: List[Dict[str, Any]] = []
    encoder.eval()
    for record in anchor_records:
        world_state = _record_world_state(record)
        info = _record_info(record)
        if not world_state:
            continue
        base_vec, base_artifacts = encode_with_artifacts(encoder, world_state, info)
        for mutation_name in [
            "correct_target",
            "wrong_target",
            "room_shift",
            "holding_toggle",
            "object_state_toggle",
            "irrelevant_perturbation",
        ]:
            mutated_world_state, mutated_info = _mutate_world_info(record, mutation_name)
            mutated_vec, mutated_artifacts = encode_with_artifacts(encoder, mutated_world_state, mutated_info)
            raw_delta = _deep_diff_count(world_state, mutated_world_state) + _deep_diff_count(info, mutated_info)
            base_feat = np.asarray(base_artifacts.get("numerical_features", []), dtype=np.float32)
            mutated_feat = np.asarray(mutated_artifacts.get("numerical_features", []), dtype=np.float32)
            feature_l2 = float(np.linalg.norm(mutated_feat - base_feat))
            state_cos = cosine_distance(base_vec, mutated_vec)
            grouped[mutation_name].append(state_cos)
            detailed_rows.append(
                {
                    "episode_id": record.get("episode_id"),
                    "step_count": record.get("step_count"),
                    "mutation": mutation_name,
                    "raw_delta_count": raw_delta,
                    "feature_l2": feature_l2,
                    "state_cosine_distance": state_cos,
                }
            )

    relevant = [
        value
        for key, values in grouped.items()
        if key != "irrelevant_perturbation"
        for value in values
    ]
    irrelevant = grouped.get("irrelevant_perturbation", [])
    return {
        "group_stats": {name: mean_and_std(values) for name, values in grouped.items()},
        "relevant_minus_irrelevant_gap": float(np.mean(relevant) - np.mean(irrelevant)) if relevant and irrelevant else 0.0,
        "rows": detailed_rows,
    }


def run_se4_stage_correspondence(
    selected_episode_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    stage_vectors: Dict[str, List[np.ndarray]] = defaultdict(list)
    all_points: List[Tuple[str, np.ndarray]] = []
    inter_stage_distances = []
    for records in selected_episode_records.values():
        anchors = select_anchor_records(records)
        ordered = []
        for stage_name in ["initial", "visible", "holding", "progress", "high_progress", "terminal"]:
            record = anchors.get(stage_name)
            if not record:
                continue
            vector = np.asarray(record.get("state_vector", []), dtype=np.float32)
            if vector.size == 0:
                continue
            stage_vectors[stage_name].append(vector)
            all_points.append((stage_name, vector))
            ordered.append((stage_name, vector))
        for idx in range(len(ordered) - 1):
            inter_stage_distances.append(cosine_distance(ordered[idx][1], ordered[idx + 1][1]))

    intra_stage_distances = []
    for stage_name, vectors in stage_vectors.items():
        for idx in range(len(vectors)):
            for jdx in range(idx + 1, len(vectors)):
                intra_stage_distances.append(cosine_distance(vectors[idx], vectors[jdx]))

    nn_correct = 0
    nn_total = 0
    for idx, (stage_name, vector) in enumerate(all_points):
        best_j = None
        best_dist = None
        for jdx, (_, other_vector) in enumerate(all_points):
            if idx == jdx:
                continue
            dist = cosine_distance(vector, other_vector)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_j = jdx
        if best_j is not None:
            nn_total += 1
            nn_correct += int(all_points[best_j][0] == stage_name)

    return {
        "inter_stage_mean": float(np.mean(inter_stage_distances)) if inter_stage_distances else 0.0,
        "intra_stage_mean": float(np.mean(intra_stage_distances)) if intra_stage_distances else 0.0,
        "stage_separability_ratio": (
            float(np.mean(inter_stage_distances) / max(np.mean(intra_stage_distances), 1e-6))
            if inter_stage_distances and intra_stage_distances
            else 0.0
        ),
        "nearest_neighbor_stage_accuracy": float(nn_correct / max(nn_total, 1)),
    }


def run_se5_fidelity(
    selected_records: Sequence[Mapping[str, Any]],
    encoder: Optional[StateEncoder],
    value_network: Optional[ValueNetwork],
    *,
    seed: int,
    device: str,
) -> Dict[str, Any]:
    if not selected_records:
        return {"skipped": True, "reason": "no records"}
    feature_sets = _feature_sets_from_records(selected_records, encoder)
    tasks = _record_source_targets(selected_records)
    source_task_names = [
        "task_object_set",
        "target_set",
        "room_set",
        "holding_state",
        "visibility_bin",
        "object_at_target_bin",
        "object_state_affordance_bin",
    ]
    target_task_names = [
        "task_progress_bin",
        "proposition_bin",
        "next_progress_gain",
        "terminal_success",
        "return_bin",
    ]
    source_results = {
        task_name: compare_feature_sets(
            feature_sets,
            tasks[task_name]["y"],
            tasks[task_name]["task_type"],
            seed=seed,
            device=device,
        )
        for task_name in source_task_names
    }
    target_results = {
        task_name: compare_feature_sets(
            feature_sets,
            tasks[task_name]["y"],
            tasks[task_name]["task_type"],
            seed=seed,
            device=device,
        )
        for task_name in target_task_names
    }

    category_task_map = {
        "task-object grounding": ["task_object_set"],
        "target/receptacle grounding": ["target_set"],
        "room grounding": ["room_set"],
        "object spatial state": ["object_at_target_bin"],
        "visibility/count": ["visibility_bin"],
        "holding/manipulation": ["holding_state"],
        "progress/outcome": ["task_progress_bin", "proposition_bin", "next_progress_gain", "terminal_success", "return_bin"],
        "object-state affordance": ["object_state_affordance_bin"],
    }
    category_rows = []
    for category in STATE_ENCODER_CATEGORY_NAMES:
        task_names = category_task_map.get(category, [])
        metrics = []
        for task_name in task_names:
            task_result = source_results.get(task_name) or target_results.get(task_name)
            if not task_result:
                continue
            state_metrics = task_result["state_vector"]["metrics"]
            metrics.append(next(iter(state_metrics.values())) if state_metrics else 0.0)
        category_rows.append(
            {
                "category": category,
                "probe_tasks": task_names,
                "state_vector_score_mean": float(np.mean(metrics)) if metrics else 0.0,
            }
        )

    occlusion_rows = []
    if encoder is not None and value_network is not None:
        encoder.eval()
        value_network.eval()
        subset = list(selected_records[: min(24, len(selected_records))])
        for category in STATE_ENCODER_CATEGORY_NAMES:
            state_deltas = []
            value_deltas = []
            for record in subset:
                world_state = _record_world_state(record)
                info = _record_info(record)
                if not world_state:
                    continue
                base_vec, _ = encode_with_artifacts(encoder, world_state, info)
                base_value, _ = value_forward(value_network, base_vec)
                occluded_world_state, occluded_info = apply_category_occlusion(record, category)
                occluded_vec, _ = encode_with_artifacts(encoder, occluded_world_state, occluded_info)
                occluded_value, _ = value_forward(value_network, occluded_vec)
                state_deltas.append(cosine_distance(base_vec, occluded_vec))
                value_deltas.append(abs(occluded_value - base_value))
            occlusion_rows.append(
                {
                    "category": category,
                    "state_cosine_drift_mean": float(np.mean(state_deltas)) if state_deltas else 0.0,
                    "value_abs_drift_mean": float(np.mean(value_deltas)) if value_deltas else 0.0,
                }
            )

    return {
        "source_reconstruction": source_results,
        "target_reconstruction": target_results,
        "category_fidelity": category_rows,
        "occlusion": occlusion_rows,
    }


def run_vn1_calculation_correctness(payloads: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    td_errors = []
    return_errors = []
    for payload in payloads:
        pairs = pair_transition_records(payload)
        if not pairs:
            continue
        rewards = np.asarray([float(state.get("reward_shaped", 0.0)) for state, _ in pairs], dtype=np.float32)
        values = np.asarray([float(state.get("predicted_value", 0.0)) for state, _ in pairs], dtype=np.float32)
        next_values = np.asarray([float(next_state.get("predicted_value", 0.0)) for _, next_state in pairs], dtype=np.float32)
        dones = np.asarray([float(state.get("done", False)) for state, _ in pairs], dtype=np.float32)
        gamma = float(payload.get("gamma", 0.99))
        gae_lambda = float(payload.get("gae_lambda", 0.95))
        _, recomputed_returns = compute_gae(rewards, values, next_values, dones, gamma=gamma, gae_lambda=gae_lambda)
        for idx, (state_record, next_record) in enumerate(pairs):
            recomputed_td = rewards[idx] + gamma * next_values[idx] * (1.0 - dones[idx]) - values[idx]
            td_errors.append(abs(recomputed_td - float(state_record.get("td_error", 0.0))))
            return_errors.append(abs(float(recomputed_returns[idx]) - float(state_record.get("return_target", 0.0))))
    return {
        "td_error_abs": mean_and_std(td_errors),
        "return_target_abs": mean_and_std(return_errors),
    }


def run_vn2_stability(
    selected_records: Sequence[Mapping[str, Any]],
    value_network: Optional[ValueNetwork],
) -> Dict[str, Any]:
    if value_network is None:
        return {"skipped": True, "reason": "checkpoint not provided"}

    def _mode_stats(train_mode: bool) -> Dict[str, float]:
        value_network.train(train_mode)
        value_stds = []
        for record in selected_records:
            state_vector = np.asarray(record.get("state_vector", []), dtype=np.float32)
            if state_vector.size == 0:
                continue
            values = []
            for _ in range(30):
                value, _ = value_forward(value_network, state_vector)
                values.append(value)
            value_stds.append(float(np.std(values)))
        return {"value_std_mean": float(np.mean(value_stds)) if value_stds else 0.0, "num_states": len(value_stds)}

    return {"eval": _mode_stats(False), "train": _mode_stats(True)}


def run_vn3_ranking(
    selected_episode_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    return_corrs = []
    progress_corrs = []
    prop_corrs = []
    ranking_acc = []
    for records in selected_episode_records.values():
        if len(records) < 2:
            continue
        values = [float(record.get("predicted_value", 0.0)) for record in records]
        returns = [float(record.get("return_target", 0.0)) for record in records]
        progress = [float(record.get("task_percent_complete", 0.0)) for record in records]
        props = [float(record.get("proposition_satisfied_fraction", 0.0)) for record in records]
        return_corrs.append(spearman_corr(values, returns))
        progress_corrs.append(spearman_corr(values, progress))
        prop_corrs.append(spearman_corr(values, props))
        correct = 0
        total = 0
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                return_gap = returns[j] - returns[i]
                if abs(return_gap) < 1e-6:
                    continue
                value_gap = values[j] - values[i]
                total += 1
                correct += int(np.sign(return_gap) == np.sign(value_gap))
        if total > 0:
            ranking_acc.append(float(correct / total))
    return {
        "return_spearman": mean_and_std(return_corrs),
        "progress_spearman": mean_and_std(progress_corrs),
        "proposition_spearman": mean_and_std(prop_corrs),
        "pairwise_ranking_accuracy": mean_and_std(ranking_acc),
    }


def run_vn4_terminal_analysis(
    selected_episode_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    finals = []
    for records in selected_episode_records.values():
        if not records:
            continue
        finals.append(dict(records[-1]))
    if not finals:
        return {"skipped": True, "reason": "no finals"}

    step_counts = np.asarray([float(record.get("step_count", 0.0)) for record in finals], dtype=np.float32)
    stuck_threshold = float(np.quantile(step_counts, 0.8)) if len(step_counts) > 0 else 0.0
    labels = []
    scores = []
    grouped_values: Dict[str, List[float]] = defaultdict(list)
    for record in finals:
        score = float(record.get("predicted_value", 0.0))
        success = float(record.get("task_state_success", 0.0)) > 0.5
        progress = float(record.get("task_percent_complete", 0.0))
        step_count = float(record.get("step_count", 0.0))
        if success:
            group = "success_terminal"
        elif step_count >= stuck_threshold and progress < 0.5:
            group = "stuck_failure"
        else:
            group = "failure_terminal"
        grouped_values[group].append(score)
        labels.append(int(success))
        scores.append(score)
    calibration_bins = []
    if scores:
        score_arr = np.asarray(scores, dtype=np.float32)
        min_score = float(score_arr.min())
        max_score = float(score_arr.max())
        if max_score > min_score:
            normalized = (score_arr - min_score) / (max_score - min_score)
            for lower in np.linspace(0.0, 0.8, 5):
                upper = lower + 0.2
                mask = (normalized >= lower) & (normalized < upper if upper < 1.0 else normalized <= upper)
                if not mask.any():
                    continue
                calibration_bins.append(
                    {
                        "bin_start": float(lower),
                        "bin_end": float(upper),
                        "mean_pred": float(normalized[mask].mean()),
                        "empirical_success": float(np.asarray(labels, dtype=np.float32)[mask].mean()),
                    }
                )
    return {
        "grouped_value_stats": {name: mean_and_std(values) for name, values in grouped_values.items()},
        "success_auroc": float(binary_auroc(labels, scores)),
        "calibration_bins": calibration_bins,
    }


def run_vn6_occlusion_dependency(
    selected_records: Sequence[Mapping[str, Any]],
    encoder: Optional[StateEncoder],
    value_network: Optional[ValueNetwork],
) -> Dict[str, Any]:
    if encoder is None or value_network is None:
        return {"skipped": True, "reason": "checkpoint not provided"}
    encoder.eval()
    value_network.eval()
    rows = []
    subset = list(selected_records[: min(24, len(selected_records))])
    for category in STATE_ENCODER_CATEGORY_NAMES:
        original_values = []
        occluded_values = []
        returns = []
        drifts = []
        for record in subset:
            world_state = _record_world_state(record)
            info = _record_info(record)
            if not world_state:
                continue
            base_vec, _ = encode_with_artifacts(encoder, world_state, info)
            base_value, _ = value_forward(value_network, base_vec)
            occ_world_state, occ_info = apply_category_occlusion(record, category)
            occ_vec, _ = encode_with_artifacts(encoder, occ_world_state, occ_info)
            occ_value, _ = value_forward(value_network, occ_vec)
            original_values.append(base_value)
            occluded_values.append(occ_value)
            returns.append(float(record.get("return_target", 0.0)))
            drifts.append(abs(occ_value - base_value))
        rows.append(
            {
                "category": category,
                "value_abs_drift_mean": float(np.mean(drifts)) if drifts else 0.0,
                "return_spearman_drop": (
                    spearman_corr(original_values, returns) - spearman_corr(occluded_values, returns)
                    if drifts
                    else 0.0
                ),
            }
        )
    return {"rows": rows}


def run_vn5_value_probes(
    selected_records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    device: str,
) -> Dict[str, Any]:
    if not selected_records:
        return {"skipped": True, "reason": "no records"}
    state_vectors = np.asarray([record.get("state_vector", []) or [] for record in selected_records], dtype=np.float32)
    value_features = np.asarray([record.get("value_features", []) or [] for record in selected_records], dtype=np.float32)

    by_episode: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in selected_records:
        by_episode[_record_episode_key(record)].append(record)
    remaining_steps = []
    next_progress_gain = []
    terminal_success = []
    return_bin = []
    for episode_id, episode_records in by_episode.items():
        episode_records = sorted(episode_records, key=lambda record: int(record.get("step_count") or 0))
        progress_values = [float(record.get("task_percent_complete", 0.0)) for record in episode_records]
        total_steps = len(episode_records)
        success = float(episode_records[-1].get("task_state_success", 0.0)) > 0.5
        for idx, record in enumerate(episode_records):
            remaining_steps.append(_bin_fraction(float(total_steps - idx - 1) / max(total_steps, 1)))
            future_window = progress_values[idx + 1 : idx + 6]
            next_progress_gain.append(float((max(future_window) - progress_values[idx]) > 0.0) if future_window else 0.0)
            terminal_success.append(float(success))
            return_bin.append(_bin_fraction(float(record.get("return_target", 0.0))))

    tasks = {
        "return_bin": {"task_type": "multiclass", "y": np.asarray(return_bin, dtype=np.int64)},
        "remaining_steps_bin": {"task_type": "multiclass", "y": np.asarray(remaining_steps, dtype=np.int64)},
        "next_progress_gain": {"task_type": "binary", "y": np.asarray(next_progress_gain, dtype=np.float32)},
        "terminal_success": {"task_type": "binary", "y": np.asarray(terminal_success, dtype=np.float32)},
    }
    feature_sets = {"state_vector": state_vectors, "value_features": value_features}
    return {
        task_name: compare_feature_sets(
            feature_sets,
            task_spec["y"],
            task_spec["task_type"],
            seed=seed,
            device=device,
        )
        for task_name, task_spec in tasks.items()
    }


def build_category_necessity_table(
    translation_rows: Sequence[Mapping[str, Any]],
    se5_results: Mapping[str, Any],
    vn6_results: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    translation_by_category: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in translation_rows:
        for category in list(row.get("text_categories", [])) + list(row.get("numerical_categories", [])):
            translation_by_category[category].append(row)
    category_fidelity = {row["category"]: row for row in se5_results.get("category_fidelity", []) or []}
    category_occlusion = {row["category"]: row for row in vn6_results.get("rows", []) or []}

    rows = []
    for category in STATE_ENCODER_CATEGORY_NAMES:
        translation_slice = translation_by_category.get(category, [])
        intermediate_rate = float(np.mean([row.get("intermediate_present_rate", 0.0) for row in translation_slice])) if translation_slice else 0.0
        mismatch_rate = float(np.mean([row.get("mismatch_rate", 0.0) for row in translation_slice])) if translation_slice else 0.0
        fidelity_score = float(category_fidelity.get(category, {}).get("state_vector_score_mean", 0.0))
        occlusion_drop = float(category_occlusion.get(category, {}).get("return_spearman_drop", 0.0))
        if category == "padding/noise":
            judgement = "redundant"
            recommended_channel = "none"
        elif fidelity_score >= 0.55 or occlusion_drop >= 0.05:
            judgement = "necessary"
            recommended_channel = "text+numerical"
        elif intermediate_rate >= 0.5 and mismatch_rate <= 0.25:
            judgement = "optional"
            recommended_channel = "text" if "grounding" in category or category in {"instruction", "object-state affordance"} else "numerical"
        else:
            judgement = "harmful" if mismatch_rate > 0.5 else "optional"
            recommended_channel = "text"
        rows.append(
            {
                "category": category,
                "judgement": judgement,
                "recommended_channel": recommended_channel,
                "intermediate_present_rate": intermediate_rate,
                "mismatch_rate": mismatch_rate,
                "state_vector_score_mean": fidelity_score,
                "return_spearman_drop": occlusion_drop,
            }
        )
    return rows


def _plot_translation_heatmap(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    labels = [row["source_field"] for row in rows]
    data = np.asarray(
        [
            [
                float(row.get("source_present_rate", 0.0)),
                float(row.get("intermediate_present_rate", 0.0)),
                float(row.get("text_covered_rate", 0.0)),
                float(row.get("numerical_covered_rate", 0.0)),
            ]
            for row in rows
        ],
        dtype=np.float32,
    )
    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(4), ["source", "intermediate", "text", "numerical"])
    ax.set_yticks(range(len(labels)), labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_counterfactual_boxplot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mutation"])].append(float(row["state_cosine_distance"]))
    labels = list(grouped.keys())
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.boxplot([grouped[label] for label in labels], labels=labels, vert=True)
    ax.set_ylabel("State cosine distance")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_fidelity_bars(se5_results: Mapping[str, Any], path: Path) -> None:
    tasks = []
    values = []
    for task_name, result in se5_results.get("source_reconstruction", {}).items():
        metrics = result.get("state_vector", {}).get("metrics", {})
        if not metrics:
            continue
        metric_value = next(iter(metrics.values()))
        tasks.append(f"source:{task_name}")
        values.append(float(metric_value))
    for task_name, result in se5_results.get("target_reconstruction", {}).items():
        metrics = result.get("state_vector", {}).get("metrics", {})
        if not metrics:
            continue
        metric_value = next(iter(metrics.values()))
        tasks.append(f"target:{task_name}")
        values.append(float(metric_value))
    if not tasks:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(tasks)), values)
    ax.set_xticks(range(len(tasks)), tasks, rotation=35, ha="right")
    ax.set_ylabel("State-vector probe score")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_category_radar(category_rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    labels = [row["category"] for row in category_rows]
    values = [float(row.get("state_vector_score_mean", 0.0)) for row in category_rows]
    if not labels:
        return
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values)
    ax.fill(angles, values, alpha=0.2)
    ax.set_xticks(angles[:-1], labels)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_value_calibration(vn4_results: Mapping[str, Any], vn3_results: Mapping[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bins = vn4_results.get("calibration_bins", []) or []
    axes[0].plot(
        [row["mean_pred"] for row in bins],
        [row["empirical_success"] for row in bins],
        marker="o",
    )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_title("Calibration")
    axes[0].set_xlabel("Normalized predicted value")
    axes[0].set_ylabel("Empirical success")

    labels = ["return", "progress", "proposition", "ranking"]
    values = [
        float(vn3_results.get("return_spearman", {}).get("mean", 0.0)),
        float(vn3_results.get("progress_spearman", {}).get("mean", 0.0)),
        float(vn3_results.get("proposition_spearman", {}).get("mean", 0.0)),
        float(vn3_results.get("pairwise_ranking_accuracy", {}).get("mean", 0.0)),
    ]
    axes[1].bar(range(len(labels)), values)
    axes[1].set_xticks(range(len(labels)), labels)
    axes[1].set_title("Value ranking")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_occlusion_bars(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    labels = [row["category"] for row in rows]
    values = [float(row.get("return_spearman_drop", 0.0)) for row in rows]
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(range(len(labels)), values)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_ylabel("Return Spearman drop")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _cross_system_table(
    export_roots: Sequence[str | Path],
    episode_lookup: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for root in export_roots:
        payloads = load_export_payloads([root])
        records = flatten_state_records(payloads)
        if not records:
            continue
        grouped = group_state_records_by_episode(payloads)
        selected_episodes = [episode_lookup[episode_id] for episode_id in grouped.keys() if episode_id in episode_lookup]
        translation = build_translation_matrix(selected_episodes, grouped)
        vn3 = run_vn3_ranking(grouped)
        vn4 = run_vn4_terminal_analysis(grouped)
        rows.append(
            {
                "system_root": str(root),
                "num_records": len(records),
                "translation_intermediate_rate": float(np.mean([row["intermediate_present_rate"] for row in translation])) if translation else 0.0,
                "value_return_spearman": float(vn3.get("return_spearman", {}).get("mean", 0.0)),
                "success_auroc": float(vn4.get("success_auroc", 0.0)),
            }
        )
    return rows


def run_suite(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = ensure_dir(args.output_dir)
    table_dir = ensure_dir(output_dir / "tables")
    plot_dir = ensure_dir(output_dir / "plots")
    review_dir = ensure_dir(output_dir / "reviews")

    dataset_episodes = load_dataset_episodes(args.dataset_path)
    episode_lookup = {_episode_key(episode): episode for episode in dataset_episodes}
    payloads = load_export_payloads(args.export_root)
    state_debug_records = load_state_debug_records(args.export_root)
    episode_records = group_state_records_by_episode(payloads)

    selected_episodes = sample_controlled_episodes(dataset_episodes, args.per_family)
    selected_episode_ids = {_episode_key(episode) for episode in selected_episodes}
    selected_episode_records = {
        episode_id: records
        for episode_id, records in episode_records.items()
        if episode_id in selected_episode_ids
    }
    selected_records = [record for records in selected_episode_records.values() for record in records]
    anchor_records = []
    for records in selected_episode_records.values():
        anchor_records.extend(select_anchor_records(records).values())

    encoder = None
    value_network = None
    if args.checkpoint_path:
        _, encoder, value_network = load_checkpoint_bundle(args.checkpoint_path, device=args.device)

    se1_source = run_se1_source_correctness(selected_episodes)
    se1_translation = build_translation_matrix(selected_episodes, selected_episode_records)
    se2 = run_se2_stability(anchor_records[: min(12, len(anchor_records))], encoder)
    se3 = run_se3_counterfactuals(anchor_records[: min(24, len(anchor_records))], encoder)
    se4 = run_se4_stage_correspondence(selected_episode_records)
    se5 = run_se5_fidelity(selected_records, encoder, value_network, seed=args.seed, device=args.device)
    vn1 = run_vn1_calculation_correctness(payloads)
    vn2 = run_vn2_stability(anchor_records[: min(12, len(anchor_records))], value_network)
    vn3 = run_vn3_ranking(selected_episode_records)
    vn4 = run_vn4_terminal_analysis(selected_episode_records)
    vn5 = run_vn5_value_probes(selected_records, seed=args.seed, device=args.device)
    vn6 = run_vn6_occlusion_dependency(anchor_records[: min(24, len(anchor_records))], encoder, value_network)
    category_necessity = build_category_necessity_table(se1_translation, se5, vn6)
    cross_system = _cross_system_table(args.export_root, episode_lookup)

    write_json(table_dir / "source_correctness.json", se1_source)
    write_json(table_dir / "translation_matrix.json", {"rows": se1_translation})
    write_json(table_dir / "category_necessity.json", {"rows": category_necessity})
    write_json(table_dir / "state_encoder_fidelity.json", se5)
    write_json(table_dir / "value_network_calculation.json", vn1)
    write_json(table_dir / "value_network_probes.json", vn5)
    write_json(table_dir / "cross_system_robustness.json", {"rows": cross_system})

    _plot_translation_heatmap(se1_translation, plot_dir / "category_coverage_heatmap.png")
    _plot_counterfactual_boxplot(se3.get("rows", []), plot_dir / "counterfactual_boxplot.png")
    _plot_fidelity_bars(se5, plot_dir / "compression_fidelity.png")
    _plot_category_radar(se5.get("category_fidelity", []), plot_dir / "category_fidelity_radar.png")
    _plot_value_calibration(vn4, vn3, plot_dir / "value_calibration_ranking.png")
    _plot_occlusion_bars(vn6.get("rows", []), plot_dir / "value_occlusion_drop.png")

    state_packets = []
    for idx, record in enumerate(selected_records[: min(50, len(selected_records))]):
        vector = np.asarray(record.get("state_vector", []), dtype=np.float32)
        neighbors = []
        if vector.size > 0:
            others = []
            for other in selected_records:
                if other is record:
                    continue
                other_vec = np.asarray(other.get("state_vector", []), dtype=np.float32)
                if other_vec.size == 0:
                    continue
                others.append((cosine_distance(vector, other_vec), other))
            for _, other in sorted(others, key=lambda item: item[0])[:3]:
                neighbors.append(
                    {
                        "episode_id": other.get("episode_id"),
                        "step_count": other.get("step_count"),
                        "task_percent_complete": other.get("task_percent_complete"),
                        "text_desc": other.get("text_desc"),
                    }
                )
        source_summary = episode_source_summary(episode_lookup.get(_record_episode_key(record), {}))
        state_packets.append(
            build_state_review_packet(
                review_id=f"se55-{idx}",
                source_summary=source_summary,
                record=dict(record),
                nearest_neighbors=neighbors,
                probe_predictions={
                    "task_percent_complete": record.get("task_percent_complete"),
                    "proposition_satisfied_fraction": record.get("proposition_satisfied_fraction"),
                    "terminal_success": record.get("task_state_success"),
                },
            )
        )

    value_packets = []
    for episode_id, records in list(selected_episode_records.items())[: min(20, len(selected_episode_records))]:
        if len(records) < 2:
            continue
        first = records[0]
        last = records[-1]
        value_packets.append(
            build_value_review_packet(
                review_id=f"vn7-{episode_id}",
                record_a=dict(first),
                record_b=dict(last),
            )
        )

    state_review_paths = save_review_packets(review_dir, "se55_packets", state_packets)
    value_review_paths = save_review_packets(review_dir, "vn7_packets", value_packets)
    llm_outputs = maybe_run_openai_reviews(
        state_packets[: min(10, len(state_packets))] + value_packets[: min(10, len(value_packets))],
        model=args.llm_model,
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
    )
    if llm_outputs:
        write_json(review_dir / "llm_reviews.json", {"rows": llm_outputs})

    summary = {
        "dataset_episode_count": len(dataset_episodes),
        "critic_export_count": len(payloads),
        "state_debug_record_count": len(state_debug_records),
        "selected_episode_count": len(selected_episodes),
        "selected_state_record_count": len(selected_records),
        "selected_anchor_record_count": len(anchor_records),
        "se1_source_correctness": se1_source,
        "se1_translation": se1_translation,
        "se2_stability": se2,
        "se3_counterfactuals": {k: v for k, v in se3.items() if k != "rows"},
        "se4_stage_correspondence": se4,
        "se5_fidelity": se5,
        "vn1_calculation": vn1,
        "vn2_stability": vn2,
        "vn3_ranking": vn3,
        "vn4_terminal": vn4,
        "vn5_probes": vn5,
        "vn6_dependency": vn6,
        "category_necessity": category_necessity,
        "cross_system": cross_system,
        "review_packets": {
            "se55": state_review_paths,
            "vn7": value_review_paths,
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run StateEncoder / ValueNetwork offline analyses.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--export-root", action="append", required=True, help="Root directory or file containing critic exports.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--per-family", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_suite(args)


if __name__ == "__main__":
    main()

