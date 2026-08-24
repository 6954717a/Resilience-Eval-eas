"""
State Encoder for A2C Critic.

Encodes a structured ``world_state_dict`` into a fixed-dimensional vector while
also exposing rich export artifacts for downstream analysis.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from habitat_llm.evaluation.experiments.common import (
    STATE_ENCODER_CATEGORY_NAMES,
    classify_feature_category,
)
from habitat_llm.evaluation.proposition_outcome import proposition_outcome_from_info

logger = logging.getLogger(__name__)

STATE_ENCODER_FEATURE_SCHEMA_VERSION = (
    "state_encoder_features_v3_target_alias_floor_unknown_mask"
)


class _HashTextEncoder:
    """
    Lightweight deterministic fallback when SentenceTransformer cannot be loaded.
    """

    def __init__(self, dim: int, device: str = "cpu") -> None:
        self.dim = int(dim)
        self.device = device

    def to(self, device: str) -> "_HashTextEncoder":
        self.device = str(device)
        return self

    def _encode_one(self, text: str) -> torch.Tensor:
        vector = torch.zeros(self.dim, dtype=torch.float32, device=self.device)
        tokens = re.findall(r"\w+", str(text or "").lower())
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], byteorder="little", signed=False) % self.dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vector[index] += sign
        norm = float(torch.linalg.norm(vector))
        if norm > 0.0:
            vector = vector / norm
        return vector

    def encode(
        self,
        sentences: Any,
        convert_to_tensor: bool = True,
        device: Optional[str] = None,
        show_progress_bar: bool = False,
    ) -> Any:
        del show_progress_bar
        target_device = str(device or self.device)
        if isinstance(sentences, str):
            encoded = self._encode_one(sentences).to(target_device)
            if convert_to_tensor:
                return encoded
            return encoded.detach().cpu().numpy()

        encoded_rows = [self._encode_one(sentence).to(target_device) for sentence in sentences]
        if not encoded_rows:
            empty = torch.zeros((0, self.dim), dtype=torch.float32, device=target_device)
            if convert_to_tensor:
                return empty
            return empty.detach().cpu().numpy()
        stacked = torch.stack(encoded_rows, dim=0)
        if convert_to_tensor:
            return stacked
        return stacked.detach().cpu().numpy()


class StateEncoder(nn.Module):
    """
    Encodes perception state into fixed-dimensional vectors.
    """

    def __init__(
        self,
        text_dim: int = 384,
        numerical_dim: int = 32,
        hidden_dim: int = 256,
        output_dim: int = 128,
        text_encoder_model: str = "all-MiniLM-L6-v2",
        text_encoder_local_files_only: bool = False,
        text_encoder_cache_dir: Optional[str] = None,
        text_encoder_fallback: str = "hash",
        device: str = "cpu",
        debug: bool = False,
        debug_dir: Optional[str] = None,
        debug_frequency: int = 1,
        debug_max_entries: int = 1000,
        debug_log_state: bool = True,
        debug_log_next_state: bool = False,
        debug_log_on_change: bool = False,
        debug_change_threshold: float = 0.0,
        debug_text_max_len: int = 200,
        debug_log_delta: bool = True,
        debug_log_task_context_once: bool = True,
        debug_log_full_features_first: bool = True,
        debug_key_feature_names: Optional[List[str]] = None,
        debug_signature_mode: str = "semantic",
        debug_signature_feature_names: Optional[List[str]] = None,
        input_variant: str = "full",
        context_variant: str = "full",
        distance_norm: float = 10.0,
        count_norm: float = 10.0,
        action_history_cap: float = 50.0,
        fusion_dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.text_dim = int(text_dim)
        self.numerical_dim = int(numerical_dim)
        self.output_dim = int(output_dim)
        self.input_variant = str(input_variant or "full")
        self.context_variant = str(context_variant or "full")
        self.text_encoder_model = str(text_encoder_model or "all-MiniLM-L6-v2")
        self.text_encoder_local_files_only = bool(text_encoder_local_files_only)
        self.text_encoder_cache_dir = text_encoder_cache_dir
        self.text_encoder_fallback = str(text_encoder_fallback or "hash").lower()
        self.distance_norm = max(float(distance_norm), 1e-6)
        self.count_norm = max(float(count_norm), 1e-6)
        self.action_history_cap = max(float(action_history_cap), 1.0)
        self.fusion_dropout_rate = max(float(fusion_dropout_rate), 0.0)
        if self.text_encoder_fallback not in {"hash", "zero", "error"}:
            self.text_encoder_fallback = "hash"
        requested_device = str(device or "cpu")
        if requested_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for StateEncoder but unavailable. Falling back to CPU.")
            requested_device = "cpu"
        self.device = requested_device

        self.debug = bool(debug)
        self.debug_frequency = max(1, int(debug_frequency))
        self.debug_max_entries = max(0, int(debug_max_entries))
        self.debug_dir = debug_dir
        self.debug_log_state = bool(debug_log_state)
        self.debug_log_next_state = bool(debug_log_next_state)
        self.debug_log_on_change = bool(debug_log_on_change)
        self.debug_change_threshold = float(debug_change_threshold)
        self.debug_text_max_len = max(0, int(debug_text_max_len))
        self.debug_log_delta = bool(debug_log_delta)
        self.debug_log_task_context_once = bool(debug_log_task_context_once)
        self.debug_log_full_features_first = bool(debug_log_full_features_first)
        self.debug_key_feature_names = debug_key_feature_names or [
            "agent_pos_x",
            "agent_pos_y",
            "agent_pos_z",
            "agent_yaw",
            "agent2_pos_x",
            "agent2_pos_y",
            "agent2_pos_z",
            "agent2_yaw",
            "task_progress",
            "proposition_satisfied_fraction",
            "min_task_obj_dist_norm",
            "min_agent_target_dist_norm",
            "min_agent_target_dist_when_holding_norm",
            "task_object_fraction_at_target",
            "task_object_visible_fraction",
            "task_target_visible_fraction",
            "any_agent_holding",
            "any_agent_holding_task_object",
            "primary_agent_holding_task_object",
        ]
        self.debug_signature_mode = (
            debug_signature_mode if debug_signature_mode in ("full", "semantic") else "full"
        )
        self.debug_signature_feature_names = debug_signature_feature_names or [
            "task_progress",
            "proposition_satisfied_fraction",
            "min_task_obj_dist_norm",
            "mean_task_obj_dist_norm",
            "min_agent_target_dist_norm",
            "min_obj_target_dist_norm",
            "mean_obj_target_dist_norm",
            "task_object_fraction_at_target",
            "task_object_visible_fraction",
            "task_target_visible_fraction",
            "task_object_visible_count_norm",
            "task_target_visible_count_norm",
            "any_agent_holding",
            "any_agent_holding_task_object",
            "held_task_object_fraction",
            "primary_agent_holding",
            "primary_agent_holding_task_object",
            "min_agent_target_dist_when_holding_norm",
            "action_history_count_norm",
        ]

        self._debug_entries = 0
        self._debug_step = 0
        self._last_feature_names: List[str] = []
        self._last_feature_values: List[float] = []
        self._last_feature_categories: List[str] = []
        self._last_text_segments: List[Dict[str, Any]] = []
        self._last_forward_record: Dict[str, Any] = {}
        self._last_diagnostics: Dict[str, Any] = {}
        self._last_logged_signature: Optional[tuple] = None
        self._last_signature_hash: Optional[str] = None
        self._last_logged_feature_values: Optional[List[float]] = None
        self._last_logged_text_hash: Optional[str] = None
        self._current_log_key: Optional[tuple] = None
        self._meta_logged_for_key: bool = False
        self._last_task_context_hash: Optional[str] = None
        self._repeat_count: int = 0
        self._repeat_start_step: Optional[int] = None
        self._repeat_last_step: Optional[int] = None
        self._last_state_kind: Optional[str] = None

        if self.debug and self.debug_dir:
            os.makedirs(self.debug_dir, exist_ok=True)

        self.text_encoder_backend = "sentence_transformer"
        self.text_encoder = self._initialize_text_encoder()
        self.text_encoder.to(self.device)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.text_dim + self.numerical_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(self.fusion_dropout_rate),
            nn.Linear(hidden_dim, self.output_dim),
            nn.ReLU(),
        )
        self.fusion_mlp.to(self.device)

    def forward(
        self,
        world_state_dict: Dict[str, Any],
        info: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        text_desc = self._generate_state_description(world_state_dict, info)

        if self._text_branch_enabled():
            text_for_encoder = text_desc if text_desc.strip() else "[no textual context]"
            with torch.no_grad():
                text_emb = self.text_encoder.encode(
                    text_for_encoder,
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=False,
                )
        else:
            text_emb = torch.zeros(self.text_dim, dtype=torch.float32, device=self.device)

        numerical_feat = self._extract_numerical_features(world_state_dict, info)
        if not self._numerical_branch_enabled():
            numerical_feat = np.zeros(self.numerical_dim, dtype=np.float32)

        if self.debug:
            self._maybe_log_debug(world_state_dict, info, text_desc)

        numerical_tensor = torch.tensor(numerical_feat, dtype=torch.float32, device=self.device)
        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(0)
        if numerical_tensor.dim() == 1:
            numerical_tensor = numerical_tensor.unsqueeze(0)

        combined = torch.cat([text_emb, numerical_tensor], dim=-1)
        state_vec = self.fusion_mlp(combined).squeeze(0)
        self._record_forward_artifacts(
            world_state_dict=world_state_dict,
            info=info,
            text_desc=text_desc,
            text_emb=text_emb.squeeze(0),
            numerical_feat=numerical_feat,
            state_vec=state_vec,
        )
        return state_vec

    def encode(
        self,
        world_state_dict: Dict[str, Any],
        info: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        return self.forward(world_state_dict, info)

    def get_last_forward_record(self) -> Dict[str, Any]:
        if not self._last_forward_record:
            return {}
        return json.loads(json.dumps(self._last_forward_record))

    def _text_branch_enabled(self) -> bool:
        return self.input_variant not in {"numerical_only", "no_text"}

    def _numerical_branch_enabled(self) -> bool:
        return self.input_variant not in {"text_only", "no_numerical"}

    def _initialize_text_encoder(self) -> Any:
        if not self._text_branch_enabled():
            self.text_encoder_backend = "disabled"
            return _HashTextEncoder(self.text_dim, self.device)

        load_errors: List[str] = []
        model_path = Path(self.text_encoder_model).expanduser()

        if model_path.exists():
            try:
                init_kwargs: Dict[str, Any] = {}
                if self.text_encoder_cache_dir:
                    init_kwargs["cache_folder"] = self.text_encoder_cache_dir
                encoder = SentenceTransformer(str(model_path), **init_kwargs)
                self.text_encoder_backend = "sentence_transformer_local_path"
                logger.info(
                    "Loaded StateEncoder text model from local path `%s`.",
                    str(model_path),
                )
                return encoder
            except Exception as exc:
                load_errors.append(f"local-path: {exc}")

        load_attempts = [self.text_encoder_local_files_only]
        if not self.text_encoder_local_files_only:
            load_attempts.append(True)

        for local_only in load_attempts:
            try:
                init_kwargs: Dict[str, Any] = {}
                if self.text_encoder_cache_dir:
                    init_kwargs["cache_folder"] = self.text_encoder_cache_dir
                try:
                    encoder = SentenceTransformer(
                        self.text_encoder_model,
                        local_files_only=local_only,
                        **init_kwargs,
                    )
                except TypeError:
                    if local_only:
                        raise RuntimeError(
                            "Installed sentence-transformers does not support local_files_only."
                        )
                    encoder = SentenceTransformer(self.text_encoder_model, **init_kwargs)
                self.text_encoder_backend = (
                    "sentence_transformer_local" if local_only else "sentence_transformer"
                )
                if local_only:
                    logger.info(
                        "Loaded StateEncoder text model `%s` in local-files-only mode.",
                        self.text_encoder_model,
                    )
                return encoder
            except Exception as exc:
                mode = "local-only" if local_only else "default"
                load_errors.append(f"{mode}: {exc}")

        if self.text_encoder_fallback == "error":
            raise RuntimeError(
                "Failed to initialize StateEncoder text encoder "
                f"`{self.text_encoder_model}`. Errors: {' | '.join(load_errors)}"
            )

        if self.text_encoder_fallback == "zero":
            self.text_encoder_backend = "zero_fallback"

            class _ZeroTextEncoder:
                def __init__(self, dim: int, device: str) -> None:
                    self.dim = dim
                    self.device = device

                def to(self, device: str) -> "_ZeroTextEncoder":
                    self.device = str(device)
                    return self

                def encode(
                    self,
                    sentences: Any,
                    convert_to_tensor: bool = True,
                    device: Optional[str] = None,
                    show_progress_bar: bool = False,
                ) -> Any:
                    del show_progress_bar
                    target_device = str(device or self.device)
                    if isinstance(sentences, str):
                        tensor = torch.zeros(self.dim, dtype=torch.float32, device=target_device)
                    else:
                        tensor = torch.zeros((len(sentences), self.dim), dtype=torch.float32, device=target_device)
                    if convert_to_tensor:
                        return tensor
                    return tensor.detach().cpu().numpy()

            logger.warning(
                "StateEncoder failed to load text model `%s`; using zero-vector fallback. Errors: %s",
                self.text_encoder_model,
                " | ".join(load_errors),
            )
            return _ZeroTextEncoder(self.text_dim, self.device)

        self.text_encoder_backend = "hash_fallback"
        logger.warning(
            "StateEncoder failed to load text model `%s`; using hash fallback. Errors: %s",
            self.text_encoder_model,
            " | ".join(load_errors),
        )
        return _HashTextEncoder(self.text_dim, self.device)

    def _record_forward_artifacts(
        self,
        world_state_dict: Dict[str, Any],
        info: Optional[Dict[str, Any]],
        text_desc: str,
        text_emb: torch.Tensor,
        numerical_feat: Sequence[float],
        state_vec: torch.Tensor,
    ) -> None:
        task_context = world_state_dict.get("task_context", {}) or {}
        semantic_payload = self._build_semantic_text_payload(task_context, info, world_state_dict)
        self._last_forward_record = {
            "feature_schema_version": STATE_ENCODER_FEATURE_SCHEMA_VERSION,
            "input_variant": self.input_variant,
            "context_variant": self.context_variant,
            "text_encoder_model": self.text_encoder_model,
            "text_encoder_backend": self.text_encoder_backend,
            "state_kind": info.get("state_kind") if isinstance(info, dict) else None,
            "task_family": info.get("task_family") if isinstance(info, dict) else None,
            "episode_id": info.get("episode_id") if isinstance(info, dict) else None,
            "step_count": info.get("step_count") if isinstance(info, dict) else None,
            "world_state_hash": self._hash_payload(world_state_dict),
            "task_context_hash": self._hash_payload(task_context),
            "text_desc": text_desc,
            "text_desc_hash": self._hash_text(text_desc),
            "semantic_text_hash": self._hash_payload(semantic_payload),
            "text_segments": list(self._last_text_segments),
            "feature_names": list(self._last_feature_names),
            "feature_categories": list(self._last_feature_categories),
            "numerical_features": [float(v) for v in numerical_feat],
            "diagnostics": dict(self._last_diagnostics),
            "text_embedding": [float(v) for v in text_emb.detach().cpu().tolist()],
            "state_vector": [float(v) for v in state_vec.detach().cpu().tolist()],
            "category_presence": self._build_category_presence(
                self._last_text_segments,
                self._last_feature_categories,
            ),
        }

    def _build_category_presence(
        self,
        text_segments: Sequence[Dict[str, Any]],
        feature_categories: Sequence[str],
    ) -> Dict[str, Dict[str, int]]:
        category_presence = {
            category: {"text_segments": 0, "numerical_features": 0, "total": 0}
            for category in STATE_ENCODER_CATEGORY_NAMES
        }
        for segment in text_segments:
            category = str(segment.get("category") or "padding/noise")
            if category not in category_presence:
                continue
            category_presence[category]["text_segments"] += 1
            category_presence[category]["total"] += 1
        for category in feature_categories:
            if category not in category_presence:
                continue
            category_presence[category]["numerical_features"] += 1
            category_presence[category]["total"] += 1
        return category_presence

    def _build_text_segments(
        self,
        world_state_dict: Dict[str, Any],
        info: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        task_context = world_state_dict.get("task_context", {}) or {}
        agent_poses = world_state_dict.get("agent_poses", {}) or {}
        agent_holdings = world_state_dict.get("agent_holdings", {}) or {}
        object_positions = world_state_dict.get("object_positions", {}) or {}
        segments: List[Dict[str, Any]] = []

        def _append(category: str, text: str, key: Optional[str] = None) -> None:
            clean = str(text or "").strip()
            if not clean:
                return
            segments.append({"category": category, "key": key or category, "text": clean})

        task_instruction = task_context.get("instruction") or ""
        if not task_instruction and info:
            task_instruction = info.get("task_instruction", "")
        if isinstance(task_instruction, str) and task_instruction.strip():
            trimmed = task_instruction.strip()
            if len(trimmed) > 200:
                trimmed = trimmed[:200] + "..."
            _append("instruction", f"Task: {trimmed}", "instruction")

        task_objects = list(
            dict.fromkeys(
                str(x)
                for x in (
                    list(task_context.get("objects", []) or [])
                    + list(task_context.get("object_handles", []) or [])
                )
                if x
            )
        )
        task_receptacles = list(
            dict.fromkeys(
                str(x)
                for x in (
                    list(task_context.get("receptacles", []) or [])
                    + list(task_context.get("receptacle_handles", []) or [])
                    + list(task_context.get("floor_target_rooms", []) or [])
                )
                if x
            )
        )
        task_rooms = [str(x) for x in (task_context.get("rooms", []) or []) if x]
        task_entities = [str(x) for x in (task_context.get("entities", []) or []) if x]
        source_object_classes = [
            str(x) for x in (task_context.get("source_object_classes", []) or []) if x
        ]
        source_furniture_names = [
            str(x) for x in (task_context.get("source_furniture_names", []) or []) if x
        ]
        source_allowed_regions = [
            str(x) for x in (task_context.get("source_allowed_regions", []) or []) if x
        ]

        if task_objects:
            _append("task-object grounding", f"Task objects: {', '.join(task_objects[:5])}.", "task_objects")
        elif source_object_classes:
            _append(
                "task-object grounding",
                f"Source object classes: {', '.join(sorted(set(source_object_classes))[:5])}.",
                "source_object_classes",
            )

        if task_receptacles:
            _append(
                "target/receptacle grounding",
                f"Target receptacles: {', '.join(task_receptacles[:5])}.",
                "task_receptacles",
            )
        elif source_furniture_names:
            _append(
                "target/receptacle grounding",
                f"Source furniture targets: {', '.join(sorted(set(source_furniture_names))[:5])}.",
                "source_furniture_names",
            )

        if task_rooms:
            _append("room grounding", f"Task rooms: {', '.join(task_rooms[:5])}.", "task_rooms")
        elif source_allowed_regions:
            _append(
                "room grounding",
                f"Source allowed regions: {', '.join(sorted(set(source_allowed_regions))[:5])}.",
                "source_allowed_regions",
            )

        extra_entities = [name for name in task_entities if name not in task_objects]
        if extra_entities:
            _append("task-object grounding", f"Task entities: {', '.join(extra_entities[:5])}.", "task_entities")

        object_state_summary = self._summarize_object_state_targets(task_context)
        if object_state_summary:
            _append(
                "object-state affordance",
                f"Task affordances: {object_state_summary}.",
                "object_state_targets",
            )

        satisfied, total = self._extract_proposition_progress(info, task_context)
        if total > 0:
            _append(
                "progress/outcome",
                f"Completed propositions: {satisfied}/{total}.",
                "completed_propositions",
            )

        planning_step = self._extract_planning_step(info)
        if planning_step is not None:
            _append("progress/outcome", f"Planning step: {planning_step}.", "planning_step")

        recent_action = self._extract_recent_action(info)
        if recent_action:
            _append("progress/outcome", f"Recent action: {recent_action}.", "recent_action")

        recent_feedback = self._extract_recent_feedback(info)
        if recent_feedback:
            _append("progress/outcome", f"Recent feedback: {recent_feedback}.", "recent_feedback")

        for agent_id in sorted(agent_poses.keys()):
            pose_data = agent_poses.get(agent_id, {}) or {}
            pos = pose_data.get("position", [0.0, 0.0, 0.0])
            yaw = float(pose_data.get("yaw", 0.0))
            direction = self._yaw_to_direction(yaw)
            _append(
                "agent kinematics",
                (
                    f"Agent {agent_id} is at position "
                    f"({float(pos[0]):.2f}, {float(pos[1]):.2f}, {float(pos[2]):.2f}) "
                    f"facing {direction}."
                ),
                f"agent_pose_{agent_id}",
            )
            held_obj = agent_holdings.get(agent_id, {}).get("held_object")
            if held_obj:
                _append(
                    "holding/manipulation",
                    f"Agent {agent_id} holds {held_obj}.",
                    f"agent_holding_{agent_id}",
                )

        candidate_names = task_objects or task_entities or list(object_positions.keys())
        if self.context_variant == "task_only":
            candidate_names = task_objects or task_entities
        for obj_name in candidate_names[:5]:
            obj_data = object_positions.get(obj_name, {}) or {}
            if "position" not in obj_data:
                continue
            pos = obj_data.get("position", [0.0, 0.0, 0.0])
            parent = obj_data.get("parent", "unknown")
            _append(
                "object spatial state",
                (
                    f"{obj_name} is at "
                    f"({float(pos[0]):.2f}, {float(pos[1]):.2f}, {float(pos[2]):.2f}) "
                    f"on {parent}."
                ),
                f"object_{obj_name}",
            )
        return segments

    def _generate_state_description(
        self,
        world_state_dict: Dict[str, Any],
        info: Optional[Dict[str, Any]] = None,
    ) -> str:
        segments = self._build_text_segments(world_state_dict, info)
        self._last_text_segments = segments
        description = " ".join(segment["text"] for segment in segments)
        if len(description) > 512:
            description = description[:512] + "..."
        return description

    def _extract_numerical_features(
        self,
        world_state_dict: Dict[str, Any],
        info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        features: List[float] = []
        feature_names: List[str] = []

        def _append(value: float, name: str) -> None:
            features.append(float(value))
            feature_names.append(name)

        agent_poses = world_state_dict.get("agent_poses", {}) or {}
        agent_ids = sorted(agent_poses.keys())
        primary_id = agent_ids[0] if agent_ids else None
        secondary_id = agent_ids[1] if len(agent_ids) > 1 else None

        def _append_agent_pose(agent_id: Optional[int], prefix: str) -> None:
            if agent_id is not None and agent_id in agent_poses:
                pose = agent_poses[agent_id] or {}
                pos = pose.get("position", [0.0, 0.0, 0.0])
                yaw = pose.get("yaw", 0.0)
                _append(float(pos[0]), f"{prefix}pos_x")
                _append(float(pos[1]), f"{prefix}pos_y")
                _append(float(pos[2]), f"{prefix}pos_z")
                _append(float(yaw), f"{prefix}yaw")
            else:
                _append(0.0, f"{prefix}pos_x")
                _append(0.0, f"{prefix}pos_y")
                _append(0.0, f"{prefix}pos_z")
                _append(0.0, f"{prefix}yaw")

        _append_agent_pose(primary_id, "agent_")
        _append_agent_pose(secondary_id, "agent2_")

        task_context = world_state_dict.get("task_context", {}) or {}
        object_positions = world_state_dict.get("object_positions", {}) or {}
        furniture_positions = world_state_dict.get("furniture_positions", {}) or {}
        agent_holdings = world_state_dict.get("agent_holdings", {}) or {}

        task_progress = float(info.get("task_percent_complete", 0.0)) if info else 0.0
        _append(task_progress, "task_progress")

        satisfied, total = self._extract_proposition_progress(info, task_context)
        proposition_fraction = float(satisfied / max(total, 1)) if total > 0 else 0.0
        _append(proposition_fraction, "proposition_satisfied_fraction")

        distance_norm = self.distance_norm
        count_norm = self.count_norm
        action_history_cap = self.action_history_cap

        def _norm_dist(value: float) -> float:
            if value < 0:
                return 0.0
            return float(min(value / distance_norm, 1.0))

        def _norm_count(value: float) -> float:
            return float(min(float(value) / count_norm, 1.0))

        def _norm_action_history(value: float) -> float:
            return float(min(float(value) / action_history_cap, 1.0))

        diagnostics: Dict[str, Any] = {
            "distance_norm": float(distance_norm),
            "count_norm": float(count_norm),
            "action_history_cap": float(action_history_cap),
        }

        def _find_entity(
            name: str, collection: Dict[str, Any]
        ) -> Optional[Dict[str, Any]]:
            direct = collection.get(name)
            if isinstance(direct, dict):
                return direct
            for entity_name, entity_data in collection.items():
                aliases = entity_data.get("aliases", []) if entity_data else []
                if name == str(entity_name) or name in {str(v) for v in aliases}:
                    return entity_data
            return None

        def _get_pos(name: str) -> Optional[np.ndarray]:
            for collection in (object_positions, furniture_positions):
                entity_data = _find_entity(name, collection)
                if entity_data and "position" in entity_data:
                    return np.asarray(entity_data["position"], dtype=np.float32)
            return None

        task_objects = [str(x) for x in (task_context.get("objects", []) or []) if x]
        task_object_handles = [
            str(x) for x in (task_context.get("object_handles", []) or []) if x
        ]
        task_entities = [str(x) for x in (task_context.get("entities", []) or []) if x]
        task_receptacles = [str(x) for x in (task_context.get("receptacles", []) or []) if x]
        task_receptacle_handles = [
            str(x) for x in (task_context.get("receptacle_handles", []) or []) if x
        ]
        task_rooms = [str(x) for x in (task_context.get("rooms", []) or []) if x]
        source_furniture_names = [
            str(x) for x in (task_context.get("source_furniture_names", []) or []) if x
        ]
        source_allowed_regions = [
            str(x) for x in (task_context.get("source_allowed_regions", []) or []) if x
        ]

        canonical_task_objects = task_object_handles or task_objects
        candidate_objects = list(dict.fromkeys(canonical_task_objects + task_entities))
        task_obj_positions = [pos for name in candidate_objects if (pos := _get_pos(name)) is not None]
        task_target_anchors: List[str] = []
        for anchor_name in (
            list(task_receptacles)
            + list(task_receptacle_handles)
            + list(source_furniture_names)
            + list(task_rooms)
            + list(source_allowed_regions)
            + list(task_context.get("floor_target_rooms", []) or [])
        ):
            if anchor_name and anchor_name not in task_target_anchors:
                task_target_anchors.append(anchor_name)

        target_positions: List[np.ndarray] = []
        resolved_target_anchors: List[str] = []
        missing_target_anchors: List[str] = []
        for anchor_name in task_target_anchors:
            anchor_pos = _get_pos(anchor_name)
            if anchor_pos is None:
                missing_target_anchors.append(anchor_name)
                continue
            target_positions.append(anchor_pos)
            resolved_target_anchors.append(anchor_name)

        diagnostics["target_anchor_total"] = len(task_target_anchors)
        diagnostics["target_anchor_resolved_count"] = len(resolved_target_anchors)
        diagnostics["target_anchor_missing_count"] = len(missing_target_anchors)
        diagnostics["target_anchor_missing"] = bool(missing_target_anchors)
        diagnostics["missing_target_anchors"] = missing_target_anchors[:8]
        diagnostics["resolved_target_anchors"] = resolved_target_anchors[:8]

        agent_pos = None
        if primary_id is not None and primary_id in agent_poses:
            agent_pos = np.asarray(
                agent_poses[primary_id].get("position", [0.0, 0.0, 0.0]),
                dtype=np.float32,
            )

        if agent_pos is not None and task_obj_positions:
            dists = [float(np.linalg.norm(agent_pos - p)) for p in task_obj_positions]
            min_task_obj_dist = min(dists)
            mean_task_obj_dist = float(sum(dists) / len(dists))
        else:
            # Unknown/unobserved is encoded as zero distance plus a zero
            # visibility fraction below; it is no longer conflated with far.
            min_task_obj_dist = 0.0
            mean_task_obj_dist = 0.0

        if agent_pos is not None and target_positions:
            target_dists = [float(np.linalg.norm(agent_pos - p)) for p in target_positions]
            min_agent_target_dist = min(target_dists)
        else:
            min_agent_target_dist = 0.0

        if task_obj_positions and target_positions:
            obj_target_dists = [
                min(float(np.linalg.norm(obj_pos - tgt_pos)) for tgt_pos in target_positions)
                for obj_pos in task_obj_positions
            ]
            min_obj_target_dist = min(obj_target_dists)
            mean_obj_target_dist = float(sum(obj_target_dists) / len(obj_target_dists))
        else:
            min_obj_target_dist = 0.0
            mean_obj_target_dist = 0.0

        if canonical_task_objects and task_target_anchors:
            target_set = set(task_target_anchors)
            at_target = 0
            for name in canonical_task_objects:
                object_data = _find_entity(name, object_positions) or {}
                parent = object_data.get("parent")
                if parent in target_set:
                    at_target += 1
            fraction_at_target = float(
                at_target / max(len(canonical_task_objects), 1)
            )
        else:
            fraction_at_target = 0.0

        _append(_norm_dist(min_task_obj_dist), "min_task_obj_dist_norm")
        _append(_norm_dist(mean_task_obj_dist), "mean_task_obj_dist_norm")
        _append(_norm_dist(min_agent_target_dist), "min_agent_target_dist_norm")
        _append(_norm_dist(min_obj_target_dist), "min_obj_target_dist_norm")
        _append(_norm_dist(mean_obj_target_dist), "mean_obj_target_dist_norm")
        _append(fraction_at_target, "task_object_fraction_at_target")
        _append(_norm_count(len(canonical_task_objects)), "task_object_count_norm")
        _append(_norm_count(len(task_target_anchors)), "task_target_count_norm")
        _append(_norm_count(len(task_entities)), "task_entity_count_norm")
        _append(_norm_count(len(task_rooms)), "task_room_count_norm")
        _append(_norm_count(total), "task_proposition_count_norm")

        visible_task_objects = [
            name
            for name in canonical_task_objects
            if _find_entity(name, object_positions) is not None
        ]
        visible_task_targets = list(resolved_target_anchors)
        task_object_visible_fraction = float(
            len(visible_task_objects) / max(len(canonical_task_objects), 1)
        )
        target_visibility_denominator = len(task_target_anchors)
        task_target_visible_fraction = float(
            len(visible_task_targets) / max(target_visibility_denominator, 1)
        )
        diagnostics["task_target_visibility_unknown"] = bool(
            target_visibility_denominator > 0 and not resolved_target_anchors
        )
        _append(task_object_visible_fraction, "task_object_visible_fraction")
        _append(task_target_visible_fraction, "task_target_visible_fraction")
        _append(_norm_count(len(visible_task_objects)), "task_object_visible_count_norm")
        _append(_norm_count(len(visible_task_targets)), "task_target_visible_count_norm")

        held_objects = [
            str(data.get("held_object"))
            for data in agent_holdings.values()
            if data.get("held_object")
        ]
        task_object_identity = set(task_objects) | set(task_object_handles)
        for name in canonical_task_objects:
            entity_data = _find_entity(name, object_positions) or {}
            task_object_identity.update(
                str(value) for value in entity_data.get("aliases", []) or []
            )
        held_task_objects = [
            name for name in held_objects if name in task_object_identity
        ]
        any_agent_holding = 1.0 if held_objects else 0.0
        any_agent_holding_task = 1.0 if held_task_objects else 0.0
        primary_held = (
            agent_holdings.get(primary_id, {}).get("held_object") if primary_id is not None else None
        )
        primary_holding = 1.0 if primary_held else 0.0
        primary_holding_task = (
            1.0 if primary_held and primary_held in task_object_identity else 0.0
        )
        held_task_fraction = float(
            len(held_task_objects) / max(len(canonical_task_objects), 1)
        )

        _append(any_agent_holding, "any_agent_holding")
        _append(any_agent_holding_task, "any_agent_holding_task_object")
        _append(primary_holding, "primary_agent_holding")
        _append(primary_holding_task, "primary_agent_holding_task_object")
        _append(held_task_fraction, "held_task_object_fraction")

        if agent_pos is not None and target_positions and primary_held:
            hold_dists = [float(np.linalg.norm(agent_pos - p)) for p in target_positions]
            min_hold_target_dist = min(hold_dists)
        else:
            min_hold_target_dist = 0.0
        _append(_norm_dist(min_hold_target_dist), "min_agent_target_dist_when_holding_norm")

        action_history = info.get("action_history", []) if isinstance(info, dict) else []
        action_history_count = (
            int(info.get("action_history_count", len(action_history)))
            if isinstance(info, dict)
            else len(action_history)
        )
        diagnostics["action_history_count"] = action_history_count
        _append(_norm_action_history(action_history_count), "action_history_count_norm")

        current_dim = len(features)
        if current_dim < self.numerical_dim:
            for pad_index in range(self.numerical_dim - current_dim):
                _append(0.0, f"pad_{pad_index}")
        elif current_dim > self.numerical_dim:
            features = features[: self.numerical_dim]
            feature_names = feature_names[: self.numerical_dim]

        self._last_feature_names = list(feature_names)
        self._last_feature_values = [float(v) for v in features]
        self._last_feature_categories = [
            classify_feature_category(name) for name in self._last_feature_names
        ]
        self._last_diagnostics = diagnostics
        return np.asarray(features, dtype=np.float32)

    def _extract_proposition_progress(
        self,
        info: Optional[Dict[str, Any]],
        task_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, int]:
        outcome = proposition_outcome_from_info(info, task_context)
        return outcome.satisfied_count, outcome.total

    def _extract_planning_step(self, info: Optional[Dict[str, Any]]) -> Optional[int]:
        if not isinstance(info, dict) or "planning_step_count" not in info:
            return None
        try:
            return int(info.get("planning_step_count", 0))
        except Exception:
            return None

    def _extract_recent_action(self, info: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(info, dict):
            return None
        action_history = info.get("action_history", []) or []
        if not action_history:
            return None
        last_action = action_history[-1]
        if isinstance(last_action, dict):
            action_payload = last_action.get("action")
            if isinstance(action_payload, (list, tuple)):
                if len(action_payload) >= 2:
                    return f"{action_payload[0]}[{action_payload[1]}]"
                if len(action_payload) == 1:
                    return str(action_payload[0])
            if action_payload is not None:
                return str(action_payload)
        action = getattr(last_action, "action", None)
        if isinstance(action, tuple) and len(action) >= 2:
            return f"{action[0]}[{action[1]}]"
        if action is not None:
            return str(action)
        to_string = getattr(last_action, "to_string", None)
        if callable(to_string):
            try:
                return str(to_string())
            except Exception:
                return None
        return str(last_action)

    def _extract_recent_feedback(self, info: Optional[Dict[str, Any]]) -> Optional[str]:
        if not isinstance(info, dict):
            return None
        action_history = info.get("action_history", []) or []
        if action_history:
            if isinstance(action_history[-1], dict):
                response = action_history[-1].get("response")
                if response:
                    return str(response)
            response = getattr(action_history[-1], "response", None)
            if response:
                return str(response)
        action_responses = info.get("action_responses", {}) or {}
        if isinstance(action_responses, dict):
            for _, value in sorted(action_responses.items()):
                if value:
                    return str(value)
        return None

    def _summarize_object_state_targets(self, task_context: Dict[str, Any]) -> str:
        summaries: List[str] = []
        source_initial_state = task_context.get("source_initial_state", []) or []
        if isinstance(source_initial_state, list):
            for state_element in source_initial_state:
                if not isinstance(state_element, dict):
                    continue
                object_states = state_element.get("object_states", {}) or {}
                if not object_states:
                    continue
                target_name = (
                    ", ".join(state_element.get("furniture_names", [])[:2])
                    or ", ".join(state_element.get("allowed_regions", [])[:2])
                    or ", ".join(state_element.get("object_classes", [])[:2])
                    or "task target"
                )
                for affordance, value in object_states.items():
                    summaries.append(f"{affordance}={bool(value)} on {target_name}")
        if summaries:
            return "; ".join(summaries[:4])

        episode_object_states = task_context.get("episode_object_states", {}) or {}
        if isinstance(episode_object_states, dict):
            for affordance, handle_map in episode_object_states.items():
                if not isinstance(handle_map, dict):
                    continue
                true_count = sum(1 for value in handle_map.values() if bool(value))
                false_count = sum(1 for value in handle_map.values() if not bool(value))
                if true_count:
                    summaries.append(f"{affordance}=True x{true_count}")
                if false_count:
                    summaries.append(f"{affordance}=False x{false_count}")
        return "; ".join(summaries[:4])

    def _build_semantic_text_payload(
        self,
        task_context: Dict[str, Any],
        info: Optional[Dict[str, Any]],
        world_state_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        instruction = task_context.get("instruction") or ""
        if not instruction and info:
            instruction = info.get("task_instruction") or ""
        if isinstance(instruction, str):
            instruction = instruction.strip()
            if len(instruction) > 200:
                instruction = instruction[:200] + "..."
        else:
            instruction = ""

        def _normalize_list(values: Sequence[Any]) -> List[str]:
            return sorted(str(value) for value in values if value is not None)

        agent_holdings = world_state_dict.get("agent_holdings", {}) or {}
        held_objects = sorted(
            {
                data.get("held_object")
                for data in agent_holdings.values()
                if data.get("held_object")
            }
        )
        satisfied, total = self._extract_proposition_progress(info, task_context)
        return {
            "instruction": instruction,
            "objects": _normalize_list(
                list(task_context.get("objects", []) or [])
                + list(task_context.get("object_handles", []) or [])
            ),
            "receptacles": _normalize_list(
                list(task_context.get("receptacles", []) or [])
                + list(task_context.get("receptacle_handles", []) or [])
                + list(task_context.get("floor_target_rooms", []) or [])
            ),
            "entities": _normalize_list(task_context.get("entities", []) or []),
            "rooms": _normalize_list(task_context.get("rooms", []) or []),
            "proposition_count": int(task_context.get("proposition_count") or 0),
            "proposition_satisfied": int(satisfied),
            "proposition_total": int(total),
            "held_objects": held_objects,
            "recent_action": self._extract_recent_action(info),
            "recent_feedback": self._extract_recent_feedback(info),
            "planning_step": self._extract_planning_step(info),
            "object_state_targets": self._summarize_object_state_targets(task_context),
        }

    def _maybe_log_debug(
        self,
        world_state_dict: Dict[str, Any],
        info: Optional[Dict[str, Any]],
        text_desc: str,
    ) -> None:
        if not self.debug or not self.debug_dir or self.debug_max_entries == 0:
            return

        state_kind = info.get("state_kind") if info else None
        if state_kind == "state" and not self.debug_log_state:
            return
        if state_kind == "next_state" and not self.debug_log_next_state:
            return

        self._debug_step += 1
        if not self.debug_log_on_change and self._debug_step % self.debug_frequency != 0:
            return

        filepath = self._get_debug_filepath(info)
        if not filepath:
            return

        episode_id = info.get("episode_id") if info else None
        episode_filename = info.get("episode_filename") if info else None
        step_count = info.get("step_count") if info else None

        log_key = (episode_id, episode_filename)
        if log_key != self._current_log_key:
            self._reset_debug_state(log_key)

        task_context = world_state_dict.get("task_context", {}) or {}
        task_context_hash = self._hash_payload(task_context)
        feature_map = {
            name: float(value)
            for name, value in zip(self._last_feature_names, self._last_feature_values)
        }
        text_hash = self._hash_text(text_desc)
        semantic_text_hash = self._hash_payload(
            self._build_semantic_text_payload(task_context, info, world_state_dict)
        )
        signature_text_hash = (
            text_hash if self.debug_signature_mode == "full" else semantic_text_hash
        )
        signature_features = (
            list(self._last_feature_values)
            if self.debug_signature_mode == "full"
            else self._get_signature_feature_values(feature_map)
        )
        signature = self._build_debug_signature(
            signature_text_hash,
            state_kind,
            task_context_hash,
            signature_features,
        )
        signature_hash = self._hash_signature(signature)
        self._last_state_kind = state_kind

        if self.debug_log_on_change and signature == self._last_logged_signature:
            self._repeat_count += 1
            if step_count is not None:
                if self._repeat_start_step is None:
                    self._repeat_start_step = step_count
                self._repeat_last_step = step_count
            return

        if self._debug_entries >= self.debug_max_entries:
            return

        if not self._meta_logged_for_key:
            meta_payload = {
                "record_type": "meta",
                "feature_schema_version": STATE_ENCODER_FEATURE_SCHEMA_VERSION,
                "episode_id": episode_id,
                "episode_filename": episode_filename,
                "text_dim": self.text_dim,
                "numerical_dim": self.numerical_dim,
                "text_encoder_model": self.text_encoder_model,
                "text_encoder_backend": self.text_encoder_backend,
                "feature_names": self._last_feature_names,
                "feature_categories": self._last_feature_categories,
                "category_names": list(STATE_ENCODER_CATEGORY_NAMES),
                "key_feature_names": self.debug_key_feature_names,
                "signature_mode": self.debug_signature_mode,
                "signature_feature_names": self.debug_signature_feature_names,
                "change_threshold": self.debug_change_threshold,
                "input_variant": self.input_variant,
                "context_variant": self.context_variant,
                "distance_norm": self.distance_norm,
                "count_norm": self.count_norm,
                "action_history_cap": self.action_history_cap,
            }
            if not self._write_debug_payload(filepath, meta_payload):
                return
            self._meta_logged_for_key = True

        if not self.debug_log_task_context_once or task_context_hash != self._last_task_context_hash:
            task_payload = {
                "record_type": "task_context",
                "episode_id": episode_id,
                "episode_filename": episode_filename,
                "task_context_hash": task_context_hash,
                "task_context": task_context,
            }
            if not self._write_debug_payload(filepath, task_payload):
                return
            self._last_task_context_hash = task_context_hash

        if self._repeat_count > 0:
            repeat_payload = {
                "record_type": "repeat",
                "episode_id": episode_id,
                "episode_filename": episode_filename,
                "state_kind": state_kind,
                "signature_hash": self._last_signature_hash,
                "repeat_count": self._repeat_count,
                "repeat_start_step": self._repeat_start_step,
                "repeat_end_step": self._repeat_last_step,
            }
            if not self._write_debug_payload(filepath, repeat_payload):
                return
            self._repeat_count = 0
            self._repeat_start_step = None
            self._repeat_last_step = None

        key_features = {
            name: feature_map[name]
            for name in self.debug_key_feature_names
            if name in feature_map
        }

        feature_snapshot = None
        feature_delta = None
        if self.debug_log_delta and self._last_logged_feature_values:
            feature_delta = {}
            for name, curr, prev in zip(
                self._last_feature_names,
                self._last_feature_values,
                self._last_logged_feature_values,
            ):
                delta = abs(curr - prev)
                if self.debug_change_threshold > 0:
                    if delta < self.debug_change_threshold:
                        continue
                elif delta == 0:
                    continue
                feature_delta[name] = [float(prev), float(curr)]
        elif self.debug_log_full_features_first or self._last_logged_feature_values is None:
            feature_snapshot = feature_map

        trimmed_text = text_desc
        if self.debug_text_max_len > 0 and len(trimmed_text) > self.debug_text_max_len:
            trimmed_text = trimmed_text[: self.debug_text_max_len] + "..."

        include_text_desc = signature_text_hash != self._last_logged_text_hash
        payload = {
            "record_type": "state",
            "episode_id": episode_id,
            "episode_filename": episode_filename,
            "step_count": step_count,
            "state_kind": state_kind,
            "signature_hash": signature_hash,
            "text_desc_hash": text_hash,
            "semantic_text_hash": semantic_text_hash,
            "signature_text_hash": signature_text_hash,
            "text_desc": trimmed_text if include_text_desc else None,
            "task_context_hash": task_context_hash,
            "key_features": key_features,
            "feature_delta": feature_delta,
            "feature_delta_count": len(feature_delta) if feature_delta else 0,
            "feature_snapshot": feature_snapshot,
            "text_segments": self._last_text_segments,
            "diagnostics": self._last_diagnostics,
        }
        if not self._write_debug_payload(filepath, payload):
            return

        self._last_logged_signature = signature
        self._last_signature_hash = signature_hash
        self._last_logged_feature_values = list(self._last_feature_values)
        self._last_logged_text_hash = signature_text_hash

    def _get_signature_feature_values(self, feature_map: Dict[str, float]) -> List[float]:
        return [float(feature_map.get(name, 0.0)) for name in self.debug_signature_feature_names]

    def _build_debug_signature(
        self,
        signature_text_hash: str,
        state_kind: Optional[str],
        task_context_hash: Optional[str],
        signature_feature_values: Optional[List[float]],
    ) -> tuple:
        values = signature_feature_values or []
        if self.debug_change_threshold > 0:
            quantized = tuple(
                int(round(value / self.debug_change_threshold))
                for value in values
            )
        else:
            quantized = tuple(values)
        return (state_kind, signature_text_hash, task_context_hash or "", quantized)

    def _hash_payload(self, payload: Dict[str, Any]) -> str:
        try:
            dumped = json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)
        except Exception:
            dumped = str(payload)
        return hashlib.md5(dumped.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_signature(signature: tuple) -> str:
        return hashlib.md5(repr(signature).encode("utf-8")).hexdigest()

    def _reset_debug_state(self, log_key: tuple) -> None:
        self._current_log_key = log_key
        self._meta_logged_for_key = False
        self._last_task_context_hash = None
        self._last_logged_signature = None
        self._last_signature_hash = None
        self._last_logged_feature_values = None
        self._last_logged_text_hash = None
        self._repeat_count = 0
        self._repeat_start_step = None
        self._repeat_last_step = None
        self._last_state_kind = None

    def _write_debug_payload(self, filepath: str, payload: Dict[str, Any]) -> bool:
        if self._debug_entries >= self.debug_max_entries:
            return False
        self._write_debug_line(filepath, payload)
        self._debug_entries += 1
        return True

    def flush_debug(self, info: Optional[Dict[str, Any]] = None) -> None:
        if not self.debug or not self.debug_dir:
            return
        if self.debug_max_entries == 0 or self._repeat_count <= 0:
            return

        filepath = self._get_debug_filepath(info)
        if not filepath:
            return

        episode_id = info.get("episode_id") if info else None
        episode_filename = info.get("episode_filename") if info else None
        state_kind = self._last_state_kind or (info.get("state_kind") if info else None)
        repeat_payload = {
            "record_type": "repeat",
            "episode_id": episode_id,
            "episode_filename": episode_filename,
            "state_kind": state_kind,
            "signature_hash": self._last_signature_hash,
            "repeat_count": self._repeat_count,
            "repeat_start_step": self._repeat_start_step,
            "repeat_end_step": self._repeat_last_step,
        }
        if not self._write_debug_payload(filepath, repeat_payload):
            return
        self._repeat_count = 0
        self._repeat_start_step = None
        self._repeat_last_step = None

    def _get_debug_filepath(self, info: Optional[Dict[str, Any]]) -> Optional[str]:
        if not self.debug_dir:
            return None
        episode_id = None
        episode_filename = None
        if info:
            episode_id = info.get("episode_id")
            episode_filename = info.get("episode_filename")
        if episode_id is None:
            filename = "state_encoder-debug.jsonl"
        else:
            safe_name = str(episode_filename or "episode")
            safe_name = safe_name.replace("\\", "_").replace("/", "_")
            filename = f"state_encoder-episode_{episode_id}_{safe_name}.jsonl"
        return os.path.join(self.debug_dir, filename)

    @staticmethod
    def _write_debug_line(filepath: str, payload: Dict[str, Any]) -> None:
        try:
            with open(filepath, "a", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, default=str)
                handle.write("\n")
        except Exception as exc:
            logger.debug("StateEncoder debug log failed: %s", exc)

    @staticmethod
    def _yaw_to_direction(yaw: float) -> str:
        yaw = float(yaw) % (2 * np.pi)
        degrees = float(np.degrees(yaw))
        if 45 <= degrees < 135:
            return "east"
        if 135 <= degrees < 225:
            return "south"
        if 225 <= degrees < 315:
            return "west"
        return "north"
