"""
State Encoder for A2C Critic

Encodes world_state_dict from perception into fixed-dimensional vectors using:
1. Pre-trained sentence transformers for text embeddings
2. Numerical feature extraction from positions and metrics
3. MLP fusion layer
"""

import json
import os
import hashlib
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


class StateEncoder(nn.Module):
    """
    Encodes perception world_state_dict into fixed-dimensional state vectors.

    Architecture:
        world_state_dict → [Text Description + Numerical Features]
                        → [SentenceTransformer + Feature Extraction]
                        → [MLP Fusion]
                        → state_vector (128-dim)
    """

    def __init__(
        self,
        text_dim: int = 384,
        numerical_dim: int = 32,
        hidden_dim: int = 256,
        output_dim: int = 128,
        text_encoder_model: str = 'all-MiniLM-L6-v2',
        device: str = 'cpu',
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
        debug_signature_feature_names: Optional[List[str]] = None
    ):
        """
        Initialize StateEncoder.

        Args:
            text_dim: Dimension of text embeddings (384 for all-MiniLM-L6-v2)
            numerical_dim: Dimension of numerical features
            hidden_dim: Hidden dimension for MLP
            output_dim: Output state vector dimension
            text_encoder_model: Pre-trained sentence transformer model
            device: Device to run on ('cpu' or 'cuda')
        """
        super().__init__()

        self.text_dim = text_dim
        self.numerical_dim = numerical_dim
        self.output_dim = output_dim
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'    # default
        self.device = device    # override default
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
        self.debug_signature_mode = debug_signature_mode if debug_signature_mode in ("full", "semantic") else "full"
        self.debug_signature_feature_names = debug_signature_feature_names or [
            "task_progress",
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
        ]
        self._debug_entries = 0
        self._debug_step = 0
        self._last_feature_names: List[str] = []
        self._last_feature_values: List[float] = []
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

        # Load pre-trained text encoder
        try:
            self.text_encoder = SentenceTransformer(text_encoder_model)
            self.text_encoder.to(device)
            logger.info(f"Loaded sentence transformer: {text_encoder_model}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            raise

        # MLP fusion network
        self.fusion_mlp = nn.Sequential(
            nn.Linear(text_dim + numerical_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

        self.fusion_mlp.to(device)

    def forward(self, world_state_dict: Dict[str, Any], info: Optional[Dict[str, Any]] = None):
        """
        Encode world state into fixed-dimensional vector.

        Args:
            world_state_dict: Dictionary with keys:
                - 'agent_poses': {agent_id: {'position': [x,y,z], 'rotation': [...], 'yaw': yaw_radians}}
                - 'object_positions': {obj_name: {'position': [x,y,z], 'parent': parent_name}}
                - 'furniture_positions': {furn_name: {'position': [x,y,z]}}
                - 'task_context': {'instruction': str, 'objects': [...], 'receptacles': [...],
                                   'entities': [...], 'rooms': [...], 'proposition_count': int}
            info: Optional info dict containing task_instruction, task_percent_complete, etc.

        Returns:
            state_vector: torch.Tensor of shape [output_dim]
        """
        # Generate text description
        text_desc = self._generate_state_description(world_state_dict, info)

        # Get text embedding
        with torch.no_grad():
            text_emb = self.text_encoder.encode(
                text_desc,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False  # Disable progress bar
            )

        # Extract numerical features
        numerical_feat = self._extract_numerical_features(world_state_dict, info)
        if self.debug:
            self._maybe_log_debug(world_state_dict, info, text_desc)
        numerical_feat = torch.tensor(numerical_feat, dtype=torch.float32, device=self.device)

        # Ensure correct dimensions
        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(0)  # [1, text_dim]
        if numerical_feat.dim() == 1:
            numerical_feat = numerical_feat.unsqueeze(0)  # [1, numerical_dim]

        # Concatenate and fuse
        combined = torch.cat([text_emb, numerical_feat], dim=-1)  # [1, text_dim + numerical_dim]
        state_vec = self.fusion_mlp(combined).squeeze(0)  # [output_dim]

        return state_vec

    def encode(self, world_state_dict: Dict[str, Any], info: Optional[Dict[str, Any]] = None):
        """Alias for forward() to match Critic interface."""
        return self.forward(world_state_dict, info)

    def _generate_state_description(
        self,
        world_state_dict: Dict[str, Any],
        info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate natural language description of world state.

        Args:
            world_state_dict: World state dictionary

        Returns:
            description: Natural language description string
        """
        desc_parts = []

        # Task instruction (if available)
        task_context = world_state_dict.get("task_context", {})
        task_instruction = task_context.get("instruction") or ""
        if not task_instruction and info:
            task_instruction = info.get("task_instruction", "")
        if isinstance(task_instruction, str) and task_instruction:
            trimmed = task_instruction.strip()
            if len(trimmed) > 200:
                trimmed = trimmed[:200] + "..."
            desc_parts.append(f"Task: {trimmed}")

        task_objects = task_context.get("objects", []) or []
        task_receptacles = task_context.get("receptacles", []) or []
        if task_objects:
            desc_parts.append(f"Task objects: {', '.join(task_objects[:3])}.")
        if task_receptacles:
            desc_parts.append(f"Target receptacles: {', '.join(task_receptacles[:3])}.")

        # Agent information
        agent_poses = world_state_dict.get('agent_poses', {})
        agent_holdings = world_state_dict.get("agent_holdings", {}) or {}
        for agent_id, pose_data in agent_poses.items():
            pos = pose_data.get('position', [0, 0, 0])
            yaw = pose_data.get('yaw', 0)

            # Convert yaw to cardinal direction
            direction = self._yaw_to_direction(yaw)

            desc_parts.append(
                f"Agent {agent_id} is at position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) facing {direction}."
            )

            held_obj = agent_holdings.get(agent_id, {}).get("held_object")
            if held_obj:
                desc_parts.append(f"Agent {agent_id} holds {held_obj}.")

        # Object information (task-relevant first, limit to 5)
        object_positions = world_state_dict.get('object_positions', {})
        if object_positions:
            task_entities = task_context.get("entities", []) or []
            candidate_names = task_objects if task_objects else task_entities
            if candidate_names:
                candidate_names = [n for n in candidate_names if n in object_positions]
            else:
                candidate_names = list(object_positions.keys())

            for obj_name in candidate_names[:5]:
                obj_data = object_positions.get(obj_name, {})
                pos = obj_data.get('position', [0, 0, 0])
                parent = obj_data.get('parent', 'unknown')
                desc_parts.append(
                    f"{obj_name} is at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) on {parent}."
                )

        # Combine all parts
        description = " ".join(desc_parts)

        # Truncate if too long (max 512 chars for efficiency)
        if len(description) > 512:
            description = description[:512] + "..."

        return description

    def _extract_numerical_features(
        self,
        world_state_dict: Dict[str, Any],
        info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        Extract numerical features from world state.

        Features (total: numerical_dim = 32):
        - Agent position (3D): [x, y, z]
        - Agent yaw (1D)
        - Task progress (1D): [0, 1]
        - Task-relevant distances (agent->object/target, object->target)
        - Fraction of task objects at target
        - Task object/target counts (normalized)
        - Padding to reach numerical_dim

        Args:
            world_state_dict: World state dictionary
            info: Optional info dict

        Returns:
            features: numpy array of shape [numerical_dim]
        """
        features: List[float] = []
        feature_names: List[str] = []

        def _append(value: float, name: str) -> None:
            features.append(value)
            feature_names.append(name)

        # Agent position and orientation (primary + secondary, 8D)
        agent_poses = world_state_dict.get('agent_poses', {})
        agent_ids = sorted(agent_poses.keys())
        primary_id = agent_ids[0] if agent_ids else None
        secondary_id = agent_ids[1] if len(agent_ids) > 1 else None

        if primary_id is not None and primary_id in agent_poses:
            agent_pose = agent_poses[primary_id]
            pos = agent_pose.get('position', [0.0, 0.0, 0.0])
            yaw = agent_pose.get('yaw', 0.0)
            _append(float(pos[0]), "agent_pos_x")
            _append(float(pos[1]), "agent_pos_y")
            _append(float(pos[2]), "agent_pos_z")
            _append(float(yaw), "agent_yaw")
        else:
            _append(0.0, "agent_pos_x")
            _append(0.0, "agent_pos_y")
            _append(0.0, "agent_pos_z")
            _append(0.0, "agent_yaw")

        if secondary_id is not None and secondary_id in agent_poses:
            agent_pose = agent_poses[secondary_id]
            pos = agent_pose.get('position', [0.0, 0.0, 0.0])
            yaw = agent_pose.get('yaw', 0.0)
            _append(float(pos[0]), "agent2_pos_x")
            _append(float(pos[1]), "agent2_pos_y")
            _append(float(pos[2]), "agent2_pos_z")
            _append(float(yaw), "agent2_yaw")
        else:
            _append(0.0, "agent2_pos_x")
            _append(0.0, "agent2_pos_y")
            _append(0.0, "agent2_pos_z")
            _append(0.0, "agent2_yaw")

        # Task progress (1D)
        if info and 'task_percent_complete' in info:
            task_progress = info['task_percent_complete']
        else:
            task_progress = 0.0
        _append(float(task_progress), "task_progress")

        # Task-relevant distances and counts
        distance_norm = 10.0
        count_norm = 10.0

        def _norm_dist(value: float) -> float:
            if value < 0:
                return 0.0
            return min(value / distance_norm, 1.0)

        object_positions = world_state_dict.get('object_positions', {})
        furniture_positions = world_state_dict.get('furniture_positions', {})
        task_context = world_state_dict.get('task_context', {}) or {}

        task_objects = task_context.get("objects", []) or []
        task_entities = task_context.get("entities", []) or []
        task_receptacles = task_context.get("receptacles", []) or []

        def _get_pos(name: str) -> Optional[np.ndarray]:
            obj_data = object_positions.get(name)
            if obj_data and "position" in obj_data:
                return np.array(obj_data["position"], dtype=np.float32)
            furn_data = furniture_positions.get(name)
            if furn_data and "position" in furn_data:
                return np.array(furn_data["position"], dtype=np.float32)
            return None

        candidate_objects = task_objects if task_objects else task_entities
        task_obj_positions = []
        for name in candidate_objects:
            pos = _get_pos(name)
            if pos is not None:
                task_obj_positions.append(pos)

        target_positions = []
        for name in task_receptacles:
            pos = _get_pos(name)
            if pos is not None:
                target_positions.append(pos)

        agent_pos = None
        if primary_id is not None and primary_id in agent_poses:
            agent_pos = np.array(agent_poses[primary_id].get('position', [0, 0, 0]), dtype=np.float32)

        if agent_pos is not None and task_obj_positions:
            dists = [float(np.linalg.norm(agent_pos - p)) for p in task_obj_positions]
            min_task_obj_dist = min(dists)
            mean_task_obj_dist = sum(dists) / len(dists)
        else:
            min_task_obj_dist = distance_norm
            mean_task_obj_dist = distance_norm

        if agent_pos is not None and target_positions:
            target_dists = [float(np.linalg.norm(agent_pos - p)) for p in target_positions]
            min_agent_target_dist = min(target_dists)
        else:
            min_agent_target_dist = distance_norm

        if task_obj_positions and target_positions:
            obj_target_dists = [
                min(float(np.linalg.norm(obj_pos - tgt_pos)) for tgt_pos in target_positions)
                for obj_pos in task_obj_positions
            ]
            min_obj_target_dist = min(obj_target_dists)
            mean_obj_target_dist = sum(obj_target_dists) / len(obj_target_dists)
        else:
            min_obj_target_dist = distance_norm
            mean_obj_target_dist = distance_norm

        if task_objects and task_receptacles:
            target_set = set(task_receptacles)
            at_target = 0
            for name in task_objects:
                parent = object_positions.get(name, {}).get("parent")
                if parent in target_set:
                    at_target += 1
            fraction_at_target = at_target / max(len(task_objects), 1)
        else:
            fraction_at_target = 0.0

        _append(_norm_dist(min_task_obj_dist), "min_task_obj_dist_norm")
        _append(_norm_dist(mean_task_obj_dist), "mean_task_obj_dist_norm")
        _append(_norm_dist(min_agent_target_dist), "min_agent_target_dist_norm")
        _append(_norm_dist(min_obj_target_dist), "min_obj_target_dist_norm")
        _append(_norm_dist(mean_obj_target_dist), "mean_obj_target_dist_norm")
        _append(float(fraction_at_target), "task_object_fraction_at_target")
        _append(min(len(task_objects) / count_norm, 1.0), "task_object_count_norm")
        _append(min(len(task_receptacles) / count_norm, 1.0), "task_target_count_norm")

        visible_task_objects = [name for name in task_objects if name in object_positions]
        visible_task_targets = [name for name in task_receptacles if _get_pos(name) is not None]
        task_object_visible_fraction = len(visible_task_objects) / max(len(task_objects), 1)
        task_target_visible_fraction = len(visible_task_targets) / max(len(task_receptacles), 1)

        _append(float(task_object_visible_fraction), "task_object_visible_fraction")
        _append(float(task_target_visible_fraction), "task_target_visible_fraction")
        _append(min(len(visible_task_objects) / count_norm, 1.0), "task_object_visible_count_norm")
        _append(min(len(visible_task_targets) / count_norm, 1.0), "task_target_visible_count_norm")

        agent_holdings = world_state_dict.get("agent_holdings", {}) or {}
        held_objects = [
            data.get("held_object")
            for data in agent_holdings.values()
            if data.get("held_object")
        ]
        held_task_objects = [name for name in held_objects if name in task_objects]
        any_agent_holding = 1.0 if held_objects else 0.0
        any_agent_holding_task = 1.0 if held_task_objects else 0.0
        held_task_fraction = len(held_task_objects) / max(len(task_objects), 1)

        _append(any_agent_holding, "any_agent_holding")
        _append(any_agent_holding_task, "any_agent_holding_task_object")
        _append(float(held_task_fraction), "held_task_object_fraction")

        primary_held = agent_holdings.get(primary_id, {}).get("held_object") if primary_id is not None else None
        primary_holding = 1.0 if primary_held else 0.0
        primary_holding_task = 1.0 if primary_held and primary_held in task_objects else 0.0
        _append(primary_holding, "primary_agent_holding")
        _append(primary_holding_task, "primary_agent_holding_task_object")

        if agent_pos is not None and target_positions and primary_held:
            holding_target_dists = [float(np.linalg.norm(agent_pos - p)) for p in target_positions]
            min_hold_target_dist = min(holding_target_dists)
        else:
            min_hold_target_dist = distance_norm
        _append(_norm_dist(min_hold_target_dist), "min_agent_target_dist_when_holding_norm")

        # Pad to numerical_dim
        current_dim = len(features)
        if current_dim < self.numerical_dim:
            pad_count = self.numerical_dim - current_dim
            for i in range(pad_count):
                _append(0.0, f"pad_{i}")
        elif current_dim > self.numerical_dim:
            features = features[:self.numerical_dim]
            feature_names = feature_names[:self.numerical_dim]

        if self.debug:
            self._last_feature_names = feature_names
            self._last_feature_values = features

        return np.array(features, dtype=np.float32)

    def _maybe_log_debug(
        self,
        world_state_dict: Dict[str, Any],
        info: Optional[Dict[str, Any]],
        text_desc: str
    ) -> None:
        if not self.debug or not self.debug_dir:
            return
        if self.debug_max_entries == 0:
            return

        state_kind = info.get("state_kind") if info else None
        if state_kind == "state" and not self.debug_log_state:
            return
        if state_kind == "next_state" and not self.debug_log_next_state:
            return

        self._debug_step += 1
        if not self.debug_log_on_change:
            if self._debug_step % self.debug_frequency != 0:
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

        feature_map = {}
        if self._last_feature_names and self._last_feature_values:
            feature_map = {
                name: float(value)
                for name, value in zip(self._last_feature_names, self._last_feature_values)
            }

        text_hash = self._hash_text(text_desc)
        semantic_text_hash = self._hash_payload(
            self._build_semantic_text_payload(task_context, info, world_state_dict)
        )
        signature_text_hash = text_hash if self.debug_signature_mode == "full" else semantic_text_hash
        if self.debug_signature_mode == "full":
            signature_features = list(self._last_feature_values) if self._last_feature_values else []
        else:
            signature_features = self._get_signature_feature_values(feature_map)
        signature = self._build_debug_signature(signature_text_hash, state_kind, task_context_hash, signature_features)
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
                "episode_id": episode_id,
                "episode_filename": episode_filename,
                "text_dim": self.text_dim,
                "numerical_dim": self.numerical_dim,
                "feature_names": self._last_feature_names,
                "key_feature_names": self.debug_key_feature_names,
                "signature_mode": self.debug_signature_mode,
                "signature_feature_names": self.debug_signature_feature_names,
                "change_threshold": self.debug_change_threshold,
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
                else:
                    if delta == 0:
                        continue
                feature_delta[name] = [float(prev), float(curr)]
        elif self.debug_log_full_features_first or self._last_logged_feature_values is None:
            feature_snapshot = feature_map

        trimmed_text = text_desc
        if self.debug_text_max_len > 0 and len(trimmed_text) > self.debug_text_max_len:
            trimmed_text = trimmed_text[:self.debug_text_max_len] + "..."

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
        }

        if not self._write_debug_payload(filepath, payload):
            return

        self._last_logged_signature = signature
        self._last_signature_hash = signature_hash
        self._last_logged_feature_values = list(self._last_feature_values)
        self._last_logged_text_hash = signature_text_hash

    def _build_semantic_text_payload(
        self,
        task_context: Dict[str, Any],
        info: Optional[Dict[str, Any]],
        world_state_dict: Dict[str, Any]
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

        def _normalize_list(values: List[Any]) -> List[str]:
            return sorted(str(value) for value in values if value is not None)

        agent_holdings = world_state_dict.get("agent_holdings", {}) or {}
        held_objects = sorted({
            data.get("held_object")
            for data in agent_holdings.values()
            if data.get("held_object")
        })

        return {
            "instruction": instruction,
            "objects": _normalize_list(task_context.get("objects", []) or []),
            "receptacles": _normalize_list(task_context.get("receptacles", []) or []),
            "entities": _normalize_list(task_context.get("entities", []) or []),
            "rooms": _normalize_list(task_context.get("rooms", []) or []),
            "proposition_count": task_context.get("proposition_count") or 0,
            "held_objects": held_objects,
        }

    def _get_signature_feature_values(self, feature_map: Dict[str, float]) -> List[float]:
        values: List[float] = []
        for name in self.debug_signature_feature_names:
            values.append(float(feature_map.get(name, 0.0)))
        return values

    def _build_debug_signature(
        self,
        signature_text_hash: str,
        state_kind: Optional[str],
        task_context_hash: Optional[str],
        signature_feature_values: Optional[List[float]]
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
        if self.debug_max_entries == 0:
            return
        if self._repeat_count <= 0:
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
            safe_name = safe_name.replace("\\\\", "_").replace("/", "_")
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
        """
        Convert yaw angle (radians) to cardinal direction.

        Args:
            yaw: Yaw angle in radians

        Returns:
            direction: 'north', 'east', 'south', or 'west'
        """
        # Normalize yaw to [0, 2π]
        yaw = yaw % (2 * np.pi)

        # Convert to degrees
        degrees = np.degrees(yaw)

        # Map to cardinal directions
        if 45 <= degrees < 135:
            return 'east'
        elif 135 <= degrees < 225:
            return 'south'
        elif 225 <= degrees < 315:
            return 'west'
        else:
            return 'north'
