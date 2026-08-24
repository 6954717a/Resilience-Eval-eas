"""
Critic module for A2C (Advantage Actor-Critic) framework.
Provides interfaces and implementations for state evaluation and advantage estimation.

Extended with neural network-based value function approximation.
"""

from abc import ABC, abstractmethod
from collections import Counter, deque
from typing import Any, Deque, Dict, List, Mapping, Optional, TextIO, Tuple, Union
import json
import logging
import os
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from habitat_llm.evaluation.eval_logging import write_critic_trajectory_export
from habitat_llm.evaluation.experiments.common import infer_task_family
from habitat_llm.evaluation.networks import StateEncoder, ValueNetwork
from habitat_llm.evaluation.proposition_outcome import proposition_outcome_from_info
from habitat_llm.evaluation.utils import compute_gae
from habitat_llm.evaluation.llm_evaluator import LLMRewardShaper, LLMOfflineAnalyzer

logger = logging.getLogger(__name__)

class BaseCritic(ABC):
    """
    Abstract base class for Critics.
    """

    def __init__(self, config: Dict[str, Any], env_interface: Any = None):
        """
        Initialize the Critic.

        :param config: Configuration dictionary for the critic.
        :param env_interface: Optional reference to the environment interface.
        """
        self.config = config
        self.env_interface = env_interface

    @abstractmethod
    def evaluate(self, state: Any, action: Any) -> float:
        """
        Evaluate a state or state-action pair.

        :param state: The current state (observations).
        :param action: The action taken (optional).
        :return: A scalar value/score (e.g., Value function V(s) or Q-value Q(s,a)).
        """
        pass

    @abstractmethod
    def update(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        info: Dict[str, Any]
    ) -> None:
        """
        Update the critic's internal model based on a transition.

        :param state: The state before the action.
        :param action: The action taken.
        :param reward: The reward received.
        :param next_state: The state resulting from the action.
        :param done: Whether the episode ended.
        :param info: Additional information dictionary.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the critic state between episodes.
        """
        pass


class A2CCritic(BaseCritic):
    """
    Extended A2C Critic with neural network-based value function approximation.

    Features:
    - StateEncoder: Encodes world_state_dict into fixed-dimensional vectors
    - ValueNetwork: Neural network for V(s) estimation
    - GAE: Generalized Advantage Estimation for advantage calculation
    - LLM Integration: Optional reward shaping and offline analysis
    """

    def __init__(self, config: Dict[str, Any], env_interface: Any = None):
        super().__init__(config, env_interface)

        # Core parameters
        self.gamma = self.config.get("gamma", 0.99)
        self.gae_lambda = self.config.get("gae_lambda", 0.95)
        self.n_step = self.config.get("n_step", 5)

        # Device configuration
        self.device = self.config.get("device", "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        self.phase = str(self.config.get("phase", "evaluation")).lower()
        if self.phase not in {"train", "training_collect", "reference", "evaluation"}:
            self.phase = "evaluation"
        self.freeze_state_encoder = bool(
            self.config.get("freeze_state_encoder", self.phase != "train")
        )
        self.update_value_network = bool(
            self.config.get("update_value_network", self.phase == "train")
        )
        self.reward_shaper_valid = bool(
            self.config.get("reward_shaper_valid", not self.config.get("use_llm_shaping", False))
        )
        self.critic_lifecycle_valid = bool(
            self.config.get("critic_lifecycle_valid", False)
        )
        self.judge_model = str(
            self.config.get("llm_model") or "unconfigured-judge"
        ).strip()
        self.initialization_seed = int(
            self.config.get("initialization_seed", 47668090)
        )

        # Initialize neural networks
        self.state_encoder_config = self._normalize_state_encoder_config(self.config)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.initialization_seed)
            self.state_encoder = StateEncoder(
                text_dim=self.state_encoder_config["text_dim"],
                numerical_dim=self.state_encoder_config["numerical_dim"],
                hidden_dim=self.state_encoder_config["hidden_dim"],
                output_dim=self.state_encoder_config["output_dim"],
                text_encoder_model=self.state_encoder_config["text_encoder_model"],
                text_encoder_local_files_only=self.state_encoder_config[
                    "text_encoder_local_files_only"
                ],
                text_encoder_cache_dir=self.state_encoder_config["text_encoder_cache_dir"],
                text_encoder_fallback=self.state_encoder_config["text_encoder_fallback"],
                device=self.device,
                debug=self.state_encoder_config["debug"],
                debug_dir=self.state_encoder_config["debug_dir"],
                debug_frequency=self.state_encoder_config["debug_frequency"],
                debug_max_entries=self.state_encoder_config["debug_max_entries"],
                debug_log_state=self.state_encoder_config["debug_log_state"],
                debug_log_next_state=self.state_encoder_config["debug_log_next_state"],
                debug_log_on_change=self.state_encoder_config["debug_log_on_change"],
                debug_change_threshold=self.state_encoder_config["debug_change_threshold"],
                debug_text_max_len=self.state_encoder_config["debug_text_max_len"],
                debug_log_delta=self.state_encoder_config["debug_log_delta"],
                debug_log_task_context_once=self.state_encoder_config[
                    "debug_log_task_context_once"
                ],
                debug_log_full_features_first=self.state_encoder_config[
                    "debug_log_full_features_first"
                ],
                debug_key_feature_names=self.state_encoder_config["debug_key_feature_names"],
                debug_signature_mode=self.state_encoder_config["debug_signature_mode"],
                debug_signature_feature_names=self.state_encoder_config[
                    "debug_signature_feature_names"
                ],
                input_variant=self.state_encoder_config["input_variant"],
                context_variant=self.state_encoder_config["context_variant"],
                distance_norm=self.state_encoder_config["distance_norm"],
                count_norm=self.state_encoder_config["count_norm"],
                action_history_cap=self.state_encoder_config["action_history_cap"],
                fusion_dropout_rate=self.state_encoder_config["fusion_dropout_rate"],
            )

            self.value_network = ValueNetwork(
                state_dim=self.state_encoder_config["output_dim"],
                hidden_dims=self.config.get("value_hidden_dims", [256, 128, 64]),
                dropout_rate=self.config.get("dropout_rate", 0.1),
                device=self.device
            )
        self.inference_mode = str(
            self.config.get("inference_mode", "current")
        ).lower()
        if self.inference_mode not in {"current", "deterministic"}:
            self.inference_mode = "current"

        # Optimizer for value network
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(),
            lr=self.config.get("value_lr", 1e-3)
        )
        self.state_encoder_id = ""
        self.value_checkpoint_id = ""
        self._load_component_checkpoints()
        if self.freeze_state_encoder:
            for parameter in self.state_encoder.parameters():
                parameter.requires_grad_(False)
        if not self.update_value_network:
            for parameter in self.value_network.parameters():
                parameter.requires_grad_(False)

        # LLM components (optional)
        self.use_llm_shaping = self.config.get("use_llm_shaping", False)
        self.use_llm_offline = self.config.get("use_llm_offline", False)

        # Determine LLM client to use
        llm_client = None

        # 1. Try to create custom client from config
        llm_base_url = self.config.get("llm_base_url")
        llm_api_key = self.config.get("llm_api_key")

        if llm_base_url or llm_api_key:
            try:
                from openai import OpenAI
                
                # Use provided key, or fallback to env var, or empty string (some local LLMs don't need key)
                api_key = llm_api_key or os.environ.get("OPENAI_API_KEY") or "dummy"
                
                llm_client = OpenAI(
                    base_url=llm_base_url,
                    api_key=api_key
                )
                logger.info(f"Created custom A2C Critic LLM client with base_url={llm_base_url}")
            except ImportError:
                logger.error("openai package not installed, cannot create custom client for A2C Critic")
            except Exception as e:
                logger.error(f"Failed to create custom A2C Critic LLM client: {e}")

        # 2. Fallback to env_interface client
        if llm_client is None and env_interface and hasattr(env_interface, 'llm_client'):
            llm_client = env_interface.llm_client

        # Initialize components with the selected client
        if self.use_llm_shaping:
            if llm_client:
                self.reward_shaper = LLMRewardShaper(llm_client, self.config)
                logger.info("LLM Reward Shaper enabled")
            else:
                logger.warning("use_llm_shaping=True but no LLM client available")
                self.reward_shaper = None
        else:
            self.reward_shaper = None

        if self.use_llm_offline:
            if llm_client:
                self.offline_analyzer = LLMOfflineAnalyzer(llm_client, self.config)
                logger.info("LLM Offline Analyzer enabled")
            else:
                logger.warning("use_llm_offline=True but no LLM client available")
                self.offline_analyzer = None
        else:
            self.offline_analyzer = None

        # Buffers
        self.buffer: List[Dict[str, Any]] = []
        self.raw_states: List[Tuple[Any, ...]] = []  # Selected event trajectory for LLM analysis
        self.last_episode_value_history: List[float] = []
        self.last_episode_value_delta_history: List[float] = []
        self.last_episode_value_variance: float = 0.0
        self.last_episode_value_delta_variance: float = 0.0
        self.last_episode_stability_trace_summary: Dict[str, Any] = {}
        self.last_analysis_path: Optional[str] = None
        self.last_export_path: Optional[str] = None
        self.analysis_export_enabled = bool(
            self.config.get("analysis_export_enabled", False)
        )
        self.analysis_export_include_world_state = bool(
            self.config.get("analysis_export_include_world_state", True)
        )
        self.analysis_export_world_state_mode = str(
            self.config.get(
                "analysis_export_world_state_mode",
                "full" if self.analysis_export_include_world_state else "none",
            )
        ).lower()
        if self.analysis_export_world_state_mode not in {"full", "compact", "none"}:
            self.analysis_export_world_state_mode = (
                "full" if self.analysis_export_include_world_state else "none"
            )
        self.analysis_export_info_mode = str(
            self.config.get("analysis_export_info_mode", "compact")
        ).lower()
        if self.analysis_export_info_mode not in {"full", "compact", "none"}:
            self.analysis_export_info_mode = "compact"
        self.analysis_export_record_mode = str(
            self.config.get("analysis_export_record_mode", "state_and_next")
        ).lower()
        if self.analysis_export_record_mode not in {"state_and_next", "state_only"}:
            self.analysis_export_record_mode = "state_and_next"
        self.analysis_export_include_text_embedding = bool(
            self.config.get("analysis_export_include_text_embedding", True)
        )
        self.analysis_export_include_reward_fields = bool(
            self.config.get("analysis_export_include_reward_fields", False)
        )
        self.analysis_export_minimal_schema = bool(
            self.config.get("analysis_export_minimal_schema", True)
        )
        self.analysis_export_include_task_context = bool(
            self.config.get("analysis_export_include_task_context", True)
        )
        self.analysis_export_include_text_desc = bool(
            self.config.get(
                "analysis_export_include_text_desc",
                not self.analysis_export_minimal_schema,
            )
        )
        self.analysis_export_include_text_segments = bool(
            self.config.get(
                "analysis_export_include_text_segments",
                not self.analysis_export_minimal_schema,
            )
        )
        self.analysis_export_include_category_presence = bool(
            self.config.get(
                "analysis_export_include_category_presence",
                not self.analysis_export_minimal_schema,
            )
        )
        self.analysis_export_include_feature_schema_per_record = bool(
            self.config.get(
                "analysis_export_include_feature_schema_per_record",
                not self.analysis_export_minimal_schema,
            )
        )
        self.analysis_export_include_next_value_features = bool(
            self.config.get(
                "analysis_export_include_next_value_features",
                not self.analysis_export_minimal_schema,
            )
        )
        self.analysis_export_info_action_history_keep_last = max(
            0,
            int(self.config.get("analysis_export_info_action_history_keep_last", 3)),
        )
        self.analysis_export_size_target = str(
            self.config.get("analysis_export_size_target", "compact-step-audit")
        )
        self.analysis_export_schema_version = int(
            self.config.get("analysis_export_schema_version", 4)
        )
        self.analysis_focus_mode = str(
            self.config.get("analysis_focus_mode", "replan_window")
        ).lower()
        self.analysis_replan_pre_sim_steps = max(
            0, int(self.config.get("analysis_replan_pre_sim_steps", 2))
        )
        self.analysis_replan_post_sim_steps = max(
            0, int(self.config.get("analysis_replan_post_sim_steps", 2))
        )
        self.analysis_export_keep_first_last = bool(
            self.config.get("analysis_export_keep_first_last", True)
        )
        self.analysis_export_keep_progress_transitions = bool(
            self.config.get("analysis_export_keep_progress_transitions", True)
        )
        self.analysis_export_keep_response_transitions = bool(
            self.config.get("analysis_export_keep_response_transitions", True)
        )
        self.analysis_export_keep_success_transition = bool(
            self.config.get("analysis_export_keep_success_transition", True)
        )
        self.analysis_export_keep_td_error_topk = max(
            0, int(self.config.get("analysis_export_keep_td_error_topk", 4))
        )
        self.analysis_export_max_selected_transitions = max(
            1, int(self.config.get("analysis_export_max_selected_transitions", 96))
        )
        self.analysis_export_full_trajectory = bool(
            self.config.get("analysis_export_full_trajectory", False)
        )
        self.analysis_export_progress_delta_threshold = float(
            self.config.get("analysis_export_progress_delta_threshold", 1e-6)
        )
        self.analysis_save_dir = self.config.get("analysis_save_dir", "./analyses")
        self.analysis_export_dir = self.config.get("analysis_export_dir")
        self.runner_output_dir = self.config.get("runner_output_dir")
        self.analysis_focus_export_enabled = bool(
            self.config.get(
                "analysis_focus_export_enabled",
                self.analysis_export_enabled,
            )
        )
        self.analysis_focus_dir = self.config.get("analysis_focus_dir")
        if not self.analysis_export_dir:
            self.analysis_export_dir = os.path.join(
                self.analysis_save_dir, "critic_exports"
            )
        if not self.analysis_focus_dir:
            focus_root = self.runner_output_dir or self._resolve_export_output_dir()
            self.analysis_focus_dir = os.path.join(focus_root, "analyze")
        if self.analysis_export_enabled:
            os.makedirs(self.analysis_export_dir, exist_ok=True)
        self._episode_feature_schema: Dict[str, Any] = {}
        self._focus_recent: Deque[Tuple[int, Dict[str, Any]]] = deque(
            maxlen=max(1, self.analysis_replan_pre_sim_steps)
        )
        self._focus_stream: Optional[TextIO] = None
        self._focus_stream_path: Optional[str] = None
        self._focus_written_indices: set[int] = set()
        self._focus_previous_planning_step: Optional[int] = None
        self._focus_previous_progress: Optional[float] = None
        self._focus_previous_success: Optional[float] = None
        self._focus_post_remaining = 0
        self._focus_last_candidate: Optional[Tuple[int, Dict[str, Any]]] = None
        self.last_focus_export_path: Optional[str] = None

        # Training statistics
        self.training_stats = {
            "value_loss": 0.0,
            "mean_predicted_value": 0.0,
            "mean_return": 0.0,
            "episodes_trained": 0,
            "inference_mode": self.inference_mode,
            "state_encoder_input_variant": self.state_encoder.input_variant,
            "state_encoder_context_variant": self.state_encoder.context_variant,
            "state_encoder_text_backend": getattr(
                self.state_encoder, "text_encoder_backend", "unknown"
            ),
            "analysis_export_schema_version": self.analysis_export_schema_version,
            "analysis_export_world_state_mode": self.analysis_export_world_state_mode,
            "analysis_export_info_mode": self.analysis_export_info_mode,
            "analysis_export_record_mode": self.analysis_export_record_mode,
            "analysis_export_minimal_schema": self.analysis_export_minimal_schema,
            "analysis_export_include_task_context": self.analysis_export_include_task_context,
            "analysis_export_size_target": self.analysis_export_size_target,
            "analysis_focus_mode": self.analysis_focus_mode,
            "analysis_replan_pre_sim_steps": self.analysis_replan_pre_sim_steps,
            "analysis_replan_post_sim_steps": self.analysis_replan_post_sim_steps,
        }

        # Checkpoint settings
        self.save_checkpoints = self.config.get("save_checkpoints", False)
        self.checkpoint_frequency = self.config.get("checkpoint_frequency", 10)
        self.checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")

        if self.save_checkpoints:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._apply_execution_mode()

        logger.info(f"Initialized A2CCritic: gamma={self.gamma}, gae_lambda={self.gae_lambda}, "
                   f"device={self.device}, llm_shaping={self.use_llm_shaping}, "
                   f"inference_mode={self.inference_mode}")

    @staticmethod
    def _nested_get(
        payload: Mapping[str, Any],
        *keys: str,
        default: Any = None,
    ) -> Any:
        current: Any = payload
        for key in keys:
            if not isinstance(current, Mapping) or key not in current:
                return default
            current = current[key]
        return current

    def _normalize_state_encoder_config(self, config: Mapping[str, Any]) -> Dict[str, Any]:
        state_encoder_cfg = config.get("state_encoder", {}) or {}

        def _get(default: Any, *nested_keys: str, flat_key: Optional[str] = None) -> Any:
            nested_value = self._nested_get(state_encoder_cfg, *nested_keys, default=None)
            if nested_value is not None:
                return nested_value
            if flat_key is not None and flat_key in config:
                return config.get(flat_key)
            return default

        normalized = {
            "text_dim": int(_get(384, "dims", "text", flat_key="text_dim")),
            "numerical_dim": int(
                _get(32, "dims", "numerical", flat_key="numerical_dim")
            ),
            "hidden_dim": int(
                _get(256, "dims", "hidden", flat_key="encoder_hidden_dim")
            ),
            "output_dim": int(_get(128, "dims", "output", flat_key="state_dim")),
            "text_encoder_model": str(
                _get(
                    "all-MiniLM-L6-v2",
                    "text_encoder",
                    "model",
                    flat_key="text_encoder_model",
                )
            ),
            "text_encoder_local_files_only": bool(
                _get(
                    False,
                    "text_encoder",
                    "local_files_only",
                    flat_key="text_encoder_local_files_only",
                )
            ),
            "text_encoder_cache_dir": _get(
                None,
                "text_encoder",
                "cache_dir",
                flat_key="text_encoder_cache_dir",
            ),
            "text_encoder_fallback": str(
                _get(
                    "hash",
                    "text_encoder",
                    "fallback",
                    flat_key="text_encoder_fallback",
                )
            ),
            "input_variant": str(
                _get("full", "variants", "input", flat_key="state_encoder_input_variant")
            ),
            "context_variant": str(
                _get(
                    "full",
                    "variants",
                    "context",
                    flat_key="state_encoder_context_variant",
                )
            ),
            "distance_norm": float(
                _get(10.0, "normalization", "distance", flat_key="distance_norm")
            ),
            "count_norm": float(
                _get(10.0, "normalization", "count", flat_key="count_norm")
            ),
            "action_history_cap": float(
                _get(
                    50.0,
                    "normalization",
                    "action_history_cap",
                    flat_key="action_history_cap",
                )
            ),
            "fusion_dropout_rate": float(
                _get(
                    0.1,
                    "fusion",
                    "dropout_rate",
                    flat_key="fusion_dropout_rate",
                )
            ),
            "debug": bool(
                _get(
                    config.get("state_encoder_debug", config.get("debug", False)),
                    "debug",
                    "enabled",
                    flat_key="state_encoder_debug",
                )
            ),
            "debug_dir": _get(None, "debug", "dir", flat_key="state_encoder_debug_dir"),
            "debug_frequency": int(
                _get(1, "debug", "frequency", flat_key="state_encoder_debug_frequency")
            ),
            "debug_max_entries": int(
                _get(
                    1000,
                    "debug",
                    "max_entries",
                    flat_key="state_encoder_debug_max_entries",
                )
            ),
            "debug_log_state": bool(
                _get(True, "debug", "log_state", flat_key="state_encoder_debug_log_state")
            ),
            "debug_log_next_state": bool(
                _get(
                    False,
                    "debug",
                    "log_next_state",
                    flat_key="state_encoder_debug_log_next_state",
                )
            ),
            "debug_log_on_change": bool(
                _get(
                    False,
                    "debug",
                    "log_on_change",
                    flat_key="state_encoder_debug_log_on_change",
                )
            ),
            "debug_change_threshold": float(
                _get(
                    0.0,
                    "debug",
                    "change_threshold",
                    flat_key="state_encoder_debug_change_threshold",
                )
            ),
            "debug_text_max_len": int(
                _get(
                    200,
                    "debug",
                    "text_max_len",
                    flat_key="state_encoder_debug_text_max_len",
                )
            ),
            "debug_log_delta": bool(
                _get(
                    True,
                    "debug",
                    "log_delta",
                    flat_key="state_encoder_debug_log_delta",
                )
            ),
            "debug_log_task_context_once": bool(
                _get(
                    True,
                    "debug",
                    "log_task_context_once",
                    flat_key="state_encoder_debug_log_task_context_once",
                )
            ),
            "debug_log_full_features_first": bool(
                _get(
                    True,
                    "debug",
                    "log_full_features_first",
                    flat_key="state_encoder_debug_log_full_features_first",
                )
            ),
            "debug_key_feature_names": _get(
                None,
                "debug",
                "key_feature_names",
                flat_key="state_encoder_debug_key_feature_names",
            ),
            "debug_signature_mode": str(
                _get(
                    "semantic",
                    "debug",
                    "signature_mode",
                    flat_key="state_encoder_debug_signature_mode",
                )
            ),
            "debug_signature_feature_names": _get(
                None,
                "debug",
                "signature_feature_names",
                flat_key="state_encoder_debug_signature_feature_names",
            ),
        }

        if normalized["debug"] and not normalized["debug_dir"]:
            analysis_dir = config.get("analysis_save_dir")
            if analysis_dir:
                normalized["debug_dir"] = os.path.join(analysis_dir, "state_encoder")

        return normalized

    def _apply_execution_mode(self) -> None:
        if self.inference_mode == "deterministic":
            self.state_encoder.eval()
            self.value_network.eval()
        else:
            self.state_encoder.train()
            self.value_network.train()

    def _load_component_checkpoints(self) -> None:
        """Load immutable StateEncoder and ValueNetwork components from config."""

        from habitat_llm.evaluation.core.critic_lifecycle import (
            checkpoint_component_id,
        )

        state_path = str(self.config.get("state_encoder_checkpoint") or "")
        value_path = str(self.config.get("value_checkpoint") or "")
        if state_path:
            checkpoint = torch.load(state_path, map_location=self.device)
            state_dict = checkpoint.get("state_encoder") if isinstance(checkpoint, dict) else None
            if not isinstance(state_dict, dict):
                raise ValueError(
                    f"StateEncoder checkpoint has no state_encoder weights: {state_path}"
                )
            self.state_encoder.load_state_dict(state_dict)
            self.state_encoder_id = checkpoint_component_id(
                state_path, "state_encoder"
            )
        if value_path:
            checkpoint = torch.load(value_path, map_location=self.device)
            state_dict = checkpoint.get("value_network") if isinstance(checkpoint, dict) else None
            if not isinstance(state_dict, dict):
                raise ValueError(
                    f"Value checkpoint has no value_network weights: {value_path}"
                )
            self.value_network.load_state_dict(state_dict)
            self.value_checkpoint_id = checkpoint_component_id(
                value_path, "value_network"
            )
        if bool(self.config.get("require_component_checkpoints", False)):
            if not self.state_encoder_id or not self.value_checkpoint_id:
                raise ValueError(
                    "Formal Critic phase requires both StateEncoder and ValueNetwork checkpoints"
                )
        self._apply_execution_mode()

    def _resolve_export_output_dir(self) -> str:
        if self.runner_output_dir:
            return self.runner_output_dir

        analysis_dir = os.path.normpath(self.analysis_save_dir)
        if os.path.basename(analysis_dir) == "analyses":
            return os.path.dirname(analysis_dir)

        export_dir = os.path.normpath(self.analysis_export_dir)
        if os.path.basename(export_dir) == "critic_exports":
            analyses_dir = os.path.dirname(export_dir)
            if os.path.basename(analyses_dir) == "analyses":
                return os.path.dirname(analyses_dir)
            return analyses_dir

        return os.path.dirname(analysis_dir)

    @staticmethod
    def _task_related_names(task_context: Mapping[str, Any]) -> List[str]:
        names = set()
        for key in (
            "objects",
            "entities",
            "receptacles",
            "rooms",
            "source_furniture_names",
            "source_allowed_regions",
        ):
            for value in (task_context.get(key, []) or []):
                if value:
                    names.add(str(value))
        return sorted(names)

    @staticmethod
    def _compact_action_history(
        action_history: Any,
        *,
        keep_last: int = 3,
    ) -> List[Dict[str, Any]]:
        compact: List[Dict[str, Any]] = []
        if not isinstance(action_history, list):
            return compact
        trimmed = action_history[-keep_last:] if keep_last > 0 else action_history
        for action_entry in trimmed:
            if isinstance(action_entry, Mapping):
                action_payload = action_entry.get("action", action_entry)
                response = action_entry.get("response")
                timestamp = action_entry.get("timestamp")
            else:
                action_payload = getattr(action_entry, "action", action_entry)
                response = getattr(action_entry, "response", None)
                timestamp = getattr(action_entry, "timestamp", None)
            if isinstance(action_payload, tuple):
                action_repr = [str(item) for item in action_payload]
            elif isinstance(action_payload, list):
                action_repr = [str(item) for item in action_payload]
            else:
                action_repr = str(action_payload)
            compact.append(
                {
                    "action": action_repr,
                    "response": str(response) if response not in (None, "") else "",
                    "timestamp": int(timestamp) if timestamp is not None else None,
                }
            )
        return compact

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        try:
            if value is None:
                return default
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def _info_sim_step_count(self, info: Optional[Dict[str, Any]]) -> int:
        if not isinstance(info, dict):
            return 0
        if "sim_step_count" in info:
            return self._safe_int(info.get("sim_step_count"), 0)
        return self._safe_int(info.get("step_count"), 0)

    def _info_loop_step_count(self, info: Optional[Dict[str, Any]]) -> int:
        if not isinstance(info, dict):
            return 0
        if "loop_step_count" in info:
            return self._safe_int(info.get("loop_step_count"), 0)
        return self._safe_int(info.get("step_count"), 0)

    def _info_planning_step_count(self, info: Optional[Dict[str, Any]]) -> int:
        if not isinstance(info, dict):
            return 0
        return self._safe_int(info.get("planning_step_count"), 0)

    def _info_progress(self, info: Optional[Dict[str, Any]]) -> float:
        if not isinstance(info, dict):
            return 0.0
        return self._safe_float(info.get("task_percent_complete"), 0.0)

    @staticmethod
    def _action_to_text(action: Any) -> str:
        if isinstance(action, dict):
            parts = []
            for agent_id, payload in sorted(action.items(), key=lambda item: str(item[0])):
                if isinstance(payload, tuple):
                    parts.append(f"{agent_id}:{payload[0]}[{payload[1]}]")
                else:
                    parts.append(f"{agent_id}:{payload}")
            return "; ".join(parts)
        if isinstance(action, tuple):
            if len(action) >= 2:
                return f"{action[0]}[{action[1]}]"
            return str(action[0])
        if isinstance(action, list):
            return "; ".join(str(item) for item in action)
        return str(action)

    def _recent_action_text(self, info: Optional[Dict[str, Any]]) -> str:
        if not isinstance(info, dict):
            return ""
        action_history = info.get("action_history", []) or []
        if action_history:
            last_action = action_history[-1]
            if isinstance(last_action, Mapping):
                action_payload = last_action.get("action", last_action)
            else:
                action_payload = getattr(last_action, "action", last_action)
            return self._action_to_text(action_payload)
        return ""

    def _recent_response_text(self, info: Optional[Dict[str, Any]]) -> str:
        if not isinstance(info, dict):
            return ""
        action_history = info.get("action_history", []) or []
        if action_history:
            last_action = action_history[-1]
            if isinstance(last_action, Mapping):
                response = last_action.get("response")
            else:
                response = getattr(last_action, "response", None)
            if response not in (None, ""):
                return str(response)
        action_responses = info.get("action_responses", {}) or {}
        if isinstance(action_responses, Mapping):
            values = [str(value) for value in action_responses.values() if value not in (None, "")]
            if values:
                return " | ".join(values[:2])
        return ""

    def _summarize_proposition_tracker(
        self,
        info: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return proposition_outcome_from_info(info).info_fields(
            include_tracker=True
        )

    def _analysis_export_manifest(self) -> Dict[str, Any]:
        return {
            "export_schema_version": self.analysis_export_schema_version,
            "export_size_target": self.analysis_export_size_target,
            "focus_mode": self.analysis_focus_mode,
            "record_mode": self.analysis_export_record_mode,
            "info_mode": self.analysis_export_info_mode,
            "world_state_mode": self.analysis_export_world_state_mode,
            "include_world_state": self.analysis_export_include_world_state,
            "include_task_context": self.analysis_export_include_task_context,
            "include_reward_fields": self.analysis_export_include_reward_fields,
            "include_text_embedding": self.analysis_export_include_text_embedding,
            "include_text_desc": self.analysis_export_include_text_desc,
            "include_text_segments": self.analysis_export_include_text_segments,
            "include_category_presence": self.analysis_export_include_category_presence,
            "include_feature_schema_per_record": (
                self.analysis_export_include_feature_schema_per_record
            ),
            "include_next_value_features": (
                self.analysis_export_include_next_value_features
            ),
            "minimal_schema": self.analysis_export_minimal_schema,
            "action_history_keep_last": (
                self.analysis_export_info_action_history_keep_last
            ),
            "replan_pre_sim_steps": self.analysis_replan_pre_sim_steps,
            "replan_post_sim_steps": self.analysis_replan_post_sim_steps,
            "keep_first_last": self.analysis_export_keep_first_last,
            "keep_progress_transitions": self.analysis_export_keep_progress_transitions,
            "keep_response_transitions": self.analysis_export_keep_response_transitions,
            "keep_success_transition": self.analysis_export_keep_success_transition,
            "keep_td_error_topk": self.analysis_export_keep_td_error_topk,
            "max_selected_transitions": self.analysis_export_max_selected_transitions,
            "full_stability_trajectory": self.analysis_export_full_trajectory,
            "full_trajectory_storage": "compact_numeric_in_memory",
            "focused_state_storage": "jsonl",
            "focused_state_path": self.last_focus_export_path,
            "critic_phase": self.phase,
            "judge_model": self.judge_model,
            "state_encoder_id": self.state_encoder_id,
            "value_checkpoint_id": self.value_checkpoint_id,
            "reward_shaper_valid": bool(self.reward_shaper_valid),
            "critic_lifecycle_valid": bool(self.critic_lifecycle_valid),
            "record_field_whitelist": [
                "episode_id",
                "transition_index",
                "step_count",
                "sim_step_count",
                "loop_step_count",
                "planning_step_count",
                "task_family",
                "state_kind",
                "planning_transition",
                "analysis_tags",
                "world_state_hash",
                "task_context_hash",
                "text_desc_hash",
                "semantic_text_hash",
                "numerical_features",
                "state_vector",
                "value_features",
                "predicted_value",
                "next_predicted_value",
                "done",
                "task_percent_complete",
                "task_state_success",
                "proposition_satisfied_fraction",
                "gae_advantage",
                "return_target",
                "td_error",
                "task_context",
                "info_snapshot",
            ],
        }

    def _export_info_snapshot(
        self,
        info: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(info, dict):
            return None
        if self.analysis_export_info_mode == "none":
            return None
        if self.analysis_export_info_mode == "full":
            return dict(info)

        action_history = info.get("action_history", []) or []
        proposition_summary = self._summarize_proposition_tracker(info)
        compact_info = {
            "episode_id": info.get("episode_id"),
            "episode_filename": info.get("episode_filename"),
            "task_instruction": info.get("task_instruction"),
            "policy_instruction": info.get(
                "policy_instruction", info.get("task_instruction")
            ),
            "reward_instruction": info.get(
                "reward_instruction", info.get("task_instruction")
            ),
            "canonical_instruction": info.get(
                "canonical_instruction",
                info.get("reward_instruction", info.get("task_instruction")),
            ),
            "task_percent_complete": float(info.get("task_percent_complete", 0.0)),
            "task_state_success": float(info.get("task_state_success", 0.0)),
            "step_count": self._info_sim_step_count(info),
            "sim_step_count": self._info_sim_step_count(info),
            "loop_step_count": self._info_loop_step_count(info),
            "planning_step_count": info.get("planning_step_count", 0),
            "task_family": info.get("task_family"),
            "action_history_count": self._safe_int(
                info.get("action_history_count", len(action_history)), 0
            ),
            "recent_action": self._recent_action_text(info),
            "recent_response": self._recent_response_text(info),
            **proposition_summary,
        }
        counterfactual_name = info.get("counterfactual_name")
        if counterfactual_name:
            compact_info["counterfactual_name"] = counterfactual_name
        occluded_category = info.get("occluded_category")
        if occluded_category:
            compact_info["occluded_category"] = occluded_category
        return compact_info

    def _export_world_state_snapshot(
        self,
        world_state: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(world_state, dict):
            return None
        if self.analysis_export_world_state_mode == "none":
            return None
        if self.analysis_export_world_state_mode == "full":
            return world_state

        task_context = dict(world_state.get("task_context", {}) or {})
        task_names = set(self._task_related_names(task_context))
        agent_holdings = dict(world_state.get("agent_holdings", {}) or {})
        held_names = {
            str(payload.get("held_object"))
            for payload in agent_holdings.values()
            if isinstance(payload, dict) and payload.get("held_object")
        }
        object_positions = dict(world_state.get("object_positions", {}) or {})
        furniture_positions = dict(world_state.get("furniture_positions", {}) or {})
        keep_object_names = set(str(name) for name in (task_context.get("objects", []) or []))
        keep_object_names.update(str(name) for name in (task_context.get("entities", []) or []))
        keep_object_names.update(held_names)
        keep_furniture_names = set(str(name) for name in (task_context.get("receptacles", []) or []))
        keep_furniture_names.update(str(name) for name in (task_context.get("source_furniture_names", []) or []))
        keep_furniture_names.update(str(name) for name in (task_context.get("rooms", []) or []))

        compact_state = {
            "task_context": task_context,
            "agent_poses": dict(world_state.get("agent_poses", {}) or {}),
            "agent_holdings": agent_holdings,
            "object_positions": {
                str(name): value
                for name, value in object_positions.items()
                if str(name) in keep_object_names or str(name) in task_names
            },
            "furniture_positions": {
                str(name): value
                for name, value in furniture_positions.items()
                if str(name) in keep_furniture_names or str(name) in task_names
            },
        }
        snapshot = world_state.get("object_state_snapshot", {})
        if snapshot and task_context.get("episode_object_states"):
            compact_state["object_state_snapshot"] = snapshot
        return compact_state

    def _buffer_world_state_snapshot(
        self,
        world_state: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """World-state payloads live in the sparse focus stream, not the core buffer."""
        return None

    @staticmethod
    def _sanitize_analysis_name(value: Any, fallback: str) -> str:
        text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
        return text.strip("._") or fallback

    def _compact_transition_info(
        self,
        info: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Retain only fields required by Stability, StageBaseline and LLM summaries."""
        if not isinstance(info, Mapping):
            return {}
        compact: Dict[str, Any] = {}
        for key in (
            "episode_id",
            "episode_filename",
            "task_instruction",
            "policy_instruction",
            "reward_instruction",
            "canonical_instruction",
            "task_percent_complete",
            "task_state_success",
            "step_count",
            "sim_step_count",
            "loop_step_count",
            "planning_step_count",
            "action_history_count",
            "task_family",
            "analysis_export_enabled",
            "counterfactual_name",
            "occluded_category",
            "perturbation_requested",
            "perturbation_realized",
            "perturbation_valid",
            "perturbation_reason",
        ):
            if key in info:
                compact[key] = info[key]

        outcome = proposition_outcome_from_info(info)
        compact.update(outcome.info_fields(include_tracker=False))

        action_history = info.get("action_history", []) or []
        if action_history:
            last_action = action_history[-1]
            if isinstance(last_action, Mapping):
                compact["action_history"] = [
                    {
                        key: last_action.get(key)
                        for key in (
                            "action",
                            "response",
                            "timestamp",
                            "agent_uid",
                        )
                        if key in last_action
                    }
                ]
            else:
                compact["action_history"] = [
                    {
                        "action": getattr(last_action, "action", None),
                        "response": getattr(last_action, "response", ""),
                        "timestamp": getattr(last_action, "timestamp", None),
                        "agent_uid": getattr(last_action, "agent_uid", None),
                    }
                ]
        action_responses = info.get("action_responses", {}) or {}
        if isinstance(action_responses, Mapping):
            compact["action_responses"] = {
                str(key): value
                for key, value in action_responses.items()
                if value not in (None, "")
            }
        return compact

    def _compact_encode_artifacts(
        self,
        artifacts: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Convert the full encoder audit record to small fixed-size arrays/hashes."""
        if not isinstance(artifacts, Mapping):
            return {}
        compact = {
            key: artifacts.get(key)
            for key in (
                "feature_schema_version",
                "input_variant",
                "context_variant",
                "world_state_hash",
                "task_context_hash",
                "text_desc_hash",
                "semantic_text_hash",
            )
            if artifacts.get(key) is not None
        }
        for key in ("numerical_features", "state_vector"):
            values = artifacts.get(key, [])
            if values is None:
                values = []
            compact[key] = np.asarray(values, dtype=np.float32)
        if not self._episode_feature_schema:
            self._episode_feature_schema = {
                "version": artifacts.get("feature_schema_version", ""),
                "feature_names": list(artifacts.get("feature_names", []) or []),
                "feature_categories": list(
                    artifacts.get("feature_categories", []) or []
                ),
            }
        return compact

    def _focus_encode_artifacts(
        self,
        artifacts: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        """Build a richer, JSON-safe record only for focused analysis steps."""
        if not isinstance(artifacts, Mapping):
            return {}
        keep = {
            "feature_schema_version",
            "input_variant",
            "context_variant",
            "state_kind",
            "task_family",
            "world_state_hash",
            "task_context_hash",
            "text_desc",
            "text_desc_hash",
            "semantic_text_hash",
            "text_segments",
            "feature_names",
            "feature_categories",
            "numerical_features",
            "state_vector",
            "category_presence",
            "diagnostics",
        }
        if self.analysis_export_include_text_embedding:
            keep.add("text_embedding")
        return {key: artifacts[key] for key in keep if key in artifacts}

    def _focus_world_state_snapshot(
        self,
        world_state: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Always use a task-scoped compact world state for focus records."""
        if not isinstance(world_state, dict):
            return None
        original_mode = self.analysis_export_world_state_mode
        try:
            self.analysis_export_world_state_mode = "compact"
            return self._export_world_state_snapshot(world_state)
        finally:
            self.analysis_export_world_state_mode = original_mode

    def _focus_candidate(
        self,
        *,
        transition_index: int,
        action: Any,
        reward_env: float,
        reward_shaped: float,
        done: bool,
        info: Dict[str, Any],
        state: Optional[Dict[str, Any]],
        next_state: Optional[Dict[str, Any]],
        state_artifacts: Dict[str, Any],
        next_state_artifacts: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "transition_index": int(transition_index),
            "episode_id": info.get("episode_id"),
            "episode_filename": info.get("episode_filename"),
            "sim_step_count": self._info_sim_step_count(info),
            "planning_step_count": self._info_planning_step_count(info),
            "task_family": info.get("task_family", "other"),
            "action": self._action_to_text(action),
            "reward_env": float(reward_env),
            "reward_shaped": float(reward_shaped),
            "done": bool(done),
            "info_snapshot": self._export_info_snapshot(info),
            "state": self._focus_world_state_snapshot(state),
            "next_state": self._focus_world_state_snapshot(next_state),
            "state_artifacts": self._focus_encode_artifacts(state_artifacts),
            "next_state_artifacts": self._focus_encode_artifacts(
                next_state_artifacts
            ),
        }

    def _ensure_focus_stream(self, info: Mapping[str, Any]) -> Optional[TextIO]:
        if not self.analysis_focus_export_enabled:
            return None
        if self._focus_stream is not None:
            return self._focus_stream
        os.makedirs(self.analysis_focus_dir, exist_ok=True)
        episode_id = self._sanitize_analysis_name(info.get("episode_id"), "unknown")
        episode_filename = self._sanitize_analysis_name(
            info.get("episode_filename"), "episode"
        )
        path = os.path.join(
            self.analysis_focus_dir,
            f"critic_focus-episode_{episode_id}_{episode_filename}.jsonl",
        )
        self._focus_stream = open(path, "w", encoding="utf-8")
        self._focus_stream_path = path
        self.last_focus_export_path = path
        header = {
            "record_type": "manifest",
            "schema_version": 1,
            "focus_mode": self.analysis_focus_mode,
            "replan_pre_sim_steps": self.analysis_replan_pre_sim_steps,
            "replan_post_sim_steps": self.analysis_replan_post_sim_steps,
            "episode_id": info.get("episode_id"),
            "episode_filename": info.get("episode_filename"),
            "judge_model": self.judge_model,
        }
        self._focus_stream.write(json.dumps(header, ensure_ascii=False, default=str) + "\n")
        self._focus_stream.flush()
        return self._focus_stream

    def _write_focus_candidate(
        self,
        index: int,
        candidate: Dict[str, Any],
        tags: List[str],
    ) -> None:
        stream = self._ensure_focus_stream(candidate)
        if stream is None:
            return
        if index in self._focus_written_indices:
            return
        payload = {
            "record_type": "focus_transition",
            "analysis_tags": sorted(set(tags)),
            **candidate,
        }
        stream.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        self._focus_written_indices.add(index)

    def _capture_focus_transition(
        self,
        index: int,
        candidate: Dict[str, Any],
        info: Dict[str, Any],
        done: bool,
    ) -> None:
        if not self.analysis_focus_export_enabled:
            return
        planning_step = self._info_planning_step_count(info)
        progress = self._info_progress(info)
        success = self._safe_float(info.get("task_state_success"), 0.0)
        replan_anchor = (
            self._focus_previous_planning_step is not None
            and planning_step != self._focus_previous_planning_step
        )
        progress_event = (
            self.analysis_export_keep_progress_transitions
            and self._focus_previous_progress is not None
            and abs(progress - self._focus_previous_progress)
            > self.analysis_export_progress_delta_threshold
        )
        success_event = (
            self.analysis_export_keep_success_transition
            and self._focus_previous_success is not None
            and success != self._focus_previous_success
        )
        response_event = bool(
            self.analysis_export_keep_response_transitions
            and self._recent_response_text(info)
        )

        if index == 0 and self.analysis_export_keep_first_last:
            self._write_focus_candidate(index, candidate, ["episode_start"])
        if replan_anchor:
            if self.analysis_replan_pre_sim_steps > 0:
                for previous_index, previous in self._focus_recent:
                    self._write_focus_candidate(
                        previous_index,
                        previous,
                        ["replan_pre_window"],
                    )
            self._write_focus_candidate(index, candidate, ["replan_anchor"])
            self._focus_post_remaining = self.analysis_replan_post_sim_steps
        elif self._focus_post_remaining > 0:
            self._write_focus_candidate(index, candidate, ["replan_post_window"])
            self._focus_post_remaining -= 1

        event_tags = []
        if progress_event:
            event_tags.append("progress_change")
        if success_event:
            event_tags.append("success_change")
        if response_event:
            event_tags.append("response_event")
        if event_tags:
            self._write_focus_candidate(index, candidate, event_tags)
        if done and self.analysis_export_keep_first_last:
            self._write_focus_candidate(index, candidate, ["episode_end"])

        self._focus_recent.append((index, candidate))
        self._focus_last_candidate = (index, candidate)
        self._focus_previous_planning_step = planning_step
        self._focus_previous_progress = progress
        self._focus_previous_success = success

    def _close_focus_stream(self, *, write_episode_end: bool = True) -> None:
        if (
            write_episode_end
            and self.analysis_export_keep_first_last
            and self._focus_last_candidate is not None
        ):
            index, candidate = self._focus_last_candidate
            self._write_focus_candidate(index, candidate, ["episode_end"])
        if self._focus_stream is not None:
            self._focus_stream.flush()
            self._focus_stream.close()
            self._focus_stream = None

    def _load_focus_candidates(self) -> Dict[int, Dict[str, Any]]:
        """Read the bounded rich-state sample back only when analysis needs it."""
        if not self.last_focus_export_path or not os.path.isfile(
            self.last_focus_export_path
        ):
            return {}
        records: Dict[int, Dict[str, Any]] = {}
        try:
            with open(self.last_focus_export_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        payload = json.loads(line)
                    except (json.JSONDecodeError, TypeError):
                        continue
                    if payload.get("record_type") != "focus_transition":
                        continue
                    index = self._safe_int(payload.get("transition_index"), -1)
                    if index >= 0:
                        records[index] = payload
        except (OSError, UnicodeError):
            logger.warning(
                "Unable to read critic focus stream: %s",
                self.last_focus_export_path,
            )
        return records

    def _get_current_episode(self) -> Optional[Any]:
        if self.env_interface is None:
            return None
        try:
            return self.env_interface.env.env.env._env.current_episode
        except Exception:
            return None

    def _get_task_family(self, info: Optional[Dict[str, Any]] = None) -> str:
        if isinstance(info, dict) and info.get("task_family"):
            return str(info["task_family"])
        episode = self._get_current_episode()
        if episode is None:
            return "other"
        return infer_task_family(episode)

    def _build_encode_info(
        self,
        info: Optional[Dict[str, Any]],
        state_kind: str,
        task_family: str,
    ) -> Optional[Dict[str, Any]]:
        if info is None:
            return None
        encode_info = dict(info)
        encode_info["state_kind"] = state_kind
        encode_info["task_family"] = task_family
        return encode_info

    @staticmethod
    def _extract_task_context_snapshot(
        world_state: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not isinstance(world_state, dict):
            return {}
        task_context = world_state.get("task_context", {}) or {}
        if isinstance(task_context, Mapping):
            return dict(task_context)
        return {}

    def _add_replan_window_tags(
        self,
        transitions: List[Dict[str, Any]],
        tags_by_index: Dict[int, set],
    ) -> None:
        if not transitions:
            return
        replan_anchor_indices: List[int] = []
        previous_planning_step: Optional[int] = None
        for index, transition in enumerate(transitions):
            planning_step = self._info_planning_step_count(transition.get("info"))
            if previous_planning_step is None:
                previous_planning_step = planning_step
                continue
            if planning_step != previous_planning_step:
                replan_anchor_indices.append(index)
            previous_planning_step = planning_step

        sim_steps = [
            self._info_sim_step_count(transition.get("info")) for transition in transitions
        ]
        for anchor_index in replan_anchor_indices:
            anchor_sim_step = sim_steps[anchor_index]
            tags_by_index.setdefault(anchor_index, set()).add("replan_anchor")
            for index, sim_step in enumerate(sim_steps):
                delta = sim_step - anchor_sim_step
                if delta < 0 and abs(delta) <= self.analysis_replan_pre_sim_steps:
                    tags_by_index.setdefault(index, set()).add("replan_pre_window")
                elif delta > 0 and delta <= self.analysis_replan_post_sim_steps:
                    tags_by_index.setdefault(index, set()).add("replan_post_window")

    def _add_progress_and_response_tags(
        self,
        transitions: List[Dict[str, Any]],
        tags_by_index: Dict[int, set],
    ) -> None:
        previous_progress: Optional[float] = None
        previous_success: Optional[float] = None
        for index, transition in enumerate(transitions):
            info = transition.get("info")
            progress = self._info_progress(info)
            success = self._safe_float((info or {}).get("task_state_success"), 0.0)
            if (
                self.analysis_export_keep_progress_transitions
                and previous_progress is not None
                and abs(progress - previous_progress)
                > self.analysis_export_progress_delta_threshold
            ):
                tags_by_index.setdefault(index, set()).add("progress_change")
            if (
                self.analysis_export_keep_success_transition
                and previous_success is not None
                and success != previous_success
            ):
                tags_by_index.setdefault(index, set()).add("success_change")
            if (
                self.analysis_export_keep_response_transitions
                and self._recent_response_text(info)
            ):
                tags_by_index.setdefault(index, set()).add("response_event")
            previous_progress = progress
            previous_success = success

    def _add_td_error_tags(
        self,
        td_errors: np.ndarray,
        tags_by_index: Dict[int, set],
    ) -> None:
        if self.analysis_export_keep_td_error_topk <= 0 or td_errors.size == 0:
            return
        ranked = np.argsort(np.abs(td_errors))[::-1][: self.analysis_export_keep_td_error_topk]
        for raw_index in ranked.tolist():
            index = int(raw_index)
            tags_by_index.setdefault(index, set()).add("td_error_peak")

    def _select_analysis_indices(
        self,
        transitions: List[Dict[str, Any]],
        td_errors: np.ndarray,
        *,
        include_full_trajectory: Optional[bool] = None,
    ) -> Tuple[List[int], Dict[int, List[str]], Dict[str, Any]]:
        tags_by_index: Dict[int, set] = {}
        total_transitions = len(transitions)
        if total_transitions == 0:
            return [], {}, {
                "focus_mode": self.analysis_focus_mode,
                "total_transitions": 0,
                "selected_transitions": 0,
                "selection_reason_counts": {},
            }

        if self.analysis_export_keep_first_last:
            tags_by_index.setdefault(0, set()).add("episode_start")
            tags_by_index.setdefault(total_transitions - 1, set()).add("episode_end")

        self._add_replan_window_tags(transitions, tags_by_index)
        self._add_progress_and_response_tags(transitions, tags_by_index)
        self._add_td_error_tags(td_errors, tags_by_index)

        full_trajectory = (
            self.analysis_export_full_trajectory
            if include_full_trajectory is None
            else bool(include_full_trajectory)
        )
        if full_trajectory:
            for index in range(total_transitions):
                tags_by_index.setdefault(index, set()).add("stability_trajectory")

        selected_items = sorted(
            tags_by_index.items(),
            key=lambda item: (
                self._info_sim_step_count(transitions[item[0]].get("info")),
                item[0],
            ),
        )
        if (
            not full_trajectory
            and len(selected_items) > self.analysis_export_max_selected_transitions
        ):
            selected_items = selected_items[: self.analysis_export_max_selected_transitions]

        tags_output = {
            index: sorted(list(reason_tags))
            for index, reason_tags in selected_items
        }
        reason_counter = Counter(
            reason
            for reasons in tags_output.values()
            for reason in reasons
        )
        summary = {
            "focus_mode": self.analysis_focus_mode,
            "total_transitions": total_transitions,
            "selected_transitions": len(selected_items),
            "selection_reason_counts": dict(reason_counter),
            "replan_pre_sim_steps": self.analysis_replan_pre_sim_steps,
            "replan_post_sim_steps": self.analysis_replan_post_sim_steps,
            "max_selected_transitions": self.analysis_export_max_selected_transitions,
            "full_stability_trajectory": full_trajectory,
        }
        return list(tags_output.keys()), tags_output, summary

    def _build_llm_action_summary(
        self,
        transition: Mapping[str, Any],
        tags: List[str],
    ) -> Dict[str, Any]:
        info = transition.get("info", {}) or {}
        return {
            "action_type": self._action_to_text(transition.get("action")),
            "sim_step_count": self._info_sim_step_count(info),
            "planning_step_count": self._info_planning_step_count(info),
            "event_tags": list(tags),
            "recent_response": self._recent_response_text(info),
        }

    def _build_offline_trajectory(
        self,
        transitions: List[Dict[str, Any]],
        selected_indices: List[int],
        tags_by_index: Dict[int, List[str]],
        focus_records: Optional[Mapping[int, Dict[str, Any]]] = None,
    ) -> List[Tuple[Any, Any, float, Any, bool, int, int, str]]:
        trajectory: List[Tuple[Any, Any, float, Any, bool, int, int, str]] = []
        focus_records = focus_records or {}
        for index in selected_indices:
            transition = transitions[index]
            info = transition.get("info", {}) or {}
            focused = focus_records.get(index, {}) or {}
            state_artifacts = focused.get("state_artifacts") or transition.get(
                "state_artifacts", {}
            )
            next_state_artifacts = focused.get(
                "next_state_artifacts"
            ) or transition.get("next_state_artifacts", {})
            state_desc = (
                state_artifacts.get("text_desc")
                or self.state_encoder._generate_state_description(
                    focused.get("state") or transition.get("state_world") or {}
                )
            )
            next_state_desc = (
                next_state_artifacts.get("text_desc")
                or self.state_encoder._generate_state_description(
                    focused.get("next_state")
                    or transition.get("next_state_world")
                    or {}
                )
            )
            event_tags = tags_by_index.get(index, [])
            trajectory.append(
                (
                    state_desc,
                    self._build_llm_action_summary(transition, event_tags),
                    float(transition.get("reward_shaped", 0.0)),
                    next_state_desc,
                    bool(transition.get("done", False)),
                    self._info_planning_step_count(info),
                    self._info_sim_step_count(info),
                    "|".join(event_tags),
                )
            )
        return trajectory

    def _copy_encode_artifacts(self) -> Dict[str, Any]:
        return self.state_encoder.get_last_forward_record()

    def _episode_export_enabled(self, transitions: List[Dict[str, Any]]) -> bool:
        if self.analysis_export_enabled:
            return True
        for transition in transitions:
            info = transition.get("info", {}) or {}
            if isinstance(info, dict) and info.get("analysis_export_enabled"):
                return True
        return False

    def _build_export_record(
        self,
        transition_index: int,
        state_kind: str,
        task_family: str,
        predicted_value: Optional[float],
        reward_env: Optional[float],
        reward_shaped: Optional[float],
        done: bool,
        info: Dict[str, Any],
        artifacts: Dict[str, Any],
        world_state: Any,
        value_features: Optional[np.ndarray] = None,
        gae_advantage: Optional[float] = None,
        return_target: Optional[float] = None,
        td_error: Optional[float] = None,
        next_predicted_value: Optional[float] = None,
        next_value_features: Optional[np.ndarray] = None,
        analysis_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        proposition_outcome = proposition_outcome_from_info(info)
        feature_names = artifacts.get("feature_names", [])
        export_world_state = self._export_world_state_snapshot(world_state)
        export_info = self._export_info_snapshot(info)
        task_context_snapshot = self._extract_task_context_snapshot(world_state)
        sim_step_count = self._info_sim_step_count(info)
        loop_step_count = self._info_loop_step_count(info)
        planning_step_count = self._info_planning_step_count(info)
        analysis_tags = list(analysis_tags or [])
        record = {
            "transition_index": transition_index,
            "episode_id": info.get("episode_id") if isinstance(info, dict) else None,
            "step_count": sim_step_count,
            "sim_step_count": sim_step_count,
            "loop_step_count": loop_step_count,
            "planning_step_count": planning_step_count,
            "state_kind": state_kind,
            "task_family": task_family,
            "planning_transition": bool("replan_anchor" in analysis_tags),
            "analysis_tags": analysis_tags,
            "world_state_hash": artifacts.get("world_state_hash"),
            "task_context_hash": artifacts.get("task_context_hash"),
            "text_desc_hash": artifacts.get("text_desc_hash"),
            "numerical_features": artifacts.get("numerical_features", []),
            "state_vector": artifacts.get("state_vector", []),
            "value_features": (
                value_features.tolist()
                if hasattr(value_features, "tolist")
                else (list(value_features) if value_features is not None else [])
            ),
            "predicted_value": predicted_value,
            "done": bool(done),
            "task_percent_complete": info.get("task_percent_complete", 0.0)
            if isinstance(info, dict)
            else 0.0,
            "task_state_success": info.get("task_state_success", 0.0)
            if isinstance(info, dict)
            else 0.0,
            "proposition_satisfied_fraction": (
                proposition_outcome.fraction
            ),
            "gae_advantage": gae_advantage,
            "return_target": return_target,
            "td_error": td_error,
            "semantic_text_hash": artifacts.get("semantic_text_hash"),
        }
        if self.analysis_export_include_reward_fields:
            record["reward_env"] = reward_env
            record["reward_shaped"] = reward_shaped
        if self.analysis_export_include_text_embedding:
            record["text_embedding"] = artifacts.get("text_embedding", [])
        if self.analysis_export_include_text_desc:
            record["text_desc"] = artifacts.get("text_desc")
        if self.analysis_export_include_text_segments:
            record["text_segments"] = artifacts.get("text_segments", [])
        if self.analysis_export_include_category_presence:
            record["category_presence"] = artifacts.get("category_presence", {})
        if self.analysis_export_include_feature_schema_per_record:
            record["feature_names"] = feature_names
            record["feature_categories"] = artifacts.get("feature_categories", [])
            record["input_variant"] = artifacts.get("input_variant")
            record["context_variant"] = artifacts.get("context_variant")
        if self.analysis_export_include_world_state and export_world_state is not None:
            record["world_state"] = export_world_state
        if self.analysis_export_include_task_context:
            if export_world_state is not None and export_world_state.get("task_context"):
                record["task_context"] = export_world_state.get("task_context", {})
            else:
                record["task_context"] = task_context_snapshot
        if export_info is not None:
            record["info_snapshot"] = export_info
        if next_predicted_value is not None:
            record["next_predicted_value"] = next_predicted_value
        if (
            self.analysis_export_include_next_value_features
            and next_value_features is not None
        ):
            record["next_value_features"] = (
                next_value_features.tolist()
                if hasattr(next_value_features, "tolist")
                else list(next_value_features)
            )
        return record

    def _export_episode_analysis(
        self,
        transitions: List[Dict[str, Any]],
        values: np.ndarray,
        next_values: np.ndarray,
        value_features: np.ndarray,
        next_value_features: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        focus_records: Optional[Mapping[int, Dict[str, Any]]] = None,
    ) -> None:
        if len(transitions) == 0:
            return
        if not self._episode_export_enabled(transitions):
            return
        os.makedirs(self.analysis_export_dir, exist_ok=True)

        infos = [transition["info"] for transition in transitions]
        task_family = transitions[0].get("task_family", self._get_task_family(infos[0]))
        rewards = np.array(
            [float(t["reward_shaped"]) for t in transitions], dtype=np.float32
        )
        td_errors = (
            np.asarray(rewards, dtype=np.float32)
            + self.gamma * np.asarray(next_values, dtype=np.float32) * (1.0 - np.asarray([float(bool(t["done"])) for t in transitions], dtype=np.float32))
            - np.asarray(values, dtype=np.float32)
        )
        selected_indices, tags_by_index, selection_summary = self._select_analysis_indices(
            transitions,
            td_errors,
        )
        records: List[Dict[str, Any]] = []
        focus_records = focus_records or {}
        for index in selected_indices:
            transition = transitions[index]
            focused = focus_records.get(index, {}) or {}
            reward_env = float(transition["reward_env"])
            reward_shaped = float(transition["reward_shaped"])
            done = bool(transition["done"])
            td_error = float(td_errors[index])
            state_record = self._build_export_record(
                transition_index=index,
                state_kind="state",
                task_family=task_family,
                predicted_value=float(values[index]),
                reward_env=reward_env,
                reward_shaped=reward_shaped,
                done=done,
                info=transition["info"],
                artifacts=focused.get("state_artifacts")
                or transition["state_artifacts"],
                world_state=focused.get("state")
                or transition.get("state_world"),
                value_features=value_features[index],
                gae_advantage=float(advantages[index]),
                return_target=float(returns[index]),
                td_error=td_error,
                next_predicted_value=float(next_values[index]),
                next_value_features=next_value_features[index],
                analysis_tags=tags_by_index.get(index, []),
            )
            records.append(state_record)
            if self.analysis_export_record_mode == "state_and_next":
                records.append(
                    self._build_export_record(
                        transition_index=index,
                        state_kind="next_state",
                        task_family=task_family,
                        predicted_value=float(next_values[index]),
                        reward_env=reward_env,
                        reward_shaped=reward_shaped,
                        done=done,
                        info=transition["info"],
                        artifacts=focused.get("next_state_artifacts")
                        or transition["next_state_artifacts"],
                        world_state=focused.get("next_state")
                        or transition.get("next_state_world"),
                        value_features=next_value_features[index],
                        analysis_tags=tags_by_index.get(index, []),
                    )
                )

        export_data = {
            "task_family": task_family,
            "schema_version": self.analysis_export_schema_version,
            "inference_mode": self.inference_mode,
            "state_encoder_input_variant": self.state_encoder.input_variant,
            "state_encoder_context_variant": self.state_encoder.context_variant,
            "analysis_export_world_state_mode": self.analysis_export_world_state_mode,
            "analysis_export_info_mode": self.analysis_export_info_mode,
            "analysis_export_record_mode": self.analysis_export_record_mode,
            "analysis_export_include_text_embedding": self.analysis_export_include_text_embedding,
            "analysis_export_minimal_schema": self.analysis_export_minimal_schema,
            "analysis_export_include_task_context": self.analysis_export_include_task_context,
            "analysis_export_size_target": self.analysis_export_size_target,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "critic_phase": self.phase,
            "judge_model": self.judge_model,
            "state_encoder_id": self.state_encoder_id,
            "value_checkpoint_id": self.value_checkpoint_id,
            "reward_shaper_valid": bool(self.reward_shaper_valid),
            "critic_lifecycle_valid": bool(self.critic_lifecycle_valid),
            "export_manifest": self._analysis_export_manifest(),
            "selection_summary": selection_summary,
            "focus_export_path": self.last_focus_export_path,
            "feature_schema": dict(self._episode_feature_schema),
            "records": records,
        }
        episode_filename = transitions[0]["info"].get("episode_filename", "episode")
        if self.env_interface is None:
            return
        self.last_export_path = write_critic_trajectory_export(
            export_data=export_data,
            output_dir=self._resolve_export_output_dir(),
            episode_filename=episode_filename,
            env_interface=self.env_interface,
        )

    def evaluate(self, state: Any, action: Any = None) -> float:
        """
        Evaluate state value V(s).

        Args:
            state: world_state_dict or pre-encoded state vector
            action: Action (not used in V(s), kept for interface compatibility)

        Returns:
            value: V(s) estimate
        """
        # Encode state if needed
        if isinstance(state, dict):
            state_vec = self.state_encoder.encode(state)
        else:
            state_vec = state

        # Ensure tensor is on correct device
        if not isinstance(state_vec, torch.Tensor):
            state_vec = torch.tensor(state_vec, dtype=torch.float32, device=self.device)
        else:
            state_vec = state_vec.to(self.device)

        # Forward pass through value network
        with torch.no_grad():
            value = self.value_network(state_vec).item()

        return value

    def update(
        self,
        state: Any,
        action: Any,
        reward: float,
        next_state: Any,
        done: bool,
        info: Dict[str, Any]
    ) -> None:
        """
        Update Critic based on transition.

        Args:
            state: Current world_state_dict
            action: Action taken
            reward: Environment reward
            next_state: Next world_state_dict
            done: Episode termination flag
            info: Additional info. ``task_instruction`` remains the
                planner-visible compatibility field; ``reward_instruction``
                carries the canonical goal for reward shaping.
        """
        # Apply LLM reward shaping if enabled
        if self.reward_shaper:
            reward_instruction = info.get(
                "reward_instruction",
                info.get("canonical_instruction", info.get("task_instruction", "")),
            )

            # Enhance info with planning step context for LLM reward shaper
            enhanced_info = dict(info) if info else {}
            enhanced_info['planning_step_count'] = self._info_planning_step_count(info)
            enhanced_info['sim_step_count'] = self._info_sim_step_count(info)

            # Add Rebound context if available
            if self.env_interface and hasattr(self.env_interface, 'planner'):
                planner = self.env_interface.planner
                # Handle both centralized (single planner) and decentralized (dict of planners)
                if isinstance(planner, dict):
                    # Decentralized: use first agent's planner
                    planner = next(iter(planner.values())) if planner else None

                if planner and hasattr(planner, 'rebound_managers'):
                    # Get first agent's rebound manager
                    rebound_mgr = next(iter(planner.rebound_managers.values()), None)
                    if rebound_mgr:
                        enhanced_info['rebound_recovery_active'] = getattr(rebound_mgr, 'recovery_in_progress', False)
                        enhanced_info['rebound_tags'] = list(getattr(rebound_mgr, 'active_recovery_types', []))
                        enhanced_info['last_replan_sim_step'] = getattr(rebound_mgr, 'last_replan_sim_step', 0)

            shaped_reward = self.reward_shaper.shape_reward(
                state,
                action,
                next_state,
                reward,
                reward_instruction,
                enhanced_info,
            )
            self.reward_shaper_valid = bool(
                self.reward_shaper_valid
                and getattr(self.reward_shaper, "valid", True)
            )
            if not self.reward_shaper_valid:
                self.critic_lifecycle_valid = False
        else:
            shaped_reward = reward

        # Encode states
        task_family = self._get_task_family(info)
        state_info = self._build_encode_info(info, "state", task_family)
        next_state_info = self._build_encode_info(info, "next_state", task_family)
        state_vec = self.state_encoder.encode(state, state_info)
        state_artifacts = self._copy_encode_artifacts()
        next_state_vec = self.state_encoder.encode(next_state, next_state_info)
        next_state_artifacts = self._copy_encode_artifacts()

        compact_info = self._compact_transition_info(info)
        transition_index = len(self.buffer)
        focus_candidate = self._focus_candidate(
            transition_index=transition_index,
            action=action,
            reward_env=float(reward),
            reward_shaped=float(shaped_reward),
            done=bool(done),
            info=compact_info,
            state=state,
            next_state=next_state,
            state_artifacts=state_artifacts,
            next_state_artifacts=next_state_artifacts,
        )
        self._capture_focus_transition(
            transition_index,
            focus_candidate,
            compact_info,
            bool(done),
        )

        # The exact numerical trajectory stays in memory for TD/GAE. Rich state,
        # text and tracker payloads are sampled into the on-disk focus stream.
        self.buffer.append(
            {
                "state_vec": state_vec.detach().cpu(),
                "action": action,
                "reward_env": float(reward),
                "reward_shaped": float(shaped_reward),
                "next_state_vec": next_state_vec.detach().cpu(),
                "done": bool(done),
                "info": compact_info,
                "state_world": None,
                "next_state_world": None,
                "state_artifacts": self._compact_encode_artifacts(state_artifacts),
                "next_state_artifacts": self._compact_encode_artifacts(
                    next_state_artifacts
                ),
                "task_family": task_family,
            }
        )

        # Process episode when done
        if done:
            self._process_episode()

    def reset(self) -> None:
        """Reset Critic state for new episode."""
        self._close_focus_stream(write_episode_end=False)
        self.buffer = []
        self.raw_states = []
        self.last_episode_value_history = []
        self.last_episode_value_delta_history = []
        self.last_episode_value_variance = 0.0
        self.last_episode_value_delta_variance = 0.0
        self.last_episode_stability_trace_summary = {}
        self.last_analysis_path = None
        self.last_export_path = None
        self.last_focus_export_path = None
        self._episode_feature_schema = {}
        self._focus_recent.clear()
        self._focus_stream_path = None
        self._focus_written_indices.clear()
        self._focus_previous_planning_step = None
        self._focus_previous_progress = None
        self._focus_previous_success = None
        self._focus_post_remaining = 0
        self._focus_last_candidate = None

    def force_end_episode(self) -> None:
        """
        Force end the current episode.
        Useful when the planner decides to end the episode but the environment done flag was False.
        This ensures the episode is processed and the model is updated.
        """
        if len(self.buffer) > 0:
            # Mark the last transition as done
            self.buffer[-1]["done"] = True

            # Also update raw_states if they exist
            if self.offline_analyzer and len(self.raw_states) > 0:
                # Update done flag in raw_states
                # raw_states structure: (state, action, reward, next_state, done, planning_step)
                last_raw = list(self.raw_states[-1])
                last_raw[4] = True
                self.raw_states[-1] = tuple(last_raw)

            # Process the episode
            self._process_episode()

    def _process_episode(self) -> None:
        """
        Process complete episode:
        1. Compute V(s) and V(s') for all transitions
        2. Calculate GAE advantages
        3. Update value network
        4. Run LLM offline analysis (if enabled)
        """
        if len(self.buffer) == 0:
            logger.warning("Empty buffer in _process_episode, skipping")
            return
        self._close_focus_stream()

        # Unpack buffer
        states = torch.stack([transition["state_vec"] for transition in self.buffer]).to(
            self.device
        )
        next_states = torch.stack(
            [transition["next_state_vec"] for transition in self.buffer]
        ).to(self.device)
        rewards = np.array(
            [transition["reward_shaped"] for transition in self.buffer],
            dtype=np.float32,
        )
        dones = np.array(
            [transition["done"] for transition in self.buffer],
            dtype=np.float32,
        )
        infos = [transition["info"] for transition in self.buffer]

        # Compute V(s) and V(s')
        with torch.no_grad():
            values_tensor, state_value_features_tensor = self.value_network.forward_with_features(states)
            next_values_tensor, next_value_features_tensor = self.value_network.forward_with_features(next_states)
            values = values_tensor.cpu().numpy()
            next_values = next_values_tensor.cpu().numpy()
            state_value_features = state_value_features_tensor.cpu().numpy()
            next_value_features = next_value_features_tensor.cpu().numpy()

        value_array = np.asarray(values, dtype=np.float32).reshape(-1)
        value_delta_array = np.diff(value_array)
        self.last_episode_value_history = [float(v) for v in value_array.tolist()]
        self.last_episode_value_delta_history = [
            float(v) for v in value_delta_array.tolist()
        ]
        self.last_episode_value_variance = (
            float(np.var(value_array)) if value_array.size > 1 else 0.0
        )
        self.last_episode_value_delta_variance = (
            float(np.var(value_delta_array)) if value_delta_array.size > 0 else 0.0
        )
        self.training_stats["last_episode_value_variance"] = (
            self.last_episode_value_variance
        )
        self.training_stats["last_episode_value_delta_variance"] = (
            self.last_episode_value_delta_variance
        )

        # Compute GAE
        advantages, returns = compute_gae(
            rewards, values, next_values, dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )
        td_errors = rewards + self.gamma * next_values * (1.0 - dones) - values
        self.last_episode_stability_trace_summary = (
            self._compute_stability_trace_summary(
                values=values,
                next_values=next_values,
                rewards=rewards,
                dones=dones,
                advantages=advantages,
                returns=returns,
                value_features=state_value_features,
            )
        )
        self.training_stats.update(
            {
                f"last_episode_{key}": value
                for key, value in self.last_episode_stability_trace_summary.items()
                if isinstance(value, (int, float, str, bool))
            }
        )
        selected_indices, tags_by_index, _ = self._select_analysis_indices(
            self.buffer,
            td_errors,
            include_full_trajectory=False,
        )
        focus_records = self._load_focus_candidates()
        self.raw_states = self._build_offline_trajectory(
            self.buffer,
            selected_indices,
            tags_by_index,
            focus_records,
        )

        # Update value network
        if self.update_value_network:
            self._update_value_network(states, returns)
        self._export_episode_analysis(
            transitions=self.buffer,
            values=values,
            next_values=next_values,
            value_features=state_value_features,
            next_value_features=next_value_features,
            advantages=advantages,
            returns=returns,
            focus_records=focus_records,
        )

        # Run LLM offline analysis (if enabled)
        if self.offline_analyzer and len(self.raw_states) > 0:
            # Offline behavioural analysis follows the policy-visible task.
            task_instruction = infos[0].get(
                "policy_instruction", infos[0].get("task_instruction", "")
            )
            success = infos[-1].get("task_state_success", False)
            percent_complete = infos[-1].get("task_percent_complete", 0.0)
            total_reward = sum(rewards)

            episode_id = infos[0].get("episode_id") if infos else None
            episode_filename = infos[0].get("episode_filename") if infos else None
            
            # Extract max planning step from selected analysis trajectory
            final_planning_step = max(
                (self._safe_int(step[5], 0) for step in self.raw_states if len(step) > 5),
                default=0,
            )

            analysis, filepath = self.offline_analyzer.analyze_episode(
                trajectory=self.raw_states,
                task_instruction=task_instruction,
                success=success,
                percent_complete=percent_complete,
                total_reward=total_reward,
                episode_id=episode_id,
                episode_filename=episode_filename,
                planning_step_count=final_planning_step
            )

            if analysis:
                logger.info(f"[LLM Analysis] {analysis.get('overall_assessment', 'N/A')}")
                if filepath:
                    self.last_analysis_path = filepath
                    logger.info(f"Recorded analysis path: {filepath}")
                else:
                    logger.warning("Analysis produced but no filepath returned (save_analyses=False?)")
            else:
                logger.debug("No analysis produced (skipped or failed)")

        if hasattr(self.state_encoder, "flush_debug"):
            try:
                self.state_encoder.flush_debug(infos[-1] if infos else None)
            except Exception as e:
                logger.debug(f"StateEncoder debug flush failed: {e}")

        # Save checkpoint if enabled
        self.training_stats["episodes_trained"] += 1
        if (self.save_checkpoints and
            self.training_stats["episodes_trained"] % self.checkpoint_frequency == 0):
            self.save_checkpoint()

        # Clear buffers
        self.buffer = []
        self.raw_states = []
        self._focus_recent.clear()
        self._focus_written_indices.clear()
        self._focus_last_candidate = None

    def _compute_stability_trace_summary(
        self,
        *,
        values: np.ndarray,
        next_values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        value_features: np.ndarray,
    ) -> Dict[str, Any]:
        """Summarize the critic trajectory for beta-stability.

        The paper definition uses an execution oscillation loss over value
        variation, TD residual spikes, and GAE advantages. At runtime we expose
        the raw diagonal proxy, while the perturbation-level beta is computed
        later against clean references.
        """
        value_arr = np.asarray(values, dtype=np.float32).reshape(-1)
        next_value_arr = np.asarray(next_values, dtype=np.float32).reshape(-1)
        reward_arr = np.asarray(rewards, dtype=np.float32).reshape(-1)
        done_arr = np.asarray(dones, dtype=np.float32).reshape(-1)
        advantage_arr = np.asarray(advantages, dtype=np.float32).reshape(-1)
        return_arr = np.asarray(returns, dtype=np.float32).reshape(-1)
        if value_arr.size == 0:
            return {
                "critic_trace_len": 0,
                "oscillation_loss": 0.0,
                "oscillation_loss_valid": False,
                "oscillation_source": "empty_critic_trace",
                "td_error_mean_abs": 0.0,
                "td_error_p95_abs": 0.0,
                "gae_mean_abs": 0.0,
                "gae_p95_abs": 0.0,
                "return_target_variance": 0.0,
            }

        td_errors = reward_arr + self.gamma * next_value_arr * (1.0 - done_arr) - value_arr
        value_deltas = np.diff(value_arr, prepend=value_arr[0])
        if value_features is not None and np.asarray(value_features).ndim == 2:
            feature_arr = np.asarray(value_features, dtype=np.float32)
            feature_deltas = np.linalg.norm(
                np.diff(feature_arr, axis=0, prepend=feature_arr[:1]),
                axis=1,
            )
        else:
            feature_deltas = np.zeros_like(value_arr)

        components = np.stack(
            [
                np.abs(value_deltas),
                np.abs(td_errors),
                np.abs(advantage_arr),
                np.abs(feature_deltas),
            ],
            axis=1,
        )
        weights = np.asarray(
            self.config.get(
                "stability_oscillation_component_weights",
                [1.0, 1.0, 1.0, 0.25],
            ),
            dtype=np.float32,
        )
        if weights.size != components.shape[1]:
            weights = np.asarray([1.0, 1.0, 1.0, 0.25], dtype=np.float32)
        step_loss = np.sqrt(np.sum((components * weights) ** 2, axis=1))
        oscillation_loss = float(np.mean(step_loss)) if step_loss.size else 0.0
        abs_td = np.abs(td_errors)
        abs_adv = np.abs(advantage_arr)
        return {
            "critic_trace_len": int(value_arr.size),
            "oscillation_loss": oscillation_loss,
            "oscillation_loss_valid": True,
            "oscillation_source": "critic_value_td_gae",
            "td_error_mean_abs": float(np.mean(abs_td)) if abs_td.size else 0.0,
            "td_error_p95_abs": float(np.percentile(abs_td, 95)) if abs_td.size else 0.0,
            "gae_mean_abs": float(np.mean(abs_adv)) if abs_adv.size else 0.0,
            "gae_p95_abs": float(np.percentile(abs_adv, 95)) if abs_adv.size else 0.0,
            "return_target_variance": (
                float(np.var(return_arr)) if return_arr.size > 1 else 0.0
            ),
        }

    def _update_value_network(self, states: torch.Tensor, returns: np.ndarray) -> None:
        """
        Update value network using computed returns.

        Loss: MSE(V(s), returns)

        Args:
            states: Batch of state vectors [T, state_dim]
            returns: Target returns [T]
        """
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Forward pass
        predicted_values = self.value_network(states)

        # Compute loss
        value_loss = F.mse_loss(predicted_values, returns_tensor)

        # Backward pass
        self.value_optimizer.zero_grad()
        value_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)

        # Update weights
        self.value_optimizer.step()

        # Update statistics
        self.training_stats["value_loss"] = value_loss.item()
        self.training_stats["mean_predicted_value"] = predicted_values.mean().item()
        self.training_stats["mean_return"] = returns.mean()

        logger.debug(f"Value network updated: loss={value_loss.item():.4f}, "
                    f"mean_pred={predicted_values.mean().item():.4f}, "
                    f"mean_return={returns.mean():.4f}")

    # === Q-value, Bellman Error, and Advantage methods ===
    def compute_q_value(self, state: Any, action: Any, reward: float,
                       next_state: Any, done: bool) -> float:
        """
        Compute Q(s,a) = r + γ V(s') (1 - done)

        This is an approximate Q-value since we only have V network.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag

        Returns:
            q_value: Q(s,a) estimate
        """
        next_value = self.evaluate(next_state)
        q_value = reward + self.gamma * next_value * (1 - done)
        return q_value

    def compute_bellman_error(self, state: Any, action: Any, reward: float,
                              next_state: Any, done: bool) -> float:
        """
        Compute TD error (Bellman Error):
        δ = r + γ V(s') (1 - done) - V(s)

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag

        Returns:
            td_error: Temporal difference error
        """
        current_value = self.evaluate(state)
        next_value = self.evaluate(next_state)

        td_error = reward + self.gamma * next_value * (1 - done) - current_value
        return td_error

    def get_advantage(self, state: Any, action: Any, reward: float,
                     next_state: Any, done: bool) -> float:
        """
        Compute 1-step advantage estimate:
        A(s,a) = Q(s,a) - V(s) = δ

        Note: In _process_episode we use GAE for more accurate advantage.
        This method provides a quick 1-step estimate.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag

        Returns:
            advantage: 1-step advantage estimate
        """
        return self.compute_bellman_error(state, action, reward, next_state, done)

    # === Statistics and Checkpoint methods ===

    def _get_recent_value_history(self, window: int = 10) -> List[float]:
        if len(self.buffer) >= 1:
            states = [transition["state_vec"] for transition in self.buffer]
            states_tensor = torch.stack(states).to(self.device)
            with torch.no_grad():
                values = self.value_network(states_tensor).cpu().numpy().flatten()
            history = [float(value) for value in values.tolist()]
        else:
            history = list(self.last_episode_value_history)

        if window > 0 and len(history) > window:
            history = history[-window:]
        return history

    def get_recent_value_statistics(self, window: int = 10) -> Dict[str, Any]:
        value_history = self._get_recent_value_history(window=window)
        if len(value_history) > 1:
            value_variance = float(np.var(value_history))
            value_deltas = np.diff(np.asarray(value_history, dtype=np.float32))
            value_delta_variance = (
                float(np.var(value_deltas)) if value_deltas.size > 0 else 0.0
            )
            delta_history = [float(value) for value in value_deltas.tolist()]
        else:
            value_variance = 0.0
            value_delta_variance = 0.0
            delta_history = []

        return {
            "value_history": value_history,
            "value_delta_history": delta_history,
            "value_variance": value_variance,
            "value_delta_variance": value_delta_variance,
        }

    def get_stability_trace_summary(self) -> Dict[str, Any]:
        """Return the last completed episode's critic-derived stability summary."""
        return dict(self.last_episode_stability_trace_summary or {})

    def compute_value_variance(self, window: int = 10) -> float:
        """
        Compute the variance of Value function V(s) over a sliding window.
        Metrics.tex: sigma^2_V
        """
        return float(
            self.get_recent_value_statistics(window=window)["value_variance"]
        )

    def compute_value_delta_variance(self, window: int = 10) -> float:
        """
        Compute the variance of first-order value deltas.
        """
        return float(
            self.get_recent_value_statistics(window=window)["value_delta_variance"]
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = self.training_stats.copy()
        stats["export_manifest"] = self._analysis_export_manifest()
        if self.last_analysis_path:
            stats["last_llm_analysis"] = self.last_analysis_path
        if self.last_export_path:
            stats["last_export_path"] = self.last_export_path
        return stats

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        """
        Save model checkpoint.

        Args:
            path: Optional custom path. If None, uses default checkpoint dir.
        """
        if path is None:
            episode_num = self.training_stats["episodes_trained"]
            filename = f"critic_checkpoint_ep{episode_num}.pt"
            path = f"{self.checkpoint_dir}/{filename}"

        try:
            torch.save({
                'state_encoder': self.state_encoder.state_dict(),
                'value_network': self.value_network.state_dict(),
                'optimizer': self.value_optimizer.state_dict(),
                'training_stats': self.training_stats,
                'config': self.config,
            }, path)
            logger.info(f"Saved checkpoint to {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self, path: str) -> None:
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.state_encoder.load_state_dict(checkpoint['state_encoder'])
            self.value_network.load_state_dict(checkpoint['value_network'])
            self.value_optimizer.load_state_dict(checkpoint['optimizer'])
            self.training_stats = checkpoint.get('training_stats', self.training_stats)
            self._apply_execution_mode()
            logger.info(f"Loaded checkpoint from {path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
