#!/usr/bin/env python3

"""
Transition Processor

Handles processing of Critic transitions, including state encoding, reward shaping,
and transition data preparation for A2C training.
"""

from typing import Any, Dict, Optional
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


class TransitionProcessor:
    """
    Processes transitions for A2C Critic training.

    Responsibilities:
    - Encode states using StateEncoder
    - Apply reward shaping (intrinsic rewards, LLM shaping)
    - Construct transition dictionaries for buffer storage
    - Handle task context and metadata extraction
    """

    def __init__(self, state_encoder: Any, config: Dict[str, Any]):
        """
        Initialize TransitionProcessor.

        Args:
            state_encoder: StateEncoder instance for state encoding
            config: Configuration dictionary
        """
        self.state_encoder = state_encoder
        self.config = config

        # Reward shaping configuration
        self.use_intrinsic_reward = config.get("use_intrinsic_reward", False)
        self.intrinsic_reward_weight = config.get("intrinsic_reward_weight", 0.1)
        self.use_llm_shaping = config.get("use_llm_shaping", False)

        # LLM reward shaper (initialized externally if needed)
        self.reward_shaper = None

    def set_reward_shaper(self, reward_shaper: Any) -> None:
        """
        Set LLM reward shaper instance.

        Args:
            reward_shaper: LLMRewardShaper instance
        """
        self.reward_shaper = reward_shaper

    def process_transition(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
        info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process a single transition for Critic training.

        Args:
            state: Previous world state dictionary
            action: Action taken
            reward: Environment reward
            next_state: Current world state dictionary
            done: Whether episode is done
            info: Episode info dictionary

        Returns:
            Processed transition dictionary
        """
        # Extract task family and state kind
        task_family = self._get_task_family(info)
        state_kind = "state"

        # Build encoding info
        encode_info = self._build_encode_info(info, state_kind, task_family)

        # Encode states
        try:
            state_vec = self.state_encoder.encode(state, encode_info)
            next_state_vec = self.state_encoder.encode(next_state, encode_info)
        except Exception as e:
            logger.error(f"Failed to encode states: {e}", exc_info=True)
            # Return dummy transition on encoding failure
            return self._create_dummy_transition(action, reward, done, info)

        # Shape reward
        reward_shaped = self._shape_reward(
            reward=reward,
            state=state,
            action=action,
            next_state=next_state,
            info=info,
        )

        # Extract task context
        task_context = self._extract_task_context(state)

        # Construct transition
        transition = {
            "state": state,
            "state_vec": state_vec,
            "action": action,
            "reward": reward,
            "reward_shaped": reward_shaped,
            "next_state": next_state,
            "next_state_vec": next_state_vec,
            "done": done,
            "info": info,
            "task_context": task_context,
            "task_family": task_family,
        }

        return transition

    def _shape_reward(
        self,
        reward: float,
        state: Dict[str, Any],
        action: Dict[str, Any],
        next_state: Dict[str, Any],
        info: Dict[str, Any],
    ) -> float:
        """
        Apply reward shaping to environment reward.

        Args:
            reward: Base environment reward
            state: Previous world state
            action: Action taken
            next_state: Current world state
            info: Episode info

        Returns:
            Shaped reward
        """
        shaped_reward = reward

        # Intrinsic reward (progress-based)
        if self.use_intrinsic_reward:
            intrinsic = self._compute_intrinsic_reward(info)
            shaped_reward += self.intrinsic_reward_weight * intrinsic

        # LLM-based reward shaping
        if self.use_llm_shaping and self.reward_shaper is not None:
            try:
                llm_reward = self.reward_shaper.shape_reward(
                    state=state,
                    action=action,
                    next_state=next_state,
                    info=info,
                )
                shaped_reward += llm_reward
            except Exception as e:
                logger.debug(f"LLM reward shaping failed: {e}")

        return shaped_reward

    def _compute_intrinsic_reward(self, info: Dict[str, Any]) -> float:
        """
        Compute intrinsic reward based on task progress.

        Args:
            info: Episode info dictionary

        Returns:
            Intrinsic reward value
        """
        # Progress-based intrinsic reward
        progress = info.get("task_percent_complete", 0.0)
        prev_progress = info.get("prev_task_percent_complete", 0.0)
        progress_delta = progress - prev_progress

        # Positive reward for progress increase
        if progress_delta > 0:
            return progress_delta

        return 0.0

    def _build_encode_info(
        self,
        info: Optional[Dict[str, Any]],
        state_kind: str,
        task_family: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Build info dictionary for state encoding.

        Args:
            info: Episode info dictionary
            state_kind: State kind label
            task_family: Task family label

        Returns:
            Encoding info dictionary
        """
        if info is None:
            return None

        encode_info = dict(info)
        encode_info["state_kind"] = state_kind
        encode_info["task_family"] = task_family

        return encode_info

    def _get_task_family(self, info: Optional[Dict[str, Any]] = None) -> str:
        """
        Extract task family from info.

        Args:
            info: Episode info dictionary

        Returns:
            Task family string
        """
        if isinstance(info, dict) and info.get("task_family"):
            return str(info["task_family"])
        return "other"

    def _extract_task_context(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract task context from world state.

        Args:
            world_state: World state dictionary

        Returns:
            Task context dictionary
        """
        if not isinstance(world_state, dict):
            return {}

        task_context = world_state.get("task_context", {}) or {}
        if isinstance(task_context, dict):
            return dict(task_context)

        return {}

    def _create_dummy_transition(
        self,
        action: Dict[str, Any],
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create a dummy transition when encoding fails.

        Args:
            action: Action taken
            reward: Environment reward
            done: Whether episode is done
            info: Episode info

        Returns:
            Dummy transition dictionary
        """
        device = self.state_encoder.device
        dummy_vec = torch.zeros(self.state_encoder.output_dim, device=device)

        return {
            "state": {},
            "state_vec": dummy_vec,
            "action": action,
            "reward": reward,
            "reward_shaped": reward,
            "next_state": {},
            "next_state_vec": dummy_vec,
            "done": done,
            "info": info,
            "task_context": {},
            "task_family": "other",
        }


__all__ = ["TransitionProcessor"]
