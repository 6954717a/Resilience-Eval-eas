#!/usr/bin/env python3

"""
Critic Updater

Encapsulates critic step updates so EvaluationRunner only keeps episode-loop logic.
"""

import logging
import copy
from typing import Any, Dict, Optional

from habitat_llm.evaluation.experiments.common import infer_task_family

logger = logging.getLogger(__name__)


class CriticUpdater:
    """
    Manages A2C critic updates during episode execution.
    """

    def __init__(
        self,
        critic: Any,
        *,
        env_interface: Any = None,
        agents: Optional[Dict[int, Any]] = None,
        log_trajectory_export: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.critic = critic
        self.env_interface = env_interface
        self.agents = agents or {}
        self.log_trajectory_export = bool(log_trajectory_export)
        self.config = config or {}
        self.update_count = 0

    def _action_history_keep_last(self) -> int:
        for key in (
            "critic_action_history_keep_last",
            "analysis_export_info_action_history_keep_last",
        ):
            try:
                value = self.config.get(key)
            except Exception:
                value = None
            if value is None:
                continue
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                continue
        return 5

    @staticmethod
    def _compact_action_history_entry(action_entry: Any) -> Dict[str, Any]:
        if isinstance(action_entry, dict):
            action_payload = action_entry.get("action")
            response = action_entry.get("response", "")
            timestamp = action_entry.get("timestamp")
            agent_uid = action_entry.get("agent_uid")
        else:
            action_payload = getattr(action_entry, "action", action_entry)
            response = getattr(action_entry, "response", "")
            timestamp = getattr(action_entry, "timestamp", None)
            agent_uid = getattr(action_entry, "agent_uid", None)

        if isinstance(action_payload, tuple):
            action_value: Any = list(action_payload)
        else:
            action_value = action_payload

        compact = {
            "action": action_value,
            "response": "" if response in (None, "") else str(response),
            "timestamp": timestamp,
        }
        if agent_uid is not None:
            compact["agent_uid"] = agent_uid
        return compact

    def set_critic(self, critic: Any) -> None:
        self.critic = critic

    def update_step(
        self,
        *,
        prev_world_state: Optional[Dict[str, Any]],
        current_world_state: Dict[str, Any],
        action_info: Dict[str, Any],
        reward: float,
        done: bool,
        step_info: Dict[str, Any],
        planner_info: Dict[str, Any],
        current_instruction: str,
        episode_filename: str,
        loop_step_count: int,
    ) -> None:
        if prev_world_state is None:
            logger.debug("Skipping critic update: no previous state")
            return
        if self.critic is None:
            logger.debug("Skipping critic update: critic is None")
            return

        critic_info = self._build_critic_info(
            step_info=step_info,
            planner_info=planner_info,
            current_instruction=current_instruction,
            episode_filename=episode_filename,
            loop_step_count=loop_step_count,
        )

        try:
            self.critic.update(
                state=prev_world_state,
                action=action_info,
                reward=reward,
                next_state=current_world_state,
                done=done,
                info=critic_info,
            )
            self.update_count += 1
        except Exception as exc:
            logger.warning("Critic update failed: %s", exc)

    def force_end_episode(self) -> None:
        if self.critic is None:
            return
        try:
            if hasattr(self.critic, "force_end_episode"):
                self.critic.force_end_episode()
            elif hasattr(self.critic, "_process_episode") and len(
                getattr(self.critic, "buffer", [])
            ) > 0:
                self.critic._process_episode()
        except Exception as exc:
            logger.warning("Critic force_end_episode failed: %s", exc)

    def _build_critic_info(
        self,
        *,
        step_info: Dict[str, Any],
        planner_info: Dict[str, Any],
        current_instruction: str,
        episode_filename: str,
        loop_step_count: int,
    ) -> Dict[str, Any]:
        planning_step_count = 0
        replanning_count = planner_info.get("replanning_count")
        if isinstance(replanning_count, dict) and replanning_count:
            planning_step_count = max(int(count) for count in replanning_count.values())
        elif replanning_count is not None:
            try:
                planning_step_count = int(replanning_count)
            except Exception:
                planning_step_count = 0

        primary_agent_id = sorted(self.agents.keys())[0] if self.agents else None
        action_history = []
        action_history_count = 0
        if (
            primary_agent_id is not None
            and getattr(self.env_interface, "agent_action_history", None)
        ):
            full_action_history = list(
                self.env_interface.agent_action_history.get(primary_agent_id, [])
            )
            action_history_count = len(full_action_history)
            keep_last = self._action_history_keep_last()
            if keep_last > 0:
                full_action_history = full_action_history[-keep_last:]
            action_history = [
                self._compact_action_history_entry(item)
                for item in full_action_history
            ]

        # Habitat measure payloads are mutable and can be updated in-place by
        # later env steps.  Freeze the per-step view here so critic trajectory
        # exports preserve the stage frontier observed at this transition.
        proposition_tracker = copy.deepcopy(
            step_info.get("auto_eval_proposition_tracker", {}) or {}
        )
        action_responses = copy.deepcopy(planner_info.get("responses", {}) or {})

        critic_info = {
            "task_instruction": current_instruction,
            "task_percent_complete": step_info.get("task_percent_complete", 0.0),
            "task_state_success": step_info.get("task_state_success", 0.0),
            "step_count": step_info.get("num_steps", loop_step_count),
            "sim_step_count": step_info.get("num_steps", loop_step_count),
            "loop_step_count": loop_step_count,
            "planning_step_count": planning_step_count,
            "action_history": action_history,
            "action_history_count": action_history_count,
            "auto_eval_proposition_tracker": proposition_tracker,
            "action_responses": action_responses,
            "episode_filename": episode_filename,
            "analysis_export_enabled": self.log_trajectory_export,
        }

        current_episode = self._get_current_episode()
        if current_episode is not None:
            critic_info["episode_id"] = getattr(current_episode, "episode_id", None)
            try:
                critic_info["task_family"] = infer_task_family(current_episode)
            except Exception:
                critic_info["task_family"] = "other"
        else:
            critic_info["episode_id"] = None
            critic_info["task_family"] = "other"

        return critic_info

    def _get_current_episode(self) -> Optional[Any]:
        try:
            return self.env_interface.env.env.env._env.current_episode
        except Exception:
            return None

    def reset(self) -> None:
        self.update_count = 0
        if self.critic is not None and hasattr(self.critic, "reset"):
            try:
                self.critic.reset()
            except Exception as exc:
                logger.debug("Critic reset failed: %s", exc)

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "update_count": self.update_count,
            "critic_enabled": self.critic is not None,
        }


__all__ = ["CriticUpdater"]
