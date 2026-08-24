#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
This module contains the base EvaluationRunner class and related utilities for running evaluations of LLM-based agents in Habitat environments.
It provides functionality for initializing agents, running episodes, collecting metrics, and storing evaluation results.
The module includes classes for tracking action and state history during evaluation runs.
"""

import copy
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

import attr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from habitat_llm.agent import Agent
from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.examples.example_utils import DebugVideoUtil
from habitat_llm.planner.planner import Planner
from habitat_llm.utils import cprint, rollout_print
from habitat_llm.utils.sim import init_agents
from habitat_llm.world_model import Entity, WorldGraph
from habitat_llm.evaluation.core import CriticUpdater, WorldStateBuilder
from habitat_llm.evaluation.proposition_outcome import (
    materialize_proposition_outcome,
)
from habitat_llm.evaluation.reporting import primary_display_items
from habitat_llm.evaluation.runner_analysis import EvaluationAnalysisCoordinator
from habitat.tasks.rearrange.utils import coll_name_matches


@attr.s(auto_attribs=True)
class ActionHistoryElement:
    """
    A class used to represent an element of action history.

    :param action: A tuple representing the action taken of format (Action Type, Action Args).
    :param timestamp: The timestamp at which the action was taken.
    :param agent_uid: The unique identifier of the agent who took the action.
    :param response: The response or feedback received after taking the action.
    :param world_graph: A dictionary mapping agent IDs to their world graph states at this point.
    :param info: Additional information dictionary containing metadata about the action.
    """

    action: tuple
    timestamp: int
    agent_uid: int
    response: str = ""
    world_graph: Dict[int, WorldGraph] = None
    info: dict = attr.ib(factory=dict)

    def to_string(self) -> str:
        """
        Convert the state history element to a string representation.

        :return: A string representation of the agent's state.
        """
        return f"{self.action[0]}[{self.action[1]}]"


@attr.s(auto_attribs=True)
class StateHistoryElement:
    """
    A class used to represent an element of state history.

    :param state: A string representing the state of the agent.
    :param timestamp: The timestamp of the state representation
    :param agent_uid: The unique identifier of the agent in the recorded state.
    """

    state: str
    timestamp: int
    agent_uid: int

    def to_string(self) -> str:
        """
        Convert the state history element to a string representation.

        :return: A string representation of the agent's state.
        """
        return self.state


# Evaluation runner, will go over episodes, run planners and store necessary data.
# Stores an episode, information about the agents and planners and uses them to run through an
# episode and store necessary data.
class EvaluationRunner:
    def __init__(
        self,
        evaluation_runner_config_arg,
        env_interface_arg: EnvironmentInterface,
        dump_world_graph: bool = False,
    ):
        """
        Initialize EvaluationRunner

        :param evaluation_runner_config_arg: The experiment configuration, including config of the agents and planners.
        :param env_interface_arg: The environment instance
        :param dump_world_graph: Whether to dump the world graph to a file.
        """
        self.env_interface = env_interface_arg
        self.evaluation_runner_config = evaluation_runner_config_arg
        self.TRUNCATE_LENGTH = self.evaluation_runner_config.truncate_length

        dataset_file = self.env_interface.conf.habitat.dataset.data_path.split("/")[-1]
        results_dir = self.env_interface.conf.paths.results_dir
        self.output_dir = f"{results_dir}/{dataset_file}/"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Log path information for debugging
        logger.info(f"EvaluationRunner path configuration:")
        logger.info(f"  - Dataset file: {dataset_file}")
        logger.info(f"  - Results dir: {results_dir}")
        logger.info(f"  - Output dir: {self.output_dir}")

        # Declare container to store agent positions
        self.agent_positions: List[Any] = []
        self.object_nodes: List[Entity] = []

        # Declare a container for storing unique agents
        self.agents: Dict[int, Agent] = {}

        self.episode_filename = ""
        # ``current_instruction`` remains the backwards-compatible planner
        # view.  Keep the pre-perturbation task and the planner-visible task
        # separate so reward shaping never evaluates a rewritten goal.
        self.canonical_instruction = ""
        self.policy_instruction = ""
        self.current_instruction = ""

        # Initialize the agents
        self.__initialize_agents()

        self.planner: Union[Dict[int, Planner], Planner] = {}
        self._initialize_planners()

        # Initialize the debug video util
        self.dvu = DebugVideoUtil(self.env_interface, self.output_dir)
        self._write_out_world_graph: bool = dump_world_graph
        self._world_graph_write_out_frequency = 5
        
        # Log subdirectory support
        self.log_subdirectory: Optional[str] = None

        # Opt-in perturbation injector (set by resilience runners to exercise
        # the Stability / Graceful-Extensibility experiments). Defaults to
        # ``None`` so the standard evaluation path is unaffected.
        self.perturbation_injector: Optional[Any] = None
        self.perturbation_audit: Dict[str, Any] = {}

        # Online Rebound tracker (t_d / t_r event stream + semantic templates).
        # Built lazily on the first call to _ensure_rebound_tracker so that
        # non-Rebound code paths do not pay the import cost.
        self.rebound_tracker: Optional[Any] = None
        self.log_critic_trajectory_export = bool(
            self.evaluation_runner_config.get("log_critic_trajectory_export", False)
        )
        self.critic = None
        self.critic_enabled = False
        self.world_state_builder = WorldStateBuilder(self.env_interface)
        self.critic_updater = CriticUpdater(
            self.critic,
            env_interface=self.env_interface,
            agents=self.agents,
            log_trajectory_export=self.log_critic_trajectory_export,
            config=self.evaluation_runner_config.get("critic", {}),
        )
        self.analysis_coordinator = EvaluationAnalysisCoordinator(
            evaluation_runner_config=self.evaluation_runner_config,
            env_interface=self.env_interface,
        )
        # Keep this attribute for compatibility with planner_demo_mp_new.
        self.adca_analyzer = self.analysis_coordinator.adca_analyzer

    def _set_critic(self, critic: Optional[Any]) -> None:
        self.critic = critic
        self.critic_enabled = critic is not None
        if hasattr(self, "critic_updater") and self.critic_updater is not None:
            self.critic_updater.set_critic(critic)


    def _initialize_planners(self):
        """
        Initialize the planners
        """
        raise NotImplementedError

    # Method to initialize the agents based in the config
    def __initialize_agents(self) -> None:
        """
        Initialize agents based on config.
        """
        agents = init_agents(self.evaluation_runner_config.agents, self.env_interface)
        for agent in agents:
            self.agents[agent.uid] = agent
            agent._dry_run = self.env_interface._dry_run
            cprint(f"successfully added agent with UID : {agent.uid}", "green")
        print("finished initializing agents!")

    # Method to print the object
    def __str__(self) -> str:
        """
        Return string with state of the evaluator
        """
        planner_type = type(self.planner)
        out = f"Centralized Planner: {planner_type}\n"
        out += f"Number of Agents: {len(self.agents)}"
        return out

    @staticmethod
    def _format_display_metric_value(value: Any) -> str:
        """Format metric values for concise console display."""
        if isinstance(value, float):
            return f"{value:.4f}"
        if isinstance(value, np.floating):
            return f"{float(value):.4f}"
        if isinstance(value, (list, tuple)):
            preview = ", ".join(
                EvaluationRunner._format_display_metric_value(v) for v in list(value)[:4]
            )
            if len(value) > 4:
                preview += ", ..."
            return f"[{preview}]"
        if isinstance(value, dict):
            items = list(sorted(value.items(), key=lambda kv: str(kv[0])))
            preview = ", ".join(
                f"{k}: {EvaluationRunner._format_display_metric_value(v)}"
                for k, v in items[:4]
            )
            if len(items) > 4:
                preview += ", ..."
            return "{" + preview + "}"
        return str(value)

    def print_primary_display_metrics(
        self, header: str, metrics: Optional[Dict[str, Any]]
    ) -> None:
        """
        Compatibility display helper used by planner demos.

        Keeps reporting logic on the runner surface while the heavy analysis
        implementation lives in dedicated modules.
        """
        cprint(header, "blue")
        if not metrics:
            print("  (no metrics)")
            return

        display_items = primary_display_items(metrics)
        for key, value in display_items:
            print(f"  {key}: {self._format_display_metric_value(value)}")

    def print_secondary_display_metrics(
        self, header: str, metrics: Optional[Dict[str, Any]]
    ) -> None:
        """Backwards-compatible alias for older callers."""
        self.print_primary_display_metrics(header, metrics)

    @property
    def current_output_dir(self) -> str:
        """Returns the output directory joined with the log_subdirectory if set."""
        if self.log_subdirectory:
            return os.path.join(self.output_dir, self.log_subdirectory)
        return self.output_dir

    @property
    def agent_list(self) -> str:
        """Returns a string listing the agent's uid"""
        return str([agent.uid for agent in self.agents.values()])

    @property
    def tool_list(self) -> List[str]:
        """Returns a list of unique tool names across all agents.

        :return: List of tool names as strings
        """
        tool_set = set()
        for agent in self.agents.values():
            for tool in agent.tools.values():
                tool_set.add(tool.name)

        return list(tool_set)

    def reset(self) -> None:
        """Reset metrics and stats to be ready for the next episode."""

        # Clear the frames to make sure that
        # video for next episode does no have frames from previous run
        self.dvu.frames.clear()

        # Clear containers used for top-down video generation
        self.agent_positions.clear()
        self.object_nodes.clear()

        # Reset filenames
        self.episode_filename = ""
        self.canonical_instruction = ""
        self.policy_instruction = ""
        self.current_instruction = ""

        # Reset planners and the agents owned by the planners
        # This will also reset skills owned by the agents to
        # make eval runner ready for next episode
        self.reset_planners()
        if self.critic_updater is not None:
            self.critic_updater.reset()
        
        # Reset analysis-side monitors
        self.analysis_coordinator.reset()


    @staticmethod
    def _optional_positive_int(value: Any) -> Optional[int]:
        if value in (None, "", False):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    @staticmethod
    def _drop_transient_planner_payload(
        planner_info: Dict[str, Any],
        *,
        keep_prompt_trace: bool = False,
    ) -> None:
        """Drop large fields that are not used by per-step JSON logs."""
        planner_info.pop("print", None)
        planner_info.pop("print_no_tags", None)
        if not keep_prompt_trace:
            planner_info.pop("prompts", None)
            planner_info.pop("traces", None)

    @staticmethod
    def _compact_planner_info_for_action_history(
        planner_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Keep the ActionHistoryElement planner payload finetuning-compatible."""
        compact: Dict[str, Any] = {}
        for key in (
            "stats",
            "replanning_count",
            "total_step_count",
            "sim_step_count",
            "high_level_actions",
            "responses",
            "is_done",
            "autofill_actions",
            "termination_reason",
        ):
            if key in planner_info:
                compact[key] = copy.deepcopy(planner_info[key])
        return compact

    @staticmethod
    def _compact_planner_info_for_episode_history(
        planner_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Keep metric/log fields without retaining planner object graphs per sim step."""
        compact: Dict[str, Any] = {}
        for key in (
            "stats",
            "replanning_count",
            "total_step_count",
            "sim_step_count",
            "high_level_actions",
            "responses",
            "replanned",
            "replan_required",
            "agent_states",
            "is_done",
            "autofill_actions",
            "termination_reason",
            "agent_collisions",
            "agent_collision_scope",
            "_context_compression_events",
        ):
            if key in planner_info:
                compact[key] = copy.deepcopy(planner_info[key])
        replanned = compact.get("replanned", {}) or {}
        if (
            isinstance(replanned, dict)
            and any(bool(value) for value in replanned.values())
            and "curr_graph" in planner_info
        ):
            compact["curr_graph"] = copy.deepcopy(planner_info["curr_graph"])
        return compact

    @staticmethod
    def _is_planner_history_event(planner_info: Dict[str, Any]) -> bool:
        replanned = planner_info.get("replanned", {}) or {}
        responses = planner_info.get("responses", {}) or {}
        return bool(
            (isinstance(replanned, dict) and any(replanned.values()))
            or (not isinstance(replanned, dict) and replanned)
            or (isinstance(responses, dict) and any(responses.values()))
            or planner_info.get("termination_reason")
            or planner_info.get("is_done")
            or planner_info.get("_context_compression_events")
        )

    def _retain_full_action_history_world_graph(self) -> bool:
        return bool(
            getattr(
                self.evaluation_runner_config,
                "retain_action_history_world_graph",
                True,
            )
        )

    @staticmethod
    def _format_action_sequence_token(agent_id: Any, action_tuple: Any) -> str:
        """Serialize one high-level action into a compact token for beta_action."""

        if isinstance(action_tuple, (list, tuple)):
            action_name = str(action_tuple[0]) if len(action_tuple) > 0 else ""
            action_args = str(action_tuple[1]) if len(action_tuple) > 1 else ""
        else:
            action_name = str(action_tuple or "")
            action_args = ""
        action_name = action_name.strip()
        action_args = action_args.strip()
        if not action_name:
            return ""
        return f"agent{agent_id}:{action_name}[{action_args}]"

    @classmethod
    def _extract_action_sequence(
        cls,
        planner_infos: List[Dict[str, Any]],
    ) -> str:
        """Build the episode-level high-level action sequence used by beta_action."""

        tokens: List[str] = []
        for step_info in planner_infos:
            high_level_actions = step_info.get("high_level_actions", {})
            if not isinstance(high_level_actions, dict):
                continue
            replanned = step_info.get("replanned")
            for agent_id in sorted(high_level_actions.keys(), key=lambda item: str(item)):
                include_action = True
                if isinstance(replanned, dict):
                    include_action = bool(
                        replanned.get(agent_id, replanned.get(str(agent_id), False))
                    )
                if not include_action:
                    continue
                token = cls._format_action_sequence_token(
                    agent_id,
                    high_level_actions.get(agent_id),
                )
                if token:
                    tokens.append(token)
        return "|".join(tokens)

    @property
    def agent_descriptions(self) -> str:
        """Returns a string listing the descriptions of all agents

        :return: A concatenated string of all agent descriptions
        """
        out = ""
        for agent in self.agents.values():
            out += agent.agent_description
        return out

    def _update_td(self, frame, ax) -> None:
        """
        Function to update the top down plot for each robot position
        and detected objects over time.

        :param frame: Current animation frame number
        :param ax: Matplotlib axis object to draw on
        """
        # Clear the current plot
        ax.clear()

        # Extract x and y positions for the current frame
        x = [position[0] for position in self.agent_positions[: frame + 1]]
        y = [position[2] for position in self.agent_positions[: frame + 1]]

        # Extract object x and y
        x_obj = [obj.properties["translation"][0] for obj in self.object_nodes[frame]]
        y_obj = [obj.properties["translation"][2] for obj in self.object_nodes[frame]]
        names = [obj.name for obj in self.object_nodes[frame]]

        # Plot the robot's path
        ax.plot(x, y, marker=".", linestyle="-")

        # Plot the objects path
        ax.scatter(x_obj, y_obj, marker="*", color="red")

        # Add text near each point
        for _, txt in enumerate(zip(x_obj, y_obj, names)):
            ax.text(txt[0], txt[1], txt[2], color="red", ha="right", va="bottom")

        # Set labels and title
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_title("Robot Movement Over Time")

        # Set axis limits
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)

        # Add grid
        ax.grid(True)

        # Set aspect ratio to be equal
        ax.set_aspect("equal")

    def _store_for_top_down_viz(self, agent_uid: Optional[int] = None) -> None:
        """
        Stores agent position and world graph object data for top-down visualization.

        :param agent_uid: Optional ID of agent whose perspective to use. If None, uses full observability.
        """
        world_graph = None
        if agent_uid is not None:
            world_graph = self.env_interface.world_graph[agent_uid]
        else:
            print(
                "No agent_uid provided. Code will generate top-down visualization from full-observability perspective"
            )
            world_graph = self.env_interface.full_world_graph
        sim = self.env_interface.sim
        self.agent_positions.append(
            sim.agents_mgr[agent_uid].articulated_agent.base_pos
        )

        self.object_nodes.append(world_graph.get_all_objects())

    def _make_td_video(self) -> None:
        """
        Creates a top-down video visualization of the episode.

        :param instruction: Task instruction being executed
        """
        os.makedirs(f"{self.current_output_dir}/videos", exist_ok=True)
        td_video_name = f"{self.current_output_dir}/videos/video-td-{self.episode_filename}.mp4"

        # Create a figure and axis for the plot
        fig, ax = plt.subplots()

        num_frames = len(self.agent_positions)

        # Create the animation
        animation = FuncAnimation(
            fig, self._update_td, fargs=(ax,), frames=num_frames, repeat=False
        )

        # Save the animation as a video file (e.g., .mp4)
        animation.save(td_video_name, writer="ffmpeg", fps=30)

    def _ensure_rebound_tracker(self, episode: Optional[Any] = None) -> Optional[Any]:
        """Lazily construct the :class:`OnlineReboundTracker`.

        Controlled by ``evaluation.resilience.rebound_windows.enabled``;
        ``evaluation.rebound_tracker`` remains a legacy config alias.
        The tracker stays attached to the runner across episodes and is
        re-initialised per-episode via ``tracker.reset(episode)``.
        """
        cfg = self.evaluation_runner_config
        tracker_cfg = None
        try:
            tracker_cfg = cfg.get("rebound_tracker") if hasattr(cfg, "get") else None
        except Exception:
            tracker_cfg = None
        if tracker_cfg is None and hasattr(cfg, "get"):
            resilience_cfg = cfg.get("resilience")
            if resilience_cfg is not None and hasattr(resilience_cfg, "get"):
                tracker_cfg = resilience_cfg.get("rebound_windows")
        enabled = True
        template_config = None
        recovery_confirmation_steps = 2
        immediate_recovery_templates = ["T1", "T2"]
        if tracker_cfg is not None:
            try:
                from omegaconf import OmegaConf as _OC
                tracker_cfg = _OC.to_container(tracker_cfg, resolve=True) or {}
            except Exception:
                tracker_cfg = {}
            if isinstance(tracker_cfg, dict):
                enabled = bool(tracker_cfg.get("enabled", True))
                template_config = tracker_cfg.get("templates")
                recovery_confirmation_steps = int(
                    tracker_cfg.get("recovery_confirmation_steps", 2)
                )
                immediate_recovery_templates = list(
                    tracker_cfg.get(
                        "immediate_recovery_templates",
                        ["T1", "T2"],
                    )
                )
        if not enabled:
            self.rebound_tracker = None
            return None
        if self.rebound_tracker is None:
            try:
                from habitat_llm.evaluation.metrics.rebound import OnlineReboundTracker

                perception = getattr(self.env_interface, "perception", None)
                self.rebound_tracker = OnlineReboundTracker(
                    episode=episode,
                    template_config=template_config,
                    entity_aliases=getattr(
                        perception, "sim_handle_to_name", None
                    ),
                    room_aliases=getattr(
                        perception, "region_id_to_name", None
                    ),
                    recovery_confirmation_steps=recovery_confirmation_steps,
                    immediate_recovery_templates=immediate_recovery_templates,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to initialise OnlineReboundTracker: %s", exc)
                self.rebound_tracker = None
        return self.rebound_tracker

    def _clear_prompt_perturbation_hooks(self) -> None:
        """Ensure prompt perturbations are applied once, at runner level."""
        if self.perturbation_injector is None:
            return
        self.perturbation_injector.clear_prompt_hooks(
            getattr(self, "planner", None)
        )

    def initialize_instruction_metadata(
        self, instruction: str, output_name: str
    ) -> None:
        """
        Start folders where the outputs will be stored.

        :param instruction: The natural language instruction for the task to be executed. If None, uses the instruction from the current episode.
        :param output_name: Name to use for output files. If empty string, generates name from instruction text.
        """
        if instruction is None:
            # Get the instruction from the episode
            resolved_instruction = (
                self.env_interface.env.env.env._env.current_episode.instruction
            )
        else:
            resolved_instruction = instruction

        # The canonical instruction is the task resolved at the episode
        # boundary, before any prompt-level perturbation.  The policy view may
        # subsequently be rewritten, while ``current_instruction`` continues
        # to alias that policy view for existing planners and callers.
        self.canonical_instruction = resolved_instruction
        self.policy_instruction = resolved_instruction
        self.current_instruction = self.policy_instruction
        if self.evaluation_runner_config.do_print:
            cprint("Instruction:", "yellow")
            print(self.current_instruction + "\n")
        # Make hyphenated instruction for creating a filename
        if len(output_name) == 0:
            self.episode_filename = self.current_instruction.replace(" ", "-")[
                :-1
            ].lower()
        else:
            self.episode_filename = output_name
        # check if name is too long, truncate to be system-friendly
        if len(self.episode_filename) > self.TRUNCATE_LENGTH:
            self.episode_filename = self.episode_filename[: self.TRUNCATE_LENGTH]

    def get_low_level_actions(
        self, instruction: str, observations: dict, world_graph: Dict[int, WorldGraph]
    ):
        """
        Given a set of observations, gets a vector of low level actions, an info dictionary and a boolean indicating that
        the run should end.

        :param instruction: String with the instruction to execute
        :param observations: Dictionary of habitat observations
        :param world_graph: The world graph from the agent.

        :return: tuple low_level_actions, info, should_end indicating 1) a dictionary from agent id to a low level action vector
        2) a dictionary with info about high level actions, an indicator of whether the task was ended.
        """
        raise NotImplementedError

    def reset_planners(self):
        """
        Reset the planners for this evaluator
        """
        raise NotImplementedError

    def get_agent_collisions(self) -> Dict[int, bool]:
        """
        Observe pairwise contact between exactly two articulated agents.

        This is not a general scene-collision sensor: it does not cover an
        agent contacting furniture, obstacles, or the static scene.  Return an
        empty mapping when the pairwise channel is unavailable so downstream
        safety accounting cannot manufacture a collision-free observation.
        """
        collision = False
        agent_ids = [
            articulated_agent.sim_obj.object_id
            for articulated_agent in self.env_interface.sim.agents_mgr.articulated_agents_iter
        ]
        if len(agent_ids) == 2:
            self.env_interface.sim.perform_discrete_collision_detection()
            contact_points = self.env_interface.sim.get_physics_contact_points()
            for cp in contact_points:
                if coll_name_matches(cp, agent_ids[0]) and coll_name_matches(
                    cp, agent_ids[1]
                ):
                    collision = True

        if len(agent_ids) != 2:
            return {}

        out = {}
        for agent_uid in self.agents:
            out[agent_uid] = collision
        return out

    def update_agent_state_history(self, planner_info: Dict[str, Any]) -> None:
        """
        Updates the state history stored in env_interface based on planner info.
        This includes logging states like "standing", "walking", "picking X", "placing on Y" etc.
        """
        # # Update the agent states in environment interface
        if "agent_states" in planner_info:
            for agent_uid in planner_info["agent_states"]:
                agent_state_at_t = planner_info["agent_states"][agent_uid]
                if len(self.env_interface.agent_state_history[agent_uid]) > 0:
                    agent_state_at_t_minus_1 = self.env_interface.agent_state_history[
                        agent_uid
                    ][-1]
                    if agent_state_at_t != agent_state_at_t_minus_1.state:
                        self.env_interface.agent_state_history[agent_uid].append(
                            StateHistoryElement(
                                agent_state_at_t,
                                planner_info["sim_step_count"],
                                agent_uid=agent_uid,
                            )
                        )
                else:
                    self.env_interface.agent_state_history[agent_uid].append(
                        StateHistoryElement(
                            agent_state_at_t,
                            planner_info["sim_step_count"],
                            agent_uid=agent_uid,
                        )
                    )

    def update_agent_action_history(self, planner_info: Dict[str, Any]) -> None:
        """
        Updates the actions history stored in env_interface based on planner info.
        This includes logging actions like "Navigate[object_id]", "Pick[object_id]" etc.

        :param planner_info: Dictionary containing planner action information
        """
        if "replan_required" not in planner_info:
            return
        high_level_actions = planner_info.get("high_level_actions", {})
        replanned_agents = [
            agent_id
            for agent_id, value in planner_info["replanned"].items()
            if value
        ]
        if not replanned_agents:
            return
        # Centralized planners replan both agents at the same instant. Build
        # one immutable-in-practice snapshot and share it across their history
        # records instead of duplicating the full WorldGraph object graph.
        world_graph_snapshot = None
        if self._retain_full_action_history_world_graph():
            world_graph_snapshot = {
                uid: wg.snapshot()
                for uid, wg in self.env_interface.world_graph.items()
            }
        # Update the agent states in environment interface
        for agent_id in replanned_agents:
            # An action must be returned if the planner replans
            if agent_id not in high_level_actions:
                logger.warning(
                    "Skipping action-history append for agent %s: missing high_level_action. planner_info keys=%s",
                    agent_id,
                    sorted(high_level_actions.keys()),
                )
                continue
            action_tuple = high_level_actions[agent_id]
            bootstrap_response = ""
            if (
                isinstance(action_tuple, tuple)
                and len(action_tuple) > 2
                and isinstance(action_tuple[2], str)
                and len(action_tuple[2]) > 0
            ):
                # Preserve diagnostic actions (e.g., auto-filled no-op) as
                # complete action-history records.
                bootstrap_response = action_tuple[2]
            action_history_object = ActionHistoryElement(
                action=action_tuple,
                timestamp=planner_info["sim_step_count"],
                agent_uid=agent_id,
                response=bootstrap_response,
                # world_graph=copy.deepcopy(self.env_interface.world_graph),
                world_graph=world_graph_snapshot,
                info={
                    "planner_info": self._compact_planner_info_for_action_history(
                        planner_info
                    ),
                    "log_time": time.time(),
                },
            )

            self.env_interface.agent_action_history[agent_id].append(
                action_history_object
            )

        # add responses the last logged action, this means the planner will replan at the next step
        if "responses" in planner_info and any(planner_info["responses"].values()):
            for agent_id, response in planner_info["responses"].items():
                # empty string response does not mean the action is over
                # skip adding the response
                if response is None or response == "":
                    continue
                # There should have been an action logged if there is a response
                if len(self.env_interface.agent_action_history[agent_id]) == 0:
                    logger.warning(
                        "Skipping response assignment for agent %s with no action history. response=%r",
                        agent_id,
                        response,
                    )
                    continue
                history = self.env_interface.agent_action_history[agent_id]
                history[-1].response = response

                # Backfill stale unresolved entries instead of hard-failing the
                # whole episode. This keeps detailed traces serializable while
                # preserving a diagnostic marker.
                for ah in history[:-1]:
                    if ah.response is None or len(ah.response) == 0:
                        ah.response = "No response recorded (auto-filled)."
                        logger.warning(
                            "Auto-filled missing response for agent %s action %s",
                            agent_id,
                            ah.action,
                        )

    def run_instruction(
        self, instruction: Optional[str] = None, output_name: str = "", log_subdirectory: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs a single instruction through the planner, taking steps until the task is done.
        Stores the information using the provided output name.

        :param instruction: Optional instruction to execute. If None, uses episode instruction.
        :param output_name: Name to use for output files. If empty, derives from instruction.
        :param log_subdirectory: Optional subdirectory name to organize logs (e.g., batch_0).

        :return: Dictionary containing execution information and metrics
        """
        # Set log subdirectory for this run
        self.log_subdirectory = log_subdirectory
        
        # Update DVU output dir to match current run
        if self.dvu:
            self.dvu.output_dir = self.current_output_dir
            
        # Log start time
        t_0 = time.time()

        # Counter to count iterations
        # of this loop, as sim step dont increase
        # for perception tools
        total_step_count = 1
        max_episode_loop_steps = self._optional_positive_int(
            self.evaluation_runner_config.get("max_episode_loop_steps", None)
        )
        max_episode_sim_steps = self._optional_positive_int(
            self.evaluation_runner_config.get("max_episode_sim_steps", None)
        )

        # Reset planners and the agents owned by the planners
        # This will also reset skills owned by the agents to
        # make eval runner ready for next episode
        self.reset_planners()

        # Reset the Rebound tracker (if enabled) so it can observe this episode.
        try:
            current_episode = getattr(
                self.env_interface.env.env.env._env, "current_episode", None
            )
        except Exception:
            current_episode = None
        tracker = self._ensure_rebound_tracker(episode=current_episode)
        if tracker is not None:
            try:
                perception = getattr(self.env_interface, "perception", None)
                tracker.reset(
                    episode=current_episode,
                    entity_aliases=getattr(
                        perception, "sim_handle_to_name", None
                    ) or {},
                    room_aliases=getattr(
                        perception, "region_id_to_name", None
                    ) or {},
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("OnlineReboundTracker.reset failed: %s", exc)

        # Apply world-level perturbation (if any) right after reset so the
        # planner sees the mutated world from the first observation on. Only
        # active when ``self.perturbation_injector`` has been set by the
        # resilience runner.
        self.perturbation_audit = {}
        if self.perturbation_injector is not None:
            # A retry can keep the same episode id/seed while rebuilding the
            # environment's world graph.  Reset injector idempotence at the
            # rollout boundary so an audit from the previous graph is never
            # reused as evidence for this run.
            self.perturbation_injector.begin_rollout()
            spec = self.perturbation_injector.spec
            self.perturbation_audit["spec"] = {
                "kind": spec.kind,
                "requested": float(spec.intensity),
                "seed": int(spec.seed),
            }
            # Prompt rewriting is runner-owned. Clear the legacy planner hook
            # to prevent a second rewrite in LLMPlanner.get_next_action().
            self._clear_prompt_perturbation_hooks()
            try:
                if spec.is_world_level:
                    world_audit = (
                        self.perturbation_injector.apply_world_perturbation_with_audit(
                            self.env_interface,
                            episode=current_episode,
                        )
                    )
                    self.perturbation_audit["world"] = world_audit.to_dict()
            except Exception as _pi_exc:  # pragma: no cover - defensive
                logger.warning(
                    "PerturbationInjector world-level apply failed: %s", _pi_exc
                )
                self.perturbation_audit["world"] = {
                    "kind": spec.kind,
                    "requested": float(spec.intensity),
                    "realized": 0.0,
                    "valid": False,
                    "reason": f"runner_world_apply_error:{type(_pi_exc).__name__}",
                    "components": {"error": str(_pi_exc)},
                    "applied": False,
                }

        # Initialize metadata
        self.initialize_instruction_metadata(instruction, output_name)

        # Apply prompt-level paraphrase (if any) to the resolved instruction.
        if (
            self.perturbation_injector is not None
            and self.perturbation_injector.spec.is_prompt_level
        ):
            try:
                rewritten, prompt_audit = (
                    self.perturbation_injector.rewrite_instruction_with_audit(
                        self.current_instruction,
                        episode=current_episode,
                    )
                )
                self.perturbation_audit["prompt"] = prompt_audit.to_dict()
                if rewritten and rewritten != self.current_instruction:
                    logger.info(
                        "PerturbationInjector rewrote instruction (%s).",
                        self.perturbation_injector.spec.kind,
                    )
                    self.policy_instruction = rewritten
                    self.current_instruction = self.policy_instruction
            except Exception as _pi_exc:  # pragma: no cover
                logger.warning(
                    "PerturbationInjector instruction-rewrite failed: %s", _pi_exc
                )
                spec = self.perturbation_injector.spec
                self.perturbation_audit["prompt"] = {
                    "kind": spec.kind,
                    "requested": float(spec.intensity),
                    "realized": 0.0,
                    "valid": False,
                    "reason": f"runner_prompt_rewrite_error:{type(_pi_exc).__name__}",
                    "components": {"error": str(_pi_exc)},
                    "applied": False,
                }
        # Initialize sensor observations
        observations = self.env_interface.get_observations()

        # Dictionary to store info about episode execution
        # Set default metrics incase the motor skills are never called
        # and episode ends
        info = {
            "task_percent_complete": 0.0,
            "task_state_success": 0.0,
            "total_step_count": total_step_count,
            "num_steps": 0.0,
            "canonical_instruction": self.canonical_instruction,
            "policy_instruction": self.policy_instruction,
            "reward_instruction": self.canonical_instruction,
            # Compatibility: historically task_instruction was the only
            # instruction field and represented the planner-visible task.
            "task_instruction": self.policy_instruction,
        }

        # List to store planner logs at each step
        planner_infos = []
        planner_info: Dict[str, Any] = {}
        latest_planner_artifacts: Dict[str, Any] = {}
        low_level_actions: List[Dict[str, Any]] = []
        should_end = False
        rebound_transition_count = 0

        # A2C Critic state tracking
        prev_world_state = None
        current_world_state = None
        if self.critic_enabled and self.critic is not None:
            try:
                prev_world_state = self.world_state_builder.build(
                    self.agents, self.policy_instruction
                )
            except Exception as e:
                logger.debug(f"Initial critic world state construction failed: {e}")
                prev_world_state = None

        # Track last replan simulation step for LLM reward shaper alignment
        last_replan_sim_step = 0

        # Plan until required
        while not should_end:
            # Print the llm response
            if (
                "print" in planner_info
                and len(planner_info["print"])
                and self.evaluation_runner_config.do_print
            ):
                rollout_print(planner_info["print"])
            # Execute low level actions
            if len(low_level_actions) > 0:
                obs, reward, done, info = self.env_interface.step(low_level_actions)
                # Refresh observations
                observations = self.env_interface.parse_observations(obs)
                # Refresh task measures before critic export.  C_rec baselines
                # and OnlineReboundTracker both derive their stage anchor from
                # the proposition tracker, so the critic transition info must
                # carry the same per-step tracker state instead of only the
                # final episode metrics.
                try:
                    curr_env_for_measures = self.env_interface.env.env.env._env
                    measures = curr_env_for_measures.task.measurements.measures
                    for measure_name in (
                        "auto_eval_proposition_tracker",
                        "task_percent_complete",
                        "task_state_success",
                    ):
                        if measure_name in measures:
                            measures[measure_name].update_metric(
                                task=curr_env_for_measures.task,
                                episode=curr_env_for_measures.current_episode,
                            )
                            info[measure_name] = measures[measure_name].get_metric()
                except Exception as exc:
                    logger.debug("Failed to refresh task measures before critic update: %s", exc)
                materialize_proposition_outcome(info)
                if self.evaluation_runner_config.save_video:
                    # Store third person frames for generating video
                    self.dvu._store_for_video(
                        observations,
                        planner_info["high_level_actions"],
                        {
                            "sim_step_count": int(info.get("num_steps", 0)),
                            "total_step_count": int(total_step_count),
                        },
                    )

                # ==================== A2C CRITIC UPDATE ====================
                # Update critic after environment step
                if self.critic_enabled and self.critic is not None:
                    current_world_state = self.world_state_builder.build(
                        self.agents, self.policy_instruction
                    )

                    # Enhance step_info with planning step context for LLM reward shaper
                    enhanced_step_info = dict(info)
                    enhanced_step_info['last_replan_sim_step'] = last_replan_sim_step

                    self.critic_updater.update_step(
                        prev_world_state=prev_world_state,
                        current_world_state=current_world_state,
                        action_info=planner_info.get("high_level_actions", {}),
                        reward=reward,
                        done=done,
                        step_info=enhanced_step_info,
                        planner_info=planner_info,
                        current_instruction=self.current_instruction,
                        policy_instruction=self.policy_instruction,
                        reward_instruction=self.canonical_instruction,
                        episode_filename=self.episode_filename,
                        loop_step_count=total_step_count,
                    )
                    prev_world_state = current_world_state
                # ===========================================================

            # Rebound observes the decision that was just executed together
            # with the state produced by that execution.  The next planner
            # decision is generated only afterwards, so action semantics and
            # progress deltas belong to the same transition.
            if self.rebound_tracker is not None and planner_info:
                rebound_transition_count += 1
                rebound_planner_info = copy.deepcopy(planner_info)
                rebound_collisions = self.get_agent_collisions()
                rebound_planner_info["agent_collisions"] = rebound_collisions
                rebound_planner_info["agent_collision_scope"] = (
                    "agent_pair_collision" if rebound_collisions else ""
                )
                try:
                    self.rebound_tracker.observe(
                        step=int(rebound_transition_count),
                        planner_info=rebound_planner_info,
                        info=info,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("OnlineReboundTracker.observe failed: %s", exc)

            # Get next low level actions
            low_level_actions, planner_info, should_end = self.get_low_level_actions(
                self.policy_instruction, observations, self.env_interface.world_graph
            )

            # Track replan events for LLM reward shaper alignment
            # Check if any agent replanned (indicates a planning step occurred)
            if planner_info.get("replanned", {}).get(0, False):  # Check agent 0
                last_replan_sim_step = info.get("num_steps", 0)

            # We terminate the episode if this loop gets stuck
            curr_env = self.env_interface.env.env.env._env

            termination_reason = None
            if total_step_count > curr_env._max_episode_steps:
                termination_reason = (
                    f"loop_step_count>{curr_env._max_episode_steps}"
                )
            if (
                max_episode_loop_steps is not None
                and total_step_count > max_episode_loop_steps
            ):
                termination_reason = (
                    f"loop_step_count>{max_episode_loop_steps}"
                )
            if max_episode_sim_steps is not None:
                try:
                    current_sim_steps = int(info.get("num_steps", 0))
                except (TypeError, ValueError):
                    current_sim_steps = 0
                if current_sim_steps >= max_episode_sim_steps:
                    termination_reason = (
                        f"sim_step_count>={max_episode_sim_steps}"
                    )
            if termination_reason is not None:
                should_end = True
                info["termination_reason"] = termination_reason
                planner_info["termination_reason"] = termination_reason
                logger.warning(
                    "Terminating episode early due to %s", termination_reason
                )

            measure_names = [
                "auto_eval_proposition_tracker",
                "task_constraint_validation",
                "task_percent_complete",
                "task_state_success",
                "task_evaluation_log",
                "task_explanation",
            ]
            measures_to_log = [
                "task_percent_complete",
                "task_state_success",
                "proposition_satisfied_fraction",
                "task_explanation",
            ]
            if should_end:
                measures = curr_env.task.measurements.measures
                for measure_name in measure_names:
                    measures[measure_name].update_metric(
                        task=curr_env.task, episode=curr_env.current_episode
                    )
                for measure_name in measure_names:
                    if measure_name in measures:
                        info[measure_name] = measures[measure_name].get_metric()

            # Keep the scalar progress field synchronized with the exact
            # tracker frontier on every planner step, including planner-only
            # and final steps that do not execute a low-level action.
            materialize_proposition_outcome(info)

            # Add performance stats and to planner_info
            planner_info["stats"] = {
                info_name: info[info_name]
                for info_name in measures_to_log
                if info_name in info
            }

            # Inject current progress into ReboundManager for independent monitoring
            # This allows Rebound to be aware of system performance without leaking metrics to the LLM Planner's context.
            if hasattr(self.planner, "rebound_managers") and "task_percent_complete" in info:
                try:
                    for agent_uid, mgr in self.planner.rebound_managers.items():
                        mgr.update_progress(info["task_percent_complete"])
                except Exception as e:
                    logger.debug(f"Failed to update rebound progress: {e}")

            # Add step count to planner_info
            planner_info["total_step_count"] = total_step_count
            planner_info["sim_step_count"] = info["num_steps"]

            # Add world description to planner_info
            # on every replanning step and at the end of planning
            if (
                "replan_required" in planner_info
                and planner_info["replan_required"]
                and any(planner_info["replan_required"].values())
            ) or should_end:
                planner_info["curr_graph"] = {
                    agent_id: self.env_interface.world_graph[agent_id].get_world_descr(
                        is_human_wg=int(agent_id) == self.env_interface.human_agent_uid
                    )
                    for agent_id in range(len(self.agents))
                }

            # --- Modular step health metrics ---
            planner_info["agent_collisions"] = self.get_agent_collisions()
            planner_info["agent_collision_scope"] = (
                "agent_pair_collision"
                if planner_info["agent_collisions"]
                else ""
            )
            self.analysis_coordinator.update_step_health_metrics(
                planner_info=planner_info,
                agent_collisions=planner_info["agent_collisions"],
                critic=self.critic if hasattr(self, "critic") else None,
            )
            # ---------------------------------

            # Carry runner-owned compression evidence with the decision.  The
            # tracker consumes it on the next loop together with that
            # decision's post-execution outcome and stores the canonical
            # transition in its episode summary.
            if self.rebound_tracker is not None:
                # ── Drain compression events from planner into planner_info ──
                try:
                    raw_planner = getattr(self, "planner", None)
                    planners_to_drain = (
                        list(raw_planner.values())
                        if isinstance(raw_planner, dict)
                        else ([raw_planner] if raw_planner is not None else [])
                    )
                    compression_events = []
                    for _pl in planners_to_drain:
                        pending = getattr(_pl, "_pending_compression_events", None)
                        if pending:
                            compression_events.extend(pending)
                            _pl._pending_compression_events = []
                    if compression_events:
                        planner_info["_context_compression_events"] = compression_events
                except Exception as _drain_exc:
                    logger.debug("Failed to drain compression events: %s", _drain_exc)

            # Update agent state and action history
            compact_planner_info = self._compact_planner_info_for_episode_history(
                planner_info
            )
            self.update_agent_state_history(compact_planner_info)
            self.update_agent_action_history(compact_planner_info)

            # Prompts/traces are only needed from the latest decision and are
            # written separately. Never retain them for every simulation step.
            for key in ("prompts", "traces", "actions_per_agent"):
                if key in planner_info:
                    latest_planner_artifacts[key] = copy.deepcopy(planner_info[key])

            history_mode = str(
                getattr(
                    self.evaluation_runner_config,
                    "planner_history_mode",
                    "all",
                )
            ).lower()
            if history_mode == "replan_window":
                if not planner_infos:
                    compact_planner_info["_history_sample_type"] = "episode_start"
                    planner_infos.append(compact_planner_info)
                elif self._is_planner_history_event(compact_planner_info):
                    compact_planner_info["_history_sample_type"] = "event"
                    if planner_infos[-1].get("_history_sample_type") == "tail":
                        planner_infos.pop()
                    planner_infos.append(compact_planner_info)
                else:
                    compact_planner_info["_history_sample_type"] = "tail"
                    if planner_infos[-1].get("_history_sample_type") == "tail":
                        planner_infos[-1] = compact_planner_info
                    else:
                        planner_infos.append(compact_planner_info)
            else:
                planner_infos.append(compact_planner_info)

            # Increment while loop step count
            total_step_count += 1

            if (
                self._write_out_world_graph
                and total_step_count % self._world_graph_write_out_frequency == 0
            ):
                # dump the world-graph somewhere to compare
                for agent_id in self.env_interface.world_graph:
                    filename = f"{self.env_interface.env.env.env._env.current_episode.episode_id}_wg_agent_{agent_id}_iter_{total_step_count}.txt"
                    filepath = os.path.join(self.output_dir, filename)
                    with open(filepath, "w") as f:
                        self.env_interface.world_graph[agent_id].display_hierarchy(
                            file_handle=f
                        )
                    print(f"WG written to:\n{filepath}")

        # ==================== A2C CRITIC EPISODE END ====================
        # Ensure critic processes the episode if it finished via planner decision (should_end=True)
        # but environment done was False
        if self.critic_enabled and self.critic is not None:
            self.critic_updater.force_end_episode()
        # ================================================================

        # Print
        if (
            "print" in planner_info
            and len(planner_info["print"])
            and self.evaluation_runner_config.do_print
        ):
            rollout_print(planner_info["print"])

        # Make video
        if self.evaluation_runner_config.save_video:
            self.dvu._make_video(play=False, postfix=self.episode_filename)

        if planner_infos and latest_planner_artifacts:
            planner_infos[-1].update(latest_planner_artifacts)

        # Log planner information per step
        self.analysis_coordinator.log_planner_data(
            output_dir=self.current_output_dir,
            episode_filename=self.episode_filename,
            agents=self.agents,
            planner_infos=planner_infos,
            current_instruction=self.current_instruction,
            canonical_instruction=self.canonical_instruction,
            policy_instruction=self.policy_instruction,
        )

        # Log overall time
        t_runtime = time.time() - t_0
        info["runtime"] = t_runtime

        # Merge dictionaries
        info |= planner_info
        # Planner metadata may contain a legacy task field.  Re-assert the
        # instruction lineage owned by the runner after merging it.
        info["canonical_instruction"] = self.canonical_instruction
        info["policy_instruction"] = self.policy_instruction
        info["reward_instruction"] = self.canonical_instruction
        info["task_instruction"] = self.policy_instruction
        # Planner metadata must never shadow or omit the environment-owned
        # proposition outcome in the final episode payload.
        materialize_proposition_outcome(info)
        if self.perturbation_audit:
            audit_payload = copy.deepcopy(self.perturbation_audit)
            info["perturbation_audit"] = audit_payload
            spec_audit = audit_payload.get("spec", {})
            effective_audit = audit_payload.get("world") or audit_payload.get(
                "prompt"
            )
            kind = str(spec_audit.get("kind", "clean") or "clean")
            is_clean = kind.strip().lower() in {"", "clean", "baseline"}
            if not isinstance(effective_audit, dict):
                effective_audit = {
                    "requested": float(spec_audit.get("requested", 0.0) or 0.0),
                    "realized": 0.0,
                    "valid": is_clean,
                    "reason": (
                        "clean_noop"
                        if is_clean
                        else "missing_runtime_perturbation_audit"
                    ),
                    "components": {},
                    "applied": False,
                }
            components = effective_audit.get("components", {})
            if not isinstance(components, dict):
                components = {"raw": components}
            info["perturbation_requested"] = float(
                effective_audit.get(
                    "requested", spec_audit.get("requested", 0.0)
                )
                or 0.0
            )
            info["perturbation_realized"] = float(
                effective_audit.get("realized", 0.0) or 0.0
            )
            info["perturbation_valid"] = bool(
                effective_audit.get("valid", False)
            )
            info["perturbation_reason"] = str(
                effective_audit.get("reason", "") or ""
            )
            info["perturbation_components"] = json.dumps(
                components,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            info["perturbation_applied"] = bool(
                effective_audit.get("applied", False)
            )
            component_exports = {
                "stress_family": "perturbation_stress_family",
                "level_index": "perturbation_level_index",
                "variant": "perturbation_variant",
                "complete_unit_count": "perturbation_complete_unit_count",
                "stress_realized": "perturbation_stress_realized",
                "text_dose": "perturbation_text_dose",
                "alignment_valid": "perturbation_alignment_valid",
                "bank_path": "perturbation_bank_path",
            }
            for component_key, output_key in component_exports.items():
                if component_key in components:
                    info[output_key] = components[component_key]
        if not info.get("action_sequence"):
            info["action_sequence"] = self._extract_action_sequence(planner_infos)

        # Finalise the Rebound tracker and stash its summary on ``info`` so the
        # downstream coordinator / metrics aggregator can bind it to
        # :meth:`ReboundCollector.compute_stage_crec_multidim`.
        if self.rebound_tracker is not None:
            try:
                tracker_summary = self.rebound_tracker.finalize()
                info["_rebound_tracker_summary"] = tracker_summary
                info["rebound_num_template_events"] = int(
                    tracker_summary.get("total_events") or 0
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("OnlineReboundTracker.finalize failed: %s", exc)

        self.analysis_coordinator.log_episode_analyses(
            output_dir=self.current_output_dir,
            episode_filename=self.episode_filename,
            current_instruction=self.current_instruction,
            canonical_instruction=self.canonical_instruction,
            policy_instruction=self.policy_instruction,
            planner=self.planner,
            planner_infos=planner_infos,
            info=info,
            critic=self.critic if hasattr(self, "critic") else None,
        )

        return info
