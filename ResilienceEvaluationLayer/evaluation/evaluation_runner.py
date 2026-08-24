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
        # Keep this attribute for compatibility with planner_demo_mp(_new).
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

        preferred_order = [
            "task_state_success",
            "task_percent_complete",
            "sim_step_count",
            "runtime",
            "replanning_count",
            # Rebound
            "rebound_c_rec",
            "rebound_c_rec_cog",
            "rebound_c_rec_phy",
            "rebound_c_rec_state_debt",
            "rebound_num_recovery_windows",
            "rebound_raw_window_count",
            "rebound_formal_window_count",
            "rebound_skipped_window_count",
            "rebound_c_rec_valid",
            "rebound_c_rec_missing_reason",
            "rebound_baseline_match_level",
            "rebound_w_rem_star",
            "rebound_g_rec_total",
            "rebound_baseline_loaded",
            "rebound_tracker_window_count",
            "rebound_tracker_window_open_final",
            # Stability
            "stability_beta",
            "stability_beta_neighborhood",
            "stability_beta_vv",
            "stability_beta_out",
            "stability_beta_oscillation",
            "stability_beta_action",
            "stability_beta_progress",
            "stability_beta_success",
            "stability_beta_perturbation",
            "stability_beta_perturbation_vv",
            "stability_beta_perturbation_out",
            "stability_beta_perturbation_oscillation",
            "stability_beta_perturbation_action",
            "stability_beta_perturbation_progress",
            "stability_beta_perturbation_success",
            "stability_beta_perturbation_valid",
            "stability_beta_perturbation_scope",
            "stability_clean_ref_valid",
            "stability_clean_ref_strategy",
            "stability_n_clean_ref",
            "stability_beta_neighborhood_valid",
            "stability_beta_neighborhood_missing_reason",
            "stability_beta_scope",
            "stability_td_error_mean_abs",
            "stability_td_error_p95_abs",
            "stability_gae_mean_abs",
            "stability_gae_p95_abs",
            "stability_return_target_variance",
            "stability_oscillation_loss",
            "stability_oscillation_loss_valid",
            "stability_oscillation_source",
            "stability_critic_trace_len",
            # Degradation / Graceful Extensibility
            "degradation_auc_loss",
            "degradation_p_cliff",
            "degradation_l_bd",
            "ge_contract_margin_min",
            "ge_audc_f",
            "ge_cell_valid",
            "ge_cell_missing_reason",
            "ge_valid_lambda_coverage",
            "ge_boundary_lambda_star",
            "ge_boundary_hit",
            "ge_hard_boundary_hit",
            "resilience_score",
            "resilience_availability",
            "copal_rsr",
            "copal_mttr",
            "copal_mtbf",
        ]
        ordered_keys = [key for key in preferred_order if key in metrics]
        if ordered_keys:
            ordered_key_set = set(ordered_keys)
            ordered_keys.extend(
                key
                for key in sorted(metrics.keys())
                if key.startswith("replanning_count_") and key not in ordered_key_set
            )
        else:
            ordered_keys = sorted(metrics.keys())

        for key in ordered_keys:
            print(f"  {key}: {self._format_display_metric_value(metrics[key])}")

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

        Controlled by ``evaluation.rebound_tracker.enabled`` (default True).
        The tracker stays attached to the runner across episodes and is
        re-initialised per-episode via ``tracker.reset(episode)``.
        """
        cfg = self.evaluation_runner_config
        tracker_cfg = None
        try:
            tracker_cfg = cfg.get("rebound_tracker") if hasattr(cfg, "get") else None
        except Exception:
            tracker_cfg = None
        enabled = True
        template_config = None
        if tracker_cfg is not None:
            try:
                from omegaconf import OmegaConf as _OC
                tracker_cfg = _OC.to_container(tracker_cfg, resolve=True) or {}
            except Exception:
                tracker_cfg = {}
            if isinstance(tracker_cfg, dict):
                enabled = bool(tracker_cfg.get("enabled", True))
                template_config = tracker_cfg.get("templates")
        if not enabled:
            self.rebound_tracker = None
            return None
        if self.rebound_tracker is None:
            try:
                from habitat_llm.evaluation.metrics.rebound import OnlineReboundTracker
                self.rebound_tracker = OnlineReboundTracker(
                    episode=episode,
                    template_config=template_config,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to initialise OnlineReboundTracker: %s", exc)
                self.rebound_tracker = None
        return self.rebound_tracker

    def _install_prompt_perturbation_hook(self) -> None:
        """Register :attr:`perturbation_injector`'s prompt hook on all planners.

        Planners that support ``_prompt_perturbation_hook`` will rewrite the
        instruction immediately before prompt assembly. Planners that don't
        support the attribute are left untouched (the hook is purely additive).
        """
        if self.perturbation_injector is None:
            return
        hook = self.perturbation_injector.prompt_hook()
        planners: List[Any] = []
        raw_planner = getattr(self, "planner", None)
        if isinstance(raw_planner, dict):
            planners.extend(raw_planner.values())
        elif raw_planner is not None:
            planners.append(raw_planner)
        for p in planners:
            try:
                setattr(p, "_prompt_perturbation_hook", hook)
            except Exception:
                pass

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
            self.current_instruction = (
                self.env_interface.env.env.env._env.current_episode.instruction
            )
        else:
            self.current_instruction = instruction
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
        Check if the agents are colliding.
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
        # Update the agent states in environment interface
        for agent_id, value in planner_info["replanned"].items():
            if value:
                # An action must be returned if the planner replans
                world_graph_snapshot = {
                    uid: wg.snapshot() 
                    for uid, wg in self.env_interface.world_graph.items()
                }
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
                tracker.reset(episode=current_episode)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("OnlineReboundTracker.reset failed: %s", exc)

        # Apply world-level perturbation (if any) right after reset so the
        # planner sees the mutated world from the first observation on. Only
        # active when ``self.perturbation_injector`` has been set by the
        # resilience runner.
        if self.perturbation_injector is not None:
            try:
                self.perturbation_injector.apply_world_perturbation(
                    self.env_interface,
                    episode=getattr(
                        self.env_interface.env.env.env._env,
                        "current_episode",
                        None,
                    ),
                )
            except Exception as _pi_exc:  # pragma: no cover - defensive
                logger.warning(
                    "PerturbationInjector world-level apply failed: %s", _pi_exc
                )
            # Also register the prompt-level hook on each planner so that
            # prompt paraphrase rewrites are applied downstream.
            self._install_prompt_perturbation_hook()

        # Initialize metadata
        self.initialize_instruction_metadata(instruction, output_name)

        # Apply prompt-level paraphrase (if any) to the resolved instruction.
        if self.perturbation_injector is not None:
            try:
                rewritten = self.perturbation_injector.rewrite_instruction(
                    self.current_instruction,
                    episode=getattr(
                        self.env_interface.env.env.env._env,
                        "current_episode",
                        None,
                    ),
                )
                if rewritten and rewritten != self.current_instruction:
                    logger.info(
                        "PerturbationInjector rewrote instruction (%s).",
                        self.perturbation_injector.spec.kind,
                    )
                    self.current_instruction = rewritten
            except Exception as _pi_exc:  # pragma: no cover
                logger.warning(
                    "PerturbationInjector instruction-rewrite failed: %s", _pi_exc
                )
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
        }

        # List to store planner logs at each step
        planner_infos = []
        planner_info: Dict[str, Any] = {}
        low_level_actions: List[Dict[str, Any]] = []
        should_end = False

        # A2C Critic state tracking
        prev_world_state = None
        current_world_state = None
        if self.critic_enabled and self.critic is not None:
            try:
                prev_world_state = self.world_state_builder.build(
                    self.agents, self.current_instruction
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
                if self.evaluation_runner_config.save_video:
                    # Store third person frames for generating video
                    self.dvu._store_for_video(
                        observations, planner_info["high_level_actions"]
                    )

                # ==================== A2C CRITIC UPDATE ====================
                # Update critic after environment step
                if self.critic_enabled and self.critic is not None:
                    current_world_state = self.world_state_builder.build(
                        self.agents, self.current_instruction
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
                        episode_filename=self.episode_filename,
                        loop_step_count=total_step_count,
                    )
                    prev_world_state = current_world_state
                # ===========================================================

            # Get next low level actions
            low_level_actions, planner_info, should_end = self.get_low_level_actions(
                self.current_instruction, observations, self.env_interface.world_graph
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
            try:
                metrics = self.env_interface.env.get_metrics()
                current_progress = metrics.get("task_percent_complete", 0.0)
            except Exception:
                current_progress = 0.0

            self.analysis_coordinator.update_step_health_metrics(
                planner_info=planner_info,
                agent_collisions=planner_info["agent_collisions"],
                current_progress=current_progress,
                critic=self.critic if hasattr(self, "critic") else None,
            )
            # ---------------------------------

            # Online Rebound tracker — emits t_d/t_r events on the planner_info
            # dict so they propagate into the per-step copy below as well as
            # the critic-exports JSONL.
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

                try:
                    self.rebound_tracker.observe(
                        step=int(total_step_count),
                        planner_info=planner_info,
                        info=info,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("OnlineReboundTracker.observe failed: %s", exc)


            # Update agent state and action history
            copy_planner_info = copy.deepcopy(planner_info)
            self._drop_transient_planner_payload(
                copy_planner_info,
                keep_prompt_trace=True,
            )
            self.update_agent_state_history(copy_planner_info)
            self.update_agent_action_history(copy_planner_info)

            # Append planner info to history
            if planner_infos:
                self._drop_transient_planner_payload(planner_infos[-1])
            planner_infos.append(copy_planner_info)

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

        # Log planner information per step
        self.analysis_coordinator.log_planner_data(
            output_dir=self.current_output_dir,
            episode_filename=self.episode_filename,
            agents=self.agents,
            planner_infos=planner_infos,
            current_instruction=self.current_instruction,
        )

        # Log overall time
        t_runtime = time.time() - t_0
        info["runtime"] = t_runtime

        # Merge dictionaries
        info |= planner_info
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
            planner=self.planner,
            planner_infos=planner_infos,
            info=info,
            critic=self.critic if hasattr(self, "critic") else None,
        )

        return info
