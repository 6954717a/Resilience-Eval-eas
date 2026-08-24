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
from matplotlib.animation import FuncAnimation

from habitat_llm.agent import Agent
from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.examples.example_utils import DebugVideoUtil
from habitat_llm.planner.planner import Planner
from habitat_llm.utils import cprint, rollout_print
from habitat_llm.utils.sim import init_agents
from habitat_llm.world_model import Entity, WorldGraph
from habitat_llm.evaluation.llm_evaluator.adca_analyzer import ADCAAnalyzer
from habitat_llm.evaluation.eval_logging import (
    log_adca_analysis,
    log_planner_data,
    log_rebound_analysis,
    save_detailed_traces,
    write_critic_stats,
)
from omegaconf import OmegaConf


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

        # Initialize ADCA Analyzer if enabled
        self.adca_analyzer = None
        # Check if ADCA is enabled in config
        adca_enabled = False
        if hasattr(self.evaluation_runner_config, 'adca'):
            adca_enabled = self.evaluation_runner_config.adca.get('enabled', False)
        
        logger.info(f"Checking ADCA configuration: enabled={adca_enabled}")

        if adca_enabled:
            llm_client = None
            adca_config = OmegaConf.to_container(self.evaluation_runner_config.adca, resolve=True)

            # 1. Try to get client from adca config
            llm_base_url = adca_config.get("llm_base_url")
            llm_api_key = adca_config.get("llm_api_key")
            
            if llm_base_url or llm_api_key:
                try:
                    from openai import OpenAI
                    api_key = llm_api_key or os.environ.get("OPENAI_API_KEY") or "dummy"
                    llm_client = OpenAI(base_url=llm_base_url, api_key=api_key)
                    logger.info(f"Created custom ADCA LLM client with base_url={llm_base_url}")
                except Exception as e:
                    logger.error(f"Failed to create custom ADCA LLM client: {e}")

            # 2. Fallback to env_interface client
            if not llm_client and hasattr(self.env_interface, 'llm_client'):
                llm_client = self.env_interface.llm_client
                logger.info("Using env_interface LLM client for ADCA")
            
            if llm_client:
                try:
                    self.adca_analyzer = ADCAAnalyzer(llm_client, adca_config)
                    logger.info("ADCA Analyzer initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize ADCA Analyzer: {e}")
            else:
                logger.warning("ADCA enabled but no LLM client found (config or env_interface)")

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

    def _log_planner_data(self, planner_infos: List[Dict[str, Any]]) -> None:
        """
        Logs planner data including prompts, traces and other info to files.

        :param planner_infos: List of dictionaries containing planner information at each step
        """
        log_planner_data(
            output_dir=self.output_dir,
            episode_filename=self.episode_filename,
            agents=self.agents,
            planner_infos=planner_infos,
            current_instruction=self.current_instruction,
            env_interface=self.env_interface,
            log_detailed_traces=self.evaluation_runner_config.log_detailed_traces,
        )

    def _save_detailed_traces(self) -> None:
        """
        Save detailed traces to a pickle file containing instruction, action history and state history.
        """
        save_detailed_traces(
            output_dir=self.output_dir,
            episode_filename=self.episode_filename,
            current_instruction=self.current_instruction,
            env_interface=self.env_interface,
        )

    def _log_rebound_analysis(self, info: Dict[str, Any]) -> None:
        """
        Collect and save rebound analysis data.
        """
        log_rebound_analysis(
            info=info,
            planner=self.planner,
            output_dir=self.output_dir,
            episode_filename=self.episode_filename,
            env_interface=self.env_interface,
            critic=getattr(self, "critic", None),
        )


    def _log_adca_analysis(self, adca_result: Dict[str, Any]) -> None:
        """
        Save ADCA analysis results to a JSON file.
        
        Args:
            adca_result: Dictionary containing ADCA analysis results with keys:
                - step_flags: List of bool (True=GOOD, False=BAD)
                - step_advantages: List of float (computed advantages)
                - llm_response: Raw LLM response
                - analysis_stats: Dict of statistics
        """
        log_adca_analysis(
            adca_result=adca_result,
            output_dir=self.output_dir,
            episode_filename=self.episode_filename,
            env_interface=self.env_interface,
            current_instruction=self.current_instruction,
            critic=getattr(self, "critic", None),
        )

    def _habitat_to_standard_coords(self, pos: List[float]) -> List[float]:
        """
        Convert coordinates from Habitat-Sim coordinate system to standard coordinate system.
        """
        if len(pos) < 3:
            return pos
        return [pos[2], -pos[0], pos[1]]

    def _construct_world_state_dict(self) -> Dict[str, Any]:
        """
        Construct world_state_dict for critic from environment interface.

        Extracts agent poses, object positions, and furniture positions
        from world graph and simulation state.
        
        Note: Coordinates are converted from Habitat-Sim coordinate system
        (X: right, Y: up, Z: forward) to standard coordinate system
        (X: forward, Y: left, Z: up) for consistency with standard 3D calculations.

        :return: Dictionary with keys: agent_poses, object_positions, furniture_positions
        """
        world_state = {
            "agent_poses": {},
            "object_positions": {},
            "furniture_positions": {}
        }

        # Extract agent poses from simulator
        sim = self.env_interface.sim
        for agent_id in self.agents.keys():
            try:
                agent_pos = sim.agents_mgr[agent_id].articulated_agent.base_pos
                agent_rot = sim.agents_mgr[agent_id].articulated_agent.base_rot

                # Convert position from Habitat-Sim to standard coordinates
                import numpy as np
                habitat_pos = [agent_pos[0], agent_pos[1], agent_pos[2]]
                standard_pos = self._habitat_to_standard_coords(habitat_pos)

                # Compute yaw from quaternion
                # In Habitat-Sim, yaw is rotation around Y-axis (vertical axis)
                # Formula for rotation around Y-axis: atan2(2*(w*y - x*z), 1 - 2*(y^2 + x^2))
                qw, qx, qy, qz = agent_rot.w, agent_rot.x, agent_rot.y, agent_rot.z
                # Extract yaw (rotation around Y-axis in Habitat-Sim, which becomes Z-axis in standard coords)
                yaw = np.arctan2(2.0 * (qw * qy - qx * qz), 1.0 - 2.0 * (qy**2 + qx**2))

                world_state["agent_poses"][agent_id] = {
                    "position": standard_pos,
                    "rotation": [qx, qy, qz, qw],
                    "yaw": float(yaw)
                }
            except Exception as e:
                logger.debug(f"Could not extract pose for agent {agent_id}: {e}")

        # Extract object and furniture positions from world graph
        try:
            for agent_id, wg in self.env_interface.world_graph.items():
                # Get all objects
                objects = wg.get_all_objects() if hasattr(wg, 'get_all_objects') else []
                for obj in objects:
                    if hasattr(obj, 'name') and hasattr(obj, 'properties'):
                        obj_name = obj.name
                        if 'translation' in obj.properties:
                            # Convert position from Habitat-Sim to standard coordinates
                            habitat_pos = obj.properties['translation']
                            if isinstance(habitat_pos, (list, tuple)) and len(habitat_pos) >= 3:
                                standard_pos = self._habitat_to_standard_coords(list(habitat_pos))
                            else:
                                standard_pos = habitat_pos
                            
                            world_state["object_positions"][obj_name] = {
                                "position": standard_pos,
                                "parent": obj.properties.get('parent_receptacle', 'unknown')
                            }

                # Get furniture
                if hasattr(wg, 'get_all_furniture'):
                    furniture = wg.get_all_furniture()
                    for furn in furniture:
                        if hasattr(furn, 'name') and hasattr(furn, 'properties'):
                            furn_name = furn.name
                            if 'translation' in furn.properties:
                                # Convert position from Habitat-Sim to standard coordinates
                                habitat_pos = furn.properties['translation']
                                if isinstance(habitat_pos, (list, tuple)) and len(habitat_pos) >= 3:
                                    standard_pos = self._habitat_to_standard_coords(list(habitat_pos))
                                else:
                                    standard_pos = habitat_pos
                                
                                world_state["furniture_positions"][furn_name] = {
                                    "position": standard_pos
                                }

                # Only need to extract from one agent in centralized setting
                break

        except Exception as e:
            logger.debug(f"Could not extract world graph state: {e}")

        return world_state

    def _make_td_video(self) -> None:
        """
        Creates a top-down video visualization of the episode.

        :param instruction: Task instruction being executed
        """
        os.makedirs(f"{self.output_dir}/videos", exist_ok=True)
        td_video_name = f"{self.output_dir}/videos/video-td-{self.episode_filename}.mp4"

        # Create a figure and axis for the plot
        fig, ax = plt.subplots()

        # Set the number of frames in the animation
        num_frames = len(self.agent_positions)

        # Create the animation
        animation = FuncAnimation(
            fig, self._update_td, fargs=(ax,), frames=num_frames, repeat=False
        )

        # Save the animation as a video file (e.g., .mp4)
        animation.save(td_video_name, writer="ffmpeg", fps=30)

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

    def update_agent_state_history(self, planner_info: Dict[str, Any]) -> None:
        """
        Updates the state history stored in env_interface based on planner info.
        This includes logging states like "standing", "walking", "picking X", "placing on Y" etc.

        :param planner_info: Dictionary containing planner state information
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
        # Update the agent states in environment interface
        for agent_id, value in planner_info["replanned"].items():
            if value:
                # An action must be returned if the planner replans
                world_graph_snapshot = {
                    uid: wg.snapshot() 
                    for uid, wg in self.env_interface.world_graph.items()
                }
                assert agent_id in planner_info["high_level_actions"]
                action_history_object = ActionHistoryElement(
                    action=planner_info["high_level_actions"][agent_id],
                    timestamp=planner_info["sim_step_count"],
                    agent_uid=agent_id,
                    # world_graph=copy.deepcopy(self.env_interface.world_graph),
                    world_graph=world_graph_snapshot,
                    info={
                        "planner_info": planner_info,
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
                if response == "":
                    continue
                # There should have been an action logged if there is a response
                assert len(self.env_interface.agent_action_history[agent_id]) > 0
                self.env_interface.agent_action_history[agent_id][
                    -1
                ].response = response
                for ah in self.env_interface.agent_action_history[agent_id]:
                    if ah.response is None or len(ah.response) == 0:
                        raise ValueError(
                            f"Agent {agent_id} has a null response on {ah.action}"
                        )

    def _construct_trajectory_for_adca(self) -> List[Dict[str, str]]:
        """
        Constructs a linear trajectory from agent action history for ADCA analysis.
        Sorts actions by timestamp across all agents.
        """
        all_actions = []
        for agent_uid, history in self.env_interface.agent_action_history.items():
            for elem in history:
                all_actions.append(elem)
        
        # Sort by timestamp
        all_actions.sort(key=lambda x: x.timestamp)
        
        trajectory = []
        for elem in all_actions:
            # Use ActionHistoryElement.to_string() for formatted action
            # e.g. "Navigate[bed]"
            action_content = elem.to_string()
            action_str = f"Agent {elem.agent_uid}: {action_content}"
            
            # Observation/Response
            obs_str = elem.response if elem.response else ""
            
            trajectory.append({
                "action": action_str,
                "observation": obs_str
            })
            
        return trajectory

    def run_instruction(
        self, instruction: Optional[str] = None, output_name: str = ""
    ) -> Dict[str, Any]:
        """
        Runs a single instruction through the planner, taking steps until the task is done.
        Stores the information using the provided output name.

        :param instruction: Optional instruction to execute. If None, uses episode instruction.
        :param output_name: Name to use for output files. If empty, derives from instruction.

        :return: Dictionary containing execution information and metrics
        """
        # Log start time
        t_0 = time.time()

        # Counter to count iterations
        # of this loop, as sim step dont increase
        # for perception tools
        total_step_count = 1

        # Reset planners and the agents owned by the planners
        # This will also reset skills owned by the agents to
        # make eval runner ready for next episode
        self.reset_planners()

        # Initialize metadata
        self.initialize_instruction_metadata(instruction, output_name)
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
                if self.evaluation_runner_config.save_video:
                    # Store third person frames for generating video
                    self.dvu._store_for_video(
                        observations, planner_info["high_level_actions"]
                    )

                # ==================== A2C CRITIC UPDATE ====================
                # Update critic after environment step
                if hasattr(self, 'critic_enabled') and self.critic_enabled and self.critic is not None:
                    # Construct current world state
                    current_world_state = self._construct_world_state_dict()

                    # If we have a previous state, perform critic update
                    if prev_world_state is not None:
                        # Extract action information
                        action_info = planner_info.get("high_level_actions", {})

                        # Prepare info dict with task context
                        critic_info = {
                            "task_instruction": self.current_instruction,
                            "task_percent_complete": info.get("task_percent_complete", 0.0),
                            "task_state_success": info.get("task_state_success", 0.0),
                            "step_count": total_step_count,
                            "action_history": self.env_interface.agent_action_history.get(list(self.agents.keys())[0], []) if self.env_interface.agent_action_history else [],
                            "auto_eval_proposition_tracker": info.get("auto_eval_proposition_tracker", {})
                        }

                        # Call critic update
                        try:
                            self.critic.update(
                                state=prev_world_state,
                                action=action_info,
                                reward=reward,
                                next_state=current_world_state,
                                done=done,
                                info=critic_info
                            )
                        except Exception as e:
                            logger.warning(f"Critic update failed: {e}")

                    # Update previous state for next iteration
                    prev_world_state = current_world_state
                # ===========================================================

            # Get next low level actions
            low_level_actions, planner_info, should_end = self.get_low_level_actions(
                self.current_instruction, observations, self.env_interface.world_graph
            )

            # We terminate the episode if this loop gets stuck
            curr_env = self.env_interface.env.env.env._env

            if total_step_count > curr_env._max_episode_steps:
                should_end = True

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
                    if measure_name in info:
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

            # Update agent state and action history
            copy_planner_info = copy.deepcopy(planner_info)
            self.update_agent_state_history(copy_planner_info)
            self.update_agent_action_history(copy_planner_info)

            # Append planner info to history
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
        if hasattr(self, 'critic_enabled') and self.critic_enabled and getattr(self, 'critic', None) is not None:
            try:
                # Use force_end_episode if available (for A2CCritic)
                if hasattr(self.critic, 'force_end_episode'):
                    self.critic.force_end_episode()
                # Or manually call _process_episode if force_end_episode is not available
                elif hasattr(self.critic, '_process_episode') and len(getattr(self.critic, 'buffer', [])) > 0:
                     self.critic._process_episode()
            except Exception as e:
                logger.warning(f"Critic force_end_episode failed: {e}")
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
        self._log_planner_data(planner_infos)

        # Log overall time
        t_runtime = time.time() - t_0
        info["runtime"] = t_runtime

        # Merge dictionaries
        info |= planner_info

        # ==================== A2C CRITIC LOGGING ====================
        # Log critic statistics if enabled
        if hasattr(self, 'critic_enabled') and self.critic_enabled and self.critic is not None:
            try:
                critic_stats = self.critic.get_statistics()
                info['critic_stats'] = critic_stats

                if self.evaluation_runner_config.get('log_critic_stats', True):
                    write_critic_stats(
                        critic_stats=critic_stats,
                        output_dir=self.output_dir,
                        episode_filename=self.episode_filename,
                        env_interface=self.env_interface,
                        critic=self.critic,
                    )

            except Exception as e:
                logger.warning(f"Failed to log critic statistics: {e}")
        # ===========================================================

        # ==================== ADCA ANALYSIS ====================
        if self.adca_analyzer:
            try:
                trajectory = self._construct_trajectory_for_adca()
                # Determine outcome score (1.0 for success, 0.0 for failure)
                # Or use task_percent_complete if appropriate
                outcome_score = 1.0 if info.get("task_state_success", 0.0) > 0.5 else 0.0
                
                adca_result = self.adca_analyzer.analyze(
                    trajectory,
                    outcome_score,
                    self.current_instruction
                )
                
                if adca_result:
                    # Save ADCA analysis results to file
                    self._log_adca_analysis(adca_result)
            except Exception as e:
                logger.error(f"ADCA Analysis failed: {e}")
        # =======================================================

        # ==================== REBOUND ANALYSIS ====================
        self._log_rebound_analysis(info)
        # ==========================================================

        return info
