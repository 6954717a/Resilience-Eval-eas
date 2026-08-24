#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains the CentralizedEvaluationRunner class, which implements a centralized evaluation strategy
where a single planner coordinates actions across multiple agents. It extends the base EvaluationRunner class
to provide functionality for running multi-agent evaluations with centralized control.
"""

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Tuple

from hydra.utils import instantiate

logger = logging.getLogger(__name__)

from habitat_llm.agent.env import EnvironmentInterface
from habitat_llm.evaluation import EvaluationRunner
from habitat_llm.planner.planner import Planner
from habitat_llm.tools.motor_skills.motor_skill_tool import MotorSkillTool

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from habitat_llm.world_model import WorldGraph


# Evaluation runner, will go over episodes, run planners and store necessary data
class CentralizedEvaluationRunner(EvaluationRunner):
    """
    Evaluation runner that implements a centralized control strategy, where a single planner
    coordinates actions across multiple agents. This class handles episode execution,
    planner initialization, and action generation for all agents in a centralized manner.
    """

    def __init__(
        self,
        evaluation_runner_config_arg: "DictConfig",  # Hydra config type
        env_arg: EnvironmentInterface,
    ) -> None:
        """
        Initialize the CentralizedEvaluationRunner.

        :param evaluation_runner_config_arg: Configuration object containing evaluation parameters
                                           including agent and planner configurations
        :param env_arg: Environment interface instance that provides access to the simulation
        """
        super().__init__(evaluation_runner_config_arg, env_arg)

        # ==================== A2C CRITIC INTEGRATION ====================
        # Initialize A2C Critic if enabled in config
        self.critic = None
        self.critic_enabled = False

        if hasattr(self.evaluation_runner_config, 'critic'):
            critic_config = self.evaluation_runner_config.critic
            self.critic_enabled = critic_config.get('enabled', False)

            if self.critic_enabled:
                try:
                    from habitat_llm.evaluation.critic import A2CCritic
                    from omegaconf import OmegaConf

                    # Convert OmegaConf to dict for critic initialization
                    critic_config_dict = OmegaConf.to_container(critic_config, resolve=True)

                    # Update paths to be relative to the runner's output directory
                    # This ensures critic outputs are stored within the experiment results folder
                    if self.output_dir:
                        # Normalize output_dir (remove trailing slash if present)
                        normalized_output_dir = self.output_dir.rstrip('/')
                        
                        # Set analysis save dir
                        if 'analysis_save_dir' in critic_config_dict:
                            # If it's a relative path, join with output_dir
                            if not os.path.isabs(critic_config_dict['analysis_save_dir']):
                                critic_config_dict['analysis_save_dir'] = os.path.join(
                                    normalized_output_dir, 
                                    critic_config_dict['analysis_save_dir'].lstrip('./')
                                )
                        else:
                             critic_config_dict['analysis_save_dir'] = os.path.join(normalized_output_dir, 'analyses')

                        # Set checkpoint dir
                        if 'checkpoint_dir' in critic_config_dict:
                             # If it's a relative path, join with output_dir
                            if not os.path.isabs(critic_config_dict['checkpoint_dir']):
                                critic_config_dict['checkpoint_dir'] = os.path.join(
                                    normalized_output_dir, 
                                    critic_config_dict['checkpoint_dir'].lstrip('./')
                                )
                        else:
                            critic_config_dict['checkpoint_dir'] = os.path.join(normalized_output_dir, 'checkpoints')

                        # Log path information for debugging
                        logger.info(f"Critic path configuration:")
                        logger.info(f"  - Runner output_dir: {self.output_dir}")
                        logger.info(f"  - Normalized output_dir: {normalized_output_dir}")
                        logger.info(f"  - Analysis save dir: {critic_config_dict.get('analysis_save_dir', 'N/A')}")
                        logger.info(f"  - Checkpoint dir: {critic_config_dict.get('checkpoint_dir', 'N/A')}")
                    else:
                        logger.warning("self.output_dir is None or empty, using default paths from config")

                    # Initialize critic with env_interface
                    self.critic = A2CCritic(
                        config=critic_config_dict,
                        env_interface=self.env_interface
                    )

                    logger.info("A2C Critic initialized successfully")
                    logger.info(f"  - Gamma: {critic_config_dict.get('gamma', 0.99)}")
                    logger.info(f"  - GAE Lambda: {critic_config_dict.get('gae_lambda', 0.95)}")
                    logger.info(f"  - LLM Shaping: {critic_config_dict.get('use_llm_shaping', False)}")
                    logger.info(f"  - LLM Offline: {critic_config_dict.get('use_llm_offline', False)}")
                    # Log final analysis_save_dir from critic instance
                    if hasattr(self.critic, 'offline_analyzer') and self.critic.offline_analyzer:
                        logger.info(f"  - Final offline analyzer save dir: {self.critic.offline_analyzer.analysis_save_dir}")

                except ImportError as e:
                    logger.error(f"Failed to import A2CCritic: {e}")
                    logger.error("Install: torch, transformers, sentence-transformers")
                    self.critic_enabled = False
                except Exception as e:
                    logger.error(f"Failed to initialize A2CCritic: {e}")
                    self.critic_enabled = False
        # ================================================================

    def get_low_level_actions(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
    ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        """
        Given a set of observations, gets a vector of low level actions for all agents.

        :param instruction: Natural language instruction describing the task to execute
        :param observations: Dictionary of observations from the environment
        :param world_graph: Dictionary mapping agent IDs to their world graph states

        :return: Tuple containing:
            - List of low level action dictionaries for each agent
            - Dictionary with planner info about high level actions
            - Boolean indicating whether execution should end
        """
        low_level_actions, planner_info, should_end = self.planner.get_next_action(
            instruction, observations, world_graph
        )
        return low_level_actions, planner_info, should_end

    def reset_planners(self) -> None:
        """Reset the centralized planner to prepare for a new episode."""
        assert isinstance(self.planner, Planner)
        self.planner.reset()

        # Reset critic if enabled
        if self.critic_enabled and self.critic is not None:
            self.critic.reset()

    def _initialize_planners(self) -> None:
        """
        Initialize the centralized planner based on the evaluation runner configuration.
        Sets up the planner with appropriate agents and configures special planning modes
        if specified.
        """
        planner_conf = self.evaluation_runner_config.planner
        planner = instantiate(planner_conf)
        self.planner: Planner = planner(env_interface=self.env_interface)

        # Set both agents to the planner
        self.planner.agents = [
            self.agents[agent_id] for agent_id in sorted(self.agents.keys())
        ]
        if (
            "plan_config" in planner_conf
            and "planning_mode" in planner_conf.plan_config
            and planner_conf.plan_config.planning_mode == "st"
        ):
            for agent in self.planner.agents:
                for tool in agent.tools.values():
                    if isinstance(tool, MotorSkillTool):
                        tool.error_mode = "st"
