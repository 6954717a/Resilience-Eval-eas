#!/usr/bin/env python3

"""
Critic Factory

Creates and configures A2C Critic instances with proper path normalization.
Extracted from centralized_evaluation_runner.py to decouple critic initialization.
"""

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CriticFactory:
    """
    Factory for creating and configuring A2C Critic instances.

    Responsibilities:
    - Create Critic from config
    - Normalize paths relative to output directory
    - Handle import errors gracefully
    - Log configuration details
    """

    @staticmethod
    def create_critic(
        critic_config: Any,
        env_interface: Any,
        output_dir: Optional[str] = None,
        log_trajectory_export: bool = False,
    ) -> Optional[Any]:
        """
        Create A2C Critic instance from configuration.

        Args:
            critic_config: Critic configuration (OmegaConf or dict)
            env_interface: Environment interface
            output_dir: Output directory for critic artifacts
            log_trajectory_export: Whether to enable trajectory export

        Returns:
            A2CCritic instance or None if disabled/failed
        """
        # Check if critic is enabled
        enabled = critic_config.get('enabled', False) if hasattr(critic_config, 'get') else getattr(critic_config, 'enabled', False)

        if not enabled:
            logger.info("A2C Critic is disabled in config")
            return None

        try:
            from habitat_llm.evaluation.critic import A2CCritic
            from omegaconf import OmegaConf

            # Convert OmegaConf to dict
            if hasattr(critic_config, "_metadata"):
                critic_config_dict = OmegaConf.to_container(critic_config, resolve=True)
            else:
                critic_config_dict = dict(critic_config)

            # Normalize paths if output_dir is provided
            if output_dir:
                critic_config_dict = CriticFactory._normalize_paths(
                    critic_config_dict,
                    output_dir,
                    log_trajectory_export,
                )

            # Create critic instance
            critic = A2CCritic(
                config=critic_config_dict,
                env_interface=env_interface
            )

            # Log configuration
            CriticFactory._log_critic_config(critic, critic_config_dict)

            logger.info("A2C Critic initialized successfully")
            return critic

        except ImportError as e:
            logger.error(f"Failed to import A2CCritic: {e}")
            logger.error("Required packages: torch, transformers, sentence-transformers")
            return None

        except Exception as e:
            logger.error(f"Failed to initialize A2CCritic: {e}", exc_info=True)
            return None

    @staticmethod
    def _normalize_paths(
        config_dict: Dict,
        output_dir: str,
        log_trajectory_export: bool,
    ) -> Dict:
        """
        Normalize critic paths relative to output directory.

        Args:
            config_dict: Critic configuration dictionary
            output_dir: Output directory path
            log_trajectory_export: Whether to enable trajectory export

        Returns:
            Updated configuration dictionary
        """
        # Normalize output_dir (remove trailing slash)
        normalized_output_dir = output_dir.rstrip('/')

        def _normalize_under_output(key: str, default_relative: str) -> None:
            """Helper to normalize a single path key."""
            configured_value = config_dict.get(key)
            if configured_value:
                if not os.path.isabs(configured_value):
                    config_dict[key] = os.path.join(
                        normalized_output_dir,
                        str(configured_value).lstrip("./"),
                    )
            else:
                config_dict[key] = os.path.join(
                    normalized_output_dir,
                    default_relative,
                )

        # Normalize standard paths
        _normalize_under_output("analysis_save_dir", "analyses")
        _normalize_under_output(
            "analysis_export_dir",
            os.path.join("analyses", "critic_exports"),
        )
        _normalize_under_output("analysis_focus_dir", "analyze")
        _normalize_under_output("checkpoint_dir", "checkpoints")

        state_encoder_cfg = config_dict.setdefault("state_encoder", {})
        debug_cfg = state_encoder_cfg.setdefault("debug", {})

        debug_dir = debug_cfg.get("dir", config_dict.get("state_encoder_debug_dir"))
        if debug_dir:
            if not os.path.isabs(debug_dir):
                debug_dir = os.path.join(
                    normalized_output_dir,
                    str(debug_dir).lstrip("./"),
                )
        else:
            debug_dir = os.path.join(
                normalized_output_dir,
                os.path.join("analyses", "state_encoder"),
            )

        debug_cfg["dir"] = debug_dir
        config_dict["state_encoder_debug_dir"] = debug_dir

        # Set runner output dir
        config_dict['runner_output_dir'] = normalized_output_dir

        # Enable trajectory export if requested
        if log_trajectory_export:
            config_dict['analysis_export_enabled'] = True

        # Log path configuration
        logger.info("Critic path configuration:")
        logger.info(f"  - Runner output_dir: {output_dir}")
        logger.info(f"  - Normalized output_dir: {normalized_output_dir}")
        logger.info(f"  - Analysis save dir: {config_dict.get('analysis_save_dir', 'N/A')}")
        logger.info(f"  - Analysis export dir: {config_dict.get('analysis_export_dir', 'N/A')}")
        logger.info(f"  - Analysis focus dir: {config_dict.get('analysis_focus_dir', 'N/A')}")
        logger.info(f"  - StateEncoder debug dir: {config_dict.get('state_encoder_debug_dir', 'N/A')}")
        logger.info(f"  - Checkpoint dir: {config_dict.get('checkpoint_dir', 'N/A')}")

        return config_dict

    @staticmethod
    def _log_critic_config(critic: Any, config_dict: Dict) -> None:
        """
        Log critic configuration details.

        Args:
            critic: A2CCritic instance
            config_dict: Configuration dictionary
        """
        logger.info("Critic configuration:")
        logger.info(f"  - Gamma: {config_dict.get('gamma', 0.99)}")
        logger.info(f"  - GAE Lambda: {config_dict.get('gae_lambda', 0.95)}")
        logger.info(f"  - LLM Shaping: {config_dict.get('use_llm_shaping', False)}")
        logger.info(f"  - LLM Offline: {config_dict.get('use_llm_offline', False)}")
        logger.info(f"  - Learning Rate: {config_dict.get('learning_rate', 1e-4)}")
        logger.info(f"  - Batch Size: {config_dict.get('batch_size', 32)}")

        # Log final analysis_save_dir from critic instance
        if hasattr(critic, 'offline_analyzer') and critic.offline_analyzer:
            logger.info(f"  - Final offline analyzer save dir: {critic.offline_analyzer.analysis_save_dir}")


__all__ = ["CriticFactory"]
