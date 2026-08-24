"""
Critic module for A2C (Advantage Actor-Critic) framework.
Provides interfaces and implementations for state evaluation and advantage estimation.

Extended with neural network-based value function approximation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from habitat_llm.evaluation.networks import StateEncoder, ValueNetwork
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

        # Initialize neural networks
        state_encoder_debug = self.config.get("state_encoder_debug", self.config.get("debug", False))
        state_encoder_debug_dir = self.config.get("state_encoder_debug_dir")
        if state_encoder_debug and not state_encoder_debug_dir:
            analysis_dir = self.config.get("analysis_save_dir")
            if analysis_dir:
                state_encoder_debug_dir = os.path.join(analysis_dir, "state_encoder")

        self.state_encoder = StateEncoder(
            text_dim=self.config.get("text_dim", 384),
            numerical_dim=self.config.get("numerical_dim", 32),
            hidden_dim=self.config.get("encoder_hidden_dim", 256),
            output_dim=self.config.get("state_dim", 128),
            device=self.device,
            debug=state_encoder_debug,
            debug_dir=state_encoder_debug_dir,
            debug_frequency=self.config.get("state_encoder_debug_frequency", 1),
            debug_max_entries=self.config.get("state_encoder_debug_max_entries", 1000),
            debug_log_state=self.config.get("state_encoder_debug_log_state", True),
            debug_log_next_state=self.config.get("state_encoder_debug_log_next_state", False),
            debug_log_on_change=self.config.get("state_encoder_debug_log_on_change", False),
            debug_change_threshold=self.config.get("state_encoder_debug_change_threshold", 0.0),
            debug_text_max_len=self.config.get("state_encoder_debug_text_max_len", 200),
            debug_log_delta=self.config.get("state_encoder_debug_log_delta", True),
            debug_log_task_context_once=self.config.get("state_encoder_debug_log_task_context_once", True),
            debug_log_full_features_first=self.config.get("state_encoder_debug_log_full_features_first", True),
            debug_key_feature_names=self.config.get("state_encoder_debug_key_feature_names"),
            debug_signature_mode=self.config.get("state_encoder_debug_signature_mode", "semantic"),
            debug_signature_feature_names=self.config.get("state_encoder_debug_signature_feature_names"),
        )

        self.value_network = ValueNetwork(
            state_dim=self.config.get("state_dim", 128),
            hidden_dims=self.config.get("value_hidden_dims", [256, 128, 64]),
            dropout_rate=self.config.get("dropout_rate", 0.1),
            device=self.device
        )

        # Optimizer for value network
        self.value_optimizer = torch.optim.Adam(
            self.value_network.parameters(),
            lr=self.config.get("value_lr", 1e-3)
        )

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
        self.buffer: List[Tuple[torch.Tensor, Any, float, torch.Tensor, bool, Dict]] = []
        self.raw_states: List[Tuple[Any, Any, float, Any, bool]] = []  # For LLM analysis
        self.last_analysis_path: Optional[str] = None

        # Training statistics
        self.training_stats = {
            "value_loss": 0.0,
            "mean_predicted_value": 0.0,
            "mean_return": 0.0,
            "episodes_trained": 0
        }

        # Checkpoint settings
        self.save_checkpoints = self.config.get("save_checkpoints", False)
        self.checkpoint_frequency = self.config.get("checkpoint_frequency", 10)
        self.checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")

        if self.save_checkpoints:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        logger.info(f"Initialized A2CCritic: gamma={self.gamma}, gae_lambda={self.gae_lambda}, "
                   f"device={self.device}, llm_shaping={self.use_llm_shaping}")

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
            info: Additional info (must contain task_instruction for LLM)
        """
        # Apply LLM reward shaping if enabled
        if self.reward_shaper:
            task_instruction = info.get("task_instruction", "")
            shaped_reward = self.reward_shaper.shape_reward(
                state, action, next_state, reward, task_instruction, info
            )
        else:
            shaped_reward = reward

        # Encode states
        state_info = dict(info) if info else None
        if state_info is not None:
            state_info["state_kind"] = "state"
        next_state_info = dict(info) if info else None
        if next_state_info is not None:
            next_state_info["state_kind"] = "next_state"
        state_vec = self.state_encoder.encode(state, state_info)
        next_state_vec = self.state_encoder.encode(next_state, next_state_info)

        # Store in buffer
        self.buffer.append((
            state_vec.detach(),
            action,
            shaped_reward,
            next_state_vec.detach(),
            done,
            info
        ))

        # Store raw states for LLM offline analysis
        if self.offline_analyzer:
            # Use StateEncoder to generate compressed text description of state
            # self.raw_states.append((state, action, shaped_reward, next_state, done))
            
            # Now we use the state str description
            state_desc = self.state_encoder._generate_state_description(state)
            next_state_desc = self.state_encoder._generate_state_description(next_state)
            
            planning_step = info.get("planning_step_count", 0) if info else 0
            self.raw_states.append((state_desc, action, shaped_reward, next_state_desc, done, planning_step))

        # Process episode when done
        if done:
            self._process_episode()

    def reset(self) -> None:
        """Reset Critic state for new episode."""
        self.buffer = []
        self.raw_states = []
        self.last_analysis_path = None

    def force_end_episode(self) -> None:
        """
        Force end the current episode.
        Useful when the planner decides to end the episode but the environment done flag was False.
        This ensures the episode is processed and the model is updated.
        """
        if len(self.buffer) > 0:
            # Mark the last transition as done
            last_transition = list(self.buffer[-1])
            last_transition[4] = True  # Set done=True
            self.buffer[-1] = tuple(last_transition)
            
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

        # Unpack buffer
        states, actions, rewards, next_states, dones, infos = zip(*self.buffer)

        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # Compute V(s) and V(s')
        with torch.no_grad():
            values = self.value_network(states).cpu().numpy()
            next_values = self.value_network(next_states).cpu().numpy()

        # Compute GAE
        advantages, returns = compute_gae(
            rewards, values, next_values, dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda
        )

        # Update value network
        self._update_value_network(states, returns)

        # Run LLM offline analysis (if enabled)
        if self.offline_analyzer and len(self.raw_states) > 0:
            task_instruction = infos[0].get("task_instruction", "")
            success = infos[-1].get("task_state_success", False)
            percent_complete = infos[-1].get("task_percent_complete", 0.0)
            total_reward = sum(rewards)

            episode_id = infos[0].get("episode_id") if infos else None
            episode_filename = infos[0].get("episode_filename") if infos else None
            
            # Extract max planning step
            final_planning_step = self.raw_states[-1][5] if len(self.raw_states[-1]) > 5 else 0

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

    def compute_value_variance(self, window: int = 10) -> float:
        """
        Compute the variance of Value function V(s) over a sliding window.
        Metrics.tex: sigma^2_V
        """
        if len(self.buffer) < 2:
            return 0.0
            
        # Get recent V(s) estimates from buffer
        start_idx = max(0, len(self.buffer) - window)
        recent_transitions = self.buffer[start_idx:]
        
        states = [t[0] for t in recent_transitions]
        if not states:
            return 0.0
            
        states_tensor = torch.stack(states).to(self.device)
        with torch.no_grad():
            values = self.value_network(states_tensor).cpu().numpy().flatten()
            
        return float(np.var(values))

    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        stats = self.training_stats.copy()
        if self.last_analysis_path:
            stats["last_llm_analysis"] = self.last_analysis_path
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
            logger.info(f"Loaded checkpoint from {path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")

