# -*- coding: utf-8 -*-
"""
Bellman Residual Function Family

Implements value function and Bellman residual families:

Value Function Family:
    V = {V_φ : S → R | φ ∈ Φ}

Bellman Residual Family:
    B = {δ_φ,θ : S × A × S → R}
    δ_φ,θ(s, a, s') = r_θ(s, a) + γ V_φ(s') - V_φ(s)

GAE Advantage:
    A_t^GAE(γ,λ) = Σ_{l=0}^∞ (γλ)^l δ_{t+l}

ADCA Advantage (suffix sum):
    A_t^ADCA = Σ_{k=t}^T [α · r_PRM(s_k, a_k) + 1_{k=T} · r_ORM]

Date: 2025-12-29
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import numpy as np
from pathlib import Path

from .function_families import RewardFunctionFamily, InputTuple


class Transition(NamedTuple):
    """
    Single transition (s, a, r, s', done).
    """
    state: Dict[str, Any]        # Current state
    action: str                   # Action taken
    reward: float                 # Immediate reward
    next_state: Dict[str, Any]   # Next state
    done: bool                    # Episode termination flag
    step_flag: Optional[bool] = None  # ADCA GOOD/BAD label


@dataclass
class Trajectory:
    """
    Complete trajectory of transitions.
    """
    episode_id: str
    task_instruction: str
    transitions: List[Transition]
    step_flags: List[bool]
    step_advantages: List[float]
    orm_score: float              # Outcome Reward Model score
    prm_mean: float               # Process Reward Model mean
    total_return: float = 0.0

    def __len__(self) -> int:
        return len(self.transitions)

    @property
    def good_step_ratio(self) -> float:
        if not self.step_flags:
            return 0.0
        return sum(self.step_flags) / len(self.step_flags)


@dataclass
class BellmanResidualFamily:
    """
    Bellman Residual Function Family.

    Implements:
        δ_φ,θ(s, a, s') = r_θ(s, a) + γ V_φ(s') - V_φ(s)

    And advantage functions:
        - GAE: A_t^GAE = Σ (γλ)^l δ_{t+l}
        - ADCA: A_t^ADCA = Σ_{k=t}^T [α · r_PRM + 1_{k=T} · r_ORM]

    Attributes:
        reward_family: Reward function family
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        prm_scale: Scaling factor for PRM rewards
    """

    reward_family: RewardFunctionFamily = field(default_factory=RewardFunctionFamily)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    prm_scale: float = 1.0

    def compute_td_error(
        self,
        reward: float,
        value_current: float,
        value_next: float,
        done: bool
    ) -> float:
        """
        Compute TD error (Bellman residual).

        δ = r + γ V(s') - V(s)  (if not done)
        δ = r - V(s)            (if done)

        Args:
            reward: Immediate reward r
            value_current: V(s)
            value_next: V(s')
            done: Episode termination flag

        Returns:
            TD error δ
        """
        if done:
            return reward - value_current
        else:
            return reward + self.gamma * value_next - value_current

    def compute_gae_advantages(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0.0
    ) -> np.ndarray:
        """
        Compute Generalized Advantage Estimation (GAE).

        A_t^GAE(γ,λ) = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}

        Where δ_t = r_t + γV(s_{t+1}) - V(s_t)

        Args:
            rewards: List of rewards [r_0, r_1, ..., r_{T-1}]
            values: List of values [V(s_0), V(s_1), ..., V(s_{T-1})]
            dones: List of done flags
            next_value: V(s_T) for bootstrapping

        Returns:
            Array of GAE advantages [A_0, A_1, ..., A_{T-1}]
        """
        n = len(rewards)
        if n == 0:
            return np.array([])

        advantages = np.zeros(n)
        last_gae = 0.0

        # Compute backwards
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            # TD error
            delta = self.compute_td_error(
                rewards[t], values[t], next_val, dones[t]
            )

            # GAE update
            if dones[t]:
                last_gae = delta
            else:
                last_gae = delta + self.gamma * self.gae_lambda * last_gae

            advantages[t] = last_gae

        return advantages

    def compute_adca_advantages(
        self,
        step_flags: List[bool],
        orm_score: float,
        prm_good_reward: float = 0.2,
        prm_bad_reward: float = -0.2
    ) -> np.ndarray:
        """
        Compute ADCA-style advantages (suffix sum).

        A_t^ADCA = Σ_{k=t}^T [α · r_PRM(s_k, a_k) + 1_{k=T} · r_ORM]

        This is the suffix-sum formulation where each step's advantage
        is the sum of all future PRM rewards plus the final ORM reward.

        Args:
            step_flags: List of GOOD/BAD labels
            orm_score: Outcome Reward Model score (task completion)
            prm_good_reward: Reward for GOOD step
            prm_bad_reward: Reward for BAD step

        Returns:
            Array of ADCA advantages
        """
        n = len(step_flags)
        if n == 0:
            return np.array([])

        # Compute PRM rewards
        prm_rewards = np.array([
            prm_good_reward if flag else prm_bad_reward
            for flag in step_flags
        ])

        # Scale PRM rewards
        prm_rewards = prm_rewards * self.prm_scale

        # Suffix sum computation
        # A_t = Σ_{k=t}^{T-1} r_PRM[k] + ORM
        advantages = np.zeros(n)
        suffix_sum = orm_score  # ORM is added at the end

        for t in reversed(range(n)):
            suffix_sum += prm_rewards[t]
            advantages[t] = suffix_sum

        return advantages

    def compute_bellman_loss(
        self,
        transitions: List[Transition],
        values: List[float]
    ) -> float:
        """
        Compute Bellman error loss.

        L_Bellman = E[(δ(s, a, s'))²]

        Args:
            transitions: List of transitions
            values: List of value estimates

        Returns:
            Mean squared Bellman error
        """
        if not transitions:
            return 0.0

        n = len(transitions)
        td_errors = []

        for i, trans in enumerate(transitions):
            # Get value estimates
            v_current = values[i] if i < len(values) else 0.0
            v_next = values[i + 1] if i + 1 < len(values) else 0.0

            # Compute TD error
            delta = self.compute_td_error(
                trans.reward, v_current, v_next, trans.done
            )
            td_errors.append(delta)

        # Mean squared error
        return float(np.mean(np.array(td_errors) ** 2))

    def compute_bellman_residual_distribution(
        self,
        trajectories: List[Trajectory]
    ) -> Dict[str, Any]:
        """
        Compute Bellman residual distribution statistics.

        Analyzes the distribution of TD errors across all trajectories.

        Args:
            trajectories: List of trajectory data

        Returns:
            Distribution statistics
        """
        all_residuals = []

        for traj in trajectories:
            # Use ADCA advantages as proxy for residuals
            residuals = traj.step_advantages
            if residuals:
                all_residuals.extend(residuals)

        if not all_residuals:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'q25': 0.0,
                'q75': 0.0,
                'n_samples': 0,
            }

        residuals_arr = np.array(all_residuals)

        return {
            'mean': float(np.mean(residuals_arr)),
            'std': float(np.std(residuals_arr)),
            'min': float(np.min(residuals_arr)),
            'max': float(np.max(residuals_arr)),
            'median': float(np.median(residuals_arr)),
            'q25': float(np.percentile(residuals_arr, 25)),
            'q75': float(np.percentile(residuals_arr, 75)),
            'n_samples': len(residuals_arr),
        }

    def compare_gae_vs_adca(
        self,
        step_flags: List[bool],
        orm_score: float,
        values: Optional[List[float]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compare GAE and ADCA advantage computations.

        Args:
            step_flags: ADCA GOOD/BAD labels
            orm_score: Final outcome score
            values: Optional value estimates for GAE

        Returns:
            Dictionary with 'gae' and 'adca' advantages
        """
        n = len(step_flags)

        # ADCA advantages
        adca_advantages = self.compute_adca_advantages(step_flags, orm_score)

        # GAE advantages (using PRM rewards if no values provided)
        if values is None:
            # Use zero values as baseline
            values = [0.0] * n

        prm_rewards = [
            self.reward_family.prm_good_reward if flag else self.reward_family.prm_bad_reward
            for flag in step_flags
        ]
        dones = [False] * (n - 1) + [True]

        gae_advantages = self.compute_gae_advantages(prm_rewards, values, dones)

        return {
            'gae': gae_advantages,
            'adca': adca_advantages,
        }

    def compute_advantage_correlation(
        self,
        trajectories: List[Trajectory]
    ) -> Dict[str, float]:
        """
        Compute correlation between ADCA advantages and trajectory outcomes.

        Args:
            trajectories: List of trajectories

        Returns:
            Correlation statistics
        """
        if not trajectories:
            return {'correlation': 0.0, 'n_samples': 0}

        # Collect mean advantages and outcomes
        mean_advantages = []
        outcomes = []

        for traj in trajectories:
            if traj.step_advantages:
                mean_adv = np.mean(traj.step_advantages)
                mean_advantages.append(mean_adv)
                outcomes.append(traj.orm_score)

        if len(mean_advantages) < 2:
            return {'correlation': 0.0, 'n_samples': len(mean_advantages)}

        # Compute Pearson correlation
        adv_arr = np.array(mean_advantages)
        out_arr = np.array(outcomes)

        correlation = np.corrcoef(adv_arr, out_arr)[0, 1]

        return {
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'mean_advantage': float(np.mean(adv_arr)),
            'std_advantage': float(np.std(adv_arr)),
            'mean_outcome': float(np.mean(out_arr)),
            'std_outcome': float(np.std(out_arr)),
            'n_samples': len(mean_advantages),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'prm_scale': self.prm_scale,
            'reward_family': self.reward_family.to_dict(),
        }


def load_trajectory_from_adca(adca_data: Dict[str, Any]) -> Trajectory:
    """
    Load trajectory from ADCA analysis data.

    Args:
        adca_data: ADCA JSON data

    Returns:
        Trajectory object
    """
    # Extract step data
    step_flags_raw = adca_data.get('step_flags', [])
    step_advantages_raw = adca_data.get('step_advantages', [])

    # Convert step flags to boolean
    step_flags = []
    for flag in step_flags_raw:
        if isinstance(flag, bool):
            step_flags.append(flag)
        elif isinstance(flag, str):
            step_flags.append(flag.upper() == 'GOOD')
        else:
            step_flags.append(bool(flag))

    # Convert advantages to float
    step_advantages = [float(a) for a in step_advantages_raw] if step_advantages_raw else []

    # Get analysis stats
    analysis_stats = adca_data.get('analysis_stats', {})
    adca_result = adca_data.get('adca_result', {})

    if adca_result:
        analysis_stats = adca_result.get('analysis_stats', analysis_stats)
        if not step_flags:
            step_flags_raw = adca_result.get('step_flags', [])
            step_flags = [
                (flag if isinstance(flag, bool) else flag.upper() == 'GOOD' if isinstance(flag, str) else bool(flag))
                for flag in step_flags_raw
            ]
        if not step_advantages:
            step_advantages = [float(a) for a in adca_result.get('step_advantages', [])]

    orm_score = analysis_stats.get('adca/orm_score', 0.0)
    prm_mean = analysis_stats.get('adca/prm_mean', 0.0)

    # Create placeholder transitions
    n = len(step_flags)
    transitions = []
    for i in range(n):
        trans = Transition(
            state={'step': i},
            action=f'step_{i}',
            reward=0.2 if step_flags[i] else -0.2,
            next_state={'step': i + 1},
            done=(i == n - 1),
            step_flag=step_flags[i] if i < len(step_flags) else None,
        )
        transitions.append(trans)

    return Trajectory(
        episode_id=adca_data.get('episode_id', adca_data.get('episode_filename', 'unknown')),
        task_instruction=adca_data.get('task_instruction', ''),
        transitions=transitions,
        step_flags=step_flags,
        step_advantages=step_advantages,
        orm_score=orm_score if orm_score else 0.0,
        prm_mean=prm_mean if prm_mean else 0.0,
        total_return=sum(step_advantages) if step_advantages else 0.0,
    )


def load_trajectories_from_adca_dir(adca_dir: Path) -> List[Trajectory]:
    """
    Load all trajectories from ADCA directory.

    Args:
        adca_dir: Path to ADCA output directory

    Returns:
        List of Trajectory objects
    """
    import json

    trajectories = []
    adca_path = Path(adca_dir)

    if not adca_path.exists():
        return trajectories

    for json_file in adca_path.glob('*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                traj = load_trajectory_from_adca(data)
                if len(traj.step_flags) > 0:
                    trajectories.append(traj)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return trajectories
