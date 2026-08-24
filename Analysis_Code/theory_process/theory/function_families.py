# -*- coding: utf-8 -*-
"""
Reward Function Family Definition

Implements the reward evaluation function family:
    R = {r_θ : X → [0, 1] | θ ∈ Θ}

Where input space X = S × A × C × T × [0,1]
- S: State space (world_state_dict)
- A: Action space (~149,085 actions)
- C: Text context (task instruction, observations)
- T: Tool call trajectory history
- [0,1]: Sub-goal completion degree

Function family decomposition:
    r_θ(x) = α · r_PRM^LLM(x) + (1-α) · r_A2C^NN(x)

Date: 2025-12-29
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import numpy as np
from pathlib import Path


class InputTuple(NamedTuple):
    """
    Input tuple for reward function evaluation.

    X = S × A × C × T × [0,1]
    """
    state: Dict[str, Any]           # S: world_state_dict
    action: str                      # A: action string
    context: str                     # C: task instruction + observations
    trajectory: List[str]            # T: tool call history
    subgoal_completion: float        # [0,1]: completion degree


@dataclass
class RewardFamilyCapacity:
    """
    Capacity metrics for the reward function family.

    These metrics are used for Rademacher complexity computation.
    """
    # VC-dimension estimates
    vc_dim_llm: float = np.inf       # LLM has infinite VC-dim (heuristic)
    vc_dim_nn: float = 0.0           # NN VC-dim ≈ O(params * log(params))

    # Parameter counts
    nn_params: int = 0               # Number of NN parameters
    llm_context_length: int = 4096   # LLM effective context

    # Lipschitz constants (for stability)
    lipschitz_nn: float = 1.0        # NN Lipschitz constant estimate

    # Effective capacity measures
    effective_capacity: float = 0.0  # Combined capacity estimate
    covering_number_log: float = 0.0 # log N(ε, F, ||·||)

    def to_dict(self) -> Dict[str, float]:
        return {
            'vc_dim_llm': self.vc_dim_llm,
            'vc_dim_nn': self.vc_dim_nn,
            'nn_params': self.nn_params,
            'llm_context_length': self.llm_context_length,
            'lipschitz_nn': self.lipschitz_nn,
            'effective_capacity': self.effective_capacity,
            'covering_number_log': self.covering_number_log,
        }


@dataclass
class RewardFunctionFamily:
    """
    Reward evaluation function family for offline batched RL.

    Implements:
        r_θ(x) = α · r_PRM^LLM(x) + (1-α) · r_A2C^NN(x)

    Where:
        - r_PRM^LLM(x) ∈ {-0.2, +0.2} (ADCA: fix_base)
        - r_A2C^NN(x) = σ(W_L · ReLU(... W_1 x ...)) (ValueNetwork)

    Attributes:
        alpha: Mixing coefficient for LLM vs NN reward
        prm_good_reward: Reward for GOOD step (default: +0.2)
        prm_bad_reward: Reward for BAD step (default: -0.2)
        gamma: Discount factor
        nn_hidden_dims: ValueNetwork architecture
    """

    # Mixing coefficient
    alpha: float = 0.5

    # PRM (Process Reward Model) parameters
    prm_good_reward: float = 0.2
    prm_bad_reward: float = -0.2

    # Discount factor
    gamma: float = 0.99

    # ValueNetwork architecture (MLP)
    nn_hidden_dims: List[int] = field(default_factory=lambda: [128, 256, 128, 64, 1])

    # State encoder dimensions
    text_embedding_dim: int = 384    # Text encoder output
    numerical_dim: int = 32          # Numerical state features
    state_dim: int = 128             # Final state representation

    def __post_init__(self):
        """Initialize derived attributes."""
        self._compute_nn_params()

    def _compute_nn_params(self) -> int:
        """Compute total NN parameters for capacity estimation."""
        params = 0
        prev_dim = self.state_dim

        for hidden_dim in self.nn_hidden_dims:
            params += prev_dim * hidden_dim + hidden_dim  # weights + bias
            prev_dim = hidden_dim

        self._nn_params = params
        return params

    def evaluate_prm(self, step_flag: bool) -> float:
        """
        Evaluate PRM (Process Reward Model) for a step.

        Args:
            step_flag: True for GOOD step, False for BAD step

        Returns:
            PRM reward: +0.2 (GOOD) or -0.2 (BAD)
        """
        return self.prm_good_reward if step_flag else self.prm_bad_reward

    def evaluate_nn_placeholder(self, state_encoding: np.ndarray) -> float:
        """
        Placeholder for NN value function evaluation.

        In practice, this would use the actual ValueNetwork.
        Here we provide a placeholder that estimates based on state features.

        Args:
            state_encoding: Encoded state vector (128-dim)

        Returns:
            Estimated value (placeholder)
        """
        # Placeholder: use L2 norm as proxy for "quality"
        if state_encoding is None or len(state_encoding) == 0:
            return 0.0

        # Normalize to [0, 1] range
        norm = np.linalg.norm(state_encoding)
        return float(np.clip(norm / 10.0, 0.0, 1.0))

    def evaluate(self, x: InputTuple, step_flag: Optional[bool] = None,
                 state_encoding: Optional[np.ndarray] = None) -> float:
        """
        Evaluate reward function for input tuple.

        r_θ(x) = α · r_PRM^LLM(x) + (1-α) · r_A2C^NN(x)

        Args:
            x: Input tuple (state, action, context, trajectory, completion)
            step_flag: Optional GOOD/BAD label from ADCA
            state_encoding: Optional encoded state for NN evaluation

        Returns:
            Combined reward value
        """
        # PRM component
        if step_flag is not None:
            r_prm = self.evaluate_prm(step_flag)
        else:
            # Default: use sub-goal completion as proxy
            r_prm = self.prm_good_reward if x.subgoal_completion > 0.5 else self.prm_bad_reward

        # NN component
        r_nn = self.evaluate_nn_placeholder(state_encoding)

        # Combined reward
        reward = self.alpha * r_prm + (1 - self.alpha) * r_nn

        return float(np.clip(reward, -1.0, 1.0))

    def compute_family_capacity(self) -> RewardFamilyCapacity:
        """
        Compute capacity metrics for the reward function family.

        Used for Rademacher complexity bounds.

        Returns:
            RewardFamilyCapacity with various capacity measures
        """
        # NN VC-dimension approximation: O(params * log(params))
        # For deep ReLU networks
        nn_params = self._nn_params
        vc_dim_nn = nn_params * np.log(nn_params + 1) if nn_params > 0 else 0

        # Effective capacity: weighted combination
        # LLM has very high capacity but bounded by context length
        llm_effective_capacity = min(self.text_embedding_dim, 4096)

        effective_capacity = (
            self.alpha * llm_effective_capacity +
            (1 - self.alpha) * vc_dim_nn
        )

        # Covering number estimate (log scale)
        # log N(ε, F, ||·||∞) ≈ VC-dim * log(1/ε)
        epsilon = 0.01
        covering_number_log = effective_capacity * np.log(1 / epsilon)

        return RewardFamilyCapacity(
            vc_dim_llm=llm_effective_capacity,
            vc_dim_nn=vc_dim_nn,
            nn_params=nn_params,
            llm_context_length=4096,
            lipschitz_nn=1.0,  # ReLU networks can be Lipschitz bounded
            effective_capacity=effective_capacity,
            covering_number_log=covering_number_log,
        )

    def compute_reward_range(self) -> Tuple[float, float]:
        """
        Compute the range of possible reward values.

        Returns:
            (min_reward, max_reward) tuple
        """
        # PRM range
        prm_min = min(self.prm_good_reward, self.prm_bad_reward)
        prm_max = max(self.prm_good_reward, self.prm_bad_reward)

        # NN range (assumed [0, 1])
        nn_min, nn_max = 0.0, 1.0

        # Combined range
        min_reward = self.alpha * prm_min + (1 - self.alpha) * nn_min
        max_reward = self.alpha * prm_max + (1 - self.alpha) * nn_max

        return (min_reward, max_reward)

    def compute_prm_statistics(self, step_flags: List[bool]) -> Dict[str, float]:
        """
        Compute PRM reward statistics from ADCA step flags.

        Args:
            step_flags: List of GOOD/BAD labels

        Returns:
            Statistics dictionary
        """
        if not step_flags:
            return {
                'mean': 0.0,
                'std': 0.0,
                'good_ratio': 0.0,
                'bad_ratio': 0.0,
                'total_steps': 0,
            }

        rewards = [self.evaluate_prm(flag) for flag in step_flags]
        good_count = sum(step_flags)
        bad_count = len(step_flags) - good_count

        return {
            'mean': float(np.mean(rewards)),
            'std': float(np.std(rewards)),
            'good_ratio': good_count / len(step_flags),
            'bad_ratio': bad_count / len(step_flags),
            'total_steps': len(step_flags),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'alpha': self.alpha,
            'prm_good_reward': self.prm_good_reward,
            'prm_bad_reward': self.prm_bad_reward,
            'gamma': self.gamma,
            'nn_hidden_dims': self.nn_hidden_dims,
            'text_embedding_dim': self.text_embedding_dim,
            'numerical_dim': self.numerical_dim,
            'state_dim': self.state_dim,
            'nn_params': self._nn_params,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'RewardFunctionFamily':
        """Create from dictionary."""
        return cls(
            alpha=d.get('alpha', 0.5),
            prm_good_reward=d.get('prm_good_reward', 0.2),
            prm_bad_reward=d.get('prm_bad_reward', -0.2),
            gamma=d.get('gamma', 0.99),
            nn_hidden_dims=d.get('nn_hidden_dims', [128, 256, 128, 64, 1]),
            text_embedding_dim=d.get('text_embedding_dim', 384),
            numerical_dim=d.get('numerical_dim', 32),
            state_dim=d.get('state_dim', 128),
        )


def load_step_flags_from_adca(adca_data: Dict[str, Any]) -> List[bool]:
    """
    Load step flags from ADCA analysis data.

    Args:
        adca_data: ADCA JSON data with step_flags

    Returns:
        List of boolean step flags
    """
    step_flags_raw = adca_data.get('step_flags', [])

    # Convert to boolean
    step_flags = []
    for flag in step_flags_raw:
        if isinstance(flag, bool):
            step_flags.append(flag)
        elif isinstance(flag, str):
            step_flags.append(flag.upper() == 'GOOD')
        else:
            step_flags.append(bool(flag))

    return step_flags


def create_input_tuple_from_adca(
    adca_data: Dict[str, Any],
    step_index: int = 0
) -> InputTuple:
    """
    Create InputTuple from ADCA data.

    Args:
        adca_data: ADCA JSON data
        step_index: Step index in trajectory

    Returns:
        InputTuple for reward evaluation
    """
    # Extract components
    task_instruction = adca_data.get('task_instruction', '')
    step_flags = load_step_flags_from_adca(adca_data)
    total_steps = len(step_flags)

    # Sub-goal completion: ratio of GOOD steps so far
    if step_index > 0 and total_steps > 0:
        good_so_far = sum(step_flags[:step_index])
        subgoal_completion = good_so_far / step_index
    else:
        subgoal_completion = 0.0

    # Create placeholder state and action
    state = {'step_index': step_index, 'total_steps': total_steps}
    action = f"step_{step_index}"
    trajectory = [f"step_{i}" for i in range(step_index)]

    return InputTuple(
        state=state,
        action=action,
        context=task_instruction,
        trajectory=trajectory,
        subgoal_completion=subgoal_completion,
    )
