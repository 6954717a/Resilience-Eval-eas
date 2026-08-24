"""
Generalized Advantage Estimation (GAE) Calculator

Implements GAE(γ, λ) for computing advantages in A2C.

Reference:
    Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
    https://arxiv.org/abs/1506.02438
"""

import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation (GAE).

    GAE Formula:
        δ_t = r_t + γ V(s_{t+1}) (1 - done_t) - V(s_t)  # TD error
        A_t^{GAE(γ,λ)} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}

    Args:
        rewards: Reward sequence [T]
        values: V(s_t) estimates [T]
        next_values: V(s_{t+1}) estimates [T]
        dones: Episode termination flags [T], 1.0 if terminal
        gamma: Discount factor
        gae_lambda: GAE λ parameter for bias-variance tradeoff

    Returns:
        advantages: GAE advantage estimates [T]
        returns: Target returns for value network update [T]
                 (returns = advantages + values)

    Notes:
        - λ = 0: One-step TD (high bias, low variance)
        - λ = 1: Monte Carlo (low bias, high variance)
        - λ ∈ (0,1): Trade-off between bias and variance
    """
    T = len(rewards)

    # Validate inputs
    assert len(values) == T, f"values length {len(values)} != T {T}"
    assert len(next_values) == T, f"next_values length {len(next_values)} != T {T}"
    assert len(dones) == T, f"dones length {len(dones)} != T {T}"

    # Initialize advantages array
    advantages = np.zeros(T, dtype=np.float32)

    # GAE accumulator
    gae = 0.0

    # Compute GAE in reverse order (from last step to first)
    for t in reversed(range(T)):
        # Compute TD error: δ_t = r_t + γ V(s_{t+1}) (1 - done_t) - V(s_t)
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]

        # Accumulate GAE: A_t = δ_t + γλ (1 - done_t) A_{t+1}
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae

        advantages[t] = gae

    # Compute returns: R_t = A_t + V(s_t)
    # This is the target for value network training
    returns = advantages + values

    logger.debug(f"Computed GAE: mean_advantage={advantages.mean():.4f}, "
                f"std_advantage={advantages.std():.4f}, mean_return={returns.mean():.4f}")

    return advantages, returns


def compute_n_step_returns(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    n: int,
    gamma: float = 0.99
) -> np.ndarray:
    """
    Compute n-step returns (alternative to GAE).

    n-step Return Formula:
        R_t^{(n)} = Σ_{k=0}^{n-1} γ^k r_{t+k} + γ^n V(s_{t+n}) (1 - done_{t+n})

    Args:
        rewards: Reward sequence [T]
        values: V(s_t) estimates [T]
        dones: Episode termination flags [T]
        n: Number of steps for n-step returns
        gamma: Discount factor

    Returns:
        returns: n-step return estimates [T]
    """
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float32)

    for t in range(T):
        # Compute n-step return
        n_step_return = 0.0
        discount = 1.0

        for k in range(min(n, T - t)):
            n_step_return += discount * rewards[t + k]
            discount *= gamma

            # Stop if episode ends
            if dones[t + k]:
                break
        else:
            # Add bootstrap value if didn't hit episode end
            if t + n < T:
                n_step_return += discount * values[t + n] * (1 - dones[t + n])

        returns[t] = n_step_return

    return returns


def normalize_advantages(advantages: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize advantages to have zero mean and unit variance.

    This can improve training stability.

    Args:
        advantages: Advantage estimates [T]
        eps: Small constant for numerical stability

    Returns:
        normalized_advantages: Normalized advantages [T]
    """
    mean = advantages.mean()
    std = advantages.std()

    if std < eps:
        logger.warning(f"Advantage std ({std}) is very small, skipping normalization")
        return advantages

    normalized = (advantages - mean) / (std + eps)

    return normalized
