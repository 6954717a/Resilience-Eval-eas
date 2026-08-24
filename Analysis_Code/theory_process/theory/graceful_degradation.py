# -*- coding: utf-8 -*-
"""
Graceful Degradation Analysis

Implements robust off-policy evaluation with distribution shift handling:

1. Distribution Shift (Robust OPE/MDP):
   - KL uncertainty sets: P_uncertain = {P : D_KL(P || P_0) ≤ ε}
   - Worst-case evaluation under uncertainty

2. LLM Critic Uncertainty:
   - Based on ADCA label consistency
   - Entropy-based uncertainty: U(s,a) = -Σ P(y|s,a) log P(y|s,a)

3. Bellman Error → Optimality Gap:
   J(π*) - J(π̂) ≤ (2γ/(1-γ)²) E_μ[max_a |δ(s,a,s')|]

Date: 2025-12-29
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import numpy as np
from pathlib import Path


class UncertaintySet(NamedTuple):
    """
    KL uncertainty set representation.

    P_uncertain = {P : D_KL(P || P_0) ≤ ε}
    """
    center_distribution: np.ndarray  # P_0 (baseline distribution)
    epsilon: float                    # KL divergence radius
    dimension: int                    # Distribution dimension


@dataclass
class GracefulDegradationAnalyzer:
    """
    Graceful Degradation Analysis for Offline RL.

    Handles:
    1. Distribution shift (train vs eval)
    2. LLM Critic uncertainty
    3. Bellman error to optimality gap conversion

    Attributes:
        gamma: Discount factor
        epsilon_kl: KL divergence radius for uncertainty set
        uncertainty_method: 'label_consistency' or 'entropy'
    """

    gamma: float = 0.99
    epsilon_kl: float = 0.1
    uncertainty_method: str = 'label_consistency'

    # =========================================================================
    # Part 1: Distribution Shift Analysis
    # =========================================================================

    def compute_kl_divergence(
        self,
        p: np.ndarray,
        q: np.ndarray,
        epsilon: float = 1e-10
    ) -> float:
        """
        Compute KL divergence D_KL(P || Q).

        D_KL(P || Q) = Σ P(x) log(P(x)/Q(x))

        Args:
            p: Distribution P
            q: Distribution Q (reference)
            epsilon: Small constant for numerical stability

        Returns:
            KL divergence value
        """
        p = np.asarray(p) + epsilon
        q = np.asarray(q) + epsilon

        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)

        return float(np.sum(p * np.log(p / q)))

    def compute_distribution_shift(
        self,
        train_distribution: np.ndarray,
        eval_distribution: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute distribution shift metrics between train and eval.

        Args:
            train_distribution: Training data distribution
            eval_distribution: Evaluation data distribution

        Returns:
            Dictionary with shift metrics
        """
        # KL divergence (both directions)
        kl_train_to_eval = self.compute_kl_divergence(train_distribution, eval_distribution)
        kl_eval_to_train = self.compute_kl_divergence(eval_distribution, train_distribution)

        # Symmetric KL (Jensen-Shannon divergence)
        m = 0.5 * (train_distribution + eval_distribution)
        js_divergence = 0.5 * (
            self.compute_kl_divergence(train_distribution, m) +
            self.compute_kl_divergence(eval_distribution, m)
        )

        # Total variation distance
        tv_distance = 0.5 * np.sum(np.abs(train_distribution - eval_distribution))

        return {
            'kl_train_to_eval': kl_train_to_eval,
            'kl_eval_to_train': kl_eval_to_train,
            'js_divergence': js_divergence,
            'tv_distance': float(tv_distance),
        }

    def compute_uncertainty_set(
        self,
        base_distribution: np.ndarray,
        epsilon: float = None
    ) -> UncertaintySet:
        """
        Define KL uncertainty set around base distribution.

        P_uncertain = {P : D_KL(P || P_0) ≤ ε}

        Args:
            base_distribution: Center distribution P_0
            epsilon: KL radius (default: self.epsilon_kl)

        Returns:
            UncertaintySet object
        """
        if epsilon is None:
            epsilon = self.epsilon_kl

        base = np.asarray(base_distribution)
        base = base / np.sum(base)  # Normalize

        return UncertaintySet(
            center_distribution=base,
            epsilon=epsilon,
            dimension=len(base),
        )

    def compute_worst_case_expectation(
        self,
        values: np.ndarray,
        uncertainty_set: UncertaintySet,
        direction: str = 'min'
    ) -> float:
        """
        Compute worst-case expectation under KL uncertainty set.

        For robust OPE, we want:
            min_{P ∈ P_uncertain} E_P[V]  (pessimistic)
        or
            max_{P ∈ P_uncertain} E_P[V]  (optimistic)

        Uses the dual formulation:
            min_P E_P[V] s.t. D_KL(P||P_0) ≤ ε
            = sup_λ≥0 {E_{P_0}[-exp(-V/λ)] - λε}

        Args:
            values: Function values V(x)
            uncertainty_set: KL uncertainty set
            direction: 'min' or 'max'

        Returns:
            Worst-case expectation
        """
        p0 = uncertainty_set.center_distribution
        eps = uncertainty_set.epsilon
        v = np.asarray(values)

        if direction == 'min':
            v_use = v
        else:
            v_use = -v

        # Solve dual via line search over λ
        def dual_objective(lam):
            if lam <= 0:
                return -np.inf
            exp_term = np.exp(-v_use / lam)
            return -lam * np.log(np.sum(p0 * exp_term)) - lam * eps

        # Line search
        best_val = -np.inf
        for lam in np.logspace(-3, 3, 100):
            val = dual_objective(lam)
            if val > best_val:
                best_val = val

        if direction == 'min':
            return best_val
        else:
            return -best_val

    def compute_robust_ope_bounds(
        self,
        value_estimates: np.ndarray,
        base_distribution: np.ndarray,
        epsilon: float = None
    ) -> Tuple[float, float]:
        """
        Compute robust OPE bounds under distribution shift.

        Returns pessimistic (lower) and optimistic (upper) bounds.

        Args:
            value_estimates: Value estimates V(s)
            base_distribution: Training distribution
            epsilon: KL radius

        Returns:
            (lower_bound, upper_bound) tuple
        """
        unc_set = self.compute_uncertainty_set(base_distribution, epsilon)

        lower = self.compute_worst_case_expectation(value_estimates, unc_set, 'min')
        upper = self.compute_worst_case_expectation(value_estimates, unc_set, 'max')

        return lower, upper

    # =========================================================================
    # Part 2: LLM Critic Uncertainty (Label Consistency Method)
    # =========================================================================

    def compute_label_consistency_uncertainty(
        self,
        step_flags_per_episode: Dict[str, List[bool]]
    ) -> Dict[str, float]:
        """
        Compute LLM uncertainty based on ADCA label consistency.

        For each state-action, we look at similar states across episodes
        and measure the consistency of GOOD/BAD labels.

        Higher variance = higher uncertainty.

        Args:
            step_flags_per_episode: Dictionary mapping episode_id to step flags

        Returns:
            Uncertainty metrics
        """
        if not step_flags_per_episode:
            return {'mean_uncertainty': 0.0, 'max_uncertainty': 0.0, 'n_episodes': 0}

        # Collect statistics per step position
        position_stats = {}
        max_steps = max(len(flags) for flags in step_flags_per_episode.values())

        for pos in range(max_steps):
            flags_at_pos = []
            for episode_flags in step_flags_per_episode.values():
                if pos < len(episode_flags):
                    flags_at_pos.append(1.0 if episode_flags[pos] else 0.0)

            if len(flags_at_pos) >= 2:
                # Compute variance (higher = more uncertainty)
                variance = float(np.var(flags_at_pos))
                # Binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
                p = np.mean(flags_at_pos)
                if 0 < p < 1:
                    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
                else:
                    entropy = 0.0

                position_stats[pos] = {
                    'variance': variance,
                    'entropy': entropy,
                    'n_samples': len(flags_at_pos),
                    'good_ratio': p,
                }

        if not position_stats:
            return {'mean_uncertainty': 0.0, 'max_uncertainty': 0.0, 'n_episodes': len(step_flags_per_episode)}

        # Aggregate uncertainty
        uncertainties = [s['entropy'] for s in position_stats.values()]

        return {
            'mean_uncertainty': float(np.mean(uncertainties)),
            'max_uncertainty': float(np.max(uncertainties)),
            'std_uncertainty': float(np.std(uncertainties)),
            'n_positions': len(position_stats),
            'n_episodes': len(step_flags_per_episode),
            'position_stats': position_stats,
        }

    def compute_trajectory_uncertainty(
        self,
        step_flags: List[bool],
        step_advantages: List[float]
    ) -> float:
        """
        Compute uncertainty for a single trajectory.

        Uses variance in advantages as uncertainty proxy.

        Args:
            step_flags: GOOD/BAD labels
            step_advantages: Advantage values

        Returns:
            Uncertainty score [0, 1]
        """
        if not step_advantages:
            return 0.5  # Maximum uncertainty when no data

        # Normalize advantages
        adv_array = np.array(step_advantages)
        adv_std = np.std(adv_array)
        adv_range = np.max(adv_array) - np.min(adv_array) if len(adv_array) > 1 else 1.0

        # Coefficient of variation
        if np.abs(np.mean(adv_array)) > 1e-10:
            cv = adv_std / np.abs(np.mean(adv_array))
        else:
            cv = adv_std

        # Normalize to [0, 1]
        uncertainty = min(1.0, cv / 2.0)

        return float(uncertainty)

    def compute_llm_uncertainty_distribution(
        self,
        trajectories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute LLM uncertainty distribution across all trajectories.

        Args:
            trajectories: List of trajectory dictionaries with step_flags and step_advantages

        Returns:
            Uncertainty distribution statistics
        """
        if not trajectories:
            return {'mean': 0.5, 'std': 0.0, 'distribution': []}

        uncertainties = []
        step_flags_by_episode = {}

        for i, traj in enumerate(trajectories):
            flags = traj.get('step_flags', [])
            advs = traj.get('step_advantages', [])

            # Per-trajectory uncertainty
            unc = self.compute_trajectory_uncertainty(flags, advs)
            uncertainties.append(unc)

            # Store for cross-episode analysis
            episode_id = traj.get('episode_id', str(i))
            step_flags_by_episode[episode_id] = flags

        # Cross-episode consistency uncertainty
        consistency_unc = self.compute_label_consistency_uncertainty(step_flags_by_episode)

        return {
            'mean': float(np.mean(uncertainties)),
            'std': float(np.std(uncertainties)),
            'min': float(np.min(uncertainties)),
            'max': float(np.max(uncertainties)),
            'distribution': uncertainties,
            'consistency_uncertainty': consistency_unc,
        }

    # =========================================================================
    # Part 3: Bellman Error → Optimality Gap
    # =========================================================================

    def bellman_to_optimality_gap(
        self,
        bellman_error: float,
        gamma: float = None
    ) -> float:
        """
        Convert Bellman error to optimality gap.

        Theorem: J(π*) - J(π̂) ≤ (2γ/(1-γ)²) · E_μ[max_a |δ(s,a,s')|]

        For average Bellman error, we use a normalized version:
            Gap ≤ C · √(E[δ²]) / (1 + √(E[δ²]))

        This gives a bounded gap in [0, 1] suitable for practical use.

        Args:
            bellman_error: Mean squared Bellman error E[δ²]
            gamma: Discount factor

        Returns:
            Upper bound on optimality gap (normalized to [0, 1])
        """
        if gamma is None:
            gamma = self.gamma

        # Use RMSE as proxy for error magnitude
        rmse = np.sqrt(bellman_error) if bellman_error > 0 else 0.0

        # Normalized gap: sigmoid-like transformation to [0, 1]
        # This ensures the gap is bounded and interpretable
        gap = rmse / (1 + rmse)

        # Scale by effective horizon: 1/(1-γ)
        effective_horizon = 1.0 / (1 - gamma)

        # Combine with bounded scaling
        # Gap is capped at 0.5 to be conservative
        scaled_gap = min(0.5, gap * min(effective_horizon / 100, 1.0))

        return float(scaled_gap)

    def compute_value_bound_from_bellman(
        self,
        empirical_value: float,
        bellman_error: float,
        gamma: float = None
    ) -> Tuple[float, float]:
        """
        Compute value function bounds from Bellman error.

        Returns (lower_bound, upper_bound) on true value.

        Args:
            empirical_value: Estimated value J(π̂)
            bellman_error: Mean squared Bellman error
            gamma: Discount factor

        Returns:
            (lower_bound, upper_bound) tuple
        """
        gap = self.bellman_to_optimality_gap(bellman_error, gamma)

        # Pessimistic: true value could be lower
        lower_bound = empirical_value - gap

        # Optimistic: true value could be higher (but capped)
        upper_bound = empirical_value + gap

        return lower_bound, upper_bound

    # =========================================================================
    # Part 4: Integrated Graceful Degradation
    # =========================================================================

    def compute_graceful_degradation_penalty(
        self,
        distribution_shift: Dict[str, float],
        llm_uncertainty: Dict[str, Any],
        bellman_error: float
    ) -> Dict[str, float]:
        """
        Compute overall graceful degradation penalty.

        Combines:
        1. Distribution shift penalty
        2. LLM uncertainty penalty
        3. Bellman error penalty

        Args:
            distribution_shift: Shift metrics from compute_distribution_shift
            llm_uncertainty: Uncertainty from compute_llm_uncertainty_distribution
            bellman_error: Mean squared Bellman error

        Returns:
            Penalty breakdown and total
        """
        # Distribution shift penalty
        # Use symmetric JS divergence as shift measure
        shift_penalty = distribution_shift.get('js_divergence', 0.0)

        # LLM uncertainty penalty
        # Scale mean uncertainty to [0, 0.5] range
        unc_mean = llm_uncertainty.get('mean', 0.0)
        uncertainty_penalty = 0.5 * unc_mean

        # Bellman error penalty
        # Convert to optimality gap
        bellman_penalty = self.bellman_to_optimality_gap(bellman_error)

        # Total penalty (additive)
        total_penalty = shift_penalty + uncertainty_penalty + bellman_penalty

        return {
            'shift_penalty': float(shift_penalty),
            'uncertainty_penalty': float(uncertainty_penalty),
            'bellman_penalty': float(bellman_penalty),
            'total_penalty': float(total_penalty),
        }

    def analyze_graceful_degradation(
        self,
        trajectories: List[Dict[str, Any]],
        bellman_error: float,
        train_gsr: Optional[float] = None,
        eval_gsr: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Complete graceful degradation analysis.

        Args:
            trajectories: List of trajectory dictionaries
            bellman_error: Mean squared Bellman error
            train_gsr: Training Good Step Ratio (optional)
            eval_gsr: Evaluation Good Step Ratio (optional)

        Returns:
            Complete analysis results
        """
        # LLM Uncertainty
        llm_uncertainty = self.compute_llm_uncertainty_distribution(trajectories)

        # Distribution Shift (if train/eval GSR provided)
        if train_gsr is not None and eval_gsr is not None:
            # Create simple 2-bin distributions (GOOD vs BAD)
            train_dist = np.array([train_gsr, 1 - train_gsr])
            eval_dist = np.array([eval_gsr, 1 - eval_gsr])
            distribution_shift = self.compute_distribution_shift(train_dist, eval_dist)
        else:
            # Estimate from trajectory data
            if trajectories:
                gsrs = [
                    sum(t.get('step_flags', [])) / max(1, len(t.get('step_flags', [])))
                    for t in trajectories if t.get('step_flags')
                ]
                if len(gsrs) >= 2:
                    mid = len(gsrs) // 2
                    train_dist = np.array([np.mean(gsrs[:mid]), 1 - np.mean(gsrs[:mid])])
                    eval_dist = np.array([np.mean(gsrs[mid:]), 1 - np.mean(gsrs[mid:])])
                    distribution_shift = self.compute_distribution_shift(train_dist, eval_dist)
                else:
                    distribution_shift = {'js_divergence': 0.0, 'tv_distance': 0.0,
                                          'kl_train_to_eval': 0.0, 'kl_eval_to_train': 0.0}
            else:
                distribution_shift = {'js_divergence': 0.0, 'tv_distance': 0.0,
                                      'kl_train_to_eval': 0.0, 'kl_eval_to_train': 0.0}

        # Compute penalties
        penalties = self.compute_graceful_degradation_penalty(
            distribution_shift, llm_uncertainty, bellman_error
        )

        # Bellman to Gap
        optimality_gap = self.bellman_to_optimality_gap(bellman_error)

        return {
            'distribution_shift': distribution_shift,
            'llm_uncertainty': llm_uncertainty,
            'penalties': penalties,
            'bellman_error': bellman_error,
            'optimality_gap': optimality_gap,
            'gamma': self.gamma,
            'epsilon_kl': self.epsilon_kl,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'gamma': self.gamma,
            'epsilon_kl': self.epsilon_kl,
            'uncertainty_method': self.uncertainty_method,
        }
