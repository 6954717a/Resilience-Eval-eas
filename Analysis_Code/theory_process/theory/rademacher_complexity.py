# -*- coding: utf-8 -*-
"""
Rademacher Complexity Analysis

Implements various Rademacher complexity measures for offline RL:

Empirical Rademacher Complexity:
    R̂_n(F) = E_σ[sup_{f∈F} (1/n) Σ σ_i f(z_i)]

Sequential Rademacher Complexity (for non-i.i.d. RL data):
    R̂_n^seq(F) = E_σ[sup_{f∈F} (1/n) Σ σ_i f(z_i)]  (with dependent samples)

Offset Rademacher Complexity (for excess risk control):
    R̂_n^offset(F, f*) = E_σ[sup_{f∈F} (1/n) Σ σ_i (f(z_i) - f*(z_i))]

Local Rademacher Complexity (for fast rates):
    R̂_n^local(F, r) = R̂_n(F ∩ B(f*, r))

Date: 2025-12-29
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from pathlib import Path


@dataclass
class ComplexityProfile:
    """
    Complete complexity profile for a function family.

    Contains empirical, sequential, offset, and local complexity measures.
    """
    # Empirical Rademacher complexity
    empirical_rc: float = 0.0
    empirical_rc_std: float = 0.0

    # Sequential Rademacher complexity
    sequential_rc: float = 0.0
    sequential_rc_std: float = 0.0

    # Offset Rademacher complexity
    offset_rc: float = 0.0
    offset_rc_std: float = 0.0

    # Local Rademacher complexity at different radii
    local_rc: Dict[float, float] = field(default_factory=dict)

    # Sample size
    n_samples: int = 0
    n_trajectories: int = 0

    # Derived quantities
    effective_complexity: float = 0.0  # Weighted combination

    def to_dict(self) -> Dict[str, Any]:
        return {
            'empirical_rc': self.empirical_rc,
            'empirical_rc_std': self.empirical_rc_std,
            'sequential_rc': self.sequential_rc,
            'sequential_rc_std': self.sequential_rc_std,
            'offset_rc': self.offset_rc,
            'offset_rc_std': self.offset_rc_std,
            'local_rc': self.local_rc,
            'n_samples': self.n_samples,
            'n_trajectories': self.n_trajectories,
            'effective_complexity': self.effective_complexity,
        }


@dataclass
class RademacherComplexityAnalyzer:
    """
    Rademacher Complexity Analysis for Offline RL.

    Computes various complexity measures using Monte Carlo sampling
    with Rademacher random variables.

    Attributes:
        n_bootstrap: Number of bootstrap samples for MC estimation
        seed: Random seed for reproducibility
    """

    n_bootstrap: int = 1000
    seed: Optional[int] = 42

    def __post_init__(self):
        self.rng = np.random.RandomState(self.seed)

    def _sample_rademacher(self, n: int) -> np.ndarray:
        """
        Sample n Rademacher random variables.

        σ_i ∈ {-1, +1} with equal probability.
        """
        return 2 * self.rng.randint(0, 2, size=n) - 1

    def compute_empirical_rademacher(
        self,
        function_values: np.ndarray,
        normalize: bool = True
    ) -> Tuple[float, float]:
        """
        Compute empirical Rademacher complexity.

        R̂_n(F) = E_σ[sup_{f∈F} (1/n) Σ σ_i f(z_i)]

        For a finite sample, this is approximated via Monte Carlo:
            R̂_n ≈ (1/B) Σ_b max_f [(1/n) Σ σ_i^(b) f(z_i)]

        When we only have one function's values (the learned function),
        we estimate complexity using the correlation with Rademacher signs.

        Args:
            function_values: Array of function values f(z_i) for i=1,...,n
            normalize: Whether to normalize by n

        Returns:
            (mean_rc, std_rc) tuple
        """
        n = len(function_values)
        if n == 0:
            return 0.0, 0.0

        function_values = np.asarray(function_values)
        rc_samples = []

        for _ in range(self.n_bootstrap):
            sigma = self._sample_rademacher(n)

            # Compute correlation with Rademacher signs
            # This estimates supremum when we have one function
            rc_b = np.mean(sigma * function_values)
            rc_samples.append(abs(rc_b))  # Take absolute value

        rc_samples = np.array(rc_samples)
        mean_rc = float(np.mean(rc_samples))
        std_rc = float(np.std(rc_samples))

        return mean_rc, std_rc

    def compute_sequential_rademacher(
        self,
        trajectories: List[np.ndarray],
        aggregate_method: str = 'mean'
    ) -> Tuple[float, float]:
        """
        Compute sequential Rademacher complexity for non-i.i.d. RL data.

        For sequential/dependent data, we account for the trajectory structure:
        - Compute per-trajectory complexity
        - Aggregate across trajectories

        The sequential RC captures the complexity when samples are correlated
        within trajectories but approximately independent across trajectories.

        Args:
            trajectories: List of arrays, each containing function values for a trajectory
            aggregate_method: 'mean' or 'max' for aggregating across trajectories

        Returns:
            (mean_rc, std_rc) tuple
        """
        if not trajectories:
            return 0.0, 0.0

        # Compute RC for each trajectory
        trajectory_rcs = []
        total_samples = 0

        for traj_values in trajectories:
            if len(traj_values) == 0:
                continue

            traj_rc, _ = self.compute_empirical_rademacher(traj_values)
            trajectory_rcs.append(traj_rc)
            total_samples += len(traj_values)

        if not trajectory_rcs:
            return 0.0, 0.0

        trajectory_rcs = np.array(trajectory_rcs)

        # Aggregate
        if aggregate_method == 'mean':
            # Weighted mean by trajectory length
            mean_rc = float(np.mean(trajectory_rcs))
        else:  # max
            mean_rc = float(np.max(trajectory_rcs))

        std_rc = float(np.std(trajectory_rcs))

        # Scale by sqrt(n_trajectories) for i.i.d. trajectory assumption
        n_traj = len(trajectories)
        scaled_rc = mean_rc / np.sqrt(n_traj)

        return scaled_rc, std_rc

    def compute_offset_rademacher(
        self,
        function_values: np.ndarray,
        reference_values: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute offset Rademacher complexity.

        R̂_n^offset(F, f*) = E_σ[sup_{f∈F} (1/n) Σ σ_i (f(z_i) - f*(z_i))]

        The offset complexity measures complexity relative to a reference
        function f*, which provides tighter excess risk bounds.

        Args:
            function_values: Array of function values f(z_i)
            reference_values: Array of reference function values f*(z_i)

        Returns:
            (mean_rc, std_rc) tuple
        """
        if len(function_values) == 0 or len(reference_values) == 0:
            return 0.0, 0.0

        # Compute offset (f - f*)
        offset = np.asarray(function_values) - np.asarray(reference_values)

        # Compute Rademacher complexity of offset
        return self.compute_empirical_rademacher(offset)

    def compute_local_rademacher(
        self,
        function_values: np.ndarray,
        radii: List[float] = None
    ) -> Dict[float, Tuple[float, float]]:
        """
        Compute local Rademacher complexity at different radii.

        R̂_n^local(F, r) = R̂_n(F ∩ B(f*, r))

        Local complexity captures the complexity in a neighborhood
        of the optimal function, enabling fast convergence rates.

        Args:
            function_values: Array of function values
            radii: List of radii to evaluate (default: [0.1, 0.2, 0.5, 1.0])

        Returns:
            Dictionary mapping radius -> (mean_rc, std_rc)
        """
        if radii is None:
            radii = [0.1, 0.2, 0.5, 1.0]

        if len(function_values) == 0:
            return {r: (0.0, 0.0) for r in radii}

        function_values = np.asarray(function_values)

        # Use mean as the "center"
        center = np.mean(function_values)

        results = {}
        for r in radii:
            # Filter values within radius of center
            mask = np.abs(function_values - center) <= r
            local_values = function_values[mask]

            if len(local_values) > 0:
                rc, std = self.compute_empirical_rademacher(local_values)
            else:
                rc, std = 0.0, 0.0

            results[r] = (rc, std)

        return results

    def compute_complexity_vs_sample_size(
        self,
        function_values: np.ndarray,
        sample_sizes: List[int] = None
    ) -> Dict[int, Tuple[float, float]]:
        """
        Compute how complexity varies with sample size.

        This helps understand the convergence rate of generalization bounds.
        Theoretically, RC should decrease as O(1/√n).

        Args:
            function_values: Full array of function values
            sample_sizes: List of sample sizes to evaluate

        Returns:
            Dictionary mapping sample_size -> (mean_rc, std_rc)
        """
        n = len(function_values)
        if n == 0:
            return {}

        if sample_sizes is None:
            # Default: evaluate at 10%, 25%, 50%, 75%, 100%
            sample_sizes = [max(10, n // 10), n // 4, n // 2, 3 * n // 4, n]
            sample_sizes = [s for s in sample_sizes if s <= n and s > 0]

        results = {}
        for size in sample_sizes:
            if size > n:
                continue

            # Sample without replacement
            indices = self.rng.choice(n, size=size, replace=False)
            subset = function_values[indices]

            rc, std = self.compute_empirical_rademacher(subset)
            results[size] = (rc, std)

        return results

    def analyze_all(
        self,
        function_values: np.ndarray,
        trajectories: Optional[List[np.ndarray]] = None,
        reference_values: Optional[np.ndarray] = None
    ) -> ComplexityProfile:
        """
        Perform complete complexity analysis.

        Args:
            function_values: Array of function values (all samples)
            trajectories: Optional list of per-trajectory values
            reference_values: Optional reference function values for offset RC

        Returns:
            ComplexityProfile with all complexity measures
        """
        n = len(function_values)

        # Empirical RC
        emp_rc, emp_std = self.compute_empirical_rademacher(function_values)

        # Sequential RC
        if trajectories is not None:
            seq_rc, seq_std = self.compute_sequential_rademacher(trajectories)
            n_traj = len(trajectories)
        else:
            seq_rc, seq_std = emp_rc, emp_std
            n_traj = 1

        # Offset RC
        if reference_values is not None:
            off_rc, off_std = self.compute_offset_rademacher(
                function_values, reference_values
            )
        else:
            # Use mean as reference (centered function)
            ref = np.full_like(function_values, np.mean(function_values))
            off_rc, off_std = self.compute_offset_rademacher(function_values, ref)

        # Local RC
        local_results = self.compute_local_rademacher(function_values)
        local_rc = {r: v[0] for r, v in local_results.items()}

        # Effective complexity: weighted average
        # Weight: empirical (0.5), sequential (0.3), offset (0.2)
        effective = 0.5 * emp_rc + 0.3 * seq_rc + 0.2 * off_rc

        return ComplexityProfile(
            empirical_rc=emp_rc,
            empirical_rc_std=emp_std,
            sequential_rc=seq_rc,
            sequential_rc_std=seq_std,
            offset_rc=off_rc,
            offset_rc_std=off_std,
            local_rc=local_rc,
            n_samples=n,
            n_trajectories=n_traj,
            effective_complexity=effective,
        )

    def compute_generalization_bound(
        self,
        complexity: ComplexityProfile,
        delta: float = 0.05,
        range_bound: float = 1.0
    ) -> Dict[str, float]:
        """
        Compute generalization bound based on Rademacher complexity.

        With probability at least 1-δ:
            L(f) ≤ L̂(f) + 2·RC + √(log(1/δ)/(2n))

        Args:
            complexity: ComplexityProfile from analysis
            delta: Confidence parameter (default: 0.05 for 95% confidence)
            range_bound: Bound on function range

        Returns:
            Dictionary with generalization bound components
        """
        n = complexity.n_samples
        if n == 0:
            return {'bound': np.inf, 'rc_term': 0.0, 'concentration_term': 0.0}

        # Rademacher term: 2 * RC
        rc_term = 2 * complexity.effective_complexity

        # Concentration term: √(log(1/δ)/(2n))
        concentration_term = range_bound * np.sqrt(np.log(1 / delta) / (2 * n))

        # Total bound
        total_bound = rc_term + concentration_term

        return {
            'bound': float(total_bound),
            'rc_term': float(rc_term),
            'concentration_term': float(concentration_term),
            'delta': delta,
            'n': n,
        }

    def compare_iid_vs_sequential(
        self,
        function_values: np.ndarray,
        trajectories: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Compare i.i.d. and sequential Rademacher complexity.

        This comparison helps understand the impact of temporal dependencies
        in RL trajectories on generalization.

        Args:
            function_values: All function values (treated as i.i.d.)
            trajectories: Same values organized by trajectory

        Returns:
            Comparison statistics
        """
        # i.i.d. RC (ignoring trajectory structure)
        iid_rc, iid_std = self.compute_empirical_rademacher(function_values)

        # Sequential RC (respecting trajectory structure)
        seq_rc, seq_std = self.compute_sequential_rademacher(trajectories)

        # Dependency penalty: how much larger is sequential RC?
        if iid_rc > 0:
            dependency_ratio = seq_rc / iid_rc
        else:
            dependency_ratio = 1.0

        return {
            'iid_rc': iid_rc,
            'iid_std': iid_std,
            'sequential_rc': seq_rc,
            'sequential_std': seq_std,
            'dependency_ratio': dependency_ratio,
            'dependency_penalty': max(0, seq_rc - iid_rc),
            'n_samples': len(function_values),
            'n_trajectories': len(trajectories),
        }


def analyze_bellman_residual_complexity(
    step_advantages: List[List[float]],
    n_bootstrap: int = 1000,
    seed: int = 42
) -> ComplexityProfile:
    """
    Convenience function to analyze Bellman residual complexity from ADCA data.

    Args:
        step_advantages: List of advantage arrays per trajectory
        n_bootstrap: Number of bootstrap samples
        seed: Random seed

    Returns:
        ComplexityProfile
    """
    analyzer = RademacherComplexityAnalyzer(n_bootstrap=n_bootstrap, seed=seed)

    # Flatten for empirical RC
    all_advantages = []
    trajectories = []

    for traj_advs in step_advantages:
        if traj_advs:
            adv_array = np.array(traj_advs)
            all_advantages.extend(traj_advs)
            trajectories.append(adv_array)

    if not all_advantages:
        return ComplexityProfile()

    all_advantages = np.array(all_advantages)

    return analyzer.analyze_all(
        function_values=all_advantages,
        trajectories=trajectories,
        reference_values=None,
    )
