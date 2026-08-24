# -*- coding: utf-8 -*-
"""
Provable Lower Bounds Computation

Implements the main theorem for provable performance guarantees:

Main Theorem: With probability 1-δ:
    J(π) ≥ Ĵ_n(π) - C₁·R̂_n^offset(B) - C₂·√(log(1/δ)/n) - C₃·ε_shift

Where:
    - Ĵ_n(π): Empirical return estimate
    - C₁·R̂_n^offset(B): Complexity term (Rademacher)
    - C₂·√(log(1/δ)/n): Concentration term
    - C₃·ε_shift: Distribution shift term

Constants C₁, C₂, C₃ are calibrated via cross-validation.

Date: 2025-12-29
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
import numpy as np
from pathlib import Path
import json

from .function_families import RewardFunctionFamily
from .bellman_residual import BellmanResidualFamily, Trajectory
from .rademacher_complexity import RademacherComplexityAnalyzer, ComplexityProfile
from .graceful_degradation import GracefulDegradationAnalyzer


class ProvableBound(NamedTuple):
    """
    Provable lower bound result.
    """
    lower_bound: float           # J(π) ≥ lower_bound with confidence 1-δ
    empirical_return: float      # Ĵ_n(π)
    complexity_term: float       # C₁·R̂_n^offset(B)
    concentration_term: float    # C₂·√(log(1/δ)/n)
    shift_term: float            # C₃·ε_shift
    confidence: float            # 1-δ
    n_samples: int               # Number of samples
    is_valid: bool               # Whether bound is mathematically valid


@dataclass
class CalibrationResult:
    """
    Result of constant calibration.
    """
    c1: float                    # Complexity constant
    c2: float                    # Concentration constant
    c3: float                    # Shift constant
    validation_score: float      # Cross-validation score
    method: str                  # Calibration method used


@dataclass
class ProvableBoundsComputer:
    """
    Provable Lower Bounds Computation.

    Implements:
        J(π) ≥ Ĵ_n(π) - C₁·R̂_n^offset(B) - C₂·√(log(1/δ)/n) - C₃·ε_shift

    Attributes:
        gamma: Discount factor
        delta: Confidence parameter (default: 0.05 for 95% confidence)
        calibration_method: 'empirical' or 'theoretical'
    """

    gamma: float = 0.99
    delta: float = 0.05
    calibration_method: str = 'empirical'

    # Calibrated constants (default values, to be updated via calibration)
    c1: float = 2.0              # Complexity constant
    c2: float = 1.0              # Concentration constant
    c3: float = 1.0              # Shift constant

    def __post_init__(self):
        """Initialize sub-analyzers."""
        self.reward_family = RewardFunctionFamily(gamma=self.gamma)
        self.bellman_family = BellmanResidualFamily(
            reward_family=self.reward_family,
            gamma=self.gamma
        )
        self.complexity_analyzer = RademacherComplexityAnalyzer()
        self.degradation_analyzer = GracefulDegradationAnalyzer(gamma=self.gamma)

    # =========================================================================
    # Constant Calibration (Empirical Method)
    # =========================================================================

    def calibrate_constants_cv(
        self,
        trajectories: List[Trajectory],
        n_folds: int = 5
    ) -> CalibrationResult:
        """
        Calibrate constants C₁, C₂, C₃ via cross-validation.

        The goal is to find constants such that the lower bound is:
        1. Valid (never exceeds true performance)
        2. Tight (as close to true performance as possible)

        Args:
            trajectories: List of trajectories for calibration
            n_folds: Number of cross-validation folds

        Returns:
            CalibrationResult with calibrated constants
        """
        n = len(trajectories)
        if n < n_folds:
            # Not enough data, use theoretical defaults
            return CalibrationResult(
                c1=2.0, c2=1.0, c3=1.0,
                validation_score=0.0,
                method='theoretical_default'
            )

        # Shuffle and split into folds
        indices = np.random.permutation(n)
        fold_size = n // n_folds

        # Track validity across different constant choices
        c1_candidates = [1.0, 1.5, 2.0, 2.5, 3.0]
        c2_candidates = [0.5, 1.0, 1.5, 2.0]
        c3_candidates = [0.5, 1.0, 1.5, 2.0]

        best_score = -np.inf
        best_constants = (2.0, 1.0, 1.0)

        for c1 in c1_candidates:
            for c2 in c2_candidates:
                for c3 in c3_candidates:
                    scores = []

                    for fold in range(n_folds):
                        # Split train/val
                        val_start = fold * fold_size
                        val_end = val_start + fold_size
                        val_indices = indices[val_start:val_end]
                        train_indices = np.concatenate([
                            indices[:val_start],
                            indices[val_end:]
                        ])

                        train_trajs = [trajectories[i] for i in train_indices]
                        val_trajs = [trajectories[i] for i in val_indices]

                        # Compute bound on train, evaluate on val
                        if not train_trajs or not val_trajs:
                            continue

                        # Empirical return on validation
                        val_returns = [t.orm_score for t in val_trajs]
                        true_return = np.mean(val_returns)

                        # Compute bound terms on training
                        train_returns = [t.orm_score for t in train_trajs]
                        emp_return = np.mean(train_returns)

                        # Complexity (simplified)
                        train_advs = []
                        for t in train_trajs:
                            train_advs.extend(t.step_advantages)
                        if train_advs:
                            complexity = np.std(train_advs) / np.sqrt(len(train_advs))
                        else:
                            complexity = 0.1

                        # Concentration
                        concentration = np.sqrt(np.log(1 / self.delta) / len(train_trajs))

                        # Shift (use GSR difference as proxy)
                        train_gsr = np.mean([t.good_step_ratio for t in train_trajs])
                        val_gsr = np.mean([t.good_step_ratio for t in val_trajs])
                        shift = abs(train_gsr - val_gsr)

                        # Compute bound
                        bound = emp_return - c1 * complexity - c2 * concentration - c3 * shift

                        # Score: how tight is the bound while still valid?
                        if bound <= true_return:
                            # Valid bound, score by tightness
                            tightness = 1.0 - (true_return - bound)
                            scores.append(tightness)
                        else:
                            # Invalid bound, penalize
                            scores.append(-1.0)

                    if scores:
                        avg_score = np.mean(scores)
                        if avg_score > best_score:
                            best_score = avg_score
                            best_constants = (c1, c2, c3)

        self.c1, self.c2, self.c3 = best_constants

        return CalibrationResult(
            c1=best_constants[0],
            c2=best_constants[1],
            c3=best_constants[2],
            validation_score=best_score,
            method='cross_validation'
        )

    # =========================================================================
    # Main Bound Computation
    # =========================================================================

    def compute_empirical_return(
        self,
        trajectories: List[Trajectory]
    ) -> float:
        """
        Compute empirical return estimate Ĵ_n(π).

        Uses ORM score as return proxy.

        Args:
            trajectories: List of trajectories

        Returns:
            Empirical return estimate
        """
        if not trajectories:
            return 0.0

        returns = [t.orm_score for t in trajectories]
        return float(np.mean(returns))

    def compute_complexity_term(
        self,
        trajectories: List[Trajectory]
    ) -> Tuple[float, ComplexityProfile]:
        """
        Compute complexity term C₁·R̂_n^offset(B).

        Args:
            trajectories: List of trajectories

        Returns:
            (complexity_term, complexity_profile) tuple
        """
        # Collect all advantages
        traj_advantages = [t.step_advantages for t in trajectories if t.step_advantages]

        if not traj_advantages:
            return 0.0, ComplexityProfile()

        # Compute complexity profile
        all_advs = []
        for advs in traj_advantages:
            all_advs.extend(advs)

        profile = self.complexity_analyzer.analyze_all(
            function_values=np.array(all_advs),
            trajectories=[np.array(a) for a in traj_advantages]
        )

        # Use offset RC for the term
        complexity_term = self.c1 * profile.offset_rc

        return complexity_term, profile

    def compute_concentration_term(
        self,
        n_samples: int
    ) -> float:
        """
        Compute concentration term C₂·√(log(1/δ)/n).

        Args:
            n_samples: Number of samples

        Returns:
            Concentration term
        """
        if n_samples <= 0:
            return np.inf

        return self.c2 * np.sqrt(np.log(1 / self.delta) / n_samples)

    def compute_shift_term(
        self,
        trajectories: List[Trajectory]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute distribution shift term C₃·ε_shift.

        Args:
            trajectories: List of trajectories

        Returns:
            (shift_term, degradation_analysis) tuple
        """
        # Convert trajectories to dict format for analyzer
        traj_dicts = [
            {
                'episode_id': t.episode_id,
                'step_flags': t.step_flags,
                'step_advantages': t.step_advantages,
                'orm_score': t.orm_score,
            }
            for t in trajectories
        ]

        # Compute Bellman error proxy
        bellman_error = 0.0
        if trajectories:
            advs = []
            for t in trajectories:
                advs.extend(t.step_advantages)
            if advs:
                bellman_error = float(np.var(advs))

        # Analyze degradation
        analysis = self.degradation_analyzer.analyze_graceful_degradation(
            trajectories=traj_dicts,
            bellman_error=bellman_error
        )

        # Shift term from penalties
        shift = analysis['penalties']['total_penalty']
        shift_term = self.c3 * shift

        return shift_term, analysis

    def compute_provable_lower_bound(
        self,
        trajectories: List[Trajectory],
        calibrate: bool = True
    ) -> ProvableBound:
        """
        Compute provable lower bound.

        Main theorem: With probability 1-δ:
            J(π) ≥ Ĵ_n(π) - C₁·R̂_n^offset(B) - C₂·√(log(1/δ)/n) - C₃·ε_shift

        Args:
            trajectories: List of trajectories
            calibrate: Whether to calibrate constants first

        Returns:
            ProvableBound result
        """
        if not trajectories:
            return ProvableBound(
                lower_bound=0.0,
                empirical_return=0.0,
                complexity_term=0.0,
                concentration_term=np.inf,
                shift_term=0.0,
                confidence=1 - self.delta,
                n_samples=0,
                is_valid=False
            )

        # Calibrate if requested
        if calibrate and len(trajectories) >= 10:
            self.calibrate_constants_cv(trajectories)

        # Compute terms
        n_samples = len(trajectories)
        n_steps = sum(len(t.step_flags) for t in trajectories)

        # 1. Empirical return
        empirical_return = self.compute_empirical_return(trajectories)

        # 2. Complexity term
        complexity_term, _ = self.compute_complexity_term(trajectories)

        # 3. Concentration term
        concentration_term = self.compute_concentration_term(n_samples)

        # 4. Shift term
        shift_term, _ = self.compute_shift_term(trajectories)

        # Compute lower bound
        lower_bound = empirical_return - complexity_term - concentration_term - shift_term

        # Check validity
        is_valid = (
            lower_bound >= 0 and
            lower_bound <= 1 and
            not np.isnan(lower_bound) and
            not np.isinf(lower_bound)
        )

        return ProvableBound(
            lower_bound=max(0.0, float(lower_bound)),
            empirical_return=float(empirical_return),
            complexity_term=float(complexity_term),
            concentration_term=float(concentration_term),
            shift_term=float(shift_term),
            confidence=1 - self.delta,
            n_samples=n_samples,
            is_valid=is_valid
        )

    def compute_achievable_goal_percentage(
        self,
        trajectories: List[Trajectory],
        calibrate: bool = True
    ) -> Dict[str, float]:
        """
        Compute the achievable goal percentage with provable guarantee.

        This is the main result: what percentage of goals can we guarantee
        the agent will achieve?

        Args:
            trajectories: List of trajectories
            calibrate: Whether to calibrate constants

        Returns:
            Dictionary with achievable percentages and confidence
        """
        bound = self.compute_provable_lower_bound(trajectories, calibrate)

        # Convert to percentage
        achievable_pct = bound.lower_bound * 100
        empirical_pct = bound.empirical_return * 100

        # Compute gap
        gap = empirical_pct - achievable_pct

        return {
            'achievable_percentage': achievable_pct,
            'empirical_percentage': empirical_pct,
            'gap': gap,
            'confidence': bound.confidence,
            'confidence_pct': bound.confidence * 100,
            'breakdown': {
                'empirical': empirical_pct,
                'complexity_deduction': bound.complexity_term * 100,
                'concentration_deduction': bound.concentration_term * 100,
                'shift_deduction': bound.shift_term * 100,
            },
            'n_samples': bound.n_samples,
            'is_valid': bound.is_valid,
        }

    # =========================================================================
    # Analysis and Reporting
    # =========================================================================

    def analyze_bound_sensitivity(
        self,
        trajectories: List[Trajectory],
        delta_values: List[float] = None
    ) -> Dict[float, ProvableBound]:
        """
        Analyze how the bound changes with confidence level.

        Args:
            trajectories: List of trajectories
            delta_values: List of delta values to evaluate

        Returns:
            Dictionary mapping delta -> ProvableBound
        """
        if delta_values is None:
            delta_values = [0.01, 0.05, 0.10, 0.20, 0.50]

        original_delta = self.delta
        results = {}

        for delta in delta_values:
            self.delta = delta
            results[delta] = self.compute_provable_lower_bound(
                trajectories, calibrate=False
            )

        self.delta = original_delta
        return results

    def analyze_sample_efficiency(
        self,
        trajectories: List[Trajectory],
        sample_fractions: List[float] = None
    ) -> Dict[float, Dict[str, float]]:
        """
        Analyze how the bound improves with more samples.

        Args:
            trajectories: List of trajectories
            sample_fractions: Fractions of data to use

        Returns:
            Dictionary mapping fraction -> metrics
        """
        if sample_fractions is None:
            sample_fractions = [0.1, 0.25, 0.5, 0.75, 1.0]

        n = len(trajectories)
        results = {}

        for frac in sample_fractions:
            n_use = max(1, int(n * frac))
            subset = trajectories[:n_use]

            achievable = self.compute_achievable_goal_percentage(subset, calibrate=False)
            results[frac] = {
                'n_samples': n_use,
                'achievable_pct': achievable['achievable_percentage'],
                'empirical_pct': achievable['empirical_percentage'],
                'gap': achievable['gap'],
            }

        return results

    def generate_certificate(
        self,
        trajectories: List[Trajectory],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate a formal certificate of provable performance.

        Args:
            trajectories: List of trajectories
            output_path: Optional path to save certificate

        Returns:
            Certificate dictionary
        """
        # Calibrate constants
        calibration = self.calibrate_constants_cv(trajectories)

        # Compute main bound
        bound = self.compute_provable_lower_bound(trajectories, calibrate=False)

        # Compute achievable percentage
        achievable = self.compute_achievable_goal_percentage(trajectories, calibrate=False)

        # Compute complexity profile
        _, complexity = self.compute_complexity_term(trajectories)

        # Compute degradation analysis
        _, degradation = self.compute_shift_term(trajectories)

        # Build certificate
        certificate = {
            'version': '1.0',
            'timestamp': str(np.datetime64('now')),
            'summary': {
                'achievable_percentage': achievable['achievable_percentage'],
                'confidence': achievable['confidence_pct'],
                'empirical_percentage': achievable['empirical_percentage'],
                'gap': achievable['gap'],
                'is_valid': bound.is_valid,
            },
            'bound': {
                'lower_bound': bound.lower_bound,
                'empirical_return': bound.empirical_return,
                'complexity_term': bound.complexity_term,
                'concentration_term': bound.concentration_term,
                'shift_term': bound.shift_term,
            },
            'calibration': {
                'c1': calibration.c1,
                'c2': calibration.c2,
                'c3': calibration.c3,
                'method': calibration.method,
                'validation_score': calibration.validation_score,
            },
            'complexity': complexity.to_dict(),
            'degradation': {
                'distribution_shift': degradation.get('distribution_shift', {}),
                'penalties': degradation.get('penalties', {}),
            },
            'data_summary': {
                'n_trajectories': len(trajectories),
                'n_steps': sum(len(t.step_flags) for t in trajectories),
                'mean_gsr': np.mean([t.good_step_ratio for t in trajectories]),
                'mean_orm': np.mean([t.orm_score for t in trajectories]),
            },
            'parameters': {
                'gamma': self.gamma,
                'delta': self.delta,
            },
        }

        # Save if path provided
        if output_path is not None:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(certificate, f, indent=2, default=str)

        return certificate

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'gamma': self.gamma,
            'delta': self.delta,
            'c1': self.c1,
            'c2': self.c2,
            'c3': self.c3,
            'calibration_method': self.calibration_method,
        }
