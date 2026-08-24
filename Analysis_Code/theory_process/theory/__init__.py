# -*- coding: utf-8 -*-
"""
Bellman/LLM Critic Theoretical Framework

This module implements the theoretical framework for offline batched RL analysis:
    Bellman/LLM Critic → Function Families → Graceful Degradation
    → Rademacher Complexity → Provable Lower Bounds

Components:
- function_families: Reward evaluation function families
- bellman_residual: Bellman residual and advantage function families
- rademacher_complexity: Rademacher complexity computation
- graceful_degradation: Distribution shift and uncertainty analysis
- provable_bounds: Provable lower bound computation
- theory_visualization: IEEE-style visualizations

Date: 2025-12-29
"""

from .function_families import (
    RewardFunctionFamily,
    InputTuple,
    RewardFamilyCapacity,
)
from .bellman_residual import (
    BellmanResidualFamily,
    Trajectory,
    Transition,
)
from .rademacher_complexity import (
    RademacherComplexityAnalyzer,
    ComplexityProfile,
)
from .graceful_degradation import (
    GracefulDegradationAnalyzer,
    UncertaintySet,
)
from .provable_bounds import (
    ProvableBoundsComputer,
    ProvableBound,
)
from .theory_visualization import (
    TheoryVisualizer,
)

__all__ = [
    # Function Families
    'RewardFunctionFamily',
    'InputTuple',
    'RewardFamilyCapacity',
    # Bellman Residual
    'BellmanResidualFamily',
    'Trajectory',
    'Transition',
    # Rademacher Complexity
    'RademacherComplexityAnalyzer',
    'ComplexityProfile',
    # Graceful Degradation
    'GracefulDegradationAnalyzer',
    'UncertaintySet',
    # Provable Bounds
    'ProvableBoundsComputer',
    'ProvableBound',
    # Visualization
    'TheoryVisualizer',
]

__version__ = '0.1.0'
