"""
Metrics Computation Modules

This package provides three levels of metrics computation:
- episode_level: L1 (Episode-Level) metrics extraction
- task_family_level: L2 (Task-Family-Level) aggregation
- evolve_level: L3 (Evolve-Level) evolution metrics
"""

from .episode_level import extract_episode_metrics, calculate_auc_loss
from .task_family_level import (
    compute_task_family_metrics,
    compute_ci_95,
    compute_les_proxy,
    compute_safety_score,
)
from .evolve_level import (
    compute_evolve_metrics,
    compute_adversarial_regret,
    compute_lower_bound,
    compute_provable_scenarios,
)

__all__ = [
    # Episode Level
    'extract_episode_metrics',
    'calculate_auc_loss',
    # Task Family Level
    'compute_task_family_metrics',
    'compute_ci_95',
    'compute_les_proxy',
    'compute_safety_score',
    # Evolve Level
    'compute_evolve_metrics',
    'compute_adversarial_regret',
    'compute_lower_bound',
    'compute_provable_scenarios',
]
