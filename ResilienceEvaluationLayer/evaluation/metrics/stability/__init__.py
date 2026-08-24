"""
Stability Metrics Module

This module separates single-rollout critic evidence from formal neighborhood
stability under controlled perturbations.

Core metric: beta is the supremum sensitivity of the stage-normalized
trajectory loss. Value/TD/GAE and action differences remain diagnostics.
"""

from .stability_metrics import StabilityMetrics
from .stability_collector import (
    BetaAggregate,
    BetaFamilyAggregate,
    StabilityCollector,
)
from .stability_logging import log_stability_metrics
from .trajectory_loss import compute_stage_trajectory_loss

__all__ = [
    "StabilityMetrics",
    "StabilityCollector",
    "BetaAggregate",
    "BetaFamilyAggregate",
    "log_stability_metrics",
    "compute_stage_trajectory_loss",
]
