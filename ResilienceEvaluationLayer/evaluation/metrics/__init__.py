"""
Metrics module.

The resilience evaluation stack is organized around three primary dimensions:

1. Rebound: Recovery windows and formal recovery cost.
2. Stability: Neighborhood consistency under perturbation.
3. Boundary (GE): Family-level stress-response capacity.

There is no single-episode ``degradation`` dimension. GE derives curve
degradation diagnostics from the multi-lambda margin, where they are actually
identifiable.
"""

from .collection import (
    CollectorRegistry,
    EpisodeCollectionContext,
    EpisodeMetricCollector,
)
from .extensibility import (
    BoundaryCollector,
    BoundaryMargin,
    BoundarySweep,
    BoundaryThresholds,
)
from .rebound import ReboundCollector, ReboundMetrics, log_rebound_metrics
from .statistical_baseline import MetricReference, StatisticalBaseline
from .stability import StabilityCollector, StabilityMetrics, log_stability_metrics

__all__ = [
    "CollectorRegistry",
    "EpisodeCollectionContext",
    "EpisodeMetricCollector",
    "ReboundMetrics",
    "ReboundCollector",
    "log_rebound_metrics",
    "StabilityMetrics",
    "StabilityCollector",
    "log_stability_metrics",
    "BoundaryMargin",
    "BoundaryThresholds",
    "BoundaryCollector",
    "BoundarySweep",
    "MetricReference",
    "StatisticalBaseline",
]
