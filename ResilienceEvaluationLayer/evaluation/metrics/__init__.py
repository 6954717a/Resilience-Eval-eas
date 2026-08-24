"""
Metrics module.

The resilience evaluation stack is organized around three primary dimensions:

1. Rebound: Recovery windows and formal recovery cost.
2. Stability: Neighborhood consistency under perturbation.
3. Graceful Extensibility: Family-level stress-response capacity.

Legacy degradation metrics remain exported for backward compatibility, but the
main narrative is now Rebound / Stability / Extensibility.
"""

from .degradation import DegradationCollector, DegradationMetrics, log_degradation_metrics
from .extensibility import BoundarySweep, ContractMargin, MarginCollector, MarginThresholds
from .rebound import ReboundCollector, ReboundMetrics, log_rebound_metrics
from .stability import StabilityCollector, StabilityMetrics, log_stability_metrics

__all__ = [
    "ReboundMetrics",
    "ReboundCollector",
    "log_rebound_metrics",
    "StabilityMetrics",
    "StabilityCollector",
    "log_stability_metrics",
    "DegradationMetrics",
    "DegradationCollector",
    "log_degradation_metrics",
    "ContractMargin",
    "MarginThresholds",
    "MarginCollector",
    "BoundarySweep",
]
