"""
Degradation Metrics Module

This module provides unified graceful degradation metrics, measuring version evolution
without catastrophic regression. Extracts metrics from DegradationMonitor.

Core Metrics:
- AUC_loss: Resilience Loss Area
- P_cliff: Cliff Probability
- T_rec: Cognitive Recovery Latency
- L_bd: Evolve Lower Bound
- NRR: Normalized Resilience Ratio
"""

from .degradation_metrics import DegradationMetrics
from .degradation_collector import DegradationCollector
from .degradation_logging import log_degradation_metrics

__all__ = [
    "DegradationMetrics",
    "DegradationCollector",
    "log_degradation_metrics",
]
