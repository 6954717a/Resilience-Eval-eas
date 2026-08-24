#!/usr/bin/env python3

"""
Degradation Collector

Collects graceful degradation metrics from DegradationMonitor.
"""

import logging
from typing import Any, Dict, Optional

from .degradation_metrics import DegradationMetrics

logger = logging.getLogger(__name__)


class DegradationCollector:
    """
    Collects Degradation metrics from DegradationMonitor.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def collect_from_monitor(
        self,
        degradation_monitor: Any,
    ) -> DegradationMetrics:
        """
        Collect Degradation metrics from DegradationMonitor.

        Args:
            degradation_monitor: DegradationMonitor instance

        Returns:
            DegradationMetrics instance
        """
        metrics = DegradationMetrics()

        if degradation_monitor is None:
            logger.warning("DegradationMonitor is None")
            return metrics

        try:
            # Get summary from monitor
            summary = degradation_monitor.get_summary()

            # Extract core metrics
            metrics.auc_loss = summary.get("total_auc_loss", 0.0)
            metrics.p_cliff = summary.get("cliff_occurred", 0.0)
            metrics.t_rec = summary.get("T_rec", 0.0)

            # Extract detailed tracking
            metrics.performance_history = list(degradation_monitor.performance_history)
            metrics.recovery_events = list(degradation_monitor.recovery_events)
            metrics.in_degraded_state = degradation_monitor.in_degraded_state
            metrics.t_drift = degradation_monitor.t_drift
            metrics.t_restored = degradation_monitor.t_restored
            metrics.total_steps = degradation_monitor.current_step
            metrics.l_bd = None
            metrics.l_bd_available = False
            metrics.l_bd_scope = "evolve_only"

            logger.debug(
                "Extracted degradation metrics: AUC_loss=%.3f, P_cliff=%.1f, T_rec=%.1f",
                metrics.auc_loss,
                metrics.p_cliff,
                metrics.t_rec,
            )

        except Exception as exc:
            logger.warning("Failed to extract metrics from DegradationMonitor: %s", exc)

        return metrics

    def collect_from_info(self, info: Dict[str, Any]) -> DegradationMetrics:
        """
        Collect Degradation metrics from episode info dict (for backward compatibility).

        Args:
            info: Episode info dictionary

        Returns:
            DegradationMetrics instance
        """
        metrics = DegradationMetrics()

        # Extract from legacy keys
        metrics.auc_loss = info.get("degradation_auc_loss", 0.0)
        metrics.p_cliff = info.get("degradation_p_cliff", 0.0)
        metrics.t_rec = info.get("degradation_t_rec", 0.0)
        metrics.l_bd = info.get("degradation_l_bd", None)
        metrics.nrr = info.get("degradation_nrr", None)
        metrics.total_steps = int(info.get("total_step_count", 0))
        metrics.l_bd_scope = str(info.get("l_bd_scope", metrics.l_bd_scope))
        metrics.l_bd_available = bool(
            info.get("l_bd_available", metrics.l_bd is not None)
        )

        return metrics

    @staticmethod
    def compute_nrr(mtbf: float, mttr: float) -> Optional[float]:
        denominator = float(mtbf) + float(mttr)
        if denominator <= 0.0:
            return None
        return float(mtbf) / denominator

    def compute_evolve_metrics(
        self,
        current_performance: float,
        historical_performances: list,
        mtbf: float,
        mttr: float,
    ) -> Dict[str, float]:
        """
        Compute L3 (Evolve-level) metrics across versions.

        Args:
            current_performance: Current version's performance
            historical_performances: List of previous versions' performances
            mtbf: Mean Time Between Failures
            mttr: Mean Time To Recovery

        Returns:
            Dictionary with L_bd, NRR, BWT, FWT, R_T
        """
        evolve_metrics = {}

        # L_bd: Evolve Lower Bound (only valid with historical versions)
        if historical_performances:
            min_historical = min(historical_performances)
            epsilon = 0.05  # 5% tolerance
            evolve_metrics["l_bd"] = min_historical - epsilon
            evolve_metrics["l_bd_available"] = True
        else:
            evolve_metrics["l_bd"] = None
            evolve_metrics["l_bd_available"] = False
        evolve_metrics["l_bd_scope"] = "evolve_only"

        # NRR: Normalized Resilience Ratio
        evolve_metrics["nrr"] = self.compute_nrr(mtbf, mttr)

        # BWT: Backward Transfer (requires old task performance data)
        # Placeholder - needs implementation with task-specific data
        evolve_metrics["bwt"] = None

        # FWT: Forward Transfer (requires new task performance data)
        # Placeholder - needs implementation with task-specific data
        evolve_metrics["fwt"] = None

        # R_T: Cumulative Adversarial Regret (requires oracle performance)
        # Placeholder - needs implementation with oracle data
        evolve_metrics["r_t"] = None

        return evolve_metrics
