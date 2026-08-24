#!/usr/bin/env python3

"""
Metrics Aggregator

Unified aggregator for collecting all three-dimensional metrics (Rebound, Stability, Degradation).
Coordinates collection order to avoid duplicate extraction (e.g., T_rec).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class EpisodeMetrics:
    """
    Unified container for all episode-level metrics.

    Holds metrics from three dimensions:
    - Rebound: Recovery from failures
    - Stability: Policy consistency
    - Degradation: Graceful performance evolution
    """
    rebound: Optional[Any] = None      # ReboundMetrics
    stability: Optional[Any] = None    # StabilityMetrics
    degradation: Optional[Any] = None  # DegradationMetrics


class MetricsAggregator:
    """
    Unified metrics aggregator for three-dimensional analysis.

    Responsibilities:
    - Coordinate collection order (Degradation → Rebound → Stability)
    - Avoid duplicate extraction (e.g., T_rec shared between Degradation and Rebound)
    - Apply dimension-level configuration switches
    - Provide unified interface for runner_analysis.py
    """

    def __init__(
        self,
        rebound_collector: Any,
        stability_collector: Any,
        degradation_collector: Any,
        config: Dict[str, Any],
    ):
        """
        Initialize MetricsAggregator.

        Args:
            rebound_collector: ReboundCollector instance
            stability_collector: StabilityCollector instance
            degradation_collector: DegradationCollector instance
            config: Metrics configuration dictionary
        """
        self.rebound_collector = rebound_collector
        self.stability_collector = stability_collector
        self.degradation_collector = degradation_collector
        self.config = config

    def collect_episode_metrics(
        self,
        planner: Any,
        planner_infos: List[Dict],
        critic: Optional[Any],
        safety_monitor: Any,
        degradation_monitor: Any,
        total_steps: int,
    ) -> EpisodeMetrics:
        """
        Collect all three-dimensional metrics for a single episode.

        Collection order:
        1. Degradation (provides T_rec)
        2. Rebound (uses T_rec from Degradation)
        3. Stability (independent)

        Args:
            planner: Planner instance with rebound_managers
            planner_infos: List of planner info dictionaries
            critic: A2CCritic instance (optional)
            safety_monitor: SafetyMonitor instance
            degradation_monitor: DegradationMonitor instance
            total_steps: Total planning steps in episode

        Returns:
            EpisodeMetrics with all collected metrics
        """
        metrics = EpisodeMetrics()

        # 1. Degradation (collect first to provide T_rec)
        if self._is_enabled("degradation"):
            try:
                metrics.degradation = self.degradation_collector.collect_from_monitor(
                    degradation_monitor=degradation_monitor,
                )
                logger.debug(f"Collected degradation metrics: T_rec={metrics.degradation.t_rec}")
            except Exception as e:
                logger.error(f"Failed to collect degradation metrics: {e}", exc_info=True)
                metrics.degradation = None

        # 2. Rebound (use T_rec from Degradation to avoid duplicate extraction)
        if self._is_enabled("rebound"):
            try:
                t_rec = metrics.degradation.t_rec if metrics.degradation else 0.0
                metrics.rebound = self.rebound_collector.collect_from_planner(
                    planner=planner,
                    total_steps=total_steps,
                    t_rec=t_rec,
                )
                logger.debug(f"Collected rebound metrics: B_epi={metrics.rebound.b_epi}, MTTR={metrics.rebound.mttr}")
            except Exception as e:
                logger.error(f"Failed to collect rebound metrics: {e}", exc_info=True)
                metrics.rebound = None

        if metrics.degradation and metrics.rebound:
            metrics.degradation.nrr = self.degradation_collector.compute_nrr(
                metrics.rebound.mtbf,
                metrics.rebound.mttr,
            )

        # 3. Stability (independent collection)
        if self._is_enabled("stability"):
            try:
                metrics.stability = self.stability_collector.collect(
                    critic=critic,
                    planner_infos=planner_infos,
                    safety_monitor=safety_monitor,
                    planner=planner,
                )
                logger.debug(f"Collected stability metrics: beta={metrics.stability.beta}, value_variance={metrics.stability.value_variance}")
            except Exception as e:
                logger.error(f"Failed to collect stability metrics: {e}", exc_info=True)
                metrics.stability = None

        return metrics

    def _is_enabled(self, dimension: str) -> bool:
        """
        Check if a dimension is enabled in configuration.

        Args:
            dimension: Dimension name ("rebound", "stability", "degradation")

        Returns:
            True if enabled, False otherwise
        """
        dimension_config = self.config.get(dimension, {})
        return dimension_config.get("enabled", True)

    def get_csv_summary(self, metrics: EpisodeMetrics) -> Dict[str, Any]:
        """
        Extract CSV-friendly summary from EpisodeMetrics.

        Args:
            metrics: EpisodeMetrics instance

        Returns:
            Dictionary with flattened metrics for CSV output
        """
        summary = {
            "rebound_c_rec": 0.0,
            "rebound_c_rec_cog": 0.0,
            "rebound_c_rec_phy": 0.0,
            "rebound_c_rec_state_debt": 0.0,
            "rebound_num_recovery_windows": 0.0,
            "rebound_raw_window_count": 0.0,
            "rebound_formal_window_count": 0.0,
            "rebound_skipped_window_count": 0.0,
            "rebound_c_rec_valid": 0.0,
            "rebound_c_rec_missing_reason": "",
            "rebound_baseline_match_level": "none",
            "rebound_w_rem_star": 0.0,
            "rebound_g_rec_total": 0.0,
            "rebound_baseline_loaded": 0.0,
            "rebound_tracker_window_count": 0.0,
            "rebound_tracker_window_open_final": 0.0,
            "stability_beta": 0.0,
            "stability_beta_neighborhood": 0.0,
            "stability_beta_vv": 0.0,
            "stability_beta_out": 0.0,
            "stability_beta_oscillation": 0.0,
            "stability_beta_action": 0.0,
            "stability_beta_progress": 0.0,
            "stability_beta_success": 0.0,
            "stability_beta_perturbation": 0.0,
            "stability_beta_perturbation_vv": 0.0,
            "stability_beta_perturbation_out": 0.0,
            "stability_beta_perturbation_oscillation": 0.0,
            "stability_beta_perturbation_action": 0.0,
            "stability_beta_perturbation_progress": 0.0,
            "stability_beta_perturbation_success": 0.0,
            "stability_beta_perturbation_valid": 0.0,
            "stability_beta_perturbation_scope": "",
            "stability_clean_ref_valid": 0.0,
            "stability_clean_ref_strategy": "",
            "stability_n_clean_ref": 0.0,
            "stability_beta_neighborhood_valid": 0.0,
            "stability_beta_neighborhood_missing_reason": "",
            "stability_beta_scope": "",
            "stability_value_variance": 0.0,
            "stability_value_delta_variance": 0.0,
            "stability_td_error_mean_abs": 0.0,
            "stability_td_error_p95_abs": 0.0,
            "stability_gae_mean_abs": 0.0,
            "stability_gae_p95_abs": 0.0,
            "stability_return_target_variance": 0.0,
            "stability_oscillation_loss": 0.0,
            "stability_oscillation_loss_valid": 0.0,
            "stability_oscillation_source": "",
            "stability_critic_trace_len": 0.0,
            "stability_n_replan": 0.0,
            "stability_p_cbf": 0.0,
            "degradation_auc_loss": 0.0,
            "degradation_p_cliff": 0.0,
            "degradation_t_rec": 0.0,
            "degradation_l_bd": None,
            "degradation_nrr": None,
            "ge_contract_margin_min": 0.0,
            "ge_audc_f": 0.0,
            "ge_cell_valid": 0.0,
            "ge_cell_missing_reason": "",
            "ge_valid_lambda_coverage": 0.0,
            "ge_boundary_lambda_star": "",
            "ge_boundary_hit": 0.0,
            "ge_hard_boundary_hit": 0.0,
            "resilience_beta": 0.0,
            "resilience_l_bd": None,
        }

        if metrics.rebound:
            summary.update(metrics.rebound.get_summary_for_csv())

        if metrics.stability:
            summary.update(metrics.stability.get_summary_for_csv())

        if metrics.degradation:
            summary.update(metrics.degradation.get_summary_for_csv())

        summary["resilience_beta"] = summary.get("stability_beta")
        summary["resilience_l_bd"] = summary.get("degradation_l_bd")

        return summary


__all__ = ["EpisodeMetrics", "MetricsAggregator"]
