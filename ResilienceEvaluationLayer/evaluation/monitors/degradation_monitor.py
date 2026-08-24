#!/usr/bin/env python3

"""
Degradation Monitor Module

This module implements the DegradationMonitor class, which tracks performance
degradation over time, calculating AUC loss and detecting performance cliffs.

Enhanced with persistence windows (L/H) per Eq.6 in Detailed_Metrics.tex.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

class DegradationMonitor:
    """
    Monitors system degradation metrics with persistence windows.
    
    Metrics tracked:
    - AUC_loss: Resilience Loss Area (integral of performance gap)
    - P_cliff: Cliff Probability (binary flag if performance drops below threshold)
    - T_rec: Cognitive Recovery Latency (t_restored - t_drift)
    - GD: Degradation Slope (if applicable)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DegradationMonitor.
        
        :param config: Configuration dictionary.
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.optimal_performance = config.get("optimal_performance", 1.0)
        self.crash_threshold = config.get("crash_threshold", 0.0)
        
        # Persistence windows - Eq.6 in Detailed_Metrics.tex
        self.persistence_L = config.get("persistence_L", 3)  # Drift detection window
        self.persistence_H = config.get("persistence_H", 3)  # Restore confirmation window
        self.deg_threshold = config.get("deg_threshold", 0.3)  # θ_deg
        self.nom_threshold = config.get("nom_threshold", 0.7)  # θ_nom
        
        self.reset()
        
    def reset(self):
        """Reset monitor state for new episode."""
        self.performance_history: List[float] = []
        self.auc_loss = 0.0
        self.cliff_detected = False
        
        # Regime tracking for T_rec
        self.below_deg_count = 0
        self.above_nom_count = 0
        self.in_degraded_state = False
        self.t_drift: Optional[int] = None
        self.t_restored: Optional[int] = None
        
        # Recovery events: [(t_drift, t_restored), ...]
        self.recovery_events: List[Tuple[int, int]] = []
        self.current_step = 0

    def update(self, step_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Update degradation metrics for the current step.
        
        :param step_data: Dictionary containing 'task_percent_complete'.
        :return: Dictionary containing degradation metrics.
        """
        if not self.enabled:
            return {"auc_loss": 0.0, "cliff_detected": 0.0, "in_degraded": 0.0}

        current_perf = step_data.get("task_percent_complete", 0.0)
        self.performance_history.append(current_perf)
        self.current_step = len(self.performance_history)
        
        # AUC Loss: Integral of (P_opt - P(t))
        performance_gap = max(0.0, self.optimal_performance - current_perf)
        self.auc_loss += performance_gap

        # Cliff detection
        if current_perf <= self.crash_threshold and self.current_step > 5:
            self.cliff_detected = True
        
        # Persistence window detection for regime transitions
        self._update_regime_state(current_perf)

        return {
            "auc_loss": self.auc_loss,
            "cliff_detected": 1.0 if self.cliff_detected else 0.0,
            "current_perf": current_perf,
            "in_degraded": 1.0 if self.in_degraded_state else 0.0,
            "t_drift": self.t_drift if self.t_drift else -1,
        }
    
    def _update_regime_state(self, current_perf: float):
        """
        Update regime state using persistence windows (L/H).
        Implements Eq.6 from Detailed_Metrics.tex.
        """
        if not self.in_degraded_state:
            # Check for drift: need L consecutive steps below θ_deg
            if current_perf <= self.deg_threshold:
                self.below_deg_count += 1
                if self.below_deg_count >= self.persistence_L:
                    self.in_degraded_state = True
                    self.t_drift = self.current_step - self.persistence_L + 1
                    self.below_deg_count = 0
            else:
                self.below_deg_count = 0
        else:
            # Check for restore: need H consecutive steps above θ_nom
            if current_perf >= self.nom_threshold:
                self.above_nom_count += 1
                if self.above_nom_count >= self.persistence_H:
                    self.in_degraded_state = False
                    self.t_restored = self.current_step
                    if self.t_drift is not None:
                        self.recovery_events.append((self.t_drift, self.t_restored))
                    self.t_drift = None
                    self.t_restored = None
                    self.above_nom_count = 0
            else:
                self.above_nom_count = 0

    def get_t_rec(self) -> float:
        """
        Get mean T_rec (Cognitive Recovery Latency).
        T_rec = t_restored - t_drift (averaged over all recovery events)
        """
        if not self.recovery_events:
            return 0.0
        total = sum(t_rest - t_drift for t_drift, t_rest in self.recovery_events)
        return total / len(self.recovery_events)

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics for the episode."""
        return {
            "total_auc_loss": self.auc_loss,
            "cliff_occurred": 1.0 if self.cliff_detected else 0.0,
            "T_rec": self.get_t_rec(),
            "recovery_event_count": float(len(self.recovery_events)),
        }

