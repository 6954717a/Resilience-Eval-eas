#!/usr/bin/env python3

"""
Degradation Metrics Data Classes
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class DegradationMetrics:
    """
    Unified graceful degradation metrics.

    `l_bd` remains evolve-only and must stay unavailable for single-run episodes
    unless historical version performance is explicitly provided.
    """

    auc_loss: float = 0.0
    p_cliff: float = 0.0
    t_rec: float = 0.0

    l_bd: Optional[float] = None
    nrr: Optional[float] = None
    bwt: Optional[float] = None
    fwt: Optional[float] = None
    r_t: Optional[float] = None

    performance_history: List[float] = field(default_factory=list)
    recovery_events: List[Tuple[int, int]] = field(default_factory=list)

    in_degraded_state: bool = False
    t_drift: Optional[int] = None
    t_restored: Optional[int] = None

    total_steps: int = 0
    episode_id: Optional[str] = None
    l_bd_scope: str = "evolve_only"
    l_bd_available: bool = False

    def compute_derived_metrics(self) -> None:
        pass

    def to_dict(self) -> Dict:
        return {
            "auc_loss": self.auc_loss,
            "p_cliff": self.p_cliff,
            "t_rec": self.t_rec,
            "l_bd": self.l_bd,
            "nrr": self.nrr,
            "bwt": self.bwt,
            "fwt": self.fwt,
            "r_t": self.r_t,
            "in_degraded_state": self.in_degraded_state,
            "recovery_event_count": len(self.recovery_events),
            "total_steps": self.total_steps,
            "episode_id": self.episode_id,
            "l_bd_scope": self.l_bd_scope,
            "l_bd_available": self.l_bd_available,
        }

    def get_summary_for_csv(self) -> Dict[str, float]:
        return {
            "degradation_auc_loss": self.auc_loss,
            "degradation_p_cliff": self.p_cliff,
            "degradation_t_rec": self.t_rec,
            "degradation_l_bd": self.l_bd,
            "degradation_nrr": self.nrr,
            "degradation_recovery_events": float(len(self.recovery_events)),
        }
