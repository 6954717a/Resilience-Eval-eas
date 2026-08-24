"""
CLARE: Continual Learning with Autonomous Routing and Expansion

Evaluation-side modules for CLARE metrics and training.
"""

from habitat_llm.evaluation.clare.continual_metrics import (
    ContinualMetricsTracker,
    log_clare_metrics,
)
from habitat_llm.evaluation.clare.evolution_trainer import (
    CLAREEvolutionTrainer,
    EvolutionResult,
)

__all__ = [
    "ContinualMetricsTracker",
    "log_clare_metrics",
    "CLAREEvolutionTrainer",
    "EvolutionResult",
]
