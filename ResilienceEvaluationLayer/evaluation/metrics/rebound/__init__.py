"""
Rebound Metrics Module
======================

This module provides unified recovery metrics from failures, consolidating
ReboundManager and CoPAL RecoveryMetrics into a single dimension-based
interface.

Core Metrics:
  - B_epi: Correction Cost (N_retry + γ·N_backtrack)
  - MTTR / MTBF / RR / T_rec
  - C_rec:  stage-based recovery cost (§1 of the resilience design doc)
"""

from .rebound_metrics import (
    FailureEvent,
    ReboundMetrics,
    StageRecoveryRecord,
)
from .rebound_collector import ReboundCollector
from .rebound_logging import log_rebound_metrics
from habitat_llm.evaluation.stage_baseline import (
    StageAnchor,
    StageBaseline,
    StageBaselineDim,
    StageBaselineEstimator,
    StageResolver,
)
from .target_graph import (
    ActionFrame,
    EpisodeTargetGraph,
    PlacementTarget,
    StageSpec,
    StateFlipTarget,
    build_episode_target_graph,
    parse_action_entry,
)
from .rebound_templates import (
    ReboundEvent,
    ReboundTemplate,
    TemplateObservation,
    build_default_templates,
)
from .cog_load_tracker import CognitiveStepRecord, CogLoadTracker
from .online_tracker import OnlineReboundTracker, WindowSummary

__all__ = [
    "ReboundMetrics",
    "FailureEvent",
    "StageRecoveryRecord",
    "ReboundCollector",
    "log_rebound_metrics",
    "StageAnchor",
    "StageResolver",
    "StageBaseline",
    "StageBaselineDim",
    "StageBaselineEstimator",
    "ActionFrame",
    "EpisodeTargetGraph",
    "PlacementTarget",
    "StageSpec",
    "StateFlipTarget",
    "build_episode_target_graph",
    "parse_action_entry",
    "ReboundEvent",
    "ReboundTemplate",
    "TemplateObservation",
    "build_default_templates",
    "CognitiveStepRecord",
    "CogLoadTracker",
    "OnlineReboundTracker",
    "WindowSummary",
]
