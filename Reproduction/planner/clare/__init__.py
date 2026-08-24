"""
CLARE: Continual Learning with Autonomous Routing and Expansion

Planner-side modules for CLARE integration.
"""

from habitat_llm.planner.clare.adapter_manager import AdapterManager, ContextAdapter
from habitat_llm.planner.clare.router import TaskRouter, RouterResult
from habitat_llm.planner.clare.expansion_controller import ExpansionController
from habitat_llm.planner.clare.planner_integration import (
    initialize_clare,
    clare_route_task,
    clare_inject_context,
    clare_record_episode,
    clare_step,
    clare_evolve_batch,
    get_clare_metrics,
)

__all__ = [
    "AdapterManager",
    "ContextAdapter",
    "TaskRouter",
    "RouterResult",
    "ExpansionController",
    "initialize_clare",
    "clare_route_task",
    "clare_inject_context",
    "clare_record_episode",
    "clare_step",
    "clare_evolve_batch",
    "get_clare_metrics",
]

