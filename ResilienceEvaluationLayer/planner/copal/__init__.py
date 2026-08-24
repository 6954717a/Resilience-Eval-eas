"""
CoPAL (Corrective Planning) Module for LLM Planner

This module implements CoPAL-style closed-loop correction through:
- ErrorTranslator: Translates technical errors to natural language
- HistoryManager: Manages action-observation history
- CorrectiveContext: Generates CoPAL-style corrective prompts

Reference: CoPAL Paper (arxiv.org/abs/2310.07263)
"""

from habitat_llm.planner.copal.error_translator import ErrorTranslator
from habitat_llm.planner.copal.history_manager import ActionHistoryManager
from habitat_llm.planner.copal.corrective_context import (
    build_copal_context,
    CoPALContextBuilder,
)
from habitat_llm.planner.copal.planner_integration import (
    get_copal_statistics,
    log_copal_analysis,
    inject_copal_context_if_needed,
    format_copal_metrics_for_log,
)

__all__ = [
    "ErrorTranslator",
    "ActionHistoryManager",
    "build_copal_context",
    "CoPALContextBuilder",
    "get_copal_statistics",
    "log_copal_analysis",
    "inject_copal_context_if_needed",
    "format_copal_metrics_for_log",
]

