#!/usr/bin/env python3

"""
Methods Module - Unified Implementation Methods

This module consolidates all implementation-specific methods (CLARE, CoPAL, CycleVLA,
InnoMono, SayCan, LLMEvaluator) into a unified structure.

These methods provide specific implementations for planning, recovery, and evaluation,
while the metrics/ module provides Rebound, Stability, and multi-run GE analysis.

Organization:
- clare/: Continual Learning with Autonomous Routing and Expansion
- copal/: Corrective Planning for error recovery
- cyclevla/: Proactive Self-Correction with VLM
- inno_mono/: Inner Monologue for closed-loop feedback
- saycan/: Say + Can grounding for affordance-based planning
- llm_evaluator/: LLM-based evaluation and reward shaping
"""

from typing import Dict, Type, Optional

# Method registry for dynamic loading
_METHOD_REGISTRY: Dict[str, Type] = {}


def register_method(name: str, method_class: Type) -> None:
    """Register a method implementation."""
    _METHOD_REGISTRY[name] = method_class


def get_method(name: str) -> Optional[Type]:
    """Get a registered method by name."""
    return _METHOD_REGISTRY.get(name)


def list_methods() -> list:
    """List all registered methods."""
    return list(_METHOD_REGISTRY.keys())


__all__ = [
    "register_method",
    "get_method",
    "list_methods",
]
