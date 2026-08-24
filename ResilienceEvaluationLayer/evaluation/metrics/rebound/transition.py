"""Canonical transition evidence consumed by the Rebound pipeline.

The runner produces planner decisions and environment outcomes at different
points in its loop.  Rebound must join the decision that was actually executed
with the state observed after that execution; otherwise action semantics and
progress deltas refer to different transitions.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping, Sequence


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


def _first_float(
    sources: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    default: float = 0.0,
) -> float:
    for source in sources:
        for name in names:
            if name in source and source[name] not in (None, ""):
                return _as_float(source[name], default)
    return float(default)


def build_rebound_transition(
    *,
    step: int,
    planner_info: Mapping[str, Any],
    info: Mapping[str, Any],
    progress: float,
    proposition_fraction: float,
    previous_progress: float,
    previous_proposition_fraction: float,
    previous_sim_step_count: float,
    previous_runtime_elapsed: float,
    previous_task_state_success: float,
    template_codes: Sequence[str],
) -> Dict[str, Any]:
    """Return one flat, formula-ready Rebound transition record."""

    stats = planner_info.get("stats") or {}
    if not isinstance(stats, Mapping):
        stats = {}
    sources = (info, planner_info, stats)

    sim_step_count = _first_float(
        sources,
        ("sim_step_count", "num_steps", "step_count"),
        previous_sim_step_count,
    )
    runtime_elapsed = _first_float(
        sources,
        ("runtime_elapsed", "elapsed_time", "runtime_seconds"),
        previous_runtime_elapsed,
    )
    task_state_success = _first_float(
        sources,
        ("task_state_success",),
        previous_task_state_success,
    )
    risk = _first_float(
        sources,
        ("cbf_penalty_step", "cbf_penalty", "safety_penalty"),
        0.0,
    )
    collisions = planner_info.get("agent_collisions") or info.get(
        "agent_collisions"
    )
    if isinstance(collisions, Mapping) and any(bool(value) for value in collisions.values()):
        risk = max(risk, 1.0)

    tags = planner_info.get("analysis_tags") or ()
    if isinstance(tags, str):
        tags = (tags,)

    return {
        "transition_step": int(step),
        "planning_step_count": int(step),
        "loop_step_count": int(step),
        "sim_step_count": float(sim_step_count),
        "delta_plan": 1.0,
        "delta_sim": max(0.0, float(sim_step_count) - previous_sim_step_count),
        "runtime_elapsed": float(runtime_elapsed),
        "delta_runtime": max(
            0.0, float(runtime_elapsed) - previous_runtime_elapsed
        ),
        "task_percent_complete": float(progress),
        "proposition_satisfied_fraction": float(proposition_fraction),
        "task_state_success": float(task_state_success),
        "delta_progress": float(progress - previous_progress),
        "delta_q": float(
            proposition_fraction - previous_proposition_fraction
        ),
        "delta_task_state_success": float(
            task_state_success - previous_task_state_success
        ),
        "cbf_penalty_step": float(risk),
        "high_level_actions": deepcopy(
            planner_info.get("high_level_actions") or {}
        ),
        "replanned": deepcopy(planner_info.get("replanned") or {}),
        "responses": deepcopy(planner_info.get("responses") or {}),
        "analysis_tags": [str(tag) for tag in tags],
        "_rebound_template_codes": sorted(
            {str(code) for code in template_codes}
        ),
    }


__all__ = ["build_rebound_transition"]
