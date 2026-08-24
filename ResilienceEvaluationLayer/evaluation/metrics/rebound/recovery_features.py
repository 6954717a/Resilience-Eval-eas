"""Shared feature primitives for the formal Rebound calculation.

The clean StageBaseline estimator and the runtime collector must use the same
normalization.  Keeping the scalar primitive here makes that contract explicit
and prevents the zero-baseline ``1 / epsilon`` amplification that previously
dominated the cognitive channel.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Set


RECOVERY_FEATURE_VERSION = "recovery_features_v3"
REBOUND_FORMULA_VERSION = "rebound_c_rec_v3"
COGNITIVE_CHANNELS = ("pln", "per", "coord")


def normalized_action_name(action: Any) -> str:
    """Return a normalized high-level action name from runtime/export forms."""

    if isinstance(action, (tuple, list)) and action:
        action = action[0]
    text = str(action or "").strip()
    if not text:
        return ""
    prefix, separator, remainder = text.partition(":")
    normalized_prefix = prefix.strip().lower()
    if separator and (
        normalized_prefix.isdigit() or normalized_prefix.startswith("agent_")
    ):
        text = remainder.strip()
    return text.split("[", 1)[0].strip().lower()


def cognitive_event_keys(planner_info: Mapping[str, Any]) -> Dict[str, Set[str]]:
    """Extract aligned cognitive event identities from runtime or compact exports.

    Both StageBaseline and the online tracker call this helper, so rising-edge
    counts cannot drift because one side prefers tags while the other prefers
    agent-scoped fields. ``Wait`` is intentionally absent.
    """

    keys: Dict[str, Set[str]] = {channel: set() for channel in COGNITIVE_CHANNELS}
    raw_tags = planner_info.get("analysis_tags") or []
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    tags = {str(tag) for tag in raw_tags}

    replanned = planner_info.get("replanned") or {}
    if isinstance(replanned, Mapping):
        keys["pln"].update(
            f"replanned:{agent_id}"
            for agent_id, active in replanned.items()
            if bool(active)
        )
    if not keys["pln"]:
        keys["pln"].update(
            f"tag:{tag}"
            for tag in tags & {"replan_anchor", "planning_transition"}
        )

    responses = planner_info.get("responses") or {}
    if isinstance(responses, Mapping):
        keys["per"].update(
            f"response:{source}"
            for source, value in responses.items()
            if bool(value)
        )
    if not keys["per"]:
        recent_response = planner_info.get("recent_response")
        if recent_response not in (None, ""):
            keys["per"].add(f"recent_response:{recent_response}")
        elif "response_event" in tags:
            keys["per"].add("tag:response_event")

    actions = planner_info.get("high_level_actions") or {}
    if isinstance(actions, Mapping):
        for agent_id, action in actions.items():
            name = normalized_action_name(action)
            if name in {"handover", "coordinate"}:
                keys["coord"].add(f"action:{agent_id}:{name}:{action!r}")
    if not keys["coord"]:
        recent_action = planner_info.get("recent_action")
        if recent_action not in (None, ""):
            for index, action in enumerate(str(recent_action).split(";")):
                name = normalized_action_name(action)
                if name in {"handover", "coordinate"}:
                    keys["coord"].add(
                        f"recent_action:{index}:{name}:{action.strip()}"
                    )
    return keys


def normalized_cognitive_excess(
    volume: float,
    baseline_volume: float,
    validity: float,
) -> float:
    r"""Return the dimensionless cognitive excess for one channel.

    .. math::

        x = \frac{(v - \bar v)^+}{\max(|\bar v|, 1)}(1-q)

    Counts therefore retain a natural unit floor: when the clean median is
    zero, one invalid extra event contributes one rather than one million.
    Invalid/non-finite inputs fail closed to a finite, non-negative feature.
    """

    try:
        observed = float(volume)
        nominal = float(baseline_volume)
        quality = float(validity)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(observed) or not math.isfinite(nominal):
        return 0.0
    if not math.isfinite(quality):
        quality = 0.0
    quality = min(1.0, max(0.0, quality))
    denominator = max(abs(nominal), 1.0)
    return max(0.0, observed - nominal) / denominator * (1.0 - quality)


def cognitive_feature_vector(
    volumes: Mapping[str, float],
    baseline_volumes: Mapping[str, float],
    validities: Mapping[str, float],
    channels: Sequence[str],
) -> Dict[str, float]:
    """Build the aligned cognitive feature vector used by baseline/runtime."""

    return {
        str(channel): normalized_cognitive_excess(
            volumes.get(channel, 0.0),
            baseline_volumes.get(channel, 0.0),
            validities.get(channel, 1.0),
        )
        for channel in channels
    }


def physical_feature_vector(
    *,
    plan_work: float,
    sim_work: float,
    plan_baseline: float,
    sim_baseline: float,
    idle_gap: float = 0.0,
    redo_event: bool = False,
) -> List[float]:
    """Build the shared three-dimensional physical recovery feature."""

    plan_base = max(abs(float(plan_baseline)), 1.0e-12)
    sim_base = max(abs(float(sim_baseline)), 1.0e-12)
    return [
        max(0.0, float(plan_work) / plan_base - 1.0),
        max(0.0, float(sim_work) / sim_base - 1.0)
        + max(0.0, float(idle_gap)),
        float(bool(redo_event)),
    ]


def state_debt_feature_vector(
    *,
    progress_debt: float,
    goal_debt: float,
    risk: float,
    progress_baseline: float,
    goal_baseline: float,
    risk_baseline: float,
) -> List[float]:
    """Build the shared progress/goal/risk debt feature."""

    progress_base = max(abs(float(progress_baseline)), 1.0e-12)
    goal_base = max(abs(float(goal_baseline)), 1.0e-12)
    risk_nominal = max(0.0, float(risk_baseline))
    risk_value = max(0.0, float(risk))
    risk_excess = (
        max(0.0, risk_value / risk_nominal - 1.0)
        if risk_nominal > 0.0
        else risk_value
    )
    return [
        max(0.0, float(progress_debt)) / progress_base,
        max(0.0, float(goal_debt)) / goal_base,
        risk_excess,
    ]


def build_recovery_feature_groups(
    *,
    volumes: Mapping[str, float],
    baseline_volumes: Mapping[str, float],
    validities: Mapping[str, float],
    plan_work: float,
    sim_work: float,
    plan_baseline: float,
    sim_baseline: float,
    idle_gap: float,
    redo_event: bool,
    progress_debt: float,
    goal_debt: float,
    risk: float,
    progress_baseline: float,
    goal_baseline: float,
    risk_baseline: float,
) -> Dict[str, List[float]]:
    """Return the sole runtime/calibration builder for all Rebound channels."""

    cognitive = cognitive_feature_vector(
        volumes,
        baseline_volumes,
        validities,
        COGNITIVE_CHANNELS,
    )
    return {
        "cog": [cognitive[channel] for channel in COGNITIVE_CHANNELS],
        "phy": physical_feature_vector(
            plan_work=plan_work,
            sim_work=sim_work,
            plan_baseline=plan_baseline,
            sim_baseline=sim_baseline,
            idle_gap=idle_gap,
            redo_event=redo_event,
        ),
        "debt": state_debt_feature_vector(
            progress_debt=progress_debt,
            goal_debt=goal_debt,
            risk=risk,
            progress_baseline=progress_baseline,
            goal_baseline=goal_baseline,
            risk_baseline=risk_baseline,
        ),
    }


__all__ = [
    "COGNITIVE_CHANNELS",
    "RECOVERY_FEATURE_VERSION",
    "REBOUND_FORMULA_VERSION",
    "build_recovery_feature_groups",
    "cognitive_event_keys",
    "cognitive_feature_vector",
    "normalized_action_name",
    "normalized_cognitive_excess",
    "physical_feature_vector",
    "state_debt_feature_vector",
]
