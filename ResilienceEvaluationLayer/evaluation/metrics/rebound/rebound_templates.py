#!/usr/bin/env python3
# isort: skip_file

r"""
Rebound Template Library
========================

The templates in this module are the *recognition front-end* for the online
Rebound pipeline.  Each template watches the per-step trajectory and emits
:class:`ReboundEvent` records with a ``kind`` field ∈ ``{"t_d", "t_r",
"stuck"}``.  Events are consumed by :class:`OnlineReboundTracker` to open
/close recovery windows and attached to ``planner_info["_rebound_events"]``
(leading underscore keeps the raw list out of the CSV / scalar-extraction
pipeline) so downstream reporting and offline ``C_rec`` aggregation can
reproduce them.

The library replaces the single "progress-stagnation" trigger with a set
of semantically meaningful patterns:

========= ================================================================
 Code      Pattern
========= ================================================================
 R1        ``nav_oscillation`` — repeated A↔B navigation without a
           Pick/Place in between (action-history only).
 R3        ``plan_thrashing`` — rapid planner churn: many replans on
           identical or semantically equivalent skill calls without any
           progress delta.
 T1        ``target_receptacle_mismatch`` — agent places an object at a
           receptacle that is *not* in the episode's target graph.
 T2        ``required_state_flip_omitted`` — agent attempts the terminal
           placement before flipping a required state (``is_clean`` etc.)
           that the target graph demands.
 T3        ``phase_region_deviation`` — agent navigates to regions that
           contain no currently-open placements / state flips for an
           extended period.
 T4        ``null_pick_place_cycle`` — ``Pick[obj] … Place[obj, same]``
           cycle with no progress delta.
========= ================================================================

All templates are *side-effect free* and can be run in parallel.  A
template returns ``None`` when it does not fire for a given step.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .target_graph import (
    ActionFrame,
    EpisodeTargetGraph,
    parse_action_entry,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Event dataclass
# ----------------------------------------------------------------------
@dataclass
class ReboundEvent:
    r"""One semantic event emitted by a template.

    Attributes
    ----------
    step:
        The planning step at which the event is observed.
    kind:
        ``"t_d"`` to open a recovery window, ``"t_r"`` to close, ``"stuck"``
        to signal a non-progressing behavioural pattern that should open a
        window but has no natural closing condition of its own.
    template:
        Short tag identifying which template produced the event (R1, T1…).
    severity:
        Template-specific magnitude used to weight
        :math:`\Omega_t` at aggregation time.
    agent_uid:
        Agent that drove the pattern when it is attributable; otherwise
        ``None``.
    payload:
        Template-specific diagnostic fields (kept small for CSV export).
    """

    step: int
    kind: str
    template: str
    severity: float = 1.0
    agent_uid: Optional[int] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "kind": self.kind,
            "template": self.template,
            "severity": float(self.severity),
            "agent_uid": self.agent_uid,
            "payload": dict(self.payload),
        }


# ----------------------------------------------------------------------
# Observation protocol (what every template receives per step)
# ----------------------------------------------------------------------
@dataclass
class TemplateObservation:
    """Per-step input bundle for templates."""

    step: int
    planner_info: Mapping[str, Any]
    info: Mapping[str, Any]
    frames: Sequence[ActionFrame]     # parsed high-level actions at this step
    prev_progress: float
    curr_progress: float
    prev_q: float                     # proposition_satisfied_fraction
    curr_q: float
    proposition_satisfied_at: Sequence[int]
    target_graph: Optional[EpisodeTargetGraph]

    @property
    def delta_progress(self) -> float:
        return float(self.curr_progress - self.prev_progress)

    @property
    def delta_q(self) -> float:
        return float(self.curr_q - self.prev_q)

    @property
    def any_manipulation(self) -> bool:
        return any(f.is_manipulation for f in self.frames)

    @property
    def any_pick(self) -> bool:
        return any(f.is_pick for f in self.frames)

    @property
    def any_place(self) -> bool:
        return any(f.is_place for f in self.frames)


# ----------------------------------------------------------------------
# Base class
# ----------------------------------------------------------------------
class ReboundTemplate:
    """Base class for semantic Rebound detectors.

    Templates keep their own small per-instance state (e.g. sliding
    windows) and expose three methods:

    * :meth:`observe` — called every step with a :class:`TemplateObservation`.
      Returns zero or more :class:`ReboundEvent`.
    * :meth:`finalize` — called at episode end; may emit a terminal ``t_r``
      for any still-open stuck window.
    * :meth:`reset` — called before a new episode.
    """

    code: str = "template"

    def __init__(self, **kwargs: Any) -> None:
        self._config = dict(kwargs)

    # Subclasses override this.
    def observe(self, obs: TemplateObservation) -> List[ReboundEvent]:
        raise NotImplementedError

    def finalize(self, last_step: int) -> List[ReboundEvent]:
        return []

    def reset(self) -> None:
        pass


# ----------------------------------------------------------------------
# R1: nav_oscillation
# ----------------------------------------------------------------------
class NavOscillationDetector(ReboundTemplate):
    """Detect A→B→A→B navigation patterns without intervening manipulation.

    The detector maintains a deque of recent *non-manipulation* nav targets.
    Once we observe an A→B→A pattern (i.e. at least one repeated target
    separated by at least one different target) AND no Pick/Place has
    occurred in the same window, an event is emitted.
    """

    code = "R1"

    def __init__(
        self,
        window_size: int = 6,
        min_oscillations: int = 2,
        cooldown: int = 4,
    ) -> None:
        super().__init__(window_size=window_size, min_oscillations=min_oscillations)
        self.window_size = int(window_size)
        self.min_oscillations = int(min_oscillations)
        self.cooldown = int(cooldown)
        self._recent: Deque[ActionFrame] = deque(maxlen=self.window_size)
        self._recent_manipulation_step: int = -(10**9)
        self._last_fired_step: int = -(10**9)
        self._open_since: Optional[int] = None

    def reset(self) -> None:
        self._recent.clear()
        self._recent_manipulation_step = -(10**9)
        self._last_fired_step = -(10**9)
        self._open_since = None

    def observe(self, obs: TemplateObservation) -> List[ReboundEvent]:
        events: List[ReboundEvent] = []
        if obs.any_manipulation:
            self._recent_manipulation_step = obs.step
        for frame in obs.frames:
            if frame.is_nav:
                self._recent.append(frame)

        # Close an open window as soon as a manipulation succeeds (Δq ≥ 0
        # and Pick/Place executed) — that is the natural recovery signal.
        if self._open_since is not None and obs.any_manipulation and obs.delta_q >= 0:
            events.append(
                ReboundEvent(
                    step=obs.step,
                    kind="t_r",
                    template=self.code,
                    severity=1.0,
                    payload={"opened_at": int(self._open_since)},
                )
            )
            self._open_since = None
            self._last_fired_step = obs.step
            return events

        if len(self._recent) < 3 or obs.step - self._last_fired_step < self.cooldown:
            return events

        # Count alternations: pairs (i, i+2) that share a primary target and
        # differ from (i+1).
        oscillations = 0
        frames = list(self._recent)
        for i in range(len(frames) - 2):
            a, b, c = frames[i], frames[i + 1], frames[i + 2]
            if (
                a.primary_arg
                and c.primary_arg
                and a.primary_arg == c.primary_arg
                and b.primary_arg
                and b.primary_arg != a.primary_arg
            ):
                oscillations += 1
        if oscillations < self.min_oscillations:
            return events

        # No manipulation happened inside the oscillation window?
        window_first_step = frames[0].step if frames else obs.step
        if self._recent_manipulation_step >= window_first_step:
            return events

        if self._open_since is None:
            self._open_since = obs.step
            events.append(
                ReboundEvent(
                    step=obs.step,
                    kind="t_d",
                    template=self.code,
                    severity=float(oscillations),
                    payload={
                        "oscillations": int(oscillations),
                        "targets": [f.primary_arg for f in frames],
                    },
                )
            )
            self._last_fired_step = obs.step
        return events

    def finalize(self, last_step: int) -> List[ReboundEvent]:
        if self._open_since is None:
            return []
        return [
            ReboundEvent(
                step=int(last_step),
                kind="t_r",
                template=self.code,
                severity=0.5,  # lower: window never closed organically
                payload={"opened_at": int(self._open_since), "finalized": True},
            )
        ]


# ----------------------------------------------------------------------
# R3: plan_thrashing
# ----------------------------------------------------------------------
class PlanThrashingDetector(ReboundTemplate):
    """Detect rapid planner replan churn without progress.

    We look at ``planner_info['replanned']`` (per-agent replan flags) and
    ``analysis_tags`` from the critic.  When ``k`` replans within a rolling
    window yield ``Δp ≤ 0`` and ``Δq ≤ 0`` the detector fires a ``t_d``.
    The window closes when a replan finally coincides with forward progress.
    """

    code = "R3"

    def __init__(
        self,
        window_size: int = 5,
        min_replans: int = 3,
        cooldown: int = 4,
    ) -> None:
        super().__init__(window_size=window_size, min_replans=min_replans)
        self.window_size = int(window_size)
        self.min_replans = int(min_replans)
        self.cooldown = int(cooldown)
        self._replan_steps: Deque[int] = deque(maxlen=32)
        self._progress_window: Deque[Tuple[float, float]] = deque(maxlen=self.window_size)
        self._last_fired_step: int = -(10**9)
        self._open_since: Optional[int] = None

    def reset(self) -> None:
        self._replan_steps.clear()
        self._progress_window.clear()
        self._last_fired_step = -(10**9)
        self._open_since = None

    def _count_recent_replans(self, current_step: int) -> int:
        lower_bound = current_step - self.window_size
        while self._replan_steps and self._replan_steps[0] < lower_bound:
            self._replan_steps.popleft()
        return len(self._replan_steps)

    def observe(self, obs: TemplateObservation) -> List[ReboundEvent]:
        events: List[ReboundEvent] = []
        replanned = obs.planner_info.get("replanned") or {}
        fired_this_step = False
        if isinstance(replanned, Mapping) and any(bool(v) for v in replanned.values()):
            self._replan_steps.append(int(obs.step))
            fired_this_step = True
        tags = obs.planner_info.get("analysis_tags") or []
        if isinstance(tags, str):
            tags = [tags]
        if any(t in ("replan_anchor", "planning_transition") for t in tags):
            if not fired_this_step:
                self._replan_steps.append(int(obs.step))

        self._progress_window.append((obs.delta_progress, obs.delta_q))

        recent = self._count_recent_replans(int(obs.step))
        pos = sum(
            1 for dp, dq in self._progress_window if dp > 1e-6 or dq > 1e-6
        )

        if self._open_since is not None:
            # Close when any real progress lands after a replan.
            if obs.delta_progress > 1e-6 or obs.delta_q > 1e-6:
                events.append(
                    ReboundEvent(
                        step=obs.step,
                        kind="t_r",
                        template=self.code,
                        severity=1.0,
                        payload={"opened_at": int(self._open_since)},
                    )
                )
                self._open_since = None
                self._last_fired_step = obs.step
            return events

        if (
            recent >= self.min_replans
            and pos == 0
            and obs.step - self._last_fired_step >= self.cooldown
        ):
            self._open_since = obs.step
            events.append(
                ReboundEvent(
                    step=obs.step,
                    kind="t_d",
                    template=self.code,
                    severity=float(recent),
                    payload={
                        "recent_replans": int(recent),
                        "window_size": int(self.window_size),
                    },
                )
            )
            self._last_fired_step = obs.step
        return events

    def finalize(self, last_step: int) -> List[ReboundEvent]:
        if self._open_since is None:
            return []
        return [
            ReboundEvent(
                step=int(last_step),
                kind="t_r",
                template=self.code,
                severity=0.5,
                payload={"opened_at": int(self._open_since), "finalized": True},
            )
        ]


# ----------------------------------------------------------------------
# T1: target_receptacle_mismatch
# ----------------------------------------------------------------------
class TargetReceptacleMismatchDetector(ReboundTemplate):
    """Fire whenever a ``Place[obj, recept]`` does not satisfy any *open*
    placement target for that object in the episode target graph.

    Unlike R1/R3 this is an instantaneous event: we emit paired ``t_d`` and
    ``t_r`` at the same step, carrying the offending ``(obj, recept)``.
    The ``severity`` is scaled by the number of remaining open placements
    for that object, because placing at a wrong receptacle while *many*
    valid targets exist is more informative than doing so near task end.
    """

    code = "T1"

    def __init__(self, cooldown: int = 1) -> None:
        super().__init__(cooldown=cooldown)
        self.cooldown = int(cooldown)
        self._last_fired_step: int = -(10**9)

    def reset(self) -> None:
        self._last_fired_step = -(10**9)

    def observe(self, obs: TemplateObservation) -> List[ReboundEvent]:
        events: List[ReboundEvent] = []
        if obs.target_graph is None:
            return events
        if obs.step - self._last_fired_step < self.cooldown:
            return events

        for frame in obs.frames:
            if not frame.is_place:
                continue
            obj = frame.primary_arg
            rec = frame.secondary_arg
            if obj is None or rec is None:
                continue
            open_placements = obs.target_graph.open_placements(obs.proposition_satisfied_at)
            matches_any_open = any(
                p.matches(obj, rec) for p in open_placements
            )
            if matches_any_open:
                continue
            # It is a mismatch — severity proportional to how many open
            # placements for this object are still unsatisfied.
            remaining = sum(
                1
                for p in open_placements
                if str(obj) in p.object_ids and not p.matches(obj, rec)
            )
            severity = float(remaining + 1)
            payload = {
                "object": obj,
                "receptacle": rec,
                "open_placements_for_object": remaining,
            }
            events.append(
                ReboundEvent(
                    step=obs.step,
                    kind="t_d",
                    template=self.code,
                    severity=severity,
                    agent_uid=frame.agent_uid,
                    payload=payload,
                )
            )
            events.append(
                ReboundEvent(
                    step=obs.step,
                    kind="t_r",
                    template=self.code,
                    severity=severity,
                    agent_uid=frame.agent_uid,
                    payload=payload,
                )
            )
            self._last_fired_step = obs.step
        return events


# ----------------------------------------------------------------------
# T2: required_state_flip_omitted
# ----------------------------------------------------------------------
class RequiredStateFlipOmittedDetector(ReboundTemplate):
    """Fire when a terminal ``Place[obj, recept]`` is executed *before* the
    required state flip on ``obj`` (is_clean/is_powered_on/…) has happened.

    We consider a state flip *outstanding* when its proposition is still
    unsatisfied at the moment the placement is attempted.
    """

    code = "T2"

    def __init__(self, cooldown: int = 1) -> None:
        super().__init__(cooldown=cooldown)
        self.cooldown = int(cooldown)
        self._last_fired_step: int = -(10**9)

    def reset(self) -> None:
        self._last_fired_step = -(10**9)

    def observe(self, obs: TemplateObservation) -> List[ReboundEvent]:
        events: List[ReboundEvent] = []
        if obs.target_graph is None:
            return events
        if obs.step - self._last_fired_step < self.cooldown:
            return events

        psa = obs.proposition_satisfied_at
        for frame in obs.frames:
            if not frame.is_place:
                continue
            obj = frame.primary_arg
            if not obj:
                continue
            outstanding_flips = [
                s for s in obs.target_graph.state_flips_for_object(str(obj))
                if 0 <= s.proposition_index < len(psa)
                and int(psa[s.proposition_index]) < 0
            ]
            if not outstanding_flips:
                continue
            payload = {
                "object": obj,
                "missing_flips": [s.predicate for s in outstanding_flips],
            }
            events.append(
                ReboundEvent(
                    step=obs.step,
                    kind="t_d",
                    template=self.code,
                    severity=float(len(outstanding_flips) + 1),
                    agent_uid=frame.agent_uid,
                    payload=payload,
                )
            )
            events.append(
                ReboundEvent(
                    step=obs.step,
                    kind="t_r",
                    template=self.code,
                    severity=float(len(outstanding_flips) + 1),
                    agent_uid=frame.agent_uid,
                    payload=payload,
                )
            )
            self._last_fired_step = obs.step
        return events


# ----------------------------------------------------------------------
# T3: phase_region_deviation
# ----------------------------------------------------------------------
class PhaseRegionDeviationDetector(ReboundTemplate):
    """Fire when the agent spends ``>N`` consecutive nav-only steps toward
    objects/receptacles that are **not** part of any currently open target.

    The detector counts how many contiguous steps have navigation actions
    whose primary argument is *not* referenced by any open placement /
    state-flip target in the graph.  Once the streak crosses the threshold,
    a ``t_d`` is emitted; the window is closed when the agent either
    (a) navigates back to a relevant target, or (b) executes a successful
    manipulation that advances ``Δq``.
    """

    code = "T3"

    def __init__(
        self,
        min_streak: int = 3,
        cooldown: int = 4,
    ) -> None:
        super().__init__(min_streak=min_streak, cooldown=cooldown)
        self.min_streak = int(min_streak)
        self.cooldown = int(cooldown)
        self._streak: int = 0
        self._last_fired_step: int = -(10**9)
        self._open_since: Optional[int] = None

    def reset(self) -> None:
        self._streak = 0
        self._last_fired_step = -(10**9)
        self._open_since = None

    def _is_irrelevant(
        self,
        frame: ActionFrame,
        graph: EpisodeTargetGraph,
        psa: Sequence[int],
    ) -> bool:
        if not frame.is_nav:
            return False
        # ``Explore[room]`` is an intentionally coarse search skill. Without a
        # room-level target graph we treat it as neutral instead of
        # automatically penalizing it as off-target drift.
        if frame.name == "explore":
            return False
        target_id = frame.primary_arg
        if not target_id:
            return False
        for p in graph.open_placements(psa):
            if str(target_id) in p.object_ids or str(target_id) in p.receptacle_ids:
                return False
        for s in graph.open_state_flips(psa):
            if str(target_id) in s.object_ids:
                return False
        return True

    def observe(self, obs: TemplateObservation) -> List[ReboundEvent]:
        events: List[ReboundEvent] = []
        if obs.target_graph is None:
            return events

        irrelevant_this_step = False
        any_nav = any(f.is_nav for f in obs.frames)
        relevant_this_step = False
        for frame in obs.frames:
            if frame.is_nav:
                if self._is_irrelevant(frame, obs.target_graph, obs.proposition_satisfied_at):
                    irrelevant_this_step = True
                else:
                    relevant_this_step = True

        if obs.any_manipulation and obs.delta_q >= 0:
            relevant_this_step = True

        if self._open_since is not None:
            if relevant_this_step:
                events.append(
                    ReboundEvent(
                        step=obs.step,
                        kind="t_r",
                        template=self.code,
                        severity=1.0,
                        payload={"opened_at": int(self._open_since)},
                    )
                )
                self._open_since = None
                self._streak = 0
                self._last_fired_step = obs.step
            return events

        if irrelevant_this_step and not relevant_this_step:
            self._streak += 1
        else:
            self._streak = 0

        if (
            self._streak >= self.min_streak
            and any_nav
            and obs.step - self._last_fired_step >= self.cooldown
        ):
            self._open_since = obs.step
            events.append(
                ReboundEvent(
                    step=obs.step,
                    kind="t_d",
                    template=self.code,
                    severity=float(self._streak),
                    payload={"streak": int(self._streak)},
                )
            )
            self._last_fired_step = obs.step
        return events

    def finalize(self, last_step: int) -> List[ReboundEvent]:
        if self._open_since is None:
            return []
        return [
            ReboundEvent(
                step=int(last_step),
                kind="t_r",
                template=self.code,
                severity=0.5,
                payload={"opened_at": int(self._open_since), "finalized": True},
            )
        ]


# ----------------------------------------------------------------------
# T4: null_pick_place_cycle
# ----------------------------------------------------------------------
class NullPickPlaceCycleDetector(ReboundTemplate):
    """Fire on ``Pick[obj] ... Place[obj, same_location]`` cycles with no
    progress delta.

    We track the most recent ``Pick`` per object and match it against a
    subsequent ``Place`` to the same object — if the ``Δq`` in between was
    non-positive AND the placement target doesn't close any open
    placement, emit an event.
    """

    code = "T4"

    def __init__(self, cooldown: int = 1, window_steps: int = 32) -> None:
        super().__init__(cooldown=cooldown, window_steps=window_steps)
        self.cooldown = int(cooldown)
        self.window_steps = int(window_steps)
        self._last_pick: Dict[str, Tuple[int, Optional[int]]] = {}  # obj -> (step, agent)
        self._q_at_pick: Dict[str, float] = {}
        self._last_fired_step: int = -(10**9)

    def reset(self) -> None:
        self._last_pick.clear()
        self._q_at_pick.clear()
        self._last_fired_step = -(10**9)

    def observe(self, obs: TemplateObservation) -> List[ReboundEvent]:
        events: List[ReboundEvent] = []
        for frame in obs.frames:
            if frame.is_pick and frame.primary_arg:
                self._last_pick[str(frame.primary_arg)] = (int(obs.step), frame.agent_uid)
                self._q_at_pick[str(frame.primary_arg)] = float(obs.curr_q)
        # Expire stale picks
        expired = [
            k for k, (step, _) in self._last_pick.items()
            if obs.step - step > self.window_steps
        ]
        for k in expired:
            self._last_pick.pop(k, None)
            self._q_at_pick.pop(k, None)

        if obs.step - self._last_fired_step < self.cooldown:
            return events
        for frame in obs.frames:
            if not frame.is_place or not frame.primary_arg:
                continue
            obj = str(frame.primary_arg)
            pick_entry = self._last_pick.get(obj)
            if pick_entry is None:
                continue
            pick_step, pick_agent = pick_entry
            q_before = self._q_at_pick.get(obj, obs.curr_q)
            closes_placement = False
            if obs.target_graph is not None and frame.secondary_arg is not None:
                closes_placement = any(
                    p.matches(obj, frame.secondary_arg)
                    for p in obs.target_graph.open_placements(obs.proposition_satisfied_at)
                )
            if closes_placement:
                self._last_pick.pop(obj, None)
                self._q_at_pick.pop(obj, None)
                continue
            if float(obs.curr_q) - float(q_before) > 1e-6:
                self._last_pick.pop(obj, None)
                self._q_at_pick.pop(obj, None)
                continue
            payload = {
                "object": obj,
                "receptacle": frame.secondary_arg,
                "picked_at": int(pick_step),
                "placed_at": int(obs.step),
                "q_delta": float(obs.curr_q) - float(q_before),
            }
            events.append(
                ReboundEvent(
                    step=obs.step,
                    kind="t_d",
                    template=self.code,
                    severity=float(2.0),
                    agent_uid=frame.agent_uid or pick_agent,
                    payload=payload,
                )
            )
            events.append(
                ReboundEvent(
                    step=obs.step,
                    kind="t_r",
                    template=self.code,
                    severity=float(2.0),
                    agent_uid=frame.agent_uid or pick_agent,
                    payload=payload,
                )
            )
            self._last_fired_step = obs.step
            self._last_pick.pop(obj, None)
            self._q_at_pick.pop(obj, None)
        return events


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------
DEFAULT_TEMPLATES: Tuple[str, ...] = ("R1", "R3", "T1", "T2", "T3", "T4")


def build_default_templates(
    config: Optional[Mapping[str, Any]] = None,
) -> List[ReboundTemplate]:
    """Instantiate the default Rebound template library."""
    cfg = dict(config or {})
    return [
        NavOscillationDetector(**(cfg.get("R1") or {})),
        PlanThrashingDetector(**(cfg.get("R3") or {})),
        TargetReceptacleMismatchDetector(**(cfg.get("T1") or {})),
        RequiredStateFlipOmittedDetector(**(cfg.get("T2") or {})),
        PhaseRegionDeviationDetector(**(cfg.get("T3") or {})),
        NullPickPlaceCycleDetector(**(cfg.get("T4") or {})),
    ]


__all__ = [
    "ReboundEvent",
    "TemplateObservation",
    "ReboundTemplate",
    "NavOscillationDetector",
    "PlanThrashingDetector",
    "TargetReceptacleMismatchDetector",
    "RequiredStateFlipOmittedDetector",
    "PhaseRegionDeviationDetector",
    "NullPickPlaceCycleDetector",
    "DEFAULT_TEMPLATES",
    "build_default_templates",
]
