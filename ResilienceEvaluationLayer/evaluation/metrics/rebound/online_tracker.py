#!/usr/bin/env python3
# isort: skip_file

r"""
Online Rebound Tracker
======================

Runtime orchestrator for the Rebound event stream.  The tracker is attached
to :class:`EvaluationRunner` via ``eval_runner.rebound_tracker`` and is
invoked once per planning step from the runner's main loop.

Per-step responsibilities
-------------------------

1. Parse ``planner_info['high_level_actions']`` into
   :class:`ActionFrame` records.
2. Ask every registered :class:`ReboundTemplate` to emit
   :class:`ReboundEvent` records for the current step.
3. Maintain window state using ``kind ∈ {t_d, t_r}``.  Each ``t_d`` adds a
   semantic cause to the active window.  Proposition-grounded T1/T2 recovery
   is immediate; heuristic ``t_r`` evidence must persist for the configured
   confirmation steps before it resolves its cause.  The window closes after
   every active cause has evidence-backed recovery.  Pattern detectors may
   backdate ``t_d`` to the first transition in the detected disruption.

   Windows are **not** forced-closed on stage switches — following the
   design doc §1.3, cross-stage windows are handled at the ``W_rem``
   aggregation layer via per-stage :math:`\varphi_k` weighting (see
   :meth:`summary_for_collector`).
4. Record per-step aggregates on the writable ``planner_info`` dict so
   downstream logging (``log_rebound_analysis``, ``critic_exports``) can
   see the events without needing a separate sink:

   ``planner_info['_rebound_events']`` ← list of event dicts emitted at t
   ``planner_info['_rebound_template_codes']`` ← list of template codes at t
   ``planner_info['rebound_window_open']`` ← bool (scalar, CSV-safe)
   ``planner_info['rebound_n_events_step']`` ← int count of events at t
   ``planner_info['analysis_tags']`` ← augmented with ``"rebound_t_d"`` /
   ``"rebound_t_r"`` tags so :class:`StageBaselineEstimator` picks them up.

   The two list-valued entries use a leading-underscore key on purpose so
   that habitat-baselines' ``extract_scalars_from_info`` (which our
   per-episode CSV pipeline relies on) skips them instead of crashing on
   ``float(list)``; anything downstream that wants the raw events should
   read the underscored keys explicitly.

Episode-end responsibilities
----------------------------

At episode close :meth:`finalize` marks a still-open window as censored; it
never invents a terminal ``t_r``.  The returned summary also carries the
canonical executed-action/outcome transitions consumed by
:class:`ReboundCollector.compute_stage_crec_multidim`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .cog_load_tracker import CogLoadTracker
from .rebound_templates import (
    ReboundEvent,
    ReboundTemplate,
    TemplateObservation,
    build_default_templates,
)
from habitat_llm.evaluation.stage_baseline import StageAnchor, StageResolver
from .target_graph import (
    ActionFrame,
    EpisodeTargetGraph,
    build_episode_target_graph,
    parse_action_entry,
)
from .transition import build_rebound_transition

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Bridge helpers
# ──────────────────────────────────────────────────────────────────────
_FAULT_SEVERITY_MAP: dict = {
    "context_overflow": 1.2,
    "tool_failure_burst": 1.5,
    "immediate_error": 1.0,
    "progress_stagnation": 0.8,
    "graph_inconsistency": 0.6,
}


def _fault_severity(faults: list) -> float:
    """Return the maximum severity for a list of fault-type strings."""
    if not faults:
        return 1.0
    return max(_FAULT_SEVERITY_MAP.get(str(f), 1.0) for f in faults)


def _event_cause(ev: ReboundEvent) -> str:
    return str(ev.payload.get("cause_id") or ev.template)


@dataclass
class _PendingRecovery:
    """A cause-level recovery candidate awaiting persistence confirmation."""

    event: ReboundEvent
    evidence_step: int
    confirmations: int = 1


@dataclass
class _OpenWindow:
    """Bookkeeping for an in-flight recovery window.

    ``anchor_t_d`` / ``anchor_t_r`` store the :class:`StageAnchor` at the
    open/close steps; all intermediate anchors are stored in
    ``stage_trace`` so :math:`W_\text{rem}` can be split across stages.
    """

    t_d: int
    anchor_t_d: StageAnchor
    triggered_by: List[ReboundEvent] = field(default_factory=list)
    active_causes: Dict[str, int] = field(default_factory=dict)
    recovered_at: Dict[str, int] = field(default_factory=dict)
    recovery_evidence_at: Dict[str, int] = field(default_factory=dict)
    pending_recoveries: Dict[str, _PendingRecovery] = field(default_factory=dict)
    stage_trace: List[Tuple[int, StageAnchor, float]] = field(default_factory=list)
    severity_sum: float = 0.0
    anchor_t_d_resolution: str = "exact"

    def record_step(self, step: int, anchor: StageAnchor, q: float) -> None:
        if self.stage_trace and self.stage_trace[-1][0] == int(step):
            return
        self.stage_trace.append((int(step), anchor, float(q)))


@dataclass
class WindowSummary:
    """Compact summary of one closed recovery window handed to the collector."""

    t_d: int
    t_r: int
    anchor_t_d: StageAnchor
    anchor_t_r: StageAnchor
    template_codes: Tuple[str, ...] = ()
    severity_sum: float = 0.0
    # Per-step q series across the window (used for φ_k fractional stage
    # completion in the multi-dim W_rem aggregation).
    step_trace: Tuple[Tuple[int, str, int, str, float], ...] = ()  # (step, family, stage, modality, q)
    recovered: bool = True
    closure_reason: str = "template_recovery"
    cause_recovery_steps: Mapping[str, int] = field(default_factory=dict)
    cause_recovery_evidence_steps: Mapping[str, int] = field(default_factory=dict)
    anchor_t_d_resolution: str = "exact"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t_d": int(self.t_d),
            "t_r": int(self.t_r),
            "duration": int(self.t_r - self.t_d + 1),
            "anchor_t_d": self.anchor_t_d.to_dict(),
            "anchor_t_r": self.anchor_t_r.to_dict(),
            "template_codes": list(self.template_codes),
            "severity_sum": float(self.severity_sum),
            "step_trace": [
                {
                    "step": int(s),
                    "family": f,
                    "stage": int(st),
                    "modality": m,
                    "q": float(q),
                }
                for (s, f, st, m, q) in self.step_trace
            ],
            "recovered": bool(self.recovered),
            "closure_reason": self.closure_reason,
            "anchor_t_d_resolution": self.anchor_t_d_resolution,
            "cause_recovery_steps": {
                str(code): int(step)
                for code, step in self.cause_recovery_steps.items()
            },
            "cause_recovery_evidence_steps": {
                str(code): int(step)
                for code, step in self.cause_recovery_evidence_steps.items()
            },
        }


class OnlineReboundTracker:
    """Runtime orchestrator for Rebound events and recovery windows."""

    def __init__(
        self,
        episode: Any = None,
        templates: Optional[Sequence[ReboundTemplate]] = None,
        template_config: Optional[Mapping[str, Any]] = None,
        resolver: Optional[StageResolver] = None,
        cog_tracker: Optional[CogLoadTracker] = None,
        family_override: Optional[str] = None,
        entity_aliases: Optional[Mapping[Any, Any]] = None,
        room_aliases: Optional[Mapping[Any, Any]] = None,
        recovery_confirmation_steps: int = 2,
        immediate_recovery_templates: Optional[Sequence[str]] = None,
    ) -> None:
        self.episode = episode
        self.templates: List[ReboundTemplate] = list(
            templates if templates is not None else build_default_templates(template_config)
        )
        self.resolver: StageResolver = resolver or StageResolver()
        self.cog_tracker: CogLoadTracker = cog_tracker or CogLoadTracker()
        self.family_override = family_override
        self.entity_aliases = entity_aliases or {}
        self.room_aliases = room_aliases or {}
        self.recovery_confirmation_steps = max(
            1, int(recovery_confirmation_steps)
        )
        self.immediate_recovery_templates = {
            str(code)
            for code in (
                immediate_recovery_templates
                if immediate_recovery_templates is not None
                else ("T1", "T2")
            )
        }

        self.target_graph: Optional[EpisodeTargetGraph] = None

        self._events_per_step: List[List[ReboundEvent]] = []
        self._anchors: List[StageAnchor] = []
        self._q_series: List[float] = []
        self._p_series: List[float] = []
        self._progress_deltas: List[float] = []
        self._q_deltas: List[float] = []
        self._transitions: List[Dict[str, Any]] = []

        self._open_window: Optional[_OpenWindow] = None
        self._closed_windows: List[WindowSummary] = []

        self._prev_progress: float = 0.0
        self._prev_q: float = 0.0
        self._prev_sim_step_count: float = 0.0
        self._prev_runtime_elapsed: float = 0.0
        self._prev_task_state_success: float = 0.0

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------
    def reset(
        self,
        episode: Any = None,
        *,
        entity_aliases: Optional[Mapping[Any, Any]] = None,
        room_aliases: Optional[Mapping[Any, Any]] = None,
    ) -> None:
        if episode is not None:
            self.episode = episode
        # EnvironmentInterface rebuilds perception and its handle/name maps on
        # every episode reset. Never retain aliases from the previous scene.
        if entity_aliases is not None:
            self.entity_aliases = entity_aliases
        if room_aliases is not None:
            self.room_aliases = room_aliases
        for tpl in self.templates:
            tpl.reset()
        self.cog_tracker.reset()
        self._events_per_step.clear()
        self._anchors.clear()
        self._q_series.clear()
        self._p_series.clear()
        self._progress_deltas.clear()
        self._q_deltas.clear()
        self._transitions.clear()
        self._open_window = None
        self._closed_windows.clear()
        self._prev_progress = 0.0
        self._prev_q = 0.0
        self._prev_sim_step_count = 0.0
        self._prev_runtime_elapsed = 0.0
        self._prev_task_state_success = 0.0
        self.target_graph = None

    def ensure_target_graph(self, info: Optional[Mapping[str, Any]] = None) -> None:
        if self.target_graph is not None:
            return
        tracker_info = None
        if isinstance(info, Mapping):
            tracker_info = info.get("auto_eval_proposition_tracker")
        try:
            self.target_graph = build_episode_target_graph(
                self.episode,
                tracker_info,
                entity_aliases=self.entity_aliases,
                room_aliases=self.room_aliases,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to build EpisodeTargetGraph: %s", exc)
            self.target_graph = None

    # ------------------------------------------------------------------
    # Per-step observation
    # ------------------------------------------------------------------
    def observe(
        self,
        step: int,
        planner_info: Dict[str, Any],
        info: Mapping[str, Any],
    ) -> List[ReboundEvent]:
        """Process one step; return events emitted by templates at this step.

        Also mutates ``planner_info`` in-place to record the events and to
        augment ``analysis_tags`` so offline pipelines can replay the
        event stream.
        """
        self.ensure_target_graph(info)

        # Resolve anchor
        anchor = self.resolver.resolve(
            info,
            episode=self.episode,
            low_level_action_count=planner_info.get("low_level_action_count"),
            fallback_action=planner_info.get("high_level_actions"),
            family_override=self.family_override,
        )

        # Parse all high-level actions this step
        frames: List[ActionFrame] = []
        high_level_actions = planner_info.get("high_level_actions") or {}
        if isinstance(high_level_actions, Mapping):
            for agent_uid, act in high_level_actions.items():
                frame = parse_action_entry(act, step=int(step), agent_uid=_safe_int(agent_uid))
                if frame is not None:
                    frames.append(frame)

        # Progress / proposition bookkeeping
        curr_progress = _safe_float(info.get("task_percent_complete"), default=self._prev_progress)
        curr_q = float(self.resolver.stage_progress(info, int(anchor.stage)))
        delta_progress = curr_progress - self._prev_progress
        delta_q = curr_q - self._prev_q
        self._progress_deltas.append(float(delta_progress))
        self._q_deltas.append(float(delta_q))
        self._p_series.append(float(curr_progress))
        self._q_series.append(float(curr_q))
        self._anchors.append(anchor)

        # Pull proposition_satisfied_at for template lookups
        psa: Sequence[int] = ()
        tracker_info = info.get("auto_eval_proposition_tracker") if isinstance(info, Mapping) else None
        if isinstance(tracker_info, Mapping):
            psa_raw = tracker_info.get("proposition_satisfied_at")
            if isinstance(psa_raw, (list, tuple)):
                psa = [int(v) if v is not None else -1 for v in psa_raw]

        obs = TemplateObservation(
            step=int(step),
            planner_info=planner_info,
            info=info,
            frames=tuple(frames),
            prev_progress=float(self._prev_progress),
            curr_progress=float(curr_progress),
            prev_q=float(self._prev_q),
            curr_q=float(curr_q),
            proposition_satisfied_at=tuple(psa),
            target_graph=self.target_graph,
        )

        emitted: List[ReboundEvent] = []
        for tpl in self.templates:
            try:
                events = tpl.observe(obs)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Template %s.observe failed: %s", tpl.code, exc)
                continue
            if not events:
                continue
            emitted.extend(events)

        # Event order must not depend on template registration order.  Add all
        # newly observed causes before resolving causes that recover on the
        # same transition, so an overlapping disruption cannot be split into
        # two artificial windows merely because its template ran later.
        for ev in emitted:
            if ev.kind == "t_d":
                self._apply_event(ev, anchor, curr_q, observed_step=int(step))
        for ev in emitted:
            if ev.kind == "t_r":
                self._apply_event(ev, anchor, curr_q, observed_step=int(step))

        # Bridge causes have no dedicated semantic template.  Positive task or
        # proposition progress is their evidence-backed recovery condition.
        if self._open_window is not None and (
            delta_progress > 1.0e-6 or delta_q > 1.0e-6
        ):
            for cause in tuple(self._open_window.active_causes):
                template = cause.split(":", 1)[0]
                if template not in {"detector_bridge", "compression_bridge"}:
                    continue
                recovery = ReboundEvent(
                    step=int(step),
                    kind="t_r",
                    template=template,
                    severity=0.0,
                    payload={
                        "source": "positive_progress",
                        "cause_id": cause,
                    },
                )
                emitted.append(recovery)
                self._apply_event(
                    recovery, anchor, curr_q, observed_step=int(step)
                )

        # ── Bridge 1: Rebound Detector faults ──────────────────────────
        # planner_integration.try_execute_recovery stamps
        # ``_rebound_detector_faults`` on planner_info when faults are
        # detected on the *same* step.  Convert them to a synthetic t_d so
        # the window state machine activates even when the standard templates
        # haven't fired yet (e.g. first step of a context_overflow event).
        detector_faults: list = list(planner_info.get("_rebound_detector_faults") or [])
        if detector_faults:
            detector_already_active = bool(
                self._open_window is not None
                and "detector_bridge" in self._open_window.active_causes
            )
            if not detector_already_active:
                synth_td = ReboundEvent(
                    step=int(step),
                    kind="t_d",
                    template="detector_bridge",
                    severity=_fault_severity(detector_faults),
                    payload={
                        "faults": sorted(detector_faults),
                        "source": "planner_detector",
                        "cause_id": "detector_bridge",
                    },
                )
                emitted.append(synth_td)
                self._apply_event(
                    synth_td, anchor, curr_q, observed_step=int(step)
                )
                logger.debug(
                    "[Tracker] synthetic t_d from detector faults %s at step %d",
                    detector_faults,
                    step,
                )
            else:
                # Ongoing faults increase burden, but are not new disruption
                # edges and therefore do not create duplicate t_d events.
                self._open_window.severity_sum += _fault_severity(detector_faults)

        # ── Bridge 2: Context compression events ───────────────────────
        # evaluation_runner drains LLMPlanner._pending_compression_events
        # into ``_context_compression_events`` before calling observe.
        # Recording the event here ensures t_c is visible to C_rec.
        compression_events: list = list(planner_info.get("_context_compression_events") or [])
        _had_compression = bool(compression_events)
        if compression_events:
            compression_already_active = bool(
                self._open_window is not None
                and "compression_bridge" in self._open_window.active_causes
            )
            if not compression_already_active:
                synth_tc = ReboundEvent(
                    step=int(step),
                    kind="t_d",
                    template="compression_bridge",
                    severity=0.5,
                    payload={
                        "events": compression_events,
                        "source": "context_compression",
                        "cause_id": "compression_bridge",
                    },
                )
                emitted.append(synth_tc)
                self._apply_event(
                    synth_tc, anchor, curr_q, observed_step=int(step)
                )
                logger.debug(
                    "[Tracker] synthetic t_d from context compression at step %d", step
                )
            else:
                self._open_window.severity_sum += 0.5 * len(compression_events)

        # Heuristic recoveries are deliberately not allowed to close a window
        # on one transient observation.  The semantic T1/T2 templates are
        # immediate because their proposition evidence is already persistent;
        # all other causes must remain stable for the configured number of
        # consecutive tracker observations.
        confirmed_recoveries = self._advance_pending_recoveries(
            step=int(step),
            anchor=anchor,
            curr_q=float(curr_q),
            stable=(
                delta_progress >= -1.0e-6
                and delta_q >= -1.0e-6
                and not detector_faults
                and not compression_events
                and not any(ev.kind == "t_d" for ev in emitted)
            ),
        )
        emitted.extend(confirmed_recoveries)

        self._events_per_step.append(emitted)

        # Cognitive volumes
        self.cog_tracker.observe_step(
            step=int(step),
            planner_info=planner_info,
            modality=anchor.modality,
        )

        # Record anchor / window trace on the open window
        if self._open_window is not None:
            self._open_window.record_step(int(step), anchor, float(curr_q))

        # Mutate planner_info for downstream logging.
        planner_info["_rebound_events"] = [ev.to_dict() for ev in emitted]
        planner_info["_rebound_template_codes"] = sorted({ev.template for ev in emitted})
        planner_info["rebound_window_open"] = self._open_window is not None
        planner_info["rebound_n_events_step"] = int(len(emitted))
        tags = planner_info.get("analysis_tags")
        if tags is None:
            tags = []
        elif isinstance(tags, str):
            tags = [tags]
        else:
            tags = list(tags)
        for ev in emitted:
            if ev.kind == "t_d":
                tags.append("rebound_t_d")
            elif ev.kind == "t_r":
                tags.append(
                    "rebound_t_r_candidate"
                    if ev.payload.get("recovery_pending_confirmation")
                    else "rebound_t_r"
                )
        if _had_compression:
            tags.append("context_compression_t_c")
        planner_info["analysis_tags"] = tags

        transition = build_rebound_transition(
            step=int(step),
            planner_info=planner_info,
            info=info,
            progress=float(curr_progress),
            proposition_fraction=float(curr_q),
            previous_progress=float(self._prev_progress),
            previous_proposition_fraction=float(self._prev_q),
            previous_sim_step_count=float(self._prev_sim_step_count),
            previous_runtime_elapsed=float(self._prev_runtime_elapsed),
            previous_task_state_success=float(self._prev_task_state_success),
            template_codes=planner_info["_rebound_template_codes"],
        )
        self._transitions.append(transition)

        # Advance prev_* counters
        self._prev_progress = curr_progress
        self._prev_q = curr_q
        self._prev_sim_step_count = float(transition["sim_step_count"])
        self._prev_runtime_elapsed = float(transition["runtime_elapsed"])
        self._prev_task_state_success = float(
            transition["task_state_success"]
        )
        return emitted

    # ------------------------------------------------------------------
    # Window state machine
    # ------------------------------------------------------------------
    def _apply_event(
        self,
        ev: ReboundEvent,
        anchor: StageAnchor,
        curr_q: float,
        *,
        observed_step: int,
    ) -> None:
        if ev.kind == "t_d":
            cause = _event_cause(ev)
            if self._open_window is None:
                anchor_t_d, anchor_resolution = self._anchor_for_step(
                    int(ev.step), anchor
                )
                self._open_window = _OpenWindow(
                    t_d=int(ev.step),
                    anchor_t_d=anchor_t_d,
                    anchor_t_d_resolution=anchor_resolution,
                )
                self._backfill_open_window(int(ev.step), int(observed_step))
            elif int(ev.step) < self._open_window.t_d:
                self._open_window.t_d = int(ev.step)
                (
                    self._open_window.anchor_t_d,
                    self._open_window.anchor_t_d_resolution,
                ) = self._anchor_for_step(
                    int(ev.step), self._open_window.anchor_t_d
                )
                self._backfill_open_window(int(ev.step), int(observed_step))
            self._open_window.triggered_by.append(ev)
            self._open_window.active_causes.setdefault(
                cause, int(ev.step)
            )
            # A relapse of the same semantic cause invalidates any candidate
            # recovery which has not yet passed the persistence gate.
            self._open_window.pending_recoveries.pop(cause, None)
            self._open_window.severity_sum += float(ev.severity)
        elif ev.kind == "t_r" and self._open_window is not None:
            cause = _event_cause(ev)
            if cause not in self._open_window.active_causes:
                return

            immediate = (
                ev.template in self.immediate_recovery_templates
                or self.recovery_confirmation_steps <= 1
                or bool(ev.payload.get("recovery_confirmed"))
            )
            if immediate:
                self._resolve_cause(
                    cause=cause,
                    recovery_step=int(ev.step),
                    evidence_step=int(
                        ev.payload.get("recovery_evidence_step", ev.step)
                    ),
                    anchor=anchor,
                    severity=float(ev.severity),
                )
                return

            ev.payload["recovery_pending_confirmation"] = True
            self._open_window.pending_recoveries.setdefault(
                cause,
                _PendingRecovery(
                    event=ev,
                    evidence_step=int(ev.step),
                    confirmations=1,
                ),
            )

    def _anchor_for_step(
        self, step: int, fallback: StageAnchor
    ) -> Tuple[StageAnchor, str]:
        index = int(step) - 1
        if 0 <= index < len(self._anchors):
            candidate = self._anchors[index]
        else:
            candidate = fallback
        if int(candidate.stage) >= 0:
            return candidate, "exact"

        # Stage -1 is terminal/unknown and cannot define remaining clean work.
        # Pattern detectors may discover a disruption late and backdate t_d;
        # resolve that onset against the last non-terminal anchor at or before
        # the onset rather than dividing by a terminal W_rem close to one.
        upper = min(max(index, 0), len(self._anchors) - 1)
        for previous in reversed(self._anchors[: upper + 1]):
            if int(previous.stage) < 0:
                continue
            if candidate.family and previous.family != candidate.family:
                continue
            return previous, "previous_nonterminal"
        if int(fallback.stage) >= 0:
            return fallback, "fallback_nonterminal"
        return candidate, "invalid_terminal_or_unknown"

    def _resolve_cause(
        self,
        *,
        cause: str,
        recovery_step: int,
        evidence_step: int,
        anchor: StageAnchor,
        severity: float,
    ) -> None:
        if self._open_window is None or cause not in self._open_window.active_causes:
            return
        # Event handling precedes the normal end-of-observe record_step call.
        # When this is the final active cause, resolving it closes the window
        # immediately, so record the recovery transition before close.  The
        # window record method de-duplicates the step if t_d/t_r coincide or a
        # backfilled trace already contains it.
        self._record_open_window_step(int(recovery_step), anchor)
        self._open_window.active_causes.pop(cause, None)
        self._open_window.pending_recoveries.pop(cause, None)
        self._open_window.recovered_at[cause] = int(recovery_step)
        self._open_window.recovery_evidence_at[cause] = int(evidence_step)
        self._open_window.severity_sum += float(severity)
        if not self._open_window.active_causes:
            self._close_open_window(
                t_r=int(recovery_step),
                anchor_t_r=anchor,
                extra_severity=0.0,
                closure_reason="all_causes_recovered",
            )

    def _record_open_window_step(
        self,
        step: int,
        fallback_anchor: StageAnchor,
    ) -> None:
        """Record an event step using its historical anchor/q when available."""

        if self._open_window is None:
            return
        index = int(step) - 1
        anchor = (
            self._anchors[index]
            if 0 <= index < len(self._anchors)
            else fallback_anchor
        )
        q = (
            float(self._q_series[index])
            if 0 <= index < len(self._q_series)
            else float(self._q_series[-1])
            if self._q_series
            else 0.0
        )
        self._open_window.record_step(int(step), anchor, q)

    def _advance_pending_recoveries(
        self,
        *,
        step: int,
        anchor: StageAnchor,
        curr_q: float,
        stable: bool,
    ) -> List[ReboundEvent]:
        if self._open_window is None:
            return []
        confirmed: List[ReboundEvent] = []
        for cause, pending in tuple(self._open_window.pending_recoveries.items()):
            if int(step) <= int(pending.evidence_step):
                continue
            if cause not in self._open_window.active_causes:
                self._open_window.pending_recoveries.pop(cause, None)
                continue
            pending.confirmations = pending.confirmations + 1 if stable else 0
            if pending.confirmations < self.recovery_confirmation_steps:
                continue
            confirmation = ReboundEvent(
                step=int(step),
                kind="t_r",
                template=pending.event.template,
                severity=0.0,
                agent_uid=pending.event.agent_uid,
                payload={
                    **dict(pending.event.payload),
                    "cause_id": cause,
                    "recovery_confirmed": True,
                    "recovery_pending_confirmation": False,
                    "recovery_evidence_step": int(pending.evidence_step),
                    "recovery_confirmation_steps": int(
                        self.recovery_confirmation_steps
                    ),
                },
            )
            confirmed.append(confirmation)
            self._resolve_cause(
                cause=cause,
                recovery_step=int(step),
                evidence_step=int(pending.evidence_step),
                anchor=anchor,
                severity=0.0,
            )
        return confirmed

    def _backfill_open_window(self, start_step: int, end_step: int) -> None:
        if self._open_window is None:
            return
        existing = {step for step, _anchor, _q in self._open_window.stage_trace}
        for trace_step in range(max(1, int(start_step)), int(end_step) + 1):
            if trace_step in existing:
                continue
            index = trace_step - 1
            if 0 <= index < len(self._anchors):
                q = self._q_series[index] if index < len(self._q_series) else 0.0
                self._open_window.record_step(
                    trace_step, self._anchors[index], q
                )
        self._open_window.stage_trace.sort(key=lambda item: item[0])

    def _close_open_window(
        self,
        t_r: int,
        anchor_t_r: StageAnchor,
        extra_severity: float = 0.0,
        *,
        recovered: bool = True,
        closure_reason: str = "template_recovery",
    ) -> None:
        if self._open_window is None:
            return
        w = self._open_window
        w.severity_sum += float(extra_severity)
        template_codes = tuple(sorted({ev.template for ev in w.triggered_by}))
        step_trace = tuple(
            (s, a.family, int(a.stage), a.modality, q)
            for (s, a, q) in w.stage_trace
        )
        summary = WindowSummary(
            t_d=int(w.t_d),
            t_r=int(t_r),
            anchor_t_d=w.anchor_t_d,
            anchor_t_r=anchor_t_r,
            template_codes=template_codes,
            severity_sum=float(w.severity_sum),
            step_trace=step_trace,
            recovered=bool(recovered),
            closure_reason=str(closure_reason),
            cause_recovery_steps=dict(w.recovered_at),
            cause_recovery_evidence_steps=dict(w.recovery_evidence_at),
            anchor_t_d_resolution=str(w.anchor_t_d_resolution),
        )
        self._closed_windows.append(summary)
        self._open_window = None

    # ------------------------------------------------------------------
    # Episode-end finalisation
    # ------------------------------------------------------------------
    def finalize(self) -> Dict[str, Any]:
        """Close any lingering window and bind cognitive validities.

        Returns a payload consumable by
        :meth:`ReboundCollector.compute_stage_crec_multidim`.
        """
        last_step = len(self._events_per_step)

        # Preserve a finite diagnostic interval, but do not invent recovery:
        # without an evidence-backed t_r this terminal window is censored.
        if self._open_window is not None:
            anchor = self._anchors[-1] if self._anchors else self._open_window.anchor_t_d
            self._close_open_window(
                t_r=int(max(last_step, self._open_window.t_d)),
                anchor_t_r=anchor,
                extra_severity=0.0,
                recovered=False,
                closure_reason="episode_tail_censored",
            )

        # Bind cognitive validity
        self.cog_tracker.bind_validity(
            progress_deltas=self._progress_deltas,
            q_deltas=self._q_deltas,
        )

        return self.summary_for_collector()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def summary_for_collector(self) -> Dict[str, Any]:
        episode_id = getattr(self.episode, "episode_id", None)
        if episode_id is None and isinstance(self.episode, Mapping):
            episode_id = self.episode.get("episode_id")
        return {
            "episode_id": (
                str(episode_id) if episode_id not in (None, "") else None
            ),
            "windows": [w.to_dict() for w in self._closed_windows],
            "anchors": [a.to_dict() for a in self._anchors],
            "progress_series": list(self._p_series),
            "q_series": list(self._q_series),
            "progress_deltas": list(self._progress_deltas),
            "q_deltas": list(self._q_deltas),
            "transitions": list(self._transitions),
            "cognitive_steps": [rec.to_dict() for rec in self.cog_tracker.records()],
            "total_events": sum(len(ev_list) for ev_list in self._events_per_step),
            "template_distribution": self._template_distribution(),
            "target_graph": self.target_graph.to_dict() if self.target_graph else None,
            "window_open_final": any(
                not window.recovered for window in self._closed_windows
            ),
            "censored_window_count": sum(
                int(not window.recovered) for window in self._closed_windows
            ),
        }

    def _template_distribution(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for step_events in self._events_per_step:
            for ev in step_events:
                counts[ev.template] = counts.get(ev.template, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Accessors (for tests + runner_analysis)
    # ------------------------------------------------------------------
    @property
    def closed_windows(self) -> List[WindowSummary]:
        return list(self._closed_windows)

    @property
    def anchors(self) -> List[StageAnchor]:
        return list(self._anchors)

    @property
    def progress_deltas(self) -> List[float]:
        return list(self._progress_deltas)

    @property
    def q_deltas(self) -> List[float]:
        return list(self._q_deltas)

    @property
    def events_per_step(self) -> List[List[ReboundEvent]]:
        return self._events_per_step

    @property
    def cognitive(self) -> CogLoadTracker:
        return self.cog_tracker


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "OnlineReboundTracker",
    "WindowSummary",
]
