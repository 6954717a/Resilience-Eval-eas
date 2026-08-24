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
3. Maintain window state using ``kind ∈ {t_d, t_r}``:

   * when **no window is open** and a ``t_d`` arrives, the tracker opens
     a new :class:`_OpenWindow` anchored at the current stage/modality;
   * when a **window is open** and a ``t_r`` arrives, the window is
     closed and appended to the tracker's list of finalised windows.

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

At episode close :meth:`finalize` calls each template's ``finalize``
hook (to terminate any still-open stuck window) and returns a compact
summary that :class:`ReboundCollector.compute_stage_crec_multidim`
consumes to produce the dimension-aware ``C_rec`` fields on
:class:`ReboundMetrics`.
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
from .stage_anchor import StageAnchor, StageResolver
from .target_graph import (
    ActionFrame,
    EpisodeTargetGraph,
    build_episode_target_graph,
    parse_action_entry,
)

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
    stage_trace: List[Tuple[int, StageAnchor, float]] = field(default_factory=list)
    severity_sum: float = 0.0

    def record_step(self, step: int, anchor: StageAnchor, q: float) -> None:
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
    ) -> None:
        self.episode = episode
        self.templates: List[ReboundTemplate] = list(
            templates if templates is not None else build_default_templates(template_config)
        )
        self.resolver: StageResolver = resolver or StageResolver()
        self.cog_tracker: CogLoadTracker = cog_tracker or CogLoadTracker()
        self.family_override = family_override

        self.target_graph: Optional[EpisodeTargetGraph] = None

        self._events_per_step: List[List[ReboundEvent]] = []
        self._anchors: List[StageAnchor] = []
        self._q_series: List[float] = []
        self._p_series: List[float] = []
        self._progress_deltas: List[float] = []
        self._q_deltas: List[float] = []

        self._open_window: Optional[_OpenWindow] = None
        self._closed_windows: List[WindowSummary] = []

        self._prev_progress: float = 0.0
        self._prev_q: float = 0.0

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------
    def reset(self, episode: Any = None) -> None:
        if episode is not None:
            self.episode = episode
        for tpl in self.templates:
            tpl.reset()
        self.cog_tracker.reset()
        self._events_per_step.clear()
        self._anchors.clear()
        self._q_series.clear()
        self._p_series.clear()
        self._progress_deltas.clear()
        self._q_deltas.clear()
        self._open_window = None
        self._closed_windows.clear()
        self._prev_progress = 0.0
        self._prev_q = 0.0
        self.target_graph = None

    def ensure_target_graph(self, info: Optional[Mapping[str, Any]] = None) -> None:
        if self.target_graph is not None:
            return
        tracker_info = None
        if isinstance(info, Mapping):
            tracker_info = info.get("auto_eval_proposition_tracker")
        try:
            self.target_graph = build_episode_target_graph(self.episode, tracker_info)
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
        curr_q = _safe_float(info.get("proposition_satisfied_fraction"), default=self._prev_q)
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
            for ev in events:
                emitted.append(ev)
                self._apply_event(ev, anchor, curr_q)

        # ── Bridge 1: Rebound Detector faults ──────────────────────────
        # planner_integration.try_execute_recovery stamps
        # ``_rebound_detector_faults`` on planner_info when faults are
        # detected on the *same* step.  Convert them to a synthetic t_d so
        # the window state machine activates even when the standard templates
        # haven't fired yet (e.g. first step of a context_overflow event).
        detector_faults: list = list(planner_info.get("_rebound_detector_faults") or [])
        if detector_faults and self._open_window is None:
            synth_td = ReboundEvent(
                step=int(step),
                kind="t_d",
                template="detector_bridge",
                severity=_fault_severity(detector_faults),
                payload={"faults": sorted(detector_faults), "source": "planner_detector"},
            )
            emitted.append(synth_td)
            self._apply_event(synth_td, anchor, curr_q)
            logger.debug(
                "[Tracker] synthetic t_d from detector faults %s at step %d",
                detector_faults,
                step,
            )

        # ── Bridge 2: Context compression events ───────────────────────
        # evaluation_runner drains LLMPlanner._pending_compression_events
        # into ``_context_compression_events`` before calling observe.
        # Recording the event here ensures t_c is visible to C_rec.
        compression_events: list = list(planner_info.get("_context_compression_events") or [])
        _had_compression = bool(compression_events)
        if compression_events:
            if self._open_window is None:
                # No window open — compression itself is the onset of a
                # cognitive recovery episode.
                synth_tc = ReboundEvent(
                    step=int(step),
                    kind="t_d",
                    template="compression_bridge",
                    severity=0.5,
                    payload={"events": compression_events, "source": "context_compression"},
                )
                emitted.append(synth_tc)
                self._apply_event(synth_tc, anchor, curr_q)
                logger.debug(
                    "[Tracker] synthetic t_d from context compression at step %d", step
                )
            else:
                # Window already open — boost severity to reflect the extra
                # cognitive overhead of re-summarising history mid-recovery.
                self._open_window.severity_sum += 0.5 * len(compression_events)

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
                tags.append("rebound_t_r")
        if _had_compression:
            tags.append("context_compression_t_c")
        planner_info["analysis_tags"] = tags

        # Advance prev_* counters
        self._prev_progress = curr_progress
        self._prev_q = curr_q
        return emitted

    # ------------------------------------------------------------------
    # Window state machine
    # ------------------------------------------------------------------
    def _apply_event(
        self,
        ev: ReboundEvent,
        anchor: StageAnchor,
        curr_q: float,
    ) -> None:
        if ev.kind == "t_d":
            if self._open_window is None:
                self._open_window = _OpenWindow(
                    t_d=int(ev.step),
                    anchor_t_d=anchor,
                )
            self._open_window.triggered_by.append(ev)
            self._open_window.severity_sum += float(ev.severity)
        elif ev.kind == "t_r" and self._open_window is not None:
            self._close_open_window(
                t_r=int(ev.step),
                anchor_t_r=anchor,
                extra_severity=float(ev.severity),
            )

    def _close_open_window(
        self,
        t_r: int,
        anchor_t_r: StageAnchor,
        extra_severity: float = 0.0,
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
        last_step = len(self._events_per_step) - 1 if self._events_per_step else 0
        for tpl in self.templates:
            try:
                events = tpl.finalize(last_step=last_step)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Template %s.finalize failed: %s", tpl.code, exc)
                continue
            if not events:
                continue
            if last_step < len(self._events_per_step):
                self._events_per_step[last_step].extend(events)
            else:  # trailing sentinel row
                self._events_per_step.append(list(events))
            for ev in events:
                if ev.kind == "t_d" and self._open_window is None:
                    # Synthetic terminal stuck window (rare but defensive).
                    self._open_window = _OpenWindow(
                        t_d=int(ev.step),
                        anchor_t_d=self._anchors[-1] if self._anchors else StageAnchor(),
                    )
                    self._open_window.triggered_by.append(ev)
                    self._open_window.severity_sum += float(ev.severity)
                elif ev.kind == "t_r" and self._open_window is not None:
                    anchor = self._anchors[-1] if self._anchors else self._open_window.anchor_t_d
                    self._close_open_window(
                        t_r=int(ev.step),
                        anchor_t_r=anchor,
                        extra_severity=float(ev.severity),
                    )

        # Any residual open window gets closed at the trajectory tail.
        if self._open_window is not None:
            anchor = self._anchors[-1] if self._anchors else self._open_window.anchor_t_d
            self._close_open_window(
                t_r=int(max(last_step, self._open_window.t_d)),
                anchor_t_r=anchor,
                extra_severity=0.0,
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
        return {
            "windows": [w.to_dict() for w in self._closed_windows],
            "anchors": [a.to_dict() for a in self._anchors],
            "progress_series": list(self._p_series),
            "q_series": list(self._q_series),
            "progress_deltas": list(self._progress_deltas),
            "q_deltas": list(self._q_deltas),
            "cognitive_steps": [rec.to_dict() for rec in self.cog_tracker.records()],
            "total_events": sum(len(ev_list) for ev_list in self._events_per_step),
            "template_distribution": self._template_distribution(),
            "target_graph": self.target_graph.to_dict() if self.target_graph else None,
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
