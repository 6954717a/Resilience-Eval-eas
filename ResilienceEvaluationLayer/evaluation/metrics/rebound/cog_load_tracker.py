#!/usr/bin/env python3
# isort: skip_file

r"""
Cognitive Load Tracker
======================

Implements the **volume × validity** model for cognitive excess work in
§1.3 of the resilience design doc:

.. math::

    W_t^\text{cog} = \sum_{c \in \{pln, per, coord\}}
        \frac{\bigl(v_t^c - \bar v^c\bigr)^+}
             {\max(|\bar v^c|, 1)}
        \bigl(1 - q_t^c\bigr)

* **Volume** :math:`v_t^c` — count of channel-``c`` activations emitted at step
  ``t``.  Planning (``pln``) volume comes from ``planner_info['replanned']`` and
  the ``replan_anchor`` / ``planning_transition`` analysis tags; Perception
  (``per``) volume is the number of tool response events flagged with
  ``response_event``; Coordination (``coord``) volume is the number of
  rising-edge cross-agent hand-off actions (``handover``/``coordinate``).
  ``Wait`` and coordination-modality idle time are physical gap diagnostics.
* **Validity** :math:`q_t^c \in [0,1]` — fraction of the L-step lookahead in
  which the channel-``c`` activation *paid off* (positive ``Δp`` or ``Δq``).
  It is computed *offline* once all ``planner_infos`` are known, so the
  online tracker just records raw volumes and the collector binds them to
  validity in :meth:`bind_validity`.

The tracker lives outside :class:`OnlineReboundTracker` because it has a
distinct two-pass lifecycle (accumulate volumes online, bind validity
offline).  The online tracker keeps a reference and calls
:meth:`observe_step` per step; the offline finaliser calls
:meth:`bind_validity` to produce the per-step ``W_cog_channels`` dict
consumed by the multi-dim ``C_rec`` computation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .recovery_features import (
    COGNITIVE_CHANNELS,
    cognitive_event_keys,
    normalized_cognitive_excess,
)

logger = logging.getLogger(__name__)

CHANNELS: Tuple[str, ...] = COGNITIVE_CHANNELS


@dataclass
class _ChannelRecord:
    volume: float = 0.0
    validity: Optional[float] = None  # filled during bind_validity

    def excess(self, baseline_volume: float) -> float:
        return max(0.0, float(self.volume) - max(0.0, float(baseline_volume)))

    def weighted_excess(self, baseline_volume: float) -> float:
        q = 1.0 if self.validity is None else float(self.validity)
        return normalized_cognitive_excess(self.volume, baseline_volume, q)


@dataclass
class CognitiveStepRecord:
    """Per-step volumes and (post-hoc) validities for the three channels."""

    step: int
    channels: Dict[str, _ChannelRecord] = field(default_factory=dict)

    def get(self, channel: str) -> _ChannelRecord:
        return self.channels.setdefault(channel, _ChannelRecord())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "channels": {
                c: {
                    "volume": float(rec.volume),
                    "validity": None if rec.validity is None else float(rec.validity),
                }
                for c, rec in self.channels.items()
            },
        }


class CogLoadTracker:
    """Accumulate per-step cognitive volumes and bind validities offline."""

    def __init__(
        self,
        lookahead_L: int = 5,
        progress_eps: float = 1e-6,
    ) -> None:
        self.lookahead_L = int(lookahead_L)
        self.progress_eps = float(progress_eps)
        self._steps: List[CognitiveStepRecord] = []
        self._active_event_keys: Dict[str, set[str]] = {
            channel: set() for channel in CHANNELS
        }

    def reset(self) -> None:
        self._steps = []
        self._active_event_keys = {channel: set() for channel in CHANNELS}

    def _edge_volume(self, channel: str, active_keys: set[str]) -> float:
        """Count rising edges so persistent state is not re-counted as work."""

        previous = self._active_event_keys.get(channel, set())
        volume = float(len(active_keys - previous))
        self._active_event_keys[channel] = set(active_keys)
        return volume

    # ------------------------------------------------------------------
    # Online phase
    # ------------------------------------------------------------------
    def observe_step(
        self,
        step: int,
        planner_info: Mapping[str, Any],
        modality: str = "planning",
    ) -> CognitiveStepRecord:
        record = CognitiveStepRecord(step=int(step))
        active = cognitive_event_keys(planner_info)
        for channel in CHANNELS:
            record.get(channel).volume = self._edge_volume(
                channel,
                active[channel],
            )

        self._steps.append(record)
        return record

    # ------------------------------------------------------------------
    # Offline phase
    # ------------------------------------------------------------------
    def bind_validity(
        self,
        progress_deltas: Sequence[float],
        q_deltas: Sequence[float],
    ) -> None:
        """Attach validity :math:`q_t^c` per channel for each recorded step.

        Validity for a given step is the fraction of the next :math:`L`
        steps in which either Δp or Δq is positive.  All three channels
        share the same validity signal because, at this resolution, we
        cannot attribute specific downstream progress to a specific
        cognitive activation — the validity essentially measures whether
        the *cognitive investment at step t paid off over the next L
        steps*.
        """
        n = len(self._steps)
        m = min(len(progress_deltas), len(q_deltas))
        if n == 0 or m == 0:
            return
        # Use observation order rather than the externally visible step label.
        # Runtime transition labels are one-based, while the delta arrays are
        # zero-based; indexing by ``rec.step`` skipped the first future outcome.
        # This now matches StageBaselineEstimator's clean-pass lookahead.
        for index, rec in enumerate(self._steps):
            start = index + 1
            end = min(m, start + self.lookahead_L)
            future = range(start, end)
            if start >= end:
                valid = 0.0
            else:
                denom = end - start
                hits = sum(
                    1
                    for j in future
                    if progress_deltas[j] > self.progress_eps
                    or q_deltas[j] > self.progress_eps
                )
                valid = float(hits) / float(denom)
            for channel in CHANNELS:
                rec.get(channel).validity = valid

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------
    def step_excess(
        self,
        baseline_volume: Mapping[str, float],
    ) -> List[Dict[str, float]]:
        """Return normalized volume-by-invalidity features per channel."""
        out: List[Dict[str, float]] = []
        for rec in self._steps:
            entry: Dict[str, float] = {}
            for channel in CHANNELS:
                base_v = float(baseline_volume.get(channel, 0.0))
                entry[channel] = rec.get(channel).weighted_excess(base_v)
            entry["sum"] = float(sum(entry.values()))
            out.append(entry)
        return out

    def window_excess(
        self,
        start_step: int,
        end_step: int,
        baseline_volume: Mapping[str, float],
    ) -> Dict[str, float]:
        """Aggregate excess cognitive work across ``[start_step, end_step]``."""
        sum_channels: Dict[str, float] = {c: 0.0 for c in CHANNELS}
        for rec in self._steps:
            if rec.step < start_step or rec.step > end_step:
                continue
            for channel in CHANNELS:
                base_v = float(baseline_volume.get(channel, 0.0))
                sum_channels[channel] += rec.get(channel).weighted_excess(base_v)
        sum_channels["sum"] = float(sum(sum_channels.values()))
        return sum_channels

    def records(self) -> List[CognitiveStepRecord]:
        return list(self._steps)


__all__ = [
    "CHANNELS",
    "CognitiveStepRecord",
    "CogLoadTracker",
]
