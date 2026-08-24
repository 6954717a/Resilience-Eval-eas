#!/usr/bin/env python3

r"""
Stage Baseline Estimator (offline)
==================================

Implements the *clean-pass* baseline estimator described in §1.2 of the
resilience design doc. Reads critic export JSONL/JSON files produced during
clean (unperturbed) evaluation runs and produces a ``stage_baseline.json``
with per-anchor nominal values:

* ``tau_star``      (:math:`\tau^*(a)`): median step duration
* ``T_bar_<mod>``   (:math:`\bar T^x(a)`): median per-step cognitive/physical
  work time for each modality ``x`` in
  ``{pln, per, coord, nav, man, redo}``
* ``delta_p_bar``   (:math:`\bar{\Delta p}(a)`): median positive delta of
  ``task_percent_complete``
* ``delta_q_bar``   (:math:`\bar{\Delta q}(a)`): median positive delta of
  ``proposition_satisfied_fraction``
* ``r_bar``         (:math:`\bar r(a)`): median normalized risk load
* ``W_rem_seg_bar`` (:math:`W_{rem}^{*,seg}(a)`): nominal stage-local remaining
  work (in SWE units), computed from the number of steps still required to
  close the current stage in the clean trajectory.

The class is intentionally standalone: it only depends on the lightweight
:class:`StageResolver` and standard library + ``numpy``. It is meant to be
run as a one-shot offline script after every clean baseline sweep.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .stage_anchor import StageAnchor, StageResolver

logger = logging.getLogger(__name__)


# Modalities tracked for the cognitive / physical work decomposition
_COGNITIVE_MODALITIES = ("pln", "per", "coord")
_PHYSICAL_MODALITIES = ("nav", "man", "redo")
ALL_WORK_MODALITIES = _COGNITIVE_MODALITIES + _PHYSICAL_MODALITIES


# ----------------------------------------------------------------------
# Per-dimension sub-baseline
# ----------------------------------------------------------------------
BASELINE_DIMENSIONS: Tuple[str, ...] = ("plan", "sim", "runtime")


@dataclass
class StageBaselineDim:
    """Per-dimension nominal values used by the multi-dim ``C_rec``.

    Each :class:`StageBaseline` carries three parallel :class:`StageBaselineDim`
    objects so that Plan-step / Sim-step / Runtime overheads can be
    compared against commensurate nominals without cross-unit mixing.

    * ``tau_bar`` — nominal per-step *work amount* in that dimension.  For
      Plan, this is typically 1 (one logical planning step per step).  For
      Sim, it is the median sim-step increment per planning step.  For
      Runtime, it is the median wall-clock elapsed per step (seconds).
    * ``W_rem_seg_bar`` — nominal stage-local remaining work in the same
      unit as ``tau_bar``.  For Plan, median number of remaining planning
      steps in the clean trajectory; for Sim, the corresponding remaining
      sim-steps; for Runtime, remaining seconds.
    """

    tau_bar: float = 1.0
    W_rem_seg_bar: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tau_bar": float(self.tau_bar),
            "W_rem_seg_bar": float(self.W_rem_seg_bar),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageBaselineDim":
        return cls(
            tau_bar=float(payload.get("tau_bar", 1.0)),
            W_rem_seg_bar=float(payload.get("W_rem_seg_bar", 1.0)),
        )


@dataclass
class StageBaseline:
    """Per-anchor baseline statistics."""

    family: str
    stage: int
    modality: str

    # Nominal step duration and work-time medians (all in the same time units)
    tau_star: float = 1.0
    T_bar: Dict[str, float] = field(default_factory=dict)

    # Positive deltas on progress / proposition fraction
    delta_p_bar: float = 1.0
    delta_q_bar: float = 1.0

    # Risk load median
    r_bar: float = 0.0

    # Nominal remaining stage-local work (in SWE units). Kept for backward
    # compatibility with callers that do not request a dimension-aware view;
    # semantically identical to ``by_dim['sim'].W_rem_seg_bar``.
    W_rem_seg_bar: float = 1.0

    # Nominal local continuation work from this disturbance anchor to clean
    # task-contract closure. This is the preferred denominator for formal
    # Rebound C_rec; the stage-local value above remains diagnostic.
    W_rem_star_swe: float = 1.0

    # Per-dimension nominals (plan / sim / runtime).
    by_dim: Dict[str, StageBaselineDim] = field(default_factory=dict)

    # Nominal channel volumes for the volume × validity cognitive work
    # model. Keys are ``cog_load_tracker.CHANNELS``.
    v_bar_channels: Dict[str, float] = field(default_factory=dict)

    # Bookkeeping
    n_samples: int = 0

    def dim(self, name: str) -> StageBaselineDim:
        if name in self.by_dim:
            return self.by_dim[name]
        # Fallbacks: sim defaults to the legacy W_rem_seg_bar;
        # plan/runtime default to unit values so they don't dominate C_rec.
        if name == "sim":
            dim = StageBaselineDim(tau_bar=max(1.0, self.tau_star), W_rem_seg_bar=self.W_rem_seg_bar)
        elif name == "plan":
            dim = StageBaselineDim(tau_bar=1.0, W_rem_seg_bar=max(1.0, self.W_rem_seg_bar))
        else:  # runtime
            dim = StageBaselineDim(tau_bar=max(1.0, self.tau_star), W_rem_seg_bar=max(1.0, self.W_rem_seg_bar))
        self.by_dim[name] = dim
        return dim

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family,
            "stage": int(self.stage),
            "modality": self.modality,
            "tau_star": float(self.tau_star),
            "T_bar": {k: float(v) for k, v in self.T_bar.items()},
            "delta_p_bar": float(self.delta_p_bar),
            "delta_q_bar": float(self.delta_q_bar),
            "r_bar": float(self.r_bar),
            "W_rem_seg_bar": float(self.W_rem_seg_bar),
            "W_rem_star_swe": float(self.W_rem_star_swe),
            "by_dim": {k: v.to_dict() for k, v in self.by_dim.items()},
            "v_bar_channels": {k: float(v) for k, v in self.v_bar_channels.items()},
            "n_samples": int(self.n_samples),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageBaseline":
        raw_dims = payload.get("by_dim") or {}
        by_dim: Dict[str, StageBaselineDim] = {}
        if isinstance(raw_dims, Mapping):
            for key, sub in raw_dims.items():
                if isinstance(sub, Mapping):
                    by_dim[str(key)] = StageBaselineDim.from_dict(sub)
        channels_payload = payload.get("v_bar_channels") or {}
        v_bar_channels: Dict[str, float] = {}
        if isinstance(channels_payload, Mapping):
            for key, val in channels_payload.items():
                try:
                    v_bar_channels[str(key)] = float(val)
                except (TypeError, ValueError):
                    continue
        return cls(
            family=str(payload.get("family", "other")),
            stage=int(payload.get("stage", -1)),
            modality=str(payload.get("modality", "planning")),
            tau_star=float(payload.get("tau_star", 1.0)),
            T_bar={k: float(v) for k, v in (payload.get("T_bar") or {}).items()},
            delta_p_bar=float(payload.get("delta_p_bar", 1.0)),
            delta_q_bar=float(payload.get("delta_q_bar", 1.0)),
            r_bar=float(payload.get("r_bar", 0.0)),
            W_rem_seg_bar=float(payload.get("W_rem_seg_bar", 1.0)),
            W_rem_star_swe=float(
                payload.get(
                    "W_rem_star_swe",
                    payload.get("W_rem_seg_bar", 1.0),
                )
            ),
            by_dim=by_dim,
            v_bar_channels=v_bar_channels,
            n_samples=int(payload.get("n_samples", 0)),
        )


def _median_positive(values: Iterable[float], default: float) -> float:
    arr = np.asarray([v for v in values if v is not None and v > 0], dtype=np.float64)
    if arr.size == 0:
        return float(default)
    return float(np.median(arr))


def _median(values: Iterable[float], default: float) -> float:
    arr = np.asarray(
        [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))],
        dtype=np.float64,
    )
    if arr.size == 0:
        return float(default)
    return float(np.median(arr))


def _get_info(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a usable ``info``-like mapping from a critic export record.

    Records may either wrap the actual info under ``info_snapshot`` or be flat.
    """
    if not isinstance(record, Mapping):
        return {}
    snap = record.get("info_snapshot")
    if isinstance(snap, Mapping):
        # Prefer the explicit snapshot but fall back to top-level fields for
        # things the flattener strips (e.g. ``task_percent_complete``).
        merged: Dict[str, Any] = dict(snap)
        for key in (
            "task_percent_complete",
            "proposition_satisfied_fraction",
            "task_state_success",
            "sim_step_count",
            "planning_step_count",
            "task_family",
        ):
            if key in record and key not in merged:
                merged[key] = record[key]
            elif key in record and merged.get(key) in (None, ""):
                merged[key] = record[key]
        return merged
    return record


def _record_iter(export_path: Path) -> Iterable[Mapping[str, Any]]:
    """Yield record dicts from a JSON or JSONL export."""
    with open(export_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    text = text.strip()
    if not text:
        return []
    if text[0] == "[":
        payload = json.loads(text)
        if isinstance(payload, list):
            yield from (r for r in payload if isinstance(r, Mapping))
            return
    if text[0] == "{":
        payload = json.loads(text)
        if isinstance(payload, Mapping):
            records = payload.get("records")
            if isinstance(records, list):
                episode_id = payload.get("episode_id")
                episode_filename = payload.get("episode_filename")
                task_family = payload.get("task_family")
                for raw in records:
                    if not isinstance(raw, Mapping):
                        continue
                    merged = dict(raw)
                    if episode_id is not None:
                        merged.setdefault("episode_id", episode_id)
                    if episode_filename is not None:
                        merged.setdefault("episode_filename", episode_filename)
                    if task_family is not None:
                        merged.setdefault("task_family", task_family)
                    yield merged
                return
            yield payload
            return
    # Fall back to JSONL
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, Mapping):
            yield obj


def _group_by_episode(records: Iterable[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        ep_id = str(rec.get("episode_id", rec.get("episode_filename", "unknown")))
        groups[ep_id].append(dict(rec))
    for ep_id in list(groups.keys()):
        groups[ep_id].sort(
            key=lambda r: (
                int(r.get("transition_index", r.get("loop_step_count", r.get("step_count", 0))) or 0)
            )
        )
    return groups


@dataclass
class _StepDerivatives:
    """Per-step derived quantities used during baseline accumulation."""

    anchor: StageAnchor
    dt_sim: float
    d_plan: float
    d_sim: float
    dt_runtime: float
    dp: float
    dq: float
    r: float
    replan_event: bool
    response_event: bool
    redo_event: bool
    remaining_stage_steps: int
    remaining_task_steps: int
    v_pln: float = 0.0
    v_per: float = 0.0
    v_coord: float = 0.0


class StageBaselineEstimator:
    """Estimate per-anchor baselines from clean critic exports."""

    def __init__(
        self,
        tau_pln: float = 0.1,
        tau_per: float = 0.1,
        tau_coord: float = 0.1,
        resolver: Optional[StageResolver] = None,
    ) -> None:
        # Per-event nominal service times for cognitive work (seconds). These
        # are overridden by the ``resilience_config.yaml``; MVP defaults are
        # conservative and unit-consistent so C_rec stays dimensionless.
        self.tau_pln = float(tau_pln)
        self.tau_per = float(tau_per)
        self.tau_coord = float(tau_coord)
        self.resolver = resolver or StageResolver()

    # ------------------------------------------------------------------
    # Core derivation
    # ------------------------------------------------------------------
    def _derive_step(
        self,
        prev: Mapping[str, Any],
        curr: Mapping[str, Any],
        episode: Any,
        family_override: Optional[str],
    ) -> _StepDerivatives:
        info_prev = _get_info(prev)
        info_curr = _get_info(curr)
        anchor = self.resolver.resolve(info_curr, episode=episode, family_override=family_override)

        # Elapsed sim/planning steps along the trajectory
        prev_sim = float(info_prev.get("sim_step_count", 0) or 0)
        curr_sim = float(info_curr.get("sim_step_count", prev_sim) or prev_sim)
        prev_plan = float(info_prev.get("planning_step_count", 0) or 0)
        curr_plan = float(info_curr.get("planning_step_count", prev_plan) or prev_plan)

        d_sim = max(0.0, curr_sim - prev_sim)
        d_plan = max(0.0, curr_plan - prev_plan)

        # Use sim-step delta as a proxy for step duration τ when wall-clock
        # timings are missing. With a unit scaling this preserves the
        # dimensional analysis of C_rec (SWE-per-step).
        dt_sim = d_sim if d_sim > 0 else 1.0

        # Progress / proposition deltas
        dp = float(info_curr.get("task_percent_complete", 0.0) or 0.0) - float(
            info_prev.get("task_percent_complete", 0.0) or 0.0
        )
        prev_stage = self.resolver.current_stage(info_prev)
        q_prev = self.resolver.stage_progress(info_prev, prev_stage)
        q_curr = self.resolver.stage_progress(info_curr, anchor.stage)
        dq = float(q_curr) - float(q_prev)

        # Risk load (normalized CBF / collision count). Best effort extraction.
        r = float(
            info_curr.get(
                "cbf_penalty_step",
                info_curr.get("cbf_penalty", info_curr.get("safety_penalty", 0.0)),
            )
            or 0.0
        )

        # Runtime (wall clock) — optional; clean exports may not carry it.
        prev_rt = float(info_prev.get("runtime_elapsed", prev.get("runtime_elapsed", 0.0)) or 0.0)
        curr_rt = float(info_curr.get("runtime_elapsed", curr.get("runtime_elapsed", prev_rt)) or prev_rt)
        dt_runtime = max(0.0, curr_rt - prev_rt) if curr_rt >= prev_rt else 0.0

        # Event-like flags from analysis_tags (written by A2CCritic)
        tags = info_curr.get("analysis_tags") or curr.get("analysis_tags") or []
        if isinstance(tags, str):
            tags = [tags]
        tag_set = {str(t) for t in tags}
        replan_event = bool(tag_set & {"replan_anchor", "planning_transition"})
        response_event = "response_event" in tag_set
        redo_event = bool(tag_set & {"retry_event", "backtrack_event"})

        # Cognitive volumes (counts per step) used by the volume × validity
        # model. Sources mirror :class:`CogLoadTracker.observe_step`.
        v_pln = 1.0 if replan_event else 0.0
        replanned = info_curr.get("replanned") or curr.get("replanned") or {}
        if isinstance(replanned, Mapping):
            v_pln += float(sum(1 for v in replanned.values() if bool(v)))
        v_per = 1.0 if response_event else 0.0
        responses = info_curr.get("responses") or curr.get("responses") or {}
        if isinstance(responses, Mapping):
            v_per += float(sum(1 for v in responses.values() if v))
        v_coord = 1.0 if anchor.modality == "coordination" else 0.0

        return _StepDerivatives(
            anchor=anchor,
            dt_sim=dt_sim,
            d_plan=d_plan,
            d_sim=d_sim,
            dt_runtime=dt_runtime,
            dp=dp,
            dq=dq,
            r=r,
            replan_event=replan_event,
            response_event=response_event,
            redo_event=redo_event,
            remaining_stage_steps=0,
            remaining_task_steps=0,
            v_pln=v_pln,
            v_per=v_per,
            v_coord=v_coord,
        )

    # ------------------------------------------------------------------
    # Stage-local remaining work computation
    # ------------------------------------------------------------------
    @staticmethod
    def _annotate_remaining_stage_steps(steps: List[_StepDerivatives]) -> None:
        """Fill ``remaining_stage_steps`` by scanning each stage run backward."""
        if not steps:
            return
        n = len(steps)
        run_end: Dict[int, int] = {}
        # Build end index (exclusive) of each contiguous stage run
        start = 0
        for i in range(1, n + 1):
            if i == n or steps[i].anchor.stage != steps[start].anchor.stage:
                for j in range(start, i):
                    run_end[j] = i
                start = i
        for i, step in enumerate(steps):
            end = run_end.get(i, i + 1)
            step.remaining_stage_steps = max(0, end - i - 1)

    @staticmethod
    def _annotate_remaining_task_steps(steps: List[_StepDerivatives]) -> None:
        """Fill clean-continuation remaining work to task-contract closure."""
        n = len(steps)
        for i, step in enumerate(steps):
            step.remaining_task_steps = max(0, n - i - 1)

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def estimate_from_episode(
        self,
        records: Sequence[Mapping[str, Any]],
        episode: Any = None,
        family_override: Optional[str] = None,
    ) -> Dict[Tuple[str, int, str], "_Accumulator"]:
        """Accumulate per-anchor samples from a single episode trajectory."""
        if len(records) < 2:
            return {}
        # derive per-step quantities
        steps: List[_StepDerivatives] = []
        for prev, curr in zip(records[:-1], records[1:]):
            steps.append(self._derive_step(prev, curr, episode, family_override))
        self._annotate_remaining_stage_steps(steps)
        self._annotate_remaining_task_steps(steps)

        accumulators: Dict[Tuple[str, int, str], _Accumulator] = defaultdict(_Accumulator)
        for step in steps:
            key = step.anchor.key()
            accumulators[key].add(step, self)
        return accumulators

    def estimate_from_exports(
        self,
        export_paths: Sequence[Path],
        family_override: Optional[str] = None,
    ) -> Dict[Tuple[str, int, str], StageBaseline]:
        """Read critic export files and produce ``(f, ℓ, m) -> StageBaseline``."""
        baselines, _by_episode, _stats = self.estimate_grouped_from_exports(
            export_paths,
            family_override=family_override,
        )
        return baselines

    def estimate_grouped_from_exports(
        self,
        export_paths: Sequence[Path],
        family_override: Optional[str] = None,
    ) -> Tuple[
        Dict[Tuple[str, int, str], StageBaseline],
        Dict[str, Dict[Tuple[str, int, str], StageBaseline]],
        Dict[str, Any],
    ]:
        """Estimate global and per-episode baselines in one export scan.

        The global hash baseline remains the precise run-specific artifact.
        The per-episode output maintains a coarser persistent cache that can be
        reused when a matching hash baseline is unavailable.
        """
        combined: Dict[Tuple[str, int, str], _Accumulator] = defaultdict(_Accumulator)
        by_episode: Dict[str, Dict[Tuple[str, int, str], _Accumulator]] = defaultdict(
            lambda: defaultdict(_Accumulator)
        )
        episode_export_counts: Dict[str, int] = defaultdict(int)
        episode_record_counts: Dict[str, int] = defaultdict(int)
        total_records = 0
        for path in export_paths:
            try:
                records = list(_record_iter(Path(path)))
            except Exception as exc:
                logger.warning("Skipping unreadable export %s: %s", path, exc)
                continue
            total_records += len(records)
            for ep_id, ep_records in _group_by_episode(records).items():
                ep_key = str(ep_id)
                episode_export_counts[ep_key] += 1
                episode_record_counts[ep_key] += len(ep_records)
                acc = self.estimate_from_episode(ep_records, family_override=family_override)
                for key, sub in acc.items():
                    combined[key].merge(sub)
                    by_episode[ep_key][key].merge(sub)
        logger.info(
            "StageBaselineEstimator processed %d critic records across %d files",
            total_records,
            len(export_paths),
        )
        combined_baselines = {key: acc.finalize(key) for key, acc in combined.items()}
        episode_baselines = {
            ep_id: {key: acc.finalize(key) for key, acc in accs.items()}
            for ep_id, accs in by_episode.items()
        }
        stats = {
            "total_records": int(total_records),
            "export_file_count": int(len(export_paths)),
            "episode_export_counts": dict(episode_export_counts),
            "episode_record_counts": dict(episode_record_counts),
        }
        return combined_baselines, episode_baselines, stats

    @staticmethod
    def merge_baseline_maps(
        existing: Mapping[Tuple[str, int, str], StageBaseline],
        incoming: Mapping[Tuple[str, int, str], StageBaseline],
    ) -> Dict[Tuple[str, int, str], StageBaseline]:
        """Merge finalized baselines with sample-count weighted summaries.

        Raw clean trajectories are not retained in the persistent episode
        cache, so this is a summary merge rather than an exact median
        recomputation.
        """

        def _clone(baseline: StageBaseline) -> StageBaseline:
            return StageBaseline.from_dict(baseline.to_dict())

        def _weight(baseline: StageBaseline) -> float:
            return float(max(1, int(getattr(baseline, "n_samples", 0) or 0)))

        def _avg(a: float, b: float, wa: float, wb: float) -> float:
            return float((float(a) * wa + float(b) * wb) / max(1.0, wa + wb))

        merged: Dict[Tuple[str, int, str], StageBaseline] = {}
        for key in sorted(set(existing.keys()) | set(incoming.keys())):
            left = existing.get(key)
            right = incoming.get(key)
            if left is None and right is not None:
                merged[key] = _clone(right)
                continue
            if right is None and left is not None:
                merged[key] = _clone(left)
                continue
            if left is None or right is None:
                continue
            wa = _weight(left)
            wb = _weight(right)
            t_bar_keys = sorted(set(left.T_bar.keys()) | set(right.T_bar.keys()))
            dim_keys = sorted(
                set(left.by_dim.keys()) | set(right.by_dim.keys()) | set(BASELINE_DIMENSIONS)
            )
            channel_keys = sorted(
                set(left.v_bar_channels.keys()) | set(right.v_bar_channels.keys())
            )
            by_dim = {
                dim: StageBaselineDim(
                    tau_bar=_avg(left.dim(dim).tau_bar, right.dim(dim).tau_bar, wa, wb),
                    W_rem_seg_bar=_avg(
                        left.dim(dim).W_rem_seg_bar,
                        right.dim(dim).W_rem_seg_bar,
                        wa,
                        wb,
                    ),
                )
                for dim in dim_keys
            }
            merged[key] = StageBaseline(
                family=right.family,
                stage=int(right.stage),
                modality=right.modality,
                tau_star=_avg(left.tau_star, right.tau_star, wa, wb),
                T_bar={
                    name: _avg(
                        left.T_bar.get(name, 0.0),
                        right.T_bar.get(name, 0.0),
                        wa,
                        wb,
                    )
                    for name in t_bar_keys
                },
                delta_p_bar=_avg(left.delta_p_bar, right.delta_p_bar, wa, wb),
                delta_q_bar=_avg(left.delta_q_bar, right.delta_q_bar, wa, wb),
                r_bar=_avg(left.r_bar, right.r_bar, wa, wb),
                W_rem_seg_bar=_avg(left.W_rem_seg_bar, right.W_rem_seg_bar, wa, wb),
                W_rem_star_swe=_avg(left.W_rem_star_swe, right.W_rem_star_swe, wa, wb),
                by_dim=by_dim,
                v_bar_channels={
                    name: _avg(
                        left.v_bar_channels.get(name, 0.0),
                        right.v_bar_channels.get(name, 0.0),
                        wa,
                        wb,
                    )
                    for name in channel_keys
                },
                n_samples=int(left.n_samples) + int(right.n_samples),
            )
        return merged

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def write(
        baselines: Mapping[Tuple[str, int, str], StageBaseline],
        output_path: Union[str, os.PathLike],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "stages": [b.to_dict() for b in baselines.values()],
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return output_path

    @staticmethod
    def load(
        path: Union[str, os.PathLike],
    ) -> Dict[Tuple[str, int, str], StageBaseline]:
        path = Path(path)
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        out: Dict[Tuple[str, int, str], StageBaseline] = {}
        for raw in payload.get("stages", []) if isinstance(payload, Mapping) else []:
            baseline = StageBaseline.from_dict(raw)
            out[(baseline.family, int(baseline.stage), baseline.modality)] = baseline
        return out


class _Accumulator:
    """Internal per-anchor sample accumulator used by the estimator."""

    __slots__ = (
        "tau",
        "T_pln",
        "T_per",
        "T_coord",
        "T_nav",
        "T_man",
        "T_redo",
        "dp_pos",
        "dq_pos",
        "risk",
        "remaining",
        "remaining_task",
        "dt_plan",
        "dt_sim",
        "dt_runtime",
        "v_pln",
        "v_per",
        "v_coord",
    )

    def __init__(self) -> None:
        self.tau: List[float] = []
        self.T_pln: List[float] = []
        self.T_per: List[float] = []
        self.T_coord: List[float] = []
        self.T_nav: List[float] = []
        self.T_man: List[float] = []
        self.T_redo: List[float] = []
        self.dp_pos: List[float] = []
        self.dq_pos: List[float] = []
        self.risk: List[float] = []
        self.remaining: List[int] = []
        self.remaining_task: List[int] = []
        self.dt_plan: List[float] = []
        self.dt_sim: List[float] = []
        self.dt_runtime: List[float] = []
        self.v_pln: List[float] = []
        self.v_per: List[float] = []
        self.v_coord: List[float] = []

    def add(self, step: _StepDerivatives, estimator: StageBaselineEstimator) -> None:
        # τ^*: step-duration sample
        self.tau.append(step.dt_sim)
        # Cognitive work (events × nominal per-event times)
        self.T_pln.append((1.0 if step.replan_event else 0.0) * estimator.tau_pln)
        self.T_per.append((1.0 if step.response_event else 0.0) * estimator.tau_per)
        # Coordination work: approximate as any planning transition in a
        # multi-agent coordination modality
        coord_weight = 1.0 if step.anchor.modality == "coordination" else 0.0
        self.T_coord.append(coord_weight * estimator.tau_coord)
        # Physical work: attribute dt_sim fully to the active modality
        if step.anchor.modality == "navigation":
            self.T_nav.append(step.dt_sim)
            self.T_man.append(0.0)
        elif step.anchor.modality == "manipulation":
            self.T_nav.append(0.0)
            self.T_man.append(step.dt_sim)
        else:
            self.T_nav.append(0.0)
            self.T_man.append(0.0)
        self.T_redo.append(step.dt_sim if step.redo_event else 0.0)
        # Progress / proposition positive deltas and risk
        self.dp_pos.append(max(0.0, step.dp))
        self.dq_pos.append(max(0.0, step.dq))
        self.risk.append(max(0.0, step.r))
        # Remaining stage-local steps (for W_rem_seg estimation)
        self.remaining.append(int(step.remaining_stage_steps))
        # Remaining clean-continuation steps to task closure (for W_rem*).
        self.remaining_task.append(int(step.remaining_task_steps))
        # Dimension-aware nominals
        self.dt_plan.append(max(1.0, step.d_plan if step.d_plan > 0 else 1.0))
        self.dt_sim.append(max(0.0, step.dt_sim))
        self.dt_runtime.append(max(0.0, step.dt_runtime))
        # Cognitive volumes
        self.v_pln.append(max(0.0, step.v_pln))
        self.v_per.append(max(0.0, step.v_per))
        self.v_coord.append(max(0.0, step.v_coord))

    def merge(self, other: "_Accumulator") -> None:
        self.tau.extend(other.tau)
        self.T_pln.extend(other.T_pln)
        self.T_per.extend(other.T_per)
        self.T_coord.extend(other.T_coord)
        self.T_nav.extend(other.T_nav)
        self.T_man.extend(other.T_man)
        self.T_redo.extend(other.T_redo)
        self.dp_pos.extend(other.dp_pos)
        self.dq_pos.extend(other.dq_pos)
        self.risk.extend(other.risk)
        self.remaining.extend(other.remaining)
        self.remaining_task.extend(other.remaining_task)
        self.dt_plan.extend(other.dt_plan)
        self.dt_sim.extend(other.dt_sim)
        self.dt_runtime.extend(other.dt_runtime)
        self.v_pln.extend(other.v_pln)
        self.v_per.extend(other.v_per)
        self.v_coord.extend(other.v_coord)

    def finalize(self, key: Tuple[str, int, str]) -> StageBaseline:
        family, stage, modality = key
        tau_star = _median_positive(self.tau, default=1.0)
        T_bar = {
            "pln": _median(self.T_pln, default=0.0),
            "per": _median(self.T_per, default=0.0),
            "coord": _median(self.T_coord, default=0.0),
            "nav": _median(self.T_nav, default=0.0),
            "man": _median(self.T_man, default=0.0),
            "redo": _median(self.T_redo, default=0.0),
        }
        delta_p_bar = _median_positive(self.dp_pos, default=1.0)
        delta_q_bar = _median_positive(self.dq_pos, default=1.0)
        r_bar = _median(self.risk, default=0.0)
        # W_rem_seg: nominal remaining stage-local work in SWE units.
        # Use the median *remaining steps* scaled by τ^* and divided by τ^*
        # (= median remaining steps) so it stays interpretable in SWE units.
        remaining_arr = np.asarray(self.remaining, dtype=np.float64)
        if remaining_arr.size == 0:
            w_rem = 1.0
            rem_median = 0.0
        else:
            rem_median = float(np.median(remaining_arr))
            w_rem = max(1.0, rem_median + 1.0)

        remaining_task_arr = np.asarray(self.remaining_task, dtype=np.float64)
        if remaining_task_arr.size == 0:
            w_rem_star = w_rem
        else:
            w_rem_star = max(1.0, float(np.median(remaining_task_arr)) + 1.0)

        # Per-dimension nominals
        def _median_pos(values: Iterable[float], default: float) -> float:
            arr = np.asarray([v for v in values if v is not None and v > 0], dtype=np.float64)
            return float(np.median(arr)) if arr.size > 0 else float(default)

        dt_plan_med = _median_pos(self.dt_plan, default=1.0)
        dt_sim_med = _median_pos(self.dt_sim, default=tau_star)
        dt_rt_med = _median_pos(self.dt_runtime, default=0.0)

        by_dim = {
            "plan": StageBaselineDim(
                tau_bar=max(1.0, dt_plan_med),
                W_rem_seg_bar=max(1.0, rem_median + 1.0),
            ),
            "sim": StageBaselineDim(
                tau_bar=max(1.0, dt_sim_med),
                W_rem_seg_bar=max(1.0, (rem_median + 1.0) * max(1.0, dt_sim_med)),
            ),
            "runtime": StageBaselineDim(
                tau_bar=max(dt_rt_med, 0.0),
                W_rem_seg_bar=max(dt_rt_med, 0.0) * max(1.0, rem_median + 1.0),
            ),
        }

        v_bar_channels = {
            "pln": _median(self.v_pln, default=0.0),
            "per": _median(self.v_per, default=0.0),
            "coord": _median(self.v_coord, default=0.0),
        }

        return StageBaseline(
            family=family,
            stage=stage,
            modality=modality,
            tau_star=tau_star,
            T_bar=T_bar,
            delta_p_bar=delta_p_bar,
            delta_q_bar=delta_q_bar,
            r_bar=r_bar,
            W_rem_seg_bar=w_rem,
            W_rem_star_swe=w_rem_star,
            by_dim=by_dim,
            v_bar_channels=v_bar_channels,
            n_samples=len(self.tau),
        )


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def _discover_export_files(root: Path) -> List[Path]:
    patterns = ("*.json", "*.jsonl")
    out: List[Path] = []
    for pat in patterns:
        out.extend(sorted(root.rglob(pat)))
    return out


def main() -> int:  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(
        description="Estimate clean stage baselines (τ*, T̄, Δ̄p, Δ̄q, r̄, W_rem^{*,seg}) from critic_exports."
    )
    parser.add_argument(
        "--exports",
        required=True,
        help="Directory containing critic_exports/*.json|jsonl from clean runs.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output stage_baseline.json",
    )
    parser.add_argument(
        "--tau-pln",
        type=float,
        default=0.1,
        help="Nominal per-event time for planning transitions (seconds).",
    )
    parser.add_argument(
        "--tau-per",
        type=float,
        default=0.1,
        help="Nominal per-event time for response events (seconds).",
    )
    parser.add_argument(
        "--tau-coord",
        type=float,
        default=0.1,
        help="Nominal per-event time for coordination events (seconds).",
    )
    parser.add_argument(
        "--family-override",
        type=str,
        default=None,
        help="Override task family for records missing the field.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    exports_root = Path(args.exports)
    if not exports_root.exists():
        logger.error("Exports directory does not exist: %s", exports_root)
        return 1
    export_paths = _discover_export_files(exports_root)
    if not export_paths:
        logger.error("No export JSON/JSONL files found under %s", exports_root)
        return 1
    estimator = StageBaselineEstimator(
        tau_pln=args.tau_pln,
        tau_per=args.tau_per,
        tau_coord=args.tau_coord,
    )
    baselines = estimator.estimate_from_exports(
        export_paths, family_override=args.family_override
    )
    out_path = estimator.write(baselines, args.output)
    logger.info("Wrote %d stage baselines -> %s", len(baselines), out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
