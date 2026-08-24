#!/usr/bin/env python3

r"""
Stage Baseline Estimator (offline)
==================================

Implements the *clean-pass* baseline estimator described in §1.2 of the
resilience design doc. Reads manifest-bearing critic JSON exports produced
during clean (unperturbed) evaluation runs and produces a ``stage_baseline.json``
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
import hashlib
import logging
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from habitat_llm.evaluation.critic_export_contract import (
    load_full_stability_export,
)

from .stage_anchor import StageAnchor, StageResolver
from .stability_features import (
    STABILITY_FEATURE_NAMES,
    extract_stability_vector,
)

logger = logging.getLogger(__name__)


STAGE_BASELINE_SCHEMA_VERSION = 3
STAGE_BASELINE_ESTIMATOR_VERSION = (
    "stage_baseline_estimator_v3.4_hierarchical_components"
)
STAGE_RESOLVER_VERSION = "stage_resolver_v3"
# Kept as local identity constants to avoid importing the Rebound package while
# the evaluation-level StageBaseline package is initialized. Executable shared
# helpers are imported lazily at their call sites.
RECOVERY_FEATURE_VERSION = "recovery_features_v3"
REBOUND_FORMULA_VERSION = "rebound_c_rec_v3"
DEFAULT_FORMAL_MIN_TRAJECTORIES = 3
DEFAULT_FORMAL_MIN_TRANSITIONS = 30

# Reserved, non-runtime action modalities.  They keep the family x stage
# component pools in the same immutable artifact as the exact
# family x stage x modality anchors without pretending that either pool was an
# observed action modality.
STABILITY_POOL_MODALITY = "__stability__"
RECOVERY_POOL_MODALITY = "__recovery__"
RESERVED_POOL_MODALITIES = frozenset(
    {STABILITY_POOL_MODALITY, RECOVERY_POOL_MODALITY}
)


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
    """Per-anchor baseline statistics.

    This is the stage-scoped specialization of the repository's resilience
    statistical-reference design. GE uses the same lifecycle at a fixed
    ``task_family`` scope instead of forcing family curves into a stage key.
    """

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

    # Appendix A.2 Stability normalization over
    # [Delta V, TD residual, GAE advantage, Delta p, Delta q, replan event].
    stability_feature_names: List[str] = field(
        default_factory=lambda: list(STABILITY_FEATURE_NAMES)
    )
    stability_mean: List[float] = field(default_factory=list)
    stability_covariance: List[List[float]] = field(default_factory=list)
    stability_n_samples: int = 0

    # Eq. (2) covariance for the normalized Rebound channel vectors.
    recovery_mean: Dict[str, List[float]] = field(default_factory=dict)
    recovery_covariance: Dict[str, List[List[float]]] = field(default_factory=dict)
    recovery_covariance_n_samples: int = 0
    # Exact additive moments for recomputing OAS after persistent merges.
    # Per group: {n_samples, sum, sum_outer}.
    recovery_sufficient_statistics: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )

    # V3 anchor provenance and covariance diagnostics.  The explicit source
    # identities let persistent episode aggregates merge idempotently instead
    # of counting a rerun of the same export twice.
    n_transitions: int = 0
    n_trajectories: int = 0
    n_seeds: int = 0
    trajectory_ids: List[str] = field(default_factory=list)
    seed_ids: List[int] = field(default_factory=list)
    source_hashes: List[str] = field(default_factory=list)
    # Immutable identity of every clean trajectory contributing to this
    # episode-stage anchor.  The mapping is intentionally keyed by trajectory
    # id so persistent merges can distinguish a replay of the same export from
    # two different rollouts of the same (episode, seed) calibration cell.
    trajectory_provenance: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # One anchor-level clean-continuation summary per independent trajectory.
    remaining_work_by_trajectory: Dict[str, float] = field(default_factory=dict)
    # Formal W_rem* is estimated at episode x stage scope, independently of
    # the active modality.  JSON serialization uses string stage keys, while
    # the in-memory representation retains integer stages.
    W_rem_star_by_episode_stage: Dict[str, Dict[int, float]] = field(
        default_factory=dict
    )
    # Number of independent clean trajectories supporting each exact
    # episode x stage denominator.  Covariance support may be pooled at family
    # scope, but W_rem* is formal only when this exact cell meets the trajectory
    # threshold.
    W_rem_support_by_episode_stage: Dict[str, Dict[int, int]] = field(
        default_factory=dict
    )
    duplicate_episode_seed_trajectories: List[Dict[str, Any]] = field(
        default_factory=list
    )
    recovery_covariance_rank: Dict[str, int] = field(default_factory=dict)
    recovery_covariance_condition: Dict[str, Optional[float]] = field(
        default_factory=dict
    )
    recovery_covariance_condition_reason: Dict[str, str] = field(
        default_factory=dict
    )
    recovery_covariance_shrinkage: Dict[str, Optional[float]] = field(
        default_factory=dict
    )
    # Recovery covariance can either be supported by this exact stage or by
    # an explicitly enabled family x modality cross-stage pool.  Local
    # sufficient statistics remain local even when the runtime covariance is
    # borrowed, so later persistent merges can recompute the pool exactly.
    recovery_covariance_scope: Dict[str, str] = field(default_factory=dict)
    recovery_covariance_source: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )
    formal_support_scope: str = "exact_stage"
    formal_min_trajectories: int = DEFAULT_FORMAL_MIN_TRAJECTORIES
    formal_min_transitions: int = DEFAULT_FORMAL_MIN_TRANSITIONS
    formal_valid: bool = True
    formal_missing_reason: str = ""
    # Machine-readable proof that a non-exact formal anchor is composed only
    # from compatible, sufficiently supported components.  Runtime consumers
    # must validate this payload before accepting
    # ``hierarchical_component_pool`` as formal evidence.
    formal_component_support: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )

    # Bookkeeping
    n_samples: int = 0

    @property
    def baseline_scope(self) -> str:
        return "stage_anchor"

    @property
    def baseline_key(self) -> Dict[str, Any]:
        return {
            "task_family": self.family,
            "stage": int(self.stage),
            "modality": self.modality,
        }

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

    def w_rem_star_for_episode_stage(
        self,
        episode_id: Any,
        stage: Optional[int] = None,
    ) -> Optional[float]:
        """Return the episode-scoped clean continuation denominator.

        ``None`` is returned when the artifact does not contain that exact
        episode-stage cell; callers can then preserve N/A semantics instead of
        silently falling back to the cross-episode diagnostic scalar.
        """

        episode_values = self.W_rem_star_by_episode_stage.get(str(episode_id))
        if not isinstance(episode_values, Mapping):
            return None
        resolved_stage = int(self.stage if stage is None else stage)
        value = episode_values.get(resolved_stage)
        if value is None:
            return None
        return float(value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_scope": self.baseline_scope,
            "baseline_key": self.baseline_key,
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
            "stability_feature_names": list(self.stability_feature_names),
            "stability_mean": [float(value) for value in self.stability_mean],
            "stability_covariance": [
                [float(value) for value in row]
                for row in self.stability_covariance
            ],
            "stability_n_samples": int(self.stability_n_samples),
            "recovery_covariance": {
                str(group): [
                    [float(value) for value in row]
                    for row in covariance
                ]
                for group, covariance in self.recovery_covariance.items()
            },
            "recovery_mean": {
                str(group): [float(value) for value in mean]
                for group, mean in self.recovery_mean.items()
            },
            "recovery_covariance_n_samples": int(
                self.recovery_covariance_n_samples
            ),
            "recovery_sufficient_statistics": {
                str(group): {
                    "n_samples": int(stats.get("n_samples", 0) or 0),
                    "sum": [float(value) for value in stats.get("sum", [])],
                    "sum_outer": [
                        [float(value) for value in row]
                        for row in stats.get("sum_outer", [])
                    ],
                }
                for group, stats in self.recovery_sufficient_statistics.items()
            },
            "n_transitions": int(self.n_transitions or self.n_samples),
            "n_trajectories": int(self.n_trajectories),
            "n_seeds": int(self.n_seeds),
            "trajectory_ids": sorted({str(value) for value in self.trajectory_ids}),
            "seed_ids": sorted({int(value) for value in self.seed_ids}),
            "source_hashes": sorted({str(value) for value in self.source_hashes}),
            "trajectory_provenance": {
                str(trajectory_id): {
                    "episode_id": str(provenance.get("episode_id", "")),
                    "seed": (
                        None
                        if provenance.get("seed") is None
                        else int(provenance["seed"])
                    ),
                    "source_hash": str(provenance.get("source_hash", "")),
                }
                for trajectory_id, provenance in sorted(
                    self.trajectory_provenance.items()
                )
                if isinstance(provenance, Mapping)
            },
            "remaining_work_by_trajectory": {
                str(trajectory_id): float(value)
                for trajectory_id, value in sorted(
                    self.remaining_work_by_trajectory.items()
                )
            },
            "W_rem_star_by_episode_stage": {
                str(episode_id): {
                    str(stage): float(value)
                    for stage, value in sorted(
                        stage_values.items(), key=lambda item: int(item[0])
                    )
                }
                for episode_id, stage_values in sorted(
                    self.W_rem_star_by_episode_stage.items()
                )
            },
            "W_rem_support_by_episode_stage": {
                str(episode_id): {
                    str(stage): int(value)
                    for stage, value in sorted(
                        stage_values.items(), key=lambda item: int(item[0])
                    )
                }
                for episode_id, stage_values in sorted(
                    self.W_rem_support_by_episode_stage.items()
                )
            },
            "duplicate_episode_seed_trajectories": [
                {
                    "episode_id": str(record.get("episode_id", "")),
                    "seed": int(record["seed"]),
                    "trajectory_ids": sorted(
                        str(value) for value in record.get("trajectory_ids", [])
                    ),
                }
                for record in self.duplicate_episode_seed_trajectories
                if record.get("seed") is not None
            ],
            "recovery_covariance_rank": {
                str(group): int(value)
                for group, value in self.recovery_covariance_rank.items()
            },
            "recovery_covariance_condition": {
                str(group): (None if value is None else float(value))
                for group, value in self.recovery_covariance_condition.items()
            },
            "recovery_covariance_condition_reason": {
                str(group): str(value)
                for group, value in self.recovery_covariance_condition_reason.items()
            },
            "recovery_covariance_shrinkage": {
                str(group): (None if value is None else float(value))
                for group, value in self.recovery_covariance_shrinkage.items()
            },
            "recovery_covariance_scope": {
                str(group): str(value)
                for group, value in self.recovery_covariance_scope.items()
            },
            "recovery_covariance_source": {
                str(group): dict(value)
                for group, value in self.recovery_covariance_source.items()
                if isinstance(value, Mapping)
            },
            "formal_support_scope": str(self.formal_support_scope),
            "formal_min_trajectories": int(self.formal_min_trajectories),
            "formal_min_transitions": int(self.formal_min_transitions),
            "formal_valid": bool(self.formal_valid),
            "formal_missing_reason": str(self.formal_missing_reason or ""),
            "formal_component_support": {
                str(component): dict(support)
                for component, support in self.formal_component_support.items()
                if isinstance(support, Mapping)
            },
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
            stability_feature_names=[
                str(name)
                for name in payload.get(
                    "stability_feature_names", STABILITY_FEATURE_NAMES
                )
            ],
            stability_mean=[
                float(value) for value in payload.get("stability_mean", [])
            ],
            stability_covariance=[
                [float(value) for value in row]
                for row in payload.get("stability_covariance", [])
                if isinstance(row, (list, tuple))
            ],
            stability_n_samples=int(payload.get("stability_n_samples", 0)),
            recovery_covariance={
                str(group): [
                    [float(value) for value in row]
                    for row in covariance
                    if isinstance(row, (list, tuple))
                ]
                for group, covariance in (
                    payload.get("recovery_covariance", {}) or {}
                ).items()
                if isinstance(covariance, (list, tuple))
            },
            recovery_mean={
                str(group): [float(value) for value in mean]
                for group, mean in (
                    payload.get("recovery_mean", {}) or {}
                ).items()
                if isinstance(mean, (list, tuple))
            },
            recovery_covariance_n_samples=int(
                payload.get("recovery_covariance_n_samples", 0)
            ),
            recovery_sufficient_statistics={
                str(group): {
                    "n_samples": int(stats.get("n_samples", 0) or 0),
                    "sum": [float(value) for value in stats.get("sum", [])],
                    "sum_outer": [
                        [float(value) for value in row]
                        for row in stats.get("sum_outer", [])
                        if isinstance(row, (list, tuple))
                    ],
                }
                for group, stats in (
                    payload.get("recovery_sufficient_statistics", {}) or {}
                ).items()
                if isinstance(stats, Mapping)
            },
            n_transitions=int(
                payload.get("n_transitions", payload.get("n_samples", 0))
            ),
            n_trajectories=int(payload.get("n_trajectories", 0)),
            n_seeds=int(payload.get("n_seeds", 0)),
            trajectory_ids=[
                str(value) for value in payload.get("trajectory_ids", [])
            ],
            seed_ids=[int(value) for value in payload.get("seed_ids", [])],
            source_hashes=[
                str(value) for value in payload.get("source_hashes", [])
            ],
            trajectory_provenance={
                str(trajectory_id): {
                    "episode_id": str(provenance.get("episode_id", "")),
                    "seed": (
                        None
                        if provenance.get("seed") is None
                        else int(provenance["seed"])
                    ),
                    "source_hash": str(provenance.get("source_hash", "")),
                }
                for trajectory_id, provenance in (
                    payload.get("trajectory_provenance", {}) or {}
                ).items()
                if isinstance(provenance, Mapping)
            },
            remaining_work_by_trajectory={
                str(trajectory_id): float(value)
                for trajectory_id, value in (
                    payload.get("remaining_work_by_trajectory", {}) or {}
                ).items()
            },
            W_rem_star_by_episode_stage={
                str(episode_id): {
                    int(stage): float(value)
                    for stage, value in stage_values.items()
                }
                for episode_id, stage_values in (
                    payload.get("W_rem_star_by_episode_stage", {}) or {}
                ).items()
                if isinstance(stage_values, Mapping)
            },
            W_rem_support_by_episode_stage={
                str(episode_id): {
                    int(stage): int(value)
                    for stage, value in stage_values.items()
                }
                for episode_id, stage_values in (
                    payload.get("W_rem_support_by_episode_stage", {}) or {}
                ).items()
                if isinstance(stage_values, Mapping)
            },
            duplicate_episode_seed_trajectories=[
                {
                    "episode_id": str(record.get("episode_id", "")),
                    "seed": int(record["seed"]),
                    "trajectory_ids": sorted(
                        str(value) for value in record.get("trajectory_ids", [])
                    ),
                }
                for record in payload.get(
                    "duplicate_episode_seed_trajectories", []
                )
                if isinstance(record, Mapping) and record.get("seed") is not None
            ],
            recovery_covariance_rank={
                str(group): int(value)
                for group, value in (
                    payload.get("recovery_covariance_rank", {}) or {}
                ).items()
            },
            recovery_covariance_condition={
                str(group): (None if value is None else float(value))
                for group, value in (
                    payload.get("recovery_covariance_condition", {}) or {}
                ).items()
            },
            recovery_covariance_condition_reason={
                str(group): str(value)
                for group, value in (
                    payload.get("recovery_covariance_condition_reason", {}) or {}
                ).items()
            },
            recovery_covariance_shrinkage={
                str(group): (None if value is None else float(value))
                for group, value in (
                    payload.get("recovery_covariance_shrinkage", {}) or {}
                ).items()
            },
            recovery_covariance_scope={
                str(group): str(value)
                for group, value in (
                    payload.get("recovery_covariance_scope", {}) or {}
                ).items()
            },
            recovery_covariance_source={
                str(group): dict(value)
                for group, value in (
                    payload.get("recovery_covariance_source", {}) or {}
                ).items()
                if isinstance(value, Mapping)
            },
            formal_support_scope=str(
                payload.get("formal_support_scope", "exact_stage")
            ),
            formal_min_trajectories=int(
                payload.get(
                    "formal_min_trajectories",
                    DEFAULT_FORMAL_MIN_TRAJECTORIES,
                )
            ),
            formal_min_transitions=int(
                payload.get(
                    "formal_min_transitions",
                    DEFAULT_FORMAL_MIN_TRANSITIONS,
                )
            ),
            formal_valid=bool(payload.get("formal_valid", True)),
            formal_missing_reason=str(payload.get("formal_missing_reason", "") or ""),
            formal_component_support={
                str(component): dict(support)
                for component, support in (
                    payload.get("formal_component_support", {}) or {}
                ).items()
                if isinstance(support, Mapping)
            },
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


def _normalized_trajectory_provenance(
    provenance: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    normalized: Dict[str, Dict[str, Any]] = {}
    for trajectory_id, raw in provenance.items():
        if not isinstance(raw, Mapping):
            continue
        raw_seed = raw.get("seed")
        try:
            seed = int(raw_seed) if raw_seed is not None else None
        except (TypeError, ValueError):
            seed = None
        normalized[str(trajectory_id)] = {
            "episode_id": str(raw.get("episode_id", "")),
            "seed": seed,
            "source_hash": str(raw.get("source_hash", "")),
        }
    return normalized


def _duplicate_episode_seed_records(
    provenance: Mapping[str, Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return calibration cells represented by multiple trajectory ids."""

    by_cell: Dict[Tuple[str, int], set[str]] = defaultdict(set)
    for trajectory_id, raw in provenance.items():
        if not isinstance(raw, Mapping):
            continue
        episode_id = str(raw.get("episode_id", ""))
        raw_seed = raw.get("seed")
        if not episode_id or raw_seed is None:
            continue
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError):
            continue
        by_cell[(episode_id, seed)].add(str(trajectory_id))
    return [
        {
            "episode_id": episode_id,
            "seed": seed,
            "trajectory_ids": sorted(trajectory_ids),
        }
        for (episode_id, seed), trajectory_ids in sorted(by_cell.items())
        if len(trajectory_ids) > 1
    ]


def _recovery_sufficient_statistics(values: np.ndarray) -> Dict[str, Any]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("recovery samples must be a two-dimensional matrix")
    return {
        "n_samples": int(matrix.shape[0]),
        "sum": np.sum(matrix, axis=0).tolist(),
        "sum_outer": (matrix.T @ matrix).tolist(),
    }


def _oas_from_sufficient_statistics(
    stats: Mapping[str, Any],
) -> Tuple[List[float], List[List[float]], float, int, Optional[float]]:
    """Recompute mean and OAS covariance from exact additive moments."""

    n_samples = int(stats.get("n_samples", 0) or 0)
    vector_sum = np.asarray(stats.get("sum", []), dtype=np.float64)
    outer_sum = np.asarray(stats.get("sum_outer", []), dtype=np.float64)
    dimensions = int(vector_sum.size)
    if dimensions <= 0 or outer_sum.shape != (dimensions, dimensions):
        raise ValueError("invalid recovery sufficient-statistics shape")
    if n_samples <= 0:
        mean = np.zeros(dimensions, dtype=np.float64)
        covariance = np.zeros((dimensions, dimensions), dtype=np.float64)
        shrinkage = 1.0
    else:
        mean = vector_sum / float(n_samples)
        empirical = outer_sum / float(n_samples) - np.outer(mean, mean)
        empirical = 0.5 * (empirical + empirical.T)
        # Remove only floating-point negative noise; empirical second moments
        # are positive semidefinite by construction.
        eigenvalues, eigenvectors = np.linalg.eigh(empirical)
        if np.any(eigenvalues < 0.0):
            empirical = (
                eigenvectors
                @ np.diag(np.maximum(eigenvalues, 0.0))
                @ eigenvectors.T
            )
        if n_samples < 2:
            covariance = np.zeros((dimensions, dimensions), dtype=np.float64)
            shrinkage = 1.0
        else:
            mu = float(np.trace(empirical)) / float(dimensions)
            alpha = float(np.mean(empirical**2))
            denominator = (float(n_samples) + 1.0) * (
                alpha - (mu * mu) / float(dimensions)
            )
            if denominator <= 0.0 or not math.isfinite(denominator):
                shrinkage = 1.0
            else:
                shrinkage = min(
                    1.0,
                    max(0.0, (alpha + mu * mu) / denominator),
                )
            covariance = (1.0 - shrinkage) * empirical
            covariance.flat[:: dimensions + 1] += shrinkage * mu
    rank = int(np.linalg.matrix_rank(covariance))
    if rank < dimensions:
        condition = None
    else:
        candidate = float(np.linalg.cond(covariance))
        condition = candidate if math.isfinite(candidate) else None
    return (
        mean.tolist(),
        covariance.tolist(),
        float(shrinkage),
        rank,
        condition,
    )


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
    records = list(records)
    if any(record.get("state_kind") == "state" for record in records):
        records = [record for record in records if record.get("state_kind") == "state"]
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_outcome_status(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[bool, str]:
    """Validate that the clean trajectory reached the full task contract."""

    if not records:
        return False, "clean_trajectory_empty"
    final_info = _get_info(records[-1])
    required = (
        "task_state_success",
        "task_percent_complete",
        "proposition_satisfied_fraction",
    )
    values: Dict[str, float] = {}
    missing: List[str] = []
    invalid: List[str] = []
    for name in required:
        raw = final_info.get(name)
        if raw in (None, ""):
            missing.append(name)
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            invalid.append(name)
            continue
        if not math.isfinite(value):
            invalid.append(name)
            continue
        values[name] = value
    if missing:
        return False, "clean_outcome_missing:" + ",".join(missing)
    if invalid:
        return False, "clean_outcome_invalid:" + ",".join(invalid)
    incomplete = [name for name, value in values.items() if value < 1.0 - 1.0e-9]
    if incomplete:
        return False, "clean_task_not_complete:" + ",".join(incomplete)
    return True, ""


def _cognitive_event_keys(record: Mapping[str, Any]) -> Dict[str, set[str]]:
    """Extract active event identities for clean/runtime edge-count parity."""

    from habitat_llm.evaluation.metrics.rebound.recovery_features import (
        cognitive_event_keys,
    )

    info = _get_info(record)
    merged = dict(record)
    merged.update(info)
    return cognitive_event_keys(merged)


def _has_wait_action(record: Mapping[str, Any]) -> bool:
    """Detect the same observable Wait forms consumed by runtime features."""

    from habitat_llm.evaluation.metrics.rebound.recovery_features import (
        normalized_action_name,
    )

    info = _get_info(record)
    merged = dict(record)
    merged.update(info)
    actions = merged.get("high_level_actions") or {}
    if isinstance(actions, Mapping) and any(
        normalized_action_name(action) == "wait" for action in actions.values()
    ):
        return True
    recent_action = merged.get("recent_action")
    if recent_action not in (None, ""):
        return any(
            normalized_action_name(action) == "wait"
            for action in str(recent_action).split(";")
        )
    return False


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
    has_wait: bool
    remaining_stage_steps: int
    remaining_task_steps: int
    remaining_task_work: float = 0.0
    v_pln: float = 0.0
    v_per: float = 0.0
    v_coord: float = 0.0
    cognitive_validity: float = 1.0
    stability_vector: Optional[List[float]] = None


class StageBaselineEstimator:
    """Estimate per-anchor baselines from clean critic exports."""

    def __init__(
        self,
        tau_pln: float = 0.1,
        tau_per: float = 0.1,
        tau_coord: float = 0.1,
        resolver: Optional[StageResolver] = None,
        formal_min_trajectories: int = DEFAULT_FORMAL_MIN_TRAJECTORIES,
        formal_min_transitions: int = DEFAULT_FORMAL_MIN_TRANSITIONS,
        require_successful_clean: bool = False,
        cognitive_lookahead: int = 5,
        allow_family_covariance_shrinkage: bool = False,
    ) -> None:
        # Per-event nominal service times for cognitive work (seconds). These
        # are overridden by the ``resilience_config.yaml``; MVP defaults are
        # conservative and unit-consistent so C_rec stays dimensionless.
        self.tau_pln = float(tau_pln)
        self.tau_per = float(tau_per)
        self.tau_coord = float(tau_coord)
        self.resolver = resolver or StageResolver()
        self.formal_min_trajectories = max(1, int(formal_min_trajectories))
        self.formal_min_transitions = max(1, int(formal_min_transitions))
        self.require_successful_clean = bool(require_successful_clean)
        self.cognitive_lookahead = max(1, int(cognitive_lookahead))
        self.allow_family_covariance_shrinkage = bool(
            allow_family_covariance_shrinkage
        )

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
        previous_events = _cognitive_event_keys(prev)
        current_events = _cognitive_event_keys(curr)
        new_events = {
            channel: current_events[channel] - previous_events[channel]
            for channel in ("pln", "per", "coord")
        }
        replan_event = bool(new_events["pln"])
        response_event = bool(new_events["per"])
        redo_event = bool(tag_set & {"retry_event", "backtrack_event"})

        # Cognitive volumes (counts per step) used by the volume × validity
        # model. Sources mirror :class:`CogLoadTracker.observe_step`.
        v_pln = float(len(new_events["pln"]))
        v_per = float(len(new_events["per"]))
        v_coord = float(len(new_events["coord"]))
        stability_vector_arr = extract_stability_vector(
            prev,
            curr,
            planning_event=replan_event,
        )
        stability_vector = (
            stability_vector_arr.tolist()
            if stability_vector_arr is not None
            else None
        )

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
            has_wait=_has_wait_action(curr),
            remaining_stage_steps=0,
            remaining_task_steps=0,
            v_pln=v_pln,
            v_per=v_per,
            v_coord=v_coord,
            stability_vector=stability_vector,
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
    def _pooled_tau_by_anchor(
        trajectories: Iterable[Sequence[_StepDerivatives]],
    ) -> Dict[Tuple[str, int, str], float]:
        grouped: Dict[Tuple[str, int, str], List[float]] = defaultdict(list)
        for steps in trajectories:
            for step in steps:
                grouped[step.anchor.key()].append(max(float(step.d_sim), 1.0))
        return {
            key: _median_positive(values, default=1.0)
            for key, values in grouped.items()
        }

    @staticmethod
    def _annotate_remaining_task_steps(
        steps: List[_StepDerivatives],
        tau_by_anchor: Optional[Mapping[Tuple[str, int, str], float]] = None,
    ) -> None:
        """Fill standardized clean-continuation work to task closure.

        Grouped calibration supplies one pooled ``tau_by_anchor`` estimated
        from every accepted clean trajectory.  The optional local fallback is
        retained only for the single-trajectory public API.
        """

        if tau_by_anchor is None:
            tau_by_anchor = StageBaselineEstimator._pooled_tau_by_anchor([steps])

        running_work = 0.0
        for reverse_index, step in enumerate(reversed(steps)):
            tau_star = max(tau_by_anchor.get(step.anchor.key(), 1.0), 1.0e-12)
            running_work += max(float(step.d_sim), 1.0) / tau_star
            step.remaining_task_work = float(running_work)
            step.remaining_task_steps = int(reverse_index)

    def _prepare_episode_steps(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        episode: Any = None,
        family_override: Optional[str] = None,
    ) -> Tuple[List[Mapping[str, Any]], List[_StepDerivatives]]:
        filtered_records = list(records)
        if any(record.get("state_kind") == "state" for record in filtered_records):
            filtered_records = [
                record
                for record in filtered_records
                if record.get("state_kind") == "state"
            ]
        if len(filtered_records) < 2:
            return filtered_records, []
        steps = [
            self._derive_step(prev, curr, episode, family_override)
            for prev, curr in zip(filtered_records[:-1], filtered_records[1:])
        ]
        self._annotate_remaining_stage_steps(steps)
        self._annotate_cognitive_validity(steps)
        return filtered_records, steps

    @staticmethod
    def _trajectory_stage_remaining_work(
        steps: Sequence[_StepDerivatives],
    ) -> Dict[Tuple[str, int], float]:
        values: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        for step in steps:
            values[(step.anchor.family, int(step.anchor.stage))].append(
                float(step.remaining_task_work)
            )
        return {
            key: float(np.median(np.asarray(stage_values, dtype=np.float64)))
            for key, stage_values in values.items()
            if stage_values
        }

    def _accumulate_prepared_steps(
        self,
        steps: Sequence[_StepDerivatives],
        *,
        trajectory_id: str,
        episode_id: str,
        seed: Optional[int],
        source_hash: Optional[str],
    ) -> Dict[Tuple[str, int, str], "_Accumulator"]:
        accumulators: Dict[Tuple[str, int, str], _Accumulator] = defaultdict(
            _Accumulator
        )
        for step in steps:
            key = step.anchor.key()
            accumulators[key].add(
                step,
                self,
                trajectory_id=trajectory_id,
                episode_id=episode_id,
                seed=seed,
                source_hash=source_hash,
            )
        return accumulators

    def _annotate_cognitive_validity(self, steps: List[_StepDerivatives]) -> None:
        for index, step in enumerate(steps):
            future = steps[index + 1 : index + 1 + self.cognitive_lookahead]
            if not future:
                step.cognitive_validity = 0.0
                continue
            productive = sum(
                1 for item in future if item.dp > 1.0e-6 or item.dq > 1.0e-6
            )
            step.cognitive_validity = float(productive) / float(len(future))

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def estimate_from_episode(
        self,
        records: Sequence[Mapping[str, Any]],
        episode: Any = None,
        family_override: Optional[str] = None,
        trajectory_id: Optional[str] = None,
        seed: Optional[int] = None,
        source_hash: Optional[str] = None,
    ) -> Dict[Tuple[str, int, str], "_Accumulator"]:
        """Accumulate per-anchor samples from a single episode trajectory."""
        filtered_records, steps = self._prepare_episode_steps(
            records,
            episode=episode,
            family_override=family_override,
        )
        if not steps:
            return {}
        self._annotate_remaining_task_steps(steps)
        episode_id = str(
            filtered_records[0].get(
                "episode_id",
                filtered_records[0].get("episode_filename", "unknown"),
            )
        )
        if trajectory_id is None:
            trajectory_id = f"episode:{episode_id}"
        return self._accumulate_prepared_steps(
            steps,
            trajectory_id=str(trajectory_id),
            episode_id=episode_id,
            seed=seed,
            source_hash=source_hash,
        )

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
        source_metadata: Optional[Mapping[str, Mapping[str, Any]]] = None,
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
        prepared_trajectories: List[Dict[str, Any]] = []
        episode_export_counts: Dict[str, int] = defaultdict(int)
        episode_record_counts: Dict[str, int] = defaultdict(int)
        total_records = 0
        accepted_export_count = 0
        skipped_export_reasons: Counter[str] = Counter()
        skipped_exports: List[Dict[str, str]] = []
        source_hashes: List[str] = []
        duplicate_source_hashes: List[str] = []
        seen_source_hashes: set[str] = set()
        accepted_episode_ids: set[str] = set()
        succeeded_trajectories: List[Dict[str, Any]] = []
        excluded_trajectories: List[Dict[str, Any]] = []
        for path in export_paths:
            path = Path(path)
            resolved_key = str(path.resolve())
            provenance = dict(
                (source_metadata or {}).get(
                    resolved_key,
                    (source_metadata or {}).get(str(path), {}),
                )
                or {}
            )
            try:
                source_hash = str(provenance.get("source_hash") or _sha256_file(path))
            except OSError as exc:
                reason = f"critic_export_unreadable:{type(exc).__name__}"
                skipped_export_reasons[reason] += 1
                skipped_exports.append({"path": str(path), "reason": reason})
                continue
            if source_hash in seen_source_hashes:
                duplicate_source_hashes.append(source_hash)
                skipped_export_reasons["duplicate_source_hash"] += 1
                skipped_exports.append(
                    {"path": str(path), "reason": "duplicate_source_hash"}
                )
                continue
            seen_source_hashes.add(source_hash)
            source_hashes.append(source_hash)
            raw_seed = provenance.get("seed")
            try:
                seed = int(raw_seed) if raw_seed is not None else None
            except (TypeError, ValueError):
                seed = None
            export = load_full_stability_export(Path(path))
            if not export.valid:
                reason = export.missing_reason or "critic_export_contract_invalid"
                skipped_export_reasons[reason] += 1
                skipped_exports.append({"path": str(path), "reason": reason})
                logger.warning("Skipping invalid critic export %s: %s", path, reason)
                continue
            records = list(export.records)
            total_records += len(records)
            export_used = False
            for ep_id, ep_records in _group_by_episode(records).items():
                if len(ep_records) < 2:
                    continue
                ep_key = str(ep_id)
                outcome_valid, outcome_reason = _clean_outcome_status(ep_records)
                if self.require_successful_clean and not outcome_valid:
                    excluded_trajectories.append(
                        {
                            "episode_id": ep_key,
                            "seed": seed,
                            "source_hash": source_hash,
                            "path": str(path),
                            "reason": outcome_reason,
                        }
                    )
                    continue
                trajectory_id = f"sha256:{source_hash}:episode:{ep_key}"
                filtered_records, steps = self._prepare_episode_steps(
                    ep_records,
                    family_override=family_override,
                )
                if not steps:
                    continue
                prepared_trajectories.append(
                    {
                        "episode_id": ep_key,
                        "records": filtered_records,
                        "steps": steps,
                        "trajectory_id": trajectory_id,
                        "seed": seed,
                        "source_hash": source_hash,
                    }
                )
                export_used = True
                accepted_episode_ids.add(ep_key)
                episode_export_counts[ep_key] += 1
                episode_record_counts[ep_key] += len(filtered_records)
                succeeded_trajectories.append(
                    {
                        "episode_id": ep_key,
                        "seed": seed,
                        "source_hash": source_hash,
                        "path": str(path),
                        "trajectory_id": trajectory_id,
                    }
                )
            if export_used:
                accepted_export_count += 1
            else:
                reason = "critic_export_no_usable_episode_trajectory"
                skipped_export_reasons[reason] += 1
                skipped_exports.append({"path": str(path), "reason": reason})

        # Formal W_rem* is a two-pass estimator.  First estimate one shared
        # nominal tau*(a) from every accepted clean trajectory; only then may
        # any individual trajectory be integrated to task closure.
        pooled_tau_by_anchor = self._pooled_tau_by_anchor(
            trajectory["steps"] for trajectory in prepared_trajectories
        )
        stage_remaining_work: Dict[
            Tuple[str, int], Dict[str, float]
        ] = defaultdict(dict)
        stage_trajectory_provenance: Dict[
            Tuple[str, int], Dict[str, Dict[str, Any]]
        ] = defaultdict(dict)
        episode_stage_work_samples: Dict[
            Tuple[str, int, str], List[float]
        ] = defaultdict(list)

        for trajectory in prepared_trajectories:
            steps = trajectory["steps"]
            trajectory_id = str(trajectory["trajectory_id"])
            episode_id = str(trajectory["episode_id"])
            seed = trajectory.get("seed")
            source_hash = str(trajectory.get("source_hash") or "")
            self._annotate_remaining_task_steps(steps, pooled_tau_by_anchor)
            acc = self._accumulate_prepared_steps(
                steps,
                trajectory_id=trajectory_id,
                episode_id=episode_id,
                seed=seed,
                source_hash=source_hash,
            )
            for key, sub in acc.items():
                combined[key].merge(sub)
                by_episode[episode_id][key].merge(sub)

            provenance_record = {
                "episode_id": episode_id,
                "seed": int(seed) if seed is not None else None,
                "source_hash": source_hash,
            }
            for (family, stage), remaining_work in (
                self._trajectory_stage_remaining_work(steps).items()
            ):
                stage_key = (str(family), int(stage))
                stage_remaining_work[stage_key][trajectory_id] = float(
                    remaining_work
                )
                stage_trajectory_provenance[stage_key][trajectory_id] = dict(
                    provenance_record
                )
                episode_stage_work_samples[
                    (str(family), int(stage), episode_id)
                ].append(float(remaining_work))

        episode_stage_work: Dict[
            Tuple[str, int], Dict[str, Dict[int, float]]
        ] = defaultdict(dict)
        episode_stage_support: Dict[
            Tuple[str, int], Dict[str, Dict[int, int]]
        ] = defaultdict(dict)
        for (family, stage, episode_id), values in episode_stage_work_samples.items():
            episode_stage_work[(family, stage)][episode_id] = {
                int(stage): float(np.median(np.asarray(values, dtype=np.float64)))
            }
            episode_stage_support[(family, stage)][episode_id] = {
                int(stage): int(len(values))
            }

        def _finalize(
            key: Tuple[str, int, str],
            acc: _Accumulator,
            *,
            episode_id: Optional[str] = None,
        ) -> StageBaseline:
            family, stage, _modality = key
            stage_key = (str(family), int(stage))
            provenance = stage_trajectory_provenance.get(stage_key, {})
            remaining_work = stage_remaining_work.get(stage_key, {})
            work_by_episode = episode_stage_work.get(stage_key, {})
            support_by_episode = episode_stage_support.get(stage_key, {})
            if episode_id is not None:
                provenance = {
                    trajectory_id: record
                    for trajectory_id, record in provenance.items()
                    if str(record.get("episode_id", "")) == str(episode_id)
                }
                remaining_work = {
                    trajectory_id: value
                    for trajectory_id, value in remaining_work.items()
                    if trajectory_id in provenance
                }
                work_by_episode = {
                    str(episode_id): dict(work_by_episode[str(episode_id)])
                } if str(episode_id) in work_by_episode else {}
                support_by_episode = {
                    str(episode_id): dict(support_by_episode[str(episode_id)])
                } if str(episode_id) in support_by_episode else {}
            return acc.finalize(
                key,
                formal_min_trajectories=self.formal_min_trajectories,
                formal_min_transitions=self.formal_min_transitions,
                stage_remaining_work_by_trajectory=remaining_work,
                stage_trajectory_provenance=provenance,
                W_rem_star_by_episode_stage=work_by_episode,
                W_rem_support_by_episode_stage=support_by_episode,
            )

        logger.info(
            "StageBaselineEstimator processed %d critic records across %d/%d files",
            total_records,
            accepted_export_count,
            len(export_paths),
        )
        combined_baselines = {
            key: _finalize(key, acc)
            for key, acc in combined.items()
        }
        episode_baselines = {
            ep_id: {
                key: _finalize(key, acc, episode_id=ep_id)
                for key, acc in accs.items()
            }
            for ep_id, accs in by_episode.items()
        }
        hierarchical_component_promotions = self._materialize_hierarchical_baselines(
            combined_baselines,
            combined,
            _finalize,
        )
        for episode_id, episode_map in episode_baselines.items():
            hierarchical_component_promotions += (
                self._materialize_hierarchical_baselines(
                    episode_map,
                    by_episode[episode_id],
                    lambda key, acc, episode_id=episode_id: _finalize(
                        key,
                        acc,
                        episode_id=episode_id,
                    ),
                )
            )
        family_covariance_shrinkage_applied = 0
        if self.allow_family_covariance_shrinkage:
            family_pools = self._build_family_covariance_pools(combined_baselines)
            family_covariance_shrinkage_applied += (
                self._apply_family_covariance_shrinkage(
                    combined_baselines,
                    family_pools,
                )
            )
            for episode_map in episode_baselines.values():
                family_covariance_shrinkage_applied += (
                    self._apply_family_covariance_shrinkage(
                        episode_map,
                        family_pools,
                    )
                )
        stats = {
            "total_records": int(total_records),
            "export_file_count": int(len(export_paths)),
            "accepted_export_file_count": int(accepted_export_count),
            "skipped_export_file_count": int(
                len(export_paths) - accepted_export_count
            ),
            "skipped_export_reasons": dict(skipped_export_reasons),
            "skipped_exports": skipped_exports,
            # Explicit aliases make the fail-closed status self-describing in
            # persisted estimator metadata and downstream diagnostics.
            "invalid_export_file_count": int(
                len(export_paths) - accepted_export_count
            ),
            "invalid_export_reasons": dict(skipped_export_reasons),
            "invalid_exports": skipped_exports,
            "episode_export_counts": dict(episode_export_counts),
            "episode_record_counts": dict(episode_record_counts),
            "accepted_episode_ids": sorted(accepted_episode_ids),
            "succeeded_trajectories": succeeded_trajectories,
            "excluded_trajectories": excluded_trajectories,
            "source_hashes": sorted(source_hashes),
            "duplicate_source_hashes": sorted(set(duplicate_source_hashes)),
            "pooled_tau_by_anchor": [
                {
                    "family": family,
                    "stage": int(stage),
                    "modality": modality,
                    "tau_star": float(value),
                }
                for (family, stage, modality), value in sorted(
                    pooled_tau_by_anchor.items()
                )
            ],
            "duplicate_episode_seed_trajectories": (
                _duplicate_episode_seed_records(
                    {
                        str(trajectory["trajectory_id"]): {
                            "episode_id": str(trajectory["episode_id"]),
                            "seed": trajectory.get("seed"),
                            "source_hash": str(
                                trajectory.get("source_hash") or ""
                            ),
                        }
                        for trajectory in prepared_trajectories
                    }
                )
            ),
            "family_covariance_shrinkage_enabled": bool(
                self.allow_family_covariance_shrinkage
            ),
            "family_covariance_shrinkage_applied_anchor_count": int(
                family_covariance_shrinkage_applied
            ),
            "hierarchical_component_promoted_anchor_count": int(
                hierarchical_component_promotions
            ),
            "stability_pool_modality": STABILITY_POOL_MODALITY,
            "recovery_pool_modality": RECOVERY_POOL_MODALITY,
            "feature_version": RECOVERY_FEATURE_VERSION,
            "estimator_version": STAGE_BASELINE_ESTIMATOR_VERSION,
        }
        return combined_baselines, episode_baselines, stats

    @staticmethod
    def _build_stage_pool_accumulators(
        accumulators: Mapping[Tuple[str, int, str], "_Accumulator"],
    ) -> Dict[Tuple[str, int], "_Accumulator"]:
        """Pool every observed action modality within each family x stage.

        Each transition belongs to exactly one exact modality accumulator, so
        merging the exact accumulators neither duplicates nor drops records.
        The reserved component pools are materialized only after this merge.
        """

        pools: Dict[Tuple[str, int], _Accumulator] = defaultdict(_Accumulator)
        for (family, stage, modality), accumulator in accumulators.items():
            if str(modality) in RESERVED_POOL_MODALITIES:
                continue
            pools[(str(family), int(stage))].merge(accumulator)
        return dict(pools)

    @staticmethod
    def _remaining_work_component_support(
        baseline: StageBaseline,
    ) -> Dict[str, Any]:
        cells: List[Dict[str, Any]] = []
        for episode_id, stage_values in sorted(
            baseline.W_rem_star_by_episode_stage.items()
        ):
            if int(baseline.stage) not in stage_values:
                continue
            support_values = baseline.W_rem_support_by_episode_stage.get(
                str(episode_id), {}
            )
            support = int(support_values.get(int(baseline.stage), 0) or 0)
            cells.append(
                {
                    "episode_id": str(episode_id),
                    "stage": int(baseline.stage),
                    "n_trajectories": support,
                    "valid": support >= int(baseline.formal_min_trajectories),
                }
            )
        valid_episode_ids = [
            str(cell["episode_id"]) for cell in cells if bool(cell["valid"])
        ]
        invalid_episode_ids = [
            str(cell["episode_id"]) for cell in cells if not bool(cell["valid"])
        ]
        return {
            "scope": "episode_stage_exact",
            # Artifact-level component support means at least one episode can
            # be evaluated formally.  Runtime remains episode-strict through
            # w_rem_star_for_episode_stage(), which rejects every under-
            # supported cell independently.
            "valid": bool(valid_episode_ids),
            "min_trajectories": int(baseline.formal_min_trajectories),
            "valid_episode_ids": valid_episode_ids,
            "invalid_episode_ids": invalid_episode_ids,
            "cells": cells,
        }

    @staticmethod
    def _recovery_normalizer_payload(
        baseline: StageBaseline,
    ) -> Dict[str, Any]:
        return {
            "v_bar_channels": {
                str(name): float(value)
                for name, value in baseline.v_bar_channels.items()
            },
            "delta_p_bar": float(baseline.delta_p_bar),
            "delta_q_bar": float(baseline.delta_q_bar),
            "r_bar": float(baseline.r_bar),
            "plan_tau_bar": float(baseline.dim("plan").tau_bar),
            "sim_tau_bar": float(baseline.dim("sim").tau_bar),
        }

    @classmethod
    def _configure_stage_pool_baseline(
        cls,
        baseline: StageBaseline,
        *,
        pool_kind: str,
    ) -> None:
        """Apply strict, component-specific formal support to a stage pool."""

        remaining_work = cls._remaining_work_component_support(baseline)
        reasons = [
            reason
            for reason in str(baseline.formal_missing_reason or "").split(";")
            if reason
        ]
        if not bool(remaining_work["valid"]):
            reasons.append("insufficient_episode_stage_remaining_work_support")

        if pool_kind == "stability":
            sample_count = int(baseline.stability_n_samples)
            component_name = "stability"
            support_scope = "family_stage_stability_pool"
            source_modality = STABILITY_POOL_MODALITY
            if sample_count < int(baseline.formal_min_transitions):
                reasons.append("insufficient_stability_transitions")
        elif pool_kind == "recovery":
            sample_count = int(baseline.recovery_covariance_n_samples)
            component_name = "recovery"
            support_scope = "family_stage_recovery_pool"
            source_modality = RECOVERY_POOL_MODALITY
            if sample_count < int(baseline.formal_min_transitions):
                reasons.append("insufficient_recovery_transitions")
        else:  # pragma: no cover - internal programming error
            raise ValueError(f"unsupported StageBaseline pool kind: {pool_kind}")

        component = {
            "scope": support_scope,
            "valid": not reasons,
            "source_modality": source_modality,
            "family": str(baseline.family),
            "stage": int(baseline.stage),
            "n_trajectories": int(baseline.n_trajectories),
            "n_transitions": sample_count,
            "min_trajectories": int(baseline.formal_min_trajectories),
            "min_transitions": int(baseline.formal_min_transitions),
        }
        if pool_kind == "recovery":
            component["coordinate_system"] = "family_stage_pooled_normalizer"
            component["normalizers"] = cls._recovery_normalizer_payload(baseline)
            for group in ("cog", "phy", "debt"):
                baseline.recovery_covariance_scope[group] = "family_stage_pool"
                baseline.recovery_covariance_source[group] = {
                    "kind": "family_stage_pool",
                    "source_modality": RECOVERY_POOL_MODALITY,
                    "family": str(baseline.family),
                    "stage": int(baseline.stage),
                    "n_samples": sample_count,
                    "n_trajectories": int(baseline.n_trajectories),
                    "min_trajectories": int(baseline.formal_min_trajectories),
                    "min_transitions": int(baseline.formal_min_transitions),
                    "coordinate_system": "family_stage_pooled_normalizer",
                    "normalizers": cls._recovery_normalizer_payload(baseline),
                }

        baseline.formal_component_support = {
            "remaining_work": remaining_work,
            component_name: component,
            "tau": {
                "scope": "family_stage_pool",
                "valid": math.isfinite(float(baseline.tau_star))
                and float(baseline.tau_star) > 0.0,
                "source_modality": source_modality,
            },
        }
        # Terminal/unknown pools remain visible for audit but can never become
        # a formal runtime anchor.
        if int(baseline.stage) < 0 and "terminal_or_unknown_stage" not in reasons:
            reasons.append("terminal_or_unknown_stage")
        reasons = list(dict.fromkeys(reasons))
        baseline.formal_valid = not reasons
        baseline.formal_support_scope = (
            support_scope if baseline.formal_valid else "diagnostic"
        )
        baseline.formal_missing_reason = ";".join(reasons)
        component["valid"] = bool(baseline.formal_valid)

    @classmethod
    def _apply_hierarchical_component_support(
        cls,
        baselines: Mapping[Tuple[str, int, str], StageBaseline],
    ) -> int:
        """Promote sparse exact anchors using auditable same-stage components.

        The exact modality service time and episode x stage denominator remain
        local.  Recovery features, their normalizers, mean, and covariance are
        borrowed together from the same family x stage pool so runtime vectors
        stay in the coordinate system in which the covariance was fitted.
        """

        allowed_local_reasons = {
            "insufficient_independent_trajectories",
            "insufficient_transitions",
        }
        applied = 0
        for key, baseline in baselines.items():
            if (
                str(baseline.modality) in RESERVED_POOL_MODALITIES
                or baseline.formal_valid
                or int(baseline.stage) < 0
            ):
                continue
            reasons = {
                reason
                for reason in str(baseline.formal_missing_reason or "").split(";")
                if reason
            }
            if not reasons or not reasons.issubset(allowed_local_reasons):
                continue
            pool = baselines.get(
                (str(baseline.family), int(baseline.stage), RECOVERY_POOL_MODALITY)
            )
            if (
                not isinstance(pool, StageBaseline)
                or not pool.formal_valid
                or pool.formal_support_scope != "family_stage_recovery_pool"
            ):
                continue
            remaining_work = cls._remaining_work_component_support(baseline)
            if not bool(remaining_work["valid"]):
                continue

            # Copy the complete recovery coordinate system, not covariance
            # alone.  tau_star remains exact because it is the service-time
            # normalization in the outer C_rec integral, not a recovery-vector
            # feature normalizer.
            local_normalizers = cls._recovery_normalizer_payload(baseline)
            baseline.v_bar_channels = dict(pool.v_bar_channels)
            baseline.delta_p_bar = float(pool.delta_p_bar)
            baseline.delta_q_bar = float(pool.delta_q_bar)
            baseline.r_bar = float(pool.r_bar)
            for dimension in ("plan", "sim"):
                local_dim = baseline.dim(dimension)
                pooled_dim = pool.dim(dimension)
                baseline.by_dim[dimension] = StageBaselineDim(
                    tau_bar=float(pooled_dim.tau_bar),
                    W_rem_seg_bar=float(local_dim.W_rem_seg_bar),
                )
            baseline.recovery_mean = {
                group: list(values)
                for group, values in pool.recovery_mean.items()
            }
            baseline.recovery_covariance = {
                group: [list(row) for row in covariance]
                for group, covariance in pool.recovery_covariance.items()
            }
            baseline.recovery_covariance_n_samples = int(
                pool.recovery_covariance_n_samples
            )
            baseline.recovery_covariance_rank = dict(
                pool.recovery_covariance_rank
            )
            baseline.recovery_covariance_condition = dict(
                pool.recovery_covariance_condition
            )
            baseline.recovery_covariance_condition_reason = dict(
                pool.recovery_covariance_condition_reason
            )
            baseline.recovery_covariance_shrinkage = dict(
                pool.recovery_covariance_shrinkage
            )
            normalizers = cls._recovery_normalizer_payload(pool)
            for group in ("cog", "phy", "debt"):
                baseline.recovery_covariance_scope[group] = "family_stage_pool"
                baseline.recovery_covariance_source[group] = {
                    "kind": "family_stage_pool",
                    "source_modality": RECOVERY_POOL_MODALITY,
                    "family": str(baseline.family),
                    "stage": int(baseline.stage),
                    "n_samples": int(pool.recovery_covariance_n_samples),
                    "n_trajectories": int(pool.n_trajectories),
                    "min_trajectories": int(baseline.formal_min_trajectories),
                    "min_transitions": int(baseline.formal_min_transitions),
                    "coordinate_system": "family_stage_pooled_normalizer",
                    "normalizers": normalizers,
                    "local_diagnostic_normalizers": local_normalizers,
                    "local_sufficient_statistics_retained": True,
                    "rescued_local_reasons": sorted(reasons),
                }
            baseline.formal_component_support = {
                "remaining_work": remaining_work,
                "recovery": {
                    "scope": "family_stage_pool",
                    "valid": True,
                    "source_modality": RECOVERY_POOL_MODALITY,
                    "family": str(baseline.family),
                    "stage": int(baseline.stage),
                    "n_trajectories": int(pool.n_trajectories),
                    "n_transitions": int(pool.recovery_covariance_n_samples),
                    "min_trajectories": int(baseline.formal_min_trajectories),
                    "min_transitions": int(baseline.formal_min_transitions),
                    "coordinate_system": "family_stage_pooled_normalizer",
                    "local_diagnostic": {
                        "n_trajectories": int(baseline.n_trajectories),
                        "n_transitions": int(baseline.n_transitions),
                        "normalizers": local_normalizers,
                        "sufficient_statistics_retained": True,
                    },
                },
                "tau": {
                    "scope": "exact_stage_modality",
                    "valid": math.isfinite(float(baseline.tau_star))
                    and float(baseline.tau_star) > 0.0,
                    "source_modality": str(baseline.modality),
                },
            }
            baseline.formal_support_scope = "hierarchical_component_pool"
            baseline.formal_valid = True
            baseline.formal_missing_reason = ""
            applied += 1
        return applied

    @classmethod
    def _materialize_hierarchical_baselines(
        cls,
        baselines: Dict[Tuple[str, int, str], StageBaseline],
        exact_accumulators: Mapping[Tuple[str, int, str], "_Accumulator"],
        finalize: Any,
    ) -> int:
        stage_pools = cls._build_stage_pool_accumulators(exact_accumulators)
        for (family, stage), accumulator in stage_pools.items():
            stability_key = (family, int(stage), STABILITY_POOL_MODALITY)
            stability = finalize(stability_key, accumulator)
            cls._configure_stage_pool_baseline(stability, pool_kind="stability")
            baselines[stability_key] = stability

            recovery_key = (family, int(stage), RECOVERY_POOL_MODALITY)
            recovery = finalize(recovery_key, accumulator)
            cls._configure_stage_pool_baseline(recovery, pool_kind="recovery")
            baselines[recovery_key] = recovery
        return cls._apply_hierarchical_component_support(baselines)

    @staticmethod
    def _build_family_covariance_pools(
        baselines: Mapping[Tuple[str, int, str], StageBaseline],
    ) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
        """Pool local normalized moments across non-terminal stages."""

        raw_pools: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for baseline in baselines.values():
            if (
                int(baseline.stage) < 0
                or str(baseline.modality) in RESERVED_POOL_MODALITIES
            ):
                continue
            provenance = _normalized_trajectory_provenance(
                baseline.trajectory_provenance
            )
            for group in ("cog", "phy", "debt"):
                raw = baseline.recovery_sufficient_statistics.get(group)
                if not isinstance(raw, Mapping):
                    continue
                n_samples = int(raw.get("n_samples", 0) or 0)
                vector_sum = np.asarray(raw.get("sum", []), dtype=np.float64)
                outer_sum = np.asarray(raw.get("sum_outer", []), dtype=np.float64)
                if (
                    n_samples <= 0
                    or vector_sum.size <= 0
                    or outer_sum.shape != (vector_sum.size, vector_sum.size)
                ):
                    continue
                key = (baseline.family, baseline.modality, group)
                pool = raw_pools.setdefault(
                    key,
                    {
                        "n_samples": 0,
                        "sum": np.zeros_like(vector_sum),
                        "sum_outer": np.zeros_like(outer_sum),
                        "trajectory_ids": set(),
                        "trajectory_provenance": {},
                        "stages": set(),
                        "shape_valid": True,
                    },
                )
                if pool["sum"].shape != vector_sum.shape:
                    pool["shape_valid"] = False
                    continue
                pool["n_samples"] += n_samples
                pool["sum"] += vector_sum
                pool["sum_outer"] += outer_sum
                pool["stages"].add(int(baseline.stage))
                for trajectory_id in baseline.trajectory_ids:
                    trajectory_id = str(trajectory_id)
                    pool["trajectory_ids"].add(trajectory_id)
                    if trajectory_id in provenance:
                        existing = pool["trajectory_provenance"].get(trajectory_id)
                        if existing is not None and existing != provenance[trajectory_id]:
                            pool["shape_valid"] = False
                        pool["trajectory_provenance"][trajectory_id] = dict(
                            provenance[trajectory_id]
                        )

        pools: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for key, raw in raw_pools.items():
            trajectory_ids = set(raw["trajectory_ids"])
            provenance = dict(raw["trajectory_provenance"])
            duplicate_cells = _duplicate_episode_seed_records(provenance)
            valid = bool(
                raw["shape_valid"]
                and int(raw["n_samples"]) >= DEFAULT_FORMAL_MIN_TRANSITIONS
                and len(trajectory_ids) >= DEFAULT_FORMAL_MIN_TRAJECTORIES
                and not (trajectory_ids - set(provenance))
                and not duplicate_cells
            )
            stats = {
                "n_samples": int(raw["n_samples"]),
                "sum": raw["sum"].tolist(),
                "sum_outer": raw["sum_outer"].tolist(),
            }
            if valid:
                mean, covariance, shrinkage, rank, condition = (
                    _oas_from_sufficient_statistics(stats)
                )
            else:
                mean, covariance, shrinkage, rank, condition = (
                    [],
                    [],
                    None,
                    0,
                    None,
                )
            pools[key] = {
                "valid": valid,
                "stats": stats,
                "mean": mean,
                "covariance": covariance,
                "shrinkage": shrinkage,
                "rank": rank,
                "condition": condition,
                "n_trajectories": len(trajectory_ids),
                "stages": sorted(int(value) for value in raw["stages"]),
                "duplicate_episode_seed_trajectories": duplicate_cells,
            }
        return pools

    @staticmethod
    def _apply_family_covariance_shrinkage(
        baselines: Mapping[Tuple[str, int, str], StageBaseline],
        pools: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    ) -> int:
        """Rescue only locally under-sampled exact non-terminal anchors."""

        allowed_local_reasons = {
            "insufficient_independent_trajectories",
            "insufficient_transitions",
        }
        applied = 0
        for baseline in baselines.values():
            reasons = {
                reason
                for reason in baseline.formal_missing_reason.split(";")
                if reason
            }
            if (
                baseline.formal_valid
                or int(baseline.stage) < 0
                or str(baseline.modality) in RESERVED_POOL_MODALITIES
                or not reasons
                or not reasons.issubset(allowed_local_reasons)
            ):
                continue
            provenance = _normalized_trajectory_provenance(
                baseline.trajectory_provenance
            )
            if (
                set(baseline.trajectory_ids) - set(provenance)
                or set(baseline.trajectory_ids)
                - set(baseline.remaining_work_by_trajectory)
                or _duplicate_episode_seed_records(provenance)
                or not any(
                    int(baseline.stage) in stage_values
                    for stage_values in baseline.W_rem_star_by_episode_stage.values()
                )
            ):
                continue
            group_pools = {
                group: pools.get((baseline.family, baseline.modality, group))
                for group in ("cog", "phy", "debt")
            }
            if not all(
                isinstance(pool, Mapping) and pool.get("valid") is True
                for pool in group_pools.values()
            ):
                continue
            local_reasons = sorted(reasons)
            group_counts: List[int] = []
            for group, pool in group_pools.items():
                assert isinstance(pool, Mapping)
                baseline.recovery_mean[group] = list(pool["mean"])
                baseline.recovery_covariance[group] = [
                    list(row) for row in pool["covariance"]
                ]
                baseline.recovery_covariance_rank[group] = int(pool["rank"])
                baseline.recovery_covariance_condition[group] = pool["condition"]
                baseline.recovery_covariance_condition_reason[group] = (
                    ""
                    if pool["condition"] is not None
                    else "rank_deficient_covariance"
                )
                baseline.recovery_covariance_shrinkage[group] = pool["shrinkage"]
                baseline.recovery_covariance_scope[group] = (
                    "family_modality_cross_stage"
                )
                baseline.recovery_covariance_source[group] = {
                    "kind": "family_modality_cross_stage",
                    "family": str(baseline.family),
                    "modality": str(baseline.modality),
                    "pooled_stages": list(pool["stages"]),
                    "n_samples": int(pool["stats"]["n_samples"]),
                    "n_trajectories": int(pool["n_trajectories"]),
                    "min_trajectories": DEFAULT_FORMAL_MIN_TRAJECTORIES,
                    "min_transitions": DEFAULT_FORMAL_MIN_TRANSITIONS,
                    "rescued_local_reasons": local_reasons,
                }
                group_counts.append(int(pool["stats"]["n_samples"]))
            baseline.recovery_covariance_n_samples = min(group_counts)
            baseline.formal_support_scope = "family_modality_cross_stage"
            baseline.formal_valid = True
            baseline.formal_missing_reason = ""
            applied += 1
        return applied

    @staticmethod
    def merge_baseline_maps(
        existing: Mapping[Tuple[str, int, str], StageBaseline],
        incoming: Mapping[Tuple[str, int, str], StageBaseline],
    ) -> Dict[Tuple[str, int, str], StageBaseline]:
        """Merge V3 baselines without changing formal statistical semantics.

        Recovery OAS is recomputed from additive raw moments, while W_rem* is
        recomputed from trajectory-identified anchor summaries. Other legacy
        scalar medians remain sample-weighted diagnostics.
        """

        # Persistent aggregation may continue an explicitly enabled family
        # shrinkage run.  Do not infer permission merely from under-sampling:
        # at least one input artifact must already carry the borrowed scope.
        family_shrinkage_requested = any(
            baseline.formal_support_scope == "family_modality_cross_stage"
            or "family_modality_cross_stage"
            in set(baseline.recovery_covariance_scope.values())
            for baseline in (*existing.values(), *incoming.values())
        )
        hierarchical_components_requested = any(
            str(baseline.modality) in RESERVED_POOL_MODALITIES
            or baseline.formal_support_scope
            in {
                "hierarchical_component_pool",
                "family_stage_recovery_pool",
                "family_stage_stability_pool",
            }
            for baseline in (*existing.values(), *incoming.values())
        )

        def _clone(baseline: StageBaseline) -> StageBaseline:
            return StageBaseline.from_dict(baseline.to_dict())

        def _weight(baseline: StageBaseline) -> float:
            return float(max(1, int(getattr(baseline, "n_samples", 0) or 0)))

        def _avg(a: float, b: float, wa: float, wb: float) -> float:
            return float((float(a) * wa + float(b) * wb) / max(1.0, wa + wb))

        def _merge_trajectory_provenance(
            *sources: Mapping[str, Mapping[str, Any]],
        ) -> Dict[str, Dict[str, Any]]:
            merged_provenance: Dict[str, Dict[str, Any]] = {}
            for source in sources:
                for trajectory_id, provenance in (
                    _normalized_trajectory_provenance(source).items()
                ):
                    existing_provenance = merged_provenance.get(trajectory_id)
                    if (
                        existing_provenance is not None
                        and existing_provenance != provenance
                    ):
                        raise ValueError(
                            "Conflicting provenance for clean trajectory "
                            f"{trajectory_id}"
                        )
                    merged_provenance[trajectory_id] = provenance
            return merged_provenance

        def _episode_stage_work_from_trajectories(
            *,
            stage: int,
            remaining_work: Mapping[str, float],
            provenance: Mapping[str, Mapping[str, Any]],
            fallback_maps: Sequence[Mapping[str, Mapping[int, float]]] = (),
        ) -> Dict[str, Dict[int, float]]:
            samples: Dict[str, List[float]] = defaultdict(list)
            for trajectory_id, value in remaining_work.items():
                record = provenance.get(str(trajectory_id))
                if not isinstance(record, Mapping):
                    continue
                episode_id = str(record.get("episode_id", ""))
                if episode_id:
                    samples[episode_id].append(float(value))
            episode_map = {
                episode_id: {
                    int(stage): float(
                        np.median(np.asarray(values, dtype=np.float64))
                    )
                }
                for episode_id, values in samples.items()
                if values
            }
            for fallback in fallback_maps:
                for episode_id, stage_values in fallback.items():
                    if str(episode_id) in episode_map:
                        continue
                    if int(stage) in stage_values:
                        episode_map[str(episode_id)] = {
                            int(stage): float(stage_values[int(stage)])
                        }
            return episode_map

        def _merge_stability_moments(
            left: StageBaseline,
            right: StageBaseline,
        ) -> Tuple[List[float], List[List[float]], int]:
            left_n = int(left.stability_n_samples or 0)
            right_n = int(right.stability_n_samples or 0)
            if left_n <= 0:
                return (
                    list(right.stability_mean),
                    [list(row) for row in right.stability_covariance],
                    right_n,
                )
            if right_n <= 0:
                return (
                    list(left.stability_mean),
                    [list(row) for row in left.stability_covariance],
                    left_n,
                )
            left_mean = np.asarray(left.stability_mean, dtype=np.float64)
            right_mean = np.asarray(right.stability_mean, dtype=np.float64)
            left_cov = np.asarray(left.stability_covariance, dtype=np.float64)
            right_cov = np.asarray(right.stability_covariance, dtype=np.float64)
            if (
                left_mean.shape != right_mean.shape
                or left_cov.shape != right_cov.shape
                or left_cov.shape != (left_mean.size, left_mean.size)
            ):
                # Incompatible schemas should not be mixed. Prefer the newer
                # incoming moments while preserving their real sample count.
                return (
                    list(right.stability_mean),
                    [list(row) for row in right.stability_covariance],
                    right_n,
                )
            total = left_n + right_n
            mean = (left_n * left_mean + right_n * right_mean) / float(total)
            left_shift = (left_mean - mean).reshape(-1, 1)
            right_shift = (right_mean - mean).reshape(-1, 1)
            covariance = (
                left_n * (left_cov + left_shift @ left_shift.T)
                + right_n * (right_cov + right_shift @ right_shift.T)
            ) / float(total)
            return mean.tolist(), covariance.tolist(), total

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
            left_hashes = {str(value) for value in left.source_hashes}
            right_hashes = {str(value) for value in right.source_hashes}
            left_trajectories = {str(value) for value in left.trajectory_ids}
            right_trajectories = {str(value) for value in right.trajectory_ids}
            recovery_normalizers_match = all(
                math.isclose(
                    float(left_value),
                    float(right_value),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
                for left_value, right_value in (
                    (left.delta_p_bar, right.delta_p_bar),
                    (left.delta_q_bar, right.delta_q_bar),
                    (left.r_bar, right.r_bar),
                    (left.dim("plan").tau_bar, right.dim("plan").tau_bar),
                    (left.dim("sim").tau_bar, right.dim("sim").tau_bar),
                    *(
                        (
                            left.v_bar_channels.get(channel, 0.0),
                            right.v_bar_channels.get(channel, 0.0),
                        )
                        for channel in ("pln", "per", "coord")
                    ),
                )
            )
            if right_trajectories and right_trajectories.issubset(left_trajectories):
                # Replaying the same clean export is an idempotent no-op.
                merged[key] = _clone(left)
                continue
            if left_trajectories & right_trajectories:
                raise ValueError(
                    "StageBaseline trajectory identities partially overlap; refusing "
                    "a summary merge that would double count clean evidence"
                )
            if (
                not right_trajectories
                and right_hashes
                and right_hashes.issubset(left_hashes)
            ):
                merged[key] = _clone(left)
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
            stability_mean, stability_covariance, stability_n_samples = (
                _merge_stability_moments(left, right)
            )
            recovery_mean: Dict[str, List[float]] = {}
            recovery_covariance: Dict[str, List[List[float]]] = {}
            recovery_sufficient_statistics: Dict[str, Dict[str, Any]] = {}
            recovery_group_counts: List[int] = []
            covariance_rank: Dict[str, int] = {}
            covariance_condition: Dict[str, Optional[float]] = {}
            covariance_condition_reason: Dict[str, str] = {}
            covariance_shrinkage: Dict[str, Optional[float]] = {}
            covariance_scope: Dict[str, str] = {}
            covariance_source: Dict[str, Dict[str, Any]] = {}
            recovery_statistics_missing: List[str] = []

            def _valid_stats(
                baseline: StageBaseline,
                group: str,
            ) -> Optional[Dict[str, Any]]:
                raw = baseline.recovery_sufficient_statistics.get(group)
                if not isinstance(raw, Mapping):
                    return None
                n_samples = int(raw.get("n_samples", 0) or 0)
                vector_sum = np.asarray(raw.get("sum", []), dtype=np.float64)
                outer_sum = np.asarray(raw.get("sum_outer", []), dtype=np.float64)
                if (
                    n_samples <= 0
                    or vector_sum.size <= 0
                    or outer_sum.shape != (vector_sum.size, vector_sum.size)
                ):
                    return None
                return {
                    "n_samples": n_samples,
                    "sum": vector_sum.tolist(),
                    "sum_outer": outer_sum.tolist(),
                }

            for group in sorted(
                set(left.recovery_covariance)
                | set(right.recovery_covariance)
                | set(left.recovery_sufficient_statistics)
                | set(right.recovery_sufficient_statistics)
            ):
                left_stats = _valid_stats(left, group)
                right_stats = _valid_stats(right, group)
                merged_stats: Optional[Dict[str, Any]] = None
                if left_stats is not None and right_stats is not None:
                    left_sum = np.asarray(left_stats["sum"], dtype=np.float64)
                    right_sum = np.asarray(right_stats["sum"], dtype=np.float64)
                    left_outer = np.asarray(left_stats["sum_outer"], dtype=np.float64)
                    right_outer = np.asarray(right_stats["sum_outer"], dtype=np.float64)
                    if left_sum.shape != right_sum.shape:
                        recovery_statistics_missing.append(group)
                    else:
                        merged_stats = {
                            "n_samples": int(left_stats["n_samples"])
                            + int(right_stats["n_samples"]),
                            "sum": (left_sum + right_sum).tolist(),
                            "sum_outer": (left_outer + right_outer).tolist(),
                        }
                elif left_stats is not None and group not in right.recovery_covariance:
                    merged_stats = left_stats
                elif right_stats is not None and group not in left.recovery_covariance:
                    merged_stats = right_stats
                else:
                    recovery_statistics_missing.append(group)

                if merged_stats is None:
                    # Never average already-shrunk covariances. Retain one side
                    # only as diagnostic evidence and invalidate formal use.
                    source = right if group in right.recovery_covariance else left
                    recovery_mean[group] = list(source.recovery_mean.get(group, []))
                    recovery_covariance[group] = [
                        list(row)
                        for row in source.recovery_covariance.get(group, [])
                    ]
                    covariance_rank[group] = int(
                        source.recovery_covariance_rank.get(group, 0) or 0
                    )
                    covariance_condition[group] = (
                        source.recovery_covariance_condition.get(group)
                    )
                    covariance_condition_reason[group] = (
                        "sufficient_statistics_missing"
                    )
                    covariance_shrinkage[group] = (
                        source.recovery_covariance_shrinkage.get(group)
                    )
                    covariance_scope[group] = "diagnostic_unmerged"
                    covariance_source[group] = {
                        "kind": "sufficient_statistics_missing"
                    }
                    continue

                mean, covariance, shrinkage, rank, condition = (
                    _oas_from_sufficient_statistics(merged_stats)
                )
                recovery_sufficient_statistics[group] = merged_stats
                recovery_mean[group] = mean
                recovery_covariance[group] = covariance
                recovery_group_counts.append(int(merged_stats["n_samples"]))
                covariance_rank[group] = rank
                covariance_condition[group] = condition
                covariance_condition_reason[group] = (
                    "" if condition is not None else "rank_deficient_covariance"
                )
                covariance_shrinkage[group] = shrinkage
                covariance_scope[group] = "stage_exact_merged"
                covariance_source[group] = {
                    "kind": "stage_exact_merged",
                    "family": str(right.family),
                    "stage": int(right.stage),
                    "modality": str(right.modality),
                    "n_samples": int(merged_stats["n_samples"]),
                }
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
            trajectory_ids = sorted(
                {str(value) for value in left.trajectory_ids}
                | {str(value) for value in right.trajectory_ids}
            )
            seed_ids = sorted(
                {int(value) for value in left.seed_ids}
                | {int(value) for value in right.seed_ids}
            )
            source_hashes = sorted(left_hashes | right_hashes)
            trajectory_provenance = _merge_trajectory_provenance(
                left.trajectory_provenance,
                right.trajectory_provenance,
            )
            remaining_work_by_trajectory = {
                str(trajectory_id): float(value)
                for trajectory_id, value in left.remaining_work_by_trajectory.items()
            }
            for trajectory_id, value in right.remaining_work_by_trajectory.items():
                trajectory_id = str(trajectory_id)
                if (
                    trajectory_id in remaining_work_by_trajectory
                    and not math.isclose(
                        remaining_work_by_trajectory[trajectory_id],
                        float(value),
                        rel_tol=1.0e-12,
                        abs_tol=1.0e-12,
                    )
                ):
                    raise ValueError(
                        "Conflicting remaining-work summaries for trajectory "
                        f"{trajectory_id}"
                    )
                remaining_work_by_trajectory[trajectory_id] = float(value)
            W_rem_star_by_episode_stage = _episode_stage_work_from_trajectories(
                stage=int(right.stage),
                remaining_work=remaining_work_by_trajectory,
                provenance=trajectory_provenance,
                fallback_maps=(
                    left.W_rem_star_by_episode_stage,
                    right.W_rem_star_by_episode_stage,
                ),
            )
            W_rem_support_by_episode_stage: Dict[str, Dict[int, int]] = {}
            for episode_id, stage_values in W_rem_star_by_episode_stage.items():
                support = sum(
                    1
                    for trajectory_id in remaining_work_by_trajectory
                    if str(
                        trajectory_provenance.get(str(trajectory_id), {}).get(
                            "episode_id", ""
                        )
                    )
                    == str(episode_id)
                )
                if support <= 0:
                    support = max(
                        int(
                            left.W_rem_support_by_episode_stage.get(
                                str(episode_id), {}
                            ).get(int(right.stage), 0)
                        ),
                        int(
                            right.W_rem_support_by_episode_stage.get(
                                str(episode_id), {}
                            ).get(int(right.stage), 0)
                        ),
                    )
                W_rem_support_by_episode_stage[str(episode_id)] = {
                    int(stage_id): int(support)
                    for stage_id in stage_values
                }
            episode_stage_values = [
                stage_values[int(right.stage)]
                for stage_values in W_rem_star_by_episode_stage.values()
                if int(right.stage) in stage_values
            ]
            if episode_stage_values:
                merged_w_rem_star = max(
                    1.0,
                    float(
                        np.median(
                            np.asarray(episode_stage_values, dtype=np.float64)
                        )
                    ),
                )
            elif remaining_work_by_trajectory:
                merged_w_rem_star = max(
                    1.0,
                    float(
                        np.median(
                            np.asarray(
                                list(remaining_work_by_trajectory.values()),
                                dtype=np.float64,
                            )
                        )
                    ),
                )
            else:
                merged_w_rem_star = _avg(
                    left.W_rem_star_swe,
                    right.W_rem_star_swe,
                    wa,
                    wb,
                )
            duplicate_episode_seed_trajectories = (
                _duplicate_episode_seed_records(trajectory_provenance)
            )
            merged_transitions = int(left.n_transitions or left.n_samples) + int(
                right.n_transitions or right.n_samples
            )
            min_trajectories = max(
                int(left.formal_min_trajectories),
                int(right.formal_min_trajectories),
            )
            min_transitions = max(
                int(left.formal_min_transitions),
                int(right.formal_min_transitions),
            )
            formal_reasons: List[str] = []
            if int(right.stage) < 0:
                formal_reasons.append("terminal_or_unknown_stage")
            if recovery_statistics_missing:
                formal_reasons.append("recovery_sufficient_statistics_missing")
            if trajectory_ids:
                if not recovery_normalizers_match:
                    formal_reasons.append("recovery_feature_normalizer_mismatch")
                if len(trajectory_ids) < min_trajectories:
                    formal_reasons.append("insufficient_independent_trajectories")
                if merged_transitions < min_transitions:
                    formal_reasons.append("insufficient_transitions")
                if set(trajectory_ids) - set(remaining_work_by_trajectory):
                    formal_reasons.append("remaining_work_trajectory_summary_missing")
                if set(trajectory_ids) - set(trajectory_provenance):
                    formal_reasons.append("trajectory_provenance_missing")
                if not any(
                    int(right.stage) in stage_values
                    for stage_values in W_rem_star_by_episode_stage.values()
                ):
                    formal_reasons.append("episode_stage_remaining_work_missing")
                if duplicate_episode_seed_trajectories:
                    formal_reasons.append("duplicate_episode_seed_trajectories")
                formal_valid = not formal_reasons
            else:
                # Trusted programmatic baselines without provenance remain
                # compatible; all estimator-produced V3 baselines carry it.
                formal_valid = bool(left.formal_valid and right.formal_valid)
                if not formal_valid:
                    formal_reasons.append("formal_provenance_missing")

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
                W_rem_star_swe=merged_w_rem_star,
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
                stability_feature_names=list(
                    right.stability_feature_names
                    or left.stability_feature_names
                    or STABILITY_FEATURE_NAMES
                ),
                stability_mean=stability_mean,
                stability_covariance=stability_covariance,
                stability_n_samples=stability_n_samples,
                recovery_mean=recovery_mean,
                recovery_covariance=recovery_covariance,
                recovery_covariance_n_samples=(
                    min(recovery_group_counts) if recovery_group_counts else 0
                ),
                recovery_sufficient_statistics=recovery_sufficient_statistics,
                n_transitions=merged_transitions,
                n_trajectories=len(trajectory_ids),
                n_seeds=len(seed_ids),
                trajectory_ids=trajectory_ids,
                seed_ids=seed_ids,
                source_hashes=source_hashes,
                trajectory_provenance=trajectory_provenance,
                remaining_work_by_trajectory=remaining_work_by_trajectory,
                W_rem_star_by_episode_stage=W_rem_star_by_episode_stage,
                W_rem_support_by_episode_stage=(
                    W_rem_support_by_episode_stage
                ),
                duplicate_episode_seed_trajectories=(
                    duplicate_episode_seed_trajectories
                ),
                recovery_covariance_rank=covariance_rank,
                recovery_covariance_condition=covariance_condition,
                recovery_covariance_condition_reason=covariance_condition_reason,
                recovery_covariance_shrinkage=covariance_shrinkage,
                recovery_covariance_scope=covariance_scope,
                recovery_covariance_source=covariance_source,
                formal_support_scope=(
                    "exact_stage_merged" if formal_valid else "diagnostic"
                ),
                formal_min_trajectories=min_trajectories,
                formal_min_transitions=min_transitions,
                formal_valid=formal_valid,
                formal_missing_reason=";".join(formal_reasons),
                formal_component_support={},
                n_samples=int(left.n_samples) + int(right.n_samples),
            )

        # A stage continuation is independent of the currently active
        # modality.  Synchronize the union after all anchor keys are merged so
        # a trajectory that lacks one modality still contributes its episode x
        # stage denominator to every sibling modality baseline.
        by_stage: Dict[Tuple[str, int], List[StageBaseline]] = defaultdict(list)
        for baseline in merged.values():
            by_stage[(baseline.family, int(baseline.stage))].append(baseline)
        for (_family, stage), stage_baselines in by_stage.items():
            remaining_work: Dict[str, float] = {}
            provenance = _merge_trajectory_provenance(
                *(baseline.trajectory_provenance for baseline in stage_baselines)
            )
            fallback_maps: List[Mapping[str, Mapping[int, float]]] = []
            for baseline in stage_baselines:
                fallback_maps.append(baseline.W_rem_star_by_episode_stage)
                for trajectory_id, value in (
                    baseline.remaining_work_by_trajectory.items()
                ):
                    if (
                        trajectory_id in remaining_work
                        and not math.isclose(
                            remaining_work[trajectory_id],
                            float(value),
                            rel_tol=1.0e-12,
                            abs_tol=1.0e-12,
                        )
                    ):
                        raise ValueError(
                            "Conflicting remaining-work summaries for trajectory "
                            f"{trajectory_id}"
                        )
                    remaining_work[str(trajectory_id)] = float(value)
            episode_map = _episode_stage_work_from_trajectories(
                stage=int(stage),
                remaining_work=remaining_work,
                provenance=provenance,
                fallback_maps=fallback_maps,
            )
            episode_values = [
                stage_values[int(stage)]
                for stage_values in episode_map.values()
                if int(stage) in stage_values
            ]
            if episode_values:
                scalar_w_rem = max(
                    1.0,
                    float(np.median(np.asarray(episode_values, dtype=np.float64))),
                )
            elif remaining_work:
                scalar_w_rem = max(
                    1.0,
                    float(
                        np.median(
                            np.asarray(
                                list(remaining_work.values()), dtype=np.float64
                            )
                        )
                    ),
                )
            else:
                scalar_w_rem = max(
                    1.0,
                    float(
                        np.median(
                            np.asarray(
                                [b.W_rem_star_swe for b in stage_baselines],
                                dtype=np.float64,
                            )
                        )
                    ),
                )
            duplicate_cells = _duplicate_episode_seed_records(provenance)
            support_map = {
                str(episode_id): {
                    int(stage): sum(
                        1
                        for trajectory_id in remaining_work
                        if str(
                            provenance.get(str(trajectory_id), {}).get(
                                "episode_id", ""
                            )
                        )
                        == str(episode_id)
                    )
                }
                for episode_id in episode_map
            }
            for baseline in stage_baselines:
                baseline.remaining_work_by_trajectory = dict(remaining_work)
                baseline.trajectory_provenance = {
                    trajectory_id: dict(record)
                    for trajectory_id, record in provenance.items()
                }
                baseline.W_rem_star_by_episode_stage = {
                    episode_id: dict(stage_values)
                    for episode_id, stage_values in episode_map.items()
                }
                baseline.W_rem_support_by_episode_stage = {
                    episode_id: dict(stage_values)
                    for episode_id, stage_values in support_map.items()
                }
                baseline.W_rem_star_swe = scalar_w_rem
                baseline.duplicate_episode_seed_trajectories = [
                    dict(record) for record in duplicate_cells
                ]
                reasons = [
                    reason
                    for reason in baseline.formal_missing_reason.split(";")
                    if reason
                ]
                if set(baseline.trajectory_ids) - set(provenance):
                    if "trajectory_provenance_missing" not in reasons:
                        reasons.append("trajectory_provenance_missing")
                    baseline.formal_valid = False
                if set(baseline.trajectory_ids) - set(remaining_work):
                    if "remaining_work_trajectory_summary_missing" not in reasons:
                        reasons.append("remaining_work_trajectory_summary_missing")
                    baseline.formal_valid = False
                if not any(
                    int(stage) in stage_values
                    for stage_values in episode_map.values()
                ):
                    if "episode_stage_remaining_work_missing" not in reasons:
                        reasons.append("episode_stage_remaining_work_missing")
                    baseline.formal_valid = False
                if duplicate_cells:
                    if "duplicate_episode_seed_trajectories" not in reasons:
                        reasons.append("duplicate_episode_seed_trajectories")
                    baseline.formal_valid = False
                if not baseline.formal_valid:
                    baseline.formal_support_scope = "diagnostic"
                baseline.formal_missing_reason = ";".join(reasons)

        if hierarchical_components_requested:
            for baseline in merged.values():
                if baseline.modality == STABILITY_POOL_MODALITY:
                    StageBaselineEstimator._configure_stage_pool_baseline(
                        baseline,
                        pool_kind="stability",
                    )
                elif baseline.modality == RECOVERY_POOL_MODALITY:
                    StageBaselineEstimator._configure_stage_pool_baseline(
                        baseline,
                        pool_kind="recovery",
                    )
            StageBaselineEstimator._apply_hierarchical_component_support(merged)

        if family_shrinkage_requested:
            # Pools are rebuilt exclusively from local additive moments.  A
            # borrowed runtime covariance is never reinterpreted as local
            # evidence during persistent aggregation.
            family_pools = StageBaselineEstimator._build_family_covariance_pools(
                merged
            )
            StageBaselineEstimator._apply_family_covariance_shrinkage(
                merged,
                family_pools,
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
        if not baselines:
            raise ValueError("refusing to write an empty StageBaseline artifact")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stage_records = [
            baselines[key].to_dict()
            for key in sorted(
                baselines,
                key=lambda item: (str(item[0]), int(item[1]), str(item[2])),
            )
        ]
        metadata_payload = dict(metadata or {})
        required_nonterminal = [
            baseline
            for baseline in baselines.values()
            if int(baseline.stage) >= 0
            and str(baseline.modality) not in RESERVED_POOL_MODALITIES
        ]
        stability_pools = [
            baseline
            for baseline in baselines.values()
            if int(baseline.stage) >= 0
            and str(baseline.modality) == STABILITY_POOL_MODALITY
        ]
        recovery_pools = [
            baseline
            for baseline in baselines.values()
            if int(baseline.stage) >= 0
            and str(baseline.modality) == RECOVERY_POOL_MODALITY
        ]
        observed_nonterminal_stages = sorted(
            {
                (str(baseline.family), int(baseline.stage))
                for baseline in required_nonterminal
            }
        )
        stability_pool_by_stage = {
            (str(baseline.family), int(baseline.stage)): baseline
            for baseline in stability_pools
        }
        recovery_pool_by_stage = {
            (str(baseline.family), int(baseline.stage)): baseline
            for baseline in recovery_pools
        }
        required_coverage_valid = bool(required_nonterminal) and all(
            bool(baseline.formal_valid) for baseline in required_nonterminal
        )
        component_pools_present = bool(stability_pools or recovery_pools)
        if component_pools_present:
            stability_coverage_valid = bool(observed_nonterminal_stages) and all(
                stage_key in stability_pool_by_stage
                and bool(stability_pool_by_stage[stage_key].formal_valid)
                for stage_key in observed_nonterminal_stages
            )
            recovery_coverage_valid = bool(observed_nonterminal_stages) and all(
                stage_key in recovery_pool_by_stage
                and bool(recovery_pool_by_stage[stage_key].formal_valid)
                for stage_key in observed_nonterminal_stages
            )
            # Exact modality anchors remain audit evidence. Sparse anchors do
            # not poison the artifact when both formal stage component pools
            # exist; runtime still fails closed on the current episode's W_rem.
            component_coverage_valid = bool(
                stability_coverage_valid and recovery_coverage_valid
            )
        else:
            # Backwards compatibility for explicit programmatic fixtures that
            # predate estimator v3.4 and therefore have no reserved pools.
            stability_coverage_valid = True
            recovery_coverage_valid = True
            component_coverage_valid = required_coverage_valid
        lifecycle_formal = bool(metadata_payload.get("formal_valid", True))
        metadata_payload["formal_valid"] = bool(
            lifecycle_formal and component_coverage_valid
        )
        formal_missing_reasons = list(
            metadata_payload.get("formal_missing_reasons", []) or []
        )
        if component_pools_present and not stability_coverage_valid:
            formal_missing_reasons.append(
                "formal_stability_pool_coverage_incomplete"
            )
        if component_pools_present and not recovery_coverage_valid:
            formal_missing_reasons.append(
                "formal_recovery_pool_coverage_incomplete"
            )
        metadata_payload["formal_missing_reasons"] = list(
            dict.fromkeys(str(value) for value in formal_missing_reasons)
        )
        metadata_payload["formal_required_nonterminal_coverage_valid"] = bool(
            required_coverage_valid
        )
        metadata_payload["formal_stability_pool_coverage_valid"] = bool(
            stability_coverage_valid
        )
        metadata_payload["formal_recovery_pool_coverage_valid"] = bool(
            recovery_coverage_valid
        )
        metadata_payload["required_nonterminal_anchor_count"] = len(
            required_nonterminal
        )
        metadata_payload["formal_required_nonterminal_anchor_count"] = sum(
            1 for baseline in required_nonterminal if baseline.formal_valid
        )
        metadata_payload["diagnostic_required_nonterminal_anchor_count"] = sum(
            1 for baseline in required_nonterminal if not baseline.formal_valid
        )
        metadata_payload["observed_nonterminal_stage_count"] = len(
            observed_nonterminal_stages
        )
        metadata_payload["stability_pool_count"] = len(stability_pools)
        metadata_payload["formal_stability_pool_count"] = sum(
            1 for baseline in stability_pools if baseline.formal_valid
        )
        metadata_payload["recovery_pool_count"] = len(recovery_pools)
        metadata_payload["formal_recovery_pool_count"] = sum(
            1 for baseline in recovery_pools if baseline.formal_valid
        )
        metadata_payload["formal_anchor_count"] = sum(
            1 for baseline in baselines.values() if baseline.formal_valid
        )
        metadata_payload["anchor_count"] = len(baselines)
        metadata_payload.setdefault(
            "baseline_id",
            hashlib.sha256(
                json.dumps(
                    {
                        "signature": metadata_payload.get("signature", {}),
                        "source_hashes": sorted(
                            {
                                source_hash
                                for baseline in baselines.values()
                                for source_hash in baseline.source_hashes
                            }
                        ),
                        "stages": stage_records,
                        "schema_version": STAGE_BASELINE_SCHEMA_VERSION,
                        "feature_version": RECOVERY_FEATURE_VERSION,
                    },
                    sort_keys=True,
                    ensure_ascii=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        )
        payload = {
            "schema_version": STAGE_BASELINE_SCHEMA_VERSION,
            "baseline_kind": "resilience_statistical_reference",
            "baseline_scope": "stage_anchor",
            "feature_version": RECOVERY_FEATURE_VERSION,
            "formula_version": REBOUND_FORMULA_VERSION,
            "resolver_version": STAGE_RESOLVER_VERSION,
            "estimator_version": STAGE_BASELINE_ESTIMATOR_VERSION,
            "stages": stage_records,
            "metadata": metadata_payload,
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload["metadata"]["artifact_integrity"] = {
            "algorithm": "sha256",
            "digest": hashlib.sha256(canonical).hexdigest(),
        }
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return output_path

    @staticmethod
    def artifact_status(
        path: Union[str, os.PathLike],
        *,
        require_formal: bool = False,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Audit schema, feature identity and serialized integrity."""

        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return False, f"baseline_unreadable:{type(exc).__name__}", {}
        if not isinstance(payload, Mapping):
            return False, "baseline_payload_not_mapping", {}
        if int(payload.get("schema_version", 0) or 0) != STAGE_BASELINE_SCHEMA_VERSION:
            return False, "stage_baseline_schema_version_mismatch", dict(payload)
        expected_versions = {
            "feature_version": RECOVERY_FEATURE_VERSION,
            "formula_version": REBOUND_FORMULA_VERSION,
            "resolver_version": STAGE_RESOLVER_VERSION,
            "estimator_version": STAGE_BASELINE_ESTIMATOR_VERSION,
        }
        for field_name, expected in expected_versions.items():
            if str(payload.get(field_name) or "") != expected:
                return False, f"{field_name}_mismatch", dict(payload)
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            return False, "baseline_metadata_missing", dict(payload)
        integrity = metadata.get("artifact_integrity")
        if not isinstance(integrity, Mapping):
            return False, "baseline_integrity_missing", dict(payload)
        expected_digest = str(integrity.get("digest") or "")
        unsigned = dict(payload)
        unsigned_metadata = dict(metadata)
        unsigned_metadata.pop("artifact_integrity", None)
        unsigned["metadata"] = unsigned_metadata
        actual_digest = hashlib.sha256(
            json.dumps(
                unsigned,
                sort_keys=True,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if not expected_digest or expected_digest != actual_digest:
            return False, "baseline_integrity_mismatch", dict(payload)
        if not str(metadata.get("baseline_id") or ""):
            return False, "baseline_id_missing", dict(payload)
        return True, "", dict(payload)

    @staticmethod
    def load(
        path: Union[str, os.PathLike],
    ) -> Dict[Tuple[str, int, str], StageBaseline]:
        path = Path(path)
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        schema_version = int(payload.get("schema_version", 0) or 0)
        out: Dict[Tuple[str, int, str], StageBaseline] = {}
        for raw in payload.get("stages", []) if isinstance(payload, Mapping) else []:
            baseline = StageBaseline.from_dict(raw)
            if schema_version != STAGE_BASELINE_SCHEMA_VERSION:
                baseline.formal_valid = False
                baseline.formal_missing_reason = "legacy_stage_baseline_schema"
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
        "dp_neg",
        "dq_neg",
        "risk",
        "remaining",
        "remaining_task",
        "remaining_task_trajectory_ids",
        "dt_plan",
        "dt_sim",
        "raw_d_plan",
        "raw_d_sim",
        "has_wait",
        "dt_runtime",
        "v_pln",
        "v_per",
        "v_coord",
        "cognitive_validity",
        "stability_vectors",
        "trajectory_ids",
        "seed_ids",
        "source_hashes",
        "trajectory_provenance",
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
        self.dp_neg: List[float] = []
        self.dq_neg: List[float] = []
        self.risk: List[float] = []
        self.remaining: List[int] = []
        self.remaining_task: List[float] = []
        self.remaining_task_trajectory_ids: List[str] = []
        self.dt_plan: List[float] = []
        self.dt_sim: List[float] = []
        self.raw_d_plan: List[float] = []
        self.raw_d_sim: List[float] = []
        self.has_wait: List[bool] = []
        self.dt_runtime: List[float] = []
        self.v_pln: List[float] = []
        self.v_per: List[float] = []
        self.v_coord: List[float] = []
        self.cognitive_validity: List[float] = []
        self.stability_vectors: List[List[float]] = []
        self.trajectory_ids: set[str] = set()
        self.seed_ids: set[int] = set()
        self.source_hashes: set[str] = set()
        self.trajectory_provenance: Dict[str, Dict[str, Any]] = {}

    def add(
        self,
        step: _StepDerivatives,
        estimator: StageBaselineEstimator,
        *,
        trajectory_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        seed: Optional[int] = None,
        source_hash: Optional[str] = None,
    ) -> None:
        # τ^*: step-duration sample
        self.tau.append(step.dt_sim)
        # Cognitive work (events × nominal per-event times)
        self.T_pln.append((1.0 if step.replan_event else 0.0) * estimator.tau_pln)
        self.T_per.append((1.0 if step.response_event else 0.0) * estimator.tau_per)
        # Coordination is an event edge, not time spent under a modality label.
        # In particular, Wait is represented as an idle/gap diagnostic.
        coord_weight = 1.0 if step.v_coord > 0.0 else 0.0
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
        self.dp_neg.append(max(0.0, -step.dp))
        self.dq_neg.append(max(0.0, -step.dq))
        self.risk.append(max(0.0, step.r))
        # Remaining stage-local steps (for W_rem_seg estimation)
        self.remaining.append(int(step.remaining_stage_steps))
        # Remaining clean-continuation steps to task closure (for W_rem*).
        self.remaining_task.append(float(step.remaining_task_work))
        self.remaining_task_trajectory_ids.append(str(trajectory_id or "unknown"))
        # Dimension-aware nominals
        self.dt_plan.append(max(1.0, step.d_plan if step.d_plan > 0 else 1.0))
        self.dt_sim.append(max(0.0, step.dt_sim))
        self.raw_d_plan.append(max(0.0, step.d_plan))
        self.raw_d_sim.append(max(0.0, step.d_sim))
        self.has_wait.append(bool(step.has_wait))
        self.dt_runtime.append(max(0.0, step.dt_runtime))
        # Cognitive volumes
        self.v_pln.append(max(0.0, step.v_pln))
        self.v_per.append(max(0.0, step.v_per))
        self.v_coord.append(max(0.0, step.v_coord))
        self.cognitive_validity.append(
            min(1.0, max(0.0, float(step.cognitive_validity)))
        )
        if step.stability_vector is not None:
            self.stability_vectors.append(list(step.stability_vector))
        if trajectory_id:
            normalized_trajectory_id = str(trajectory_id)
            self.trajectory_ids.add(normalized_trajectory_id)
            provenance = {
                "episode_id": str(episode_id or ""),
                "seed": int(seed) if seed is not None else None,
                "source_hash": str(source_hash or ""),
            }
            existing_provenance = self.trajectory_provenance.get(
                normalized_trajectory_id
            )
            if (
                existing_provenance is not None
                and existing_provenance != provenance
            ):
                raise ValueError(
                    "Conflicting provenance for clean trajectory "
                    f"{normalized_trajectory_id}"
                )
            self.trajectory_provenance[normalized_trajectory_id] = provenance
        if seed is not None:
            self.seed_ids.add(int(seed))
        if source_hash:
            self.source_hashes.add(str(source_hash))

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
        self.dp_neg.extend(other.dp_neg)
        self.dq_neg.extend(other.dq_neg)
        self.risk.extend(other.risk)
        self.remaining.extend(other.remaining)
        self.remaining_task.extend(other.remaining_task)
        self.remaining_task_trajectory_ids.extend(other.remaining_task_trajectory_ids)
        self.dt_plan.extend(other.dt_plan)
        self.dt_sim.extend(other.dt_sim)
        self.raw_d_plan.extend(other.raw_d_plan)
        self.raw_d_sim.extend(other.raw_d_sim)
        self.has_wait.extend(other.has_wait)
        self.dt_runtime.extend(other.dt_runtime)
        self.v_pln.extend(other.v_pln)
        self.v_per.extend(other.v_per)
        self.v_coord.extend(other.v_coord)
        self.cognitive_validity.extend(other.cognitive_validity)
        self.stability_vectors.extend(other.stability_vectors)
        self.trajectory_ids.update(other.trajectory_ids)
        self.seed_ids.update(other.seed_ids)
        self.source_hashes.update(other.source_hashes)
        for trajectory_id, provenance in other.trajectory_provenance.items():
            existing = self.trajectory_provenance.get(trajectory_id)
            if existing is not None and existing != provenance:
                raise ValueError(
                    "Conflicting provenance for clean trajectory "
                    f"{trajectory_id}"
                )
            self.trajectory_provenance[trajectory_id] = dict(provenance)

    def finalize(
        self,
        key: Tuple[str, int, str],
        *,
        formal_min_trajectories: int = DEFAULT_FORMAL_MIN_TRAJECTORIES,
        formal_min_transitions: int = DEFAULT_FORMAL_MIN_TRANSITIONS,
        stage_remaining_work_by_trajectory: Optional[Mapping[str, float]] = None,
        stage_trajectory_provenance: Optional[
            Mapping[str, Mapping[str, Any]]
        ] = None,
        W_rem_star_by_episode_stage: Optional[
            Mapping[str, Mapping[int, float]]
        ] = None,
        W_rem_support_by_episode_stage: Optional[
            Mapping[str, Mapping[int, int]]
        ] = None,
    ) -> StageBaseline:
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

        remaining_by_trajectory: Dict[str, List[float]] = defaultdict(list)
        for trajectory_id, remaining_work in zip(
            self.remaining_task_trajectory_ids,
            self.remaining_task,
        ):
            remaining_by_trajectory[str(trajectory_id)].append(float(remaining_work))
        trajectory_work_summary = {
            trajectory_id: float(np.median(np.asarray(values, dtype=np.float64)))
            for trajectory_id, values in remaining_by_trajectory.items()
            if values
        }
        if stage_remaining_work_by_trajectory is not None:
            trajectory_work_summary = {
                str(trajectory_id): float(value)
                for trajectory_id, value in stage_remaining_work_by_trajectory.items()
            }
        normalized_episode_stage_work = {
            str(episode_id): {
                int(stage_id): float(value)
                for stage_id, value in stage_values.items()
            }
            for episode_id, stage_values in (
                W_rem_star_by_episode_stage or {}
            ).items()
        }
        normalized_episode_stage_support = {
            str(episode_id): {
                int(stage_id): int(value)
                for stage_id, value in stage_values.items()
            }
            for episode_id, stage_values in (
                W_rem_support_by_episode_stage or {}
            ).items()
        }
        episode_stage_values = [
            float(stage_values[int(stage)])
            for stage_values in normalized_episode_stage_work.values()
            if int(stage) in stage_values
        ]
        if episode_stage_values:
            # Equal episode weighting is deliberate: reruns/seeds within one
            # episode first collapse to the episode x stage median.
            w_rem_star = max(
                1.0,
                float(np.median(np.asarray(episode_stage_values, dtype=np.float64))),
            )
        else:
            trajectory_work = list(trajectory_work_summary.values())
            remaining_task_arr = np.asarray(trajectory_work, dtype=np.float64)
            if remaining_task_arr.size == 0:
                w_rem_star = w_rem
            else:
                w_rem_star = max(1.0, float(np.median(remaining_task_arr)))

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
        stability_n_samples = len(self.stability_vectors)
        if stability_n_samples:
            stability_array = np.asarray(self.stability_vectors, dtype=np.float64)
            stability_mean = np.mean(stability_array, axis=0)
            if stability_n_samples >= 2:
                stability_covariance = np.cov(
                    stability_array,
                    rowvar=False,
                    bias=True,
                )
            else:
                stability_covariance = np.zeros(
                    (stability_array.shape[1], stability_array.shape[1]),
                    dtype=np.float64,
                )
        else:
            stability_mean = np.asarray([], dtype=np.float64)
            stability_covariance = np.asarray([], dtype=np.float64)

        recovery_n_samples = min(
            len(self.v_pln),
            len(self.dt_plan),
            len(self.dp_neg),
            len(self.risk),
            len(self.cognitive_validity),
            len(self.raw_d_plan),
            len(self.raw_d_sim),
            len(self.has_wait),
        )
        recovery_mean: Dict[str, List[float]] = {}
        recovery_covariance: Dict[str, List[List[float]]] = {}
        recovery_covariance_rank: Dict[str, int] = {}
        recovery_covariance_condition: Dict[str, Optional[float]] = {}
        recovery_covariance_condition_reason: Dict[str, str] = {}
        recovery_covariance_shrinkage: Dict[str, Optional[float]] = {}
        recovery_covariance_scope: Dict[str, str] = {}
        recovery_covariance_source: Dict[str, Dict[str, Any]] = {}
        recovery_sufficient_statistics: Dict[str, Dict[str, Any]] = {}
        if recovery_n_samples:
            from habitat_llm.evaluation.metrics.rebound.recovery_features import (
                build_recovery_feature_groups,
            )

            grouped_rows: Dict[str, List[List[float]]] = {
                "cog": [],
                "phy": [],
                "debt": [],
            }
            for index in range(recovery_n_samples):
                validity = float(self.cognitive_validity[index])
                no_forward_progress = (
                    self.dp_pos[index] <= 1.0e-6
                    and self.dq_pos[index] <= 1.0e-6
                )
                idle_gap = (
                    max(
                        self.raw_d_plan[index] / max(float(dt_plan_med), 1.0e-12),
                        self.raw_d_sim[index] / max(float(dt_sim_med), 1.0e-12),
                    )
                    if self.has_wait[index] and no_forward_progress
                    else 0.0
                )
                groups = build_recovery_feature_groups(
                    volumes={
                        "pln": self.v_pln[index],
                        "per": self.v_per[index],
                        "coord": self.v_coord[index],
                    },
                    baseline_volumes=v_bar_channels,
                    validities={
                        "pln": validity,
                        "per": validity,
                        "coord": validity,
                    },
                    plan_work=self.raw_d_plan[index],
                    sim_work=self.raw_d_sim[index],
                    plan_baseline=dt_plan_med,
                    sim_baseline=dt_sim_med,
                    idle_gap=idle_gap,
                    redo_event=self.T_redo[index] > 0.0,
                    progress_debt=self.dp_neg[index],
                    goal_debt=self.dq_neg[index],
                    risk=self.risk[index],
                    progress_baseline=delta_p_bar,
                    goal_baseline=delta_q_bar,
                    risk_baseline=r_bar,
                )
                for group, values in groups.items():
                    grouped_rows[group].append(values)
            cog_array = np.asarray(grouped_rows["cog"], dtype=np.float64)
            phy_array = np.asarray(grouped_rows["phy"], dtype=np.float64)
            debt_array = np.asarray(grouped_rows["debt"], dtype=np.float64)

            for group, values in (
                ("cog", cog_array),
                ("phy", phy_array),
                ("debt", debt_array),
            ):
                stats = _recovery_sufficient_statistics(values)
                mean, covariance, shrinkage, rank, condition = (
                    _oas_from_sufficient_statistics(stats)
                )
                recovery_sufficient_statistics[group] = stats
                recovery_mean[group] = mean
                recovery_covariance[group] = covariance
                recovery_covariance_shrinkage[group] = shrinkage
                recovery_covariance_rank[group] = rank
                recovery_covariance_condition[group] = condition
                recovery_covariance_condition_reason[group] = (
                    "" if condition is not None else "rank_deficient_covariance"
                )
                recovery_covariance_scope[group] = "stage_exact"
                recovery_covariance_source[group] = {
                    "kind": "stage_exact",
                    "family": str(family),
                    "stage": int(stage),
                    "modality": str(modality),
                    "n_samples": int(stats["n_samples"]),
                    "n_trajectories": int(len(self.trajectory_ids)),
                }

        n_transitions = len(self.tau)
        n_trajectories = len(self.trajectory_ids)
        trajectory_provenance = _normalized_trajectory_provenance(
            stage_trajectory_provenance or self.trajectory_provenance
        )
        duplicate_episode_seed_trajectories = _duplicate_episode_seed_records(
            trajectory_provenance
        )
        formal_reasons: List[str] = []
        if int(stage) < 0:
            formal_reasons.append("terminal_or_unknown_stage")
        if n_trajectories < int(formal_min_trajectories):
            formal_reasons.append("insufficient_independent_trajectories")
        if n_transitions < int(formal_min_transitions):
            formal_reasons.append("insufficient_transitions")
        if set(self.trajectory_ids) - set(trajectory_provenance):
            formal_reasons.append("trajectory_provenance_missing")
        if set(self.trajectory_ids) - set(trajectory_work_summary):
            formal_reasons.append("remaining_work_trajectory_summary_missing")
        if not any(
            int(stage) in stage_values
            for stage_values in normalized_episode_stage_work.values()
        ):
            formal_reasons.append("episode_stage_remaining_work_missing")
        if duplicate_episode_seed_trajectories:
            formal_reasons.append("duplicate_episode_seed_trajectories")

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
            stability_feature_names=list(STABILITY_FEATURE_NAMES),
            stability_mean=stability_mean.tolist(),
            stability_covariance=stability_covariance.tolist(),
            stability_n_samples=stability_n_samples,
            recovery_mean=recovery_mean,
            recovery_covariance=recovery_covariance,
            recovery_covariance_n_samples=recovery_n_samples,
            recovery_sufficient_statistics=recovery_sufficient_statistics,
            n_transitions=n_transitions,
            n_trajectories=n_trajectories,
            n_seeds=len(self.seed_ids),
            trajectory_ids=sorted(self.trajectory_ids),
            seed_ids=sorted(self.seed_ids),
            source_hashes=sorted(self.source_hashes),
            trajectory_provenance=trajectory_provenance,
            remaining_work_by_trajectory=trajectory_work_summary,
            W_rem_star_by_episode_stage=normalized_episode_stage_work,
            W_rem_support_by_episode_stage=normalized_episode_stage_support,
            duplicate_episode_seed_trajectories=(
                duplicate_episode_seed_trajectories
            ),
            recovery_covariance_rank=recovery_covariance_rank,
            recovery_covariance_condition=recovery_covariance_condition,
            recovery_covariance_condition_reason=recovery_covariance_condition_reason,
            recovery_covariance_shrinkage=recovery_covariance_shrinkage,
            recovery_covariance_scope=recovery_covariance_scope,
            recovery_covariance_source=recovery_covariance_source,
            formal_support_scope=(
                "exact_stage" if not formal_reasons else "diagnostic"
            ),
            formal_min_trajectories=int(formal_min_trajectories),
            formal_min_transitions=int(formal_min_transitions),
            formal_valid=not formal_reasons,
            formal_missing_reason=";".join(formal_reasons),
            n_samples=len(self.tau),
        )


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def _discover_export_files(root: Path) -> List[Path]:
    patterns = ("*.json",)
    out: List[Path] = []
    for pat in patterns:
        out.extend(sorted(root.rglob(pat)))
    return out


def main() -> int:  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(
        description="Estimate clean stage baselines from critic_exports."
    )
    parser.add_argument(
        "--exports",
        required=True,
        help="Directory containing manifest-bearing critic_exports/*.json files.",
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
        logger.error("No critic export JSON files found under %s", exports_root)
        return 1
    estimator = StageBaselineEstimator(
        tau_pln=args.tau_pln,
        tau_per=args.tau_per,
        tau_coord=args.tau_coord,
    )
    baselines, _episode_baselines, stats = estimator.estimate_grouped_from_exports(
        export_paths, family_override=args.family_override
    )
    if not baselines:
        logger.error(
            "No valid full-trajectory critic export produced a StageBaseline. "
            "Reasons: %s",
            stats.get("skipped_export_reasons", {}),
        )
        return 2
    out_path = estimator.write(
        baselines,
        args.output,
        metadata={"estimator_stats": stats},
    )
    logger.info("Wrote %d stage baselines -> %s", len(baselines), out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
