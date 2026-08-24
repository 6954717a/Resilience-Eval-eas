#!/usr/bin/env python3

"""
Stage Anchor Resolver
=====================

Implements the stage-based perturbation anchor ``a_t = (f, ℓ_t, m_t)`` used by
the Rebound :math:`C_{rec}` pipeline (see resilience design doc §1.1).

This module is intentionally dependency-light: it reads from ``info`` dicts
produced by ``EnvironmentInterface.step`` and from the ``action_history``
strings stored on either the env_interface or the critic export records.
It can therefore be used both **in-process** (consumed by the runtime
:class:`ReboundCollector`) and **offline** (consumed by the
:class:`StageBaselineEstimator`).

Public API:
    - :class:`StageAnchor`: lightweight dataclass describing ``(f, ℓ_t, m_t)``
    - :class:`StageResolver`: stateless helpers to derive anchors from info dicts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

# Keep the skill -> modality mapping here so it is the single source of truth
# for both runtime collection and offline baseline estimation.
_NAVIGATION_SKILL_PREFIXES = ("navigate", "explore", "goto", "findagent")
_MANIPULATION_SKILL_PREFIXES = (
    "pick",
    "place",
    "open",
    "close",
    "clean",
    "power",
    "fill",
    "pour",
    "rearrange_pick",
    "rearrange_place",
)
_COORDINATION_SKILL_PREFIXES = ("rearrange", "wait", "handover", "coordinate")


@dataclass(frozen=True)
class StageAnchor:
    r"""Stage-level anchor ``a_t = (f, \ell_t, m_t)``.

    Attributes
    ----------
    family:
        Task family as returned by
        :func:`habitat_llm.evaluation.experiments.common.infer_task_family`.
    stage:
        Constrained subgoal frontier stage ``ell_t`` (0-indexed). ``-1``
        means all staged propositions have been satisfied (terminal) or no
        staging info was available.
    modality:
        One of ``{"planning", "navigation", "manipulation", "coordination"}``.
    progress:
        Optional sub-stage progress :math:`\eta_t \in [0,1]` computed as the
        satisfied fraction within the current stage (for temporal DAGs) or as
        the global ``task_percent_complete`` fallback.
    """

    family: str = "other"
    stage: int = -1
    modality: str = "planning"
    progress: float = 0.0

    def key(self) -> Tuple[str, int, str]:
        r"""Group key ``(f, \ell, m)`` for baseline bucketing."""
        return (self.family, int(self.stage), self.modality)

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "stage": int(self.stage),
            "modality": self.modality,
            "progress": float(self.progress),
        }


def _classify_modality(action_string: Optional[str]) -> str:
    """Map a raw skill name / high-level action string to a modality label."""
    if not action_string:
        return "planning"
    if not isinstance(action_string, str):
        action_string = str(action_string)
    name = action_string.strip().lower()
    if not name:
        return "planning"
    # action_history entries look like ``"Navigate[obj_x]"`` or
    # ``"Pick[obj_y]"``; strip the bracket payload.
    bracket_idx = name.find("[")
    if bracket_idx >= 0:
        name = name[:bracket_idx]
    # ``high_level_actions`` may arrive as ``("Navigate", "obj_x", ...)`` tuples
    if "," in name and name.startswith("("):
        head = name.strip("()").split(",", 1)[0].strip().strip("'\"")
        name = head
    if any(name.startswith(p) for p in _NAVIGATION_SKILL_PREFIXES):
        return "navigation"
    if any(name.startswith(p) for p in _MANIPULATION_SKILL_PREFIXES):
        return "manipulation"
    if any(name.startswith(p) for p in _COORDINATION_SKILL_PREFIXES):
        return "coordination"
    return "planning"


def _extract_action_string(action_entry: Any) -> Optional[str]:
    """Best-effort extractor for the leading skill name from a history entry."""
    if action_entry is None:
        return None
    if isinstance(action_entry, str):
        return action_entry
    if isinstance(action_entry, Mapping):
        for key in ("action", "name", "skill", "high_level_action"):
            value = action_entry.get(key)
            if isinstance(value, str) and value:
                return value
        return None
    if isinstance(action_entry, Sequence) and not isinstance(action_entry, (bytes, bytearray)):
        if len(action_entry) > 0:
            head = action_entry[0]
            if isinstance(head, str):
                return head
            if isinstance(head, Mapping):
                return _extract_action_string(head)
    # Fallback to ``repr`` so tuples from habitat ActionHistoryElement still map.
    return str(action_entry)


class StageResolver:
    """Stateless resolver for stage anchors.

    Accepts either an ``info`` dict (runtime) or a compact record mapping
    (offline; e.g. critic_exports JSON lines).
    """

    def __init__(self, default_family: str = "other") -> None:
        self.default_family = default_family

    @staticmethod
    def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _task_progress(self, info: Mapping[str, Any]) -> Optional[float]:
        progress = self._safe_float(info.get("task_percent_complete"))
        if progress is None:
            return None
        return max(0.0, min(1.0, progress))

    def _stage_from_progress(self, progress: Optional[float], total: int) -> int:
        if progress is None or total <= 0:
            return -1
        if progress >= 1.0 - 1e-6:
            return -1
        return max(0, min(total - 1, int(progress * total)))

    # ------------------------------------------------------------------
    # ℓ_t: constrained subgoal frontier stage
    # ------------------------------------------------------------------
    def current_stage(self, info: Optional[Mapping[str, Any]]) -> int:
        r"""Compute the constrained subgoal frontier ``\ell_t``.

        Runtime records carry
        ``info["auto_eval_proposition_tracker"]["proposition_satisfied_at"]``.
        Compact critic exports may only carry a flattened
        ``proposition_satisfied_at`` or ``proposition_satisfied_count`` /
        ``proposition_total`` pair.  All three representations describe the
        same stage anchor; using them here keeps clean-baseline estimation and
        online Rebound windows on the same key space without falling back to a
        family-level baseline.
        """
        if not isinstance(info, Mapping):
            return -1
        tracker = info.get("auto_eval_proposition_tracker") or {}
        satisfied_at = (
            tracker.get("proposition_satisfied_at")
            if isinstance(tracker, Mapping)
            else None
        )
        if not isinstance(satisfied_at, (list, tuple)):
            satisfied_at = info.get("proposition_satisfied_at")
        progress = self._task_progress(info)
        if not isinstance(satisfied_at, (list, tuple)):
            satisfied = self._safe_int(info.get("proposition_satisfied_count"), -1)
            total = self._safe_int(info.get("proposition_total"), -1)
            if total > 0:
                if 0 <= satisfied < total:
                    return int(satisfied)
                progress_stage = self._stage_from_progress(progress, total)
                return progress_stage if progress_stage >= 0 else -1
            return -1
        for idx, value in enumerate(satisfied_at):
            # -1 / None / unresolved => unsatisfied
            v = self._safe_int(value, -1) if value is not None else -1
            if v < 0:
                return int(idx)
        # The tracker metric may be stored by mutable reference and later show
        # terminal satisfied_at values for every historical critic record.  If
        # task progress still says the transition is non-terminal, recover the
        # frontier from progress rather than collapsing the clean baseline to
        # stage=-1.
        progress_stage = self._stage_from_progress(progress, len(satisfied_at))
        if progress_stage >= 0:
            return progress_stage
        return -1

    def stage_progress(
        self, info: Optional[Mapping[str, Any]], stage: int
    ) -> float:
        r"""Compute ``\eta_t``: satisfied fraction among all propositions.

        Uses ``proposition_satisfied_fraction`` when present; otherwise
        falls back to deriving it from ``proposition_satisfied_at``.
        """
        if not isinstance(info, Mapping):
            return 0.0
        task_progress = self._task_progress(info)
        q = info.get("proposition_satisfied_fraction")
        if q is None:
            tracker = info.get("auto_eval_proposition_tracker") or {}
            satisfied_at = (
                tracker.get("proposition_satisfied_at")
                if isinstance(tracker, Mapping)
                else None
            )
            if not isinstance(satisfied_at, (list, tuple)):
                satisfied_at = info.get("proposition_satisfied_at")
            if isinstance(satisfied_at, (list, tuple)) and satisfied_at:
                total = len(satisfied_at)
                satisfied = sum(
                    1 for v in satisfied_at if isinstance(v, (int, float)) and int(v) >= 0
                )
                q = satisfied / total if total > 0 else 0.0
            else:
                try:
                    satisfied = int(info.get("proposition_satisfied_count", 0))
                    total = int(info.get("proposition_total", 0))
                except (TypeError, ValueError):
                    satisfied = 0
                    total = 0
                q = satisfied / total if total > 0 else 0.0
        q_float = self._safe_float(q)
        if (
            task_progress is not None
            and q_float is not None
            and q_float >= 1.0 - 1e-6
            and task_progress < 1.0 - 1e-6
        ):
            return float(task_progress)
        if q_float is not None:
            return float(q_float)
        if task_progress is not None:
            return float(task_progress)
        try:
            return float(q)
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------
    # m_t: planning / navigation / manipulation / coordination
    # ------------------------------------------------------------------
    def current_modality(
        self,
        info: Optional[Mapping[str, Any]],
        low_level_action_count: Optional[int] = None,
        fallback_action: Optional[str] = None,
    ) -> str:
        r"""Derive modality ``m_t`` from the most recent executable action.

        Preference order:
          1. ``info["action_history"]`` last entry (same format as PartnR export)
          2. explicit ``fallback_action`` (used when resolver is called from
             runtime with ``planner_info["high_level_actions"]``)
          3. ``low_level_action_count == 0`` => ``planning``
        """
        action_str: Optional[str] = None
        if isinstance(info, Mapping):
            history = info.get("action_history")
            if isinstance(history, (list, tuple)) and history:
                action_str = _extract_action_string(history[-1])
        if not action_str and fallback_action is not None:
            action_str = _extract_action_string(fallback_action)
        if not action_str:
            if low_level_action_count is not None and int(low_level_action_count) == 0:
                return "planning"
            return "planning"
        return _classify_modality(action_str)

    # ------------------------------------------------------------------
    # f: task family (delegates to shared helper)
    # ------------------------------------------------------------------
    def resolve_family(
        self,
        episode_or_info: Any,
        fallback: Optional[str] = None,
    ) -> str:
        if fallback:
            return fallback
        # 1) info may already carry a precomputed ``task_family`` (critic exports do).
        if isinstance(episode_or_info, Mapping):
            candidate = episode_or_info.get("task_family")
            if isinstance(candidate, str) and candidate:
                return candidate
        # 2) fall back to the canonical inference routine
        try:
            from habitat_llm.evaluation.experiments.common import infer_task_family

            return infer_task_family(episode_or_info)
        except Exception:
            return self.default_family

    # ------------------------------------------------------------------
    # convenience: build full StageAnchor
    # ------------------------------------------------------------------
    def resolve(
        self,
        info: Optional[Mapping[str, Any]],
        episode: Any = None,
        low_level_action_count: Optional[int] = None,
        fallback_action: Optional[str] = None,
        family_override: Optional[str] = None,
    ) -> StageAnchor:
        family = self.resolve_family(episode if episode is not None else info, family_override)
        stage = self.current_stage(info)
        modality = self.current_modality(
            info,
            low_level_action_count=low_level_action_count,
            fallback_action=fallback_action,
        )
        progress = self.stage_progress(info, stage)
        return StageAnchor(family=family, stage=stage, modality=modality, progress=progress)

    # ------------------------------------------------------------------
    # Sequence helpers
    # ------------------------------------------------------------------
    def resolve_sequence(
        self,
        infos: Iterable[Mapping[str, Any]],
        episode: Any = None,
        family_override: Optional[str] = None,
    ) -> List[StageAnchor]:
        """Resolve anchors for an entire trajectory of ``info`` dicts.

        Used offline by :class:`StageBaselineEstimator` and aggregated
        in :func:`ReboundCollector.compute_stage_crec`.
        """
        anchors: List[StageAnchor] = []
        for info in infos:
            anchors.append(
                self.resolve(
                    info,
                    episode=episode,
                    family_override=family_override,
                )
            )
        return anchors

    @staticmethod
    def stage_switches(anchors: Sequence[StageAnchor]) -> int:
        r"""Count how many times ``(\ell, m)`` changes along the trajectory."""
        if not anchors:
            return 0
        switches = 0
        prev_key: Optional[Tuple[int, str]] = None
        for anchor in anchors:
            key = (int(anchor.stage), anchor.modality)
            if prev_key is not None and key != prev_key:
                switches += 1
            prev_key = key
        return switches
