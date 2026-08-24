#!/usr/bin/env python3
# isort: skip_file

r"""
Episode Target Graph
====================

Semantic representation of *what the episode is actually trying to achieve*
on top of PartnR's ``evaluation_propositions``.  It is the shared substrate
used by the semantic Rebound templates (T1–T4) and by the
:class:`OnlineReboundTracker` to decide whether an action is making progress
against the **intended** target or merely wandering.

Design notes
------------
* PartnR's ``evaluation_propositions`` are flat predicates such as
  ``is_on_top(obj, rec)``, ``is_inside(obj, rec)``,
  ``is_clean(obj, True)``, ``is_powered_on(obj, True)``,
  ``is_filled(obj, True)`` …  Each has an integer index that is referenced by
  ``evaluation_proposition_dependencies`` to encode *staging* (``depends_on``
  + ``relation_type=before``/``after``).
* We group propositions into two categories:

  - **placement** propositions → :class:`PlacementTarget`
  - **state-flip** propositions → :class:`StateFlipTarget`

  Navigation-only movements that are not tied to a placement target are
  classified as ``irrelevant`` and penalised by template **T3**
  (``phase_region_deviation``).
* A **stage** is a connected component in the topologically sorted
  dependency DAG.  A stage carries the set of propositions that must be
  satisfied before the *next* stage unlocks; the stage index is exactly
  the ``ell_t`` used by :class:`StageResolver`.

The graph is intentionally *read-only* after construction: it is built once
per episode from the (mutated) ``env_interface`` at rollout start and then
queried by the templates with constant-time lookups.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------
PLACEMENT_PREDICATES: Tuple[str, ...] = (
    "is_on_top",
    "is_inside",
    "is_next_to",
    "is_on_floor",
)
STATE_FLIP_PREDICATES: Tuple[str, ...] = (
    "is_clean",
    "is_powered_on",
    "is_powered_off",
    "is_filled",
    "is_empty",
    "is_open",
    "is_closed",
)


@dataclass(frozen=True)
class PlacementTarget:
    """One intended object→receptacle placement."""

    proposition_index: int
    predicate: str                 # one of PLACEMENT_PREDICATES
    object_ids: Tuple[str, ...]    # all semantically equivalent object handles
    receptacle_ids: Tuple[str, ...]
    number: int = 1                # how many of ``object_ids`` must satisfy

    def matches(
        self,
        object_id: Optional[str],
        receptacle_id: Optional[str],
    ) -> bool:
        if object_id is None or receptacle_id is None:
            return False
        return (
            str(object_id) in self.object_ids
            and str(receptacle_id) in self.receptacle_ids
        )


@dataclass(frozen=True)
class StateFlipTarget:
    """One intended state flip (``is_clean``/``is_powered_on``/...)."""

    proposition_index: int
    predicate: str                 # one of STATE_FLIP_PREDICATES
    object_ids: Tuple[str, ...]
    target_value: bool = True

    def matches(self, object_id: Optional[str]) -> bool:
        if object_id is None:
            return False
        return str(object_id) in self.object_ids


@dataclass(frozen=True)
class StageSpec:
    r"""One stage :math:`\ell` in the episode DAG."""

    stage_index: int
    proposition_indices: Tuple[int, ...]

    def total_propositions(self) -> int:
        return len(self.proposition_indices)


@dataclass
class EpisodeTargetGraph:
    """The shared semantic target graph for a single episode.

    Exposes constant-time query helpers used by all Rebound templates.
    """

    placements: Tuple[PlacementTarget, ...] = field(default_factory=tuple)
    state_flips: Tuple[StateFlipTarget, ...] = field(default_factory=tuple)
    stages: Tuple[StageSpec, ...] = field(default_factory=tuple)

    # Reverse indices (lazily built so the object stays easily serialisable)
    _placement_by_object: Dict[str, Tuple[PlacementTarget, ...]] = field(
        default_factory=dict, repr=False
    )
    _state_flip_by_object: Dict[str, Tuple[StateFlipTarget, ...]] = field(
        default_factory=dict, repr=False
    )
    _stage_of_prop: Dict[int, int] = field(default_factory=dict, repr=False)
    _raw_propositions: Tuple[Mapping[str, Any], ...] = field(
        default_factory=tuple, repr=False
    )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def placements_for_object(self, object_id: str) -> Tuple[PlacementTarget, ...]:
        return self._placement_by_object.get(str(object_id), ())

    def state_flips_for_object(self, object_id: str) -> Tuple[StateFlipTarget, ...]:
        return self._state_flip_by_object.get(str(object_id), ())

    def stage_of_proposition(self, prop_index: int) -> int:
        return int(self._stage_of_prop.get(int(prop_index), -1))

    def is_placement_target(
        self, object_id: Optional[str], receptacle_id: Optional[str]
    ) -> bool:
        for p in self.placements_for_object(str(object_id) if object_id else ""):
            if p.matches(object_id, receptacle_id):
                return True
        return False

    def is_state_flip_target(self, object_id: Optional[str]) -> bool:
        return bool(self.state_flips_for_object(str(object_id) if object_id else ""))

    def active_stage_indices(
        self, proposition_satisfied_at: Sequence[int]
    ) -> Tuple[int, ...]:
        """Return the current staging frontier.

        A proposition is *active* if it is not yet satisfied AND all of its
        predecessor stages are fully satisfied.  We return the set of stage
        indices that contain at least one active proposition.
        """
        if not proposition_satisfied_at:
            return tuple(s.stage_index for s in self.stages)
        active: List[int] = []
        for stage in self.stages:
            is_active = False
            for idx in stage.proposition_indices:
                if 0 <= idx < len(proposition_satisfied_at):
                    if int(proposition_satisfied_at[idx]) < 0:
                        is_active = True
                        break
            if is_active:
                active.append(stage.stage_index)
        return tuple(active) if active else tuple()

    def open_placements(
        self, proposition_satisfied_at: Sequence[int]
    ) -> Tuple[PlacementTarget, ...]:
        return tuple(
            p for p in self.placements
            if 0 <= p.proposition_index < len(proposition_satisfied_at)
            and int(proposition_satisfied_at[p.proposition_index]) < 0
        )

    def open_state_flips(
        self, proposition_satisfied_at: Sequence[int]
    ) -> Tuple[StateFlipTarget, ...]:
        return tuple(
            s for s in self.state_flips
            if 0 <= s.proposition_index < len(proposition_satisfied_at)
            and int(proposition_satisfied_at[s.proposition_index]) < 0
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "placements": [
                {
                    "proposition_index": int(p.proposition_index),
                    "predicate": p.predicate,
                    "object_ids": list(p.object_ids),
                    "receptacle_ids": list(p.receptacle_ids),
                    "number": int(p.number),
                }
                for p in self.placements
            ],
            "state_flips": [
                {
                    "proposition_index": int(s.proposition_index),
                    "predicate": s.predicate,
                    "object_ids": list(s.object_ids),
                    "target_value": bool(s.target_value),
                }
                for s in self.state_flips
            ],
            "stages": [
                {
                    "stage_index": int(st.stage_index),
                    "proposition_indices": list(st.proposition_indices),
                }
                for st in self.stages
            ],
        }


# ----------------------------------------------------------------------
# Construction helpers
# ----------------------------------------------------------------------
def _as_mapping(prop: Any) -> Mapping[str, Any]:
    """Best-effort coercion of a Proposition instance into a mapping."""
    if isinstance(prop, Mapping):
        return prop
    payload: Dict[str, Any] = {}
    for attr in ("function_name", "args", "number"):
        if hasattr(prop, attr):
            payload[attr] = getattr(prop, attr)
    return payload


def _as_tuple(values: Any) -> Tuple[str, ...]:
    if values is None:
        return tuple()
    if isinstance(values, str):
        return (values,)
    if isinstance(values, Iterable):
        return tuple(str(v) for v in values if v is not None)
    return (str(values),)


def _extract_placement(
    idx: int, prop_map: Mapping[str, Any]
) -> Optional[PlacementTarget]:
    predicate = str(prop_map.get("function_name") or "")
    if predicate not in PLACEMENT_PREDICATES:
        return None
    args = prop_map.get("args") or {}
    if not isinstance(args, Mapping):
        return None

    # PartnR uses either ``object_names`` + ``receptacle_names`` (list form)
    # or ``entity_handles_a_name`` + ``entity_handles_b_name``. Support both.
    object_ids: Tuple[str, ...] = _as_tuple(
        args.get("object_names")
        or args.get("entity_handles_a_name")
        or args.get("object_handles")
    )
    receptacle_ids: Tuple[str, ...] = _as_tuple(
        args.get("receptacle_names")
        or args.get("entity_handles_b_name")
        or args.get("receptacle_handles")
    )
    number = int(args.get("number", 1) or 1)
    if not object_ids or not receptacle_ids:
        return None
    return PlacementTarget(
        proposition_index=int(idx),
        predicate=predicate,
        object_ids=object_ids,
        receptacle_ids=receptacle_ids,
        number=number,
    )


def _extract_state_flip(
    idx: int, prop_map: Mapping[str, Any]
) -> Optional[StateFlipTarget]:
    predicate = str(prop_map.get("function_name") or "")
    if predicate not in STATE_FLIP_PREDICATES:
        return None
    args = prop_map.get("args") or {}
    if not isinstance(args, Mapping):
        return None
    object_ids: Tuple[str, ...] = _as_tuple(
        args.get("object_names")
        or args.get("entity_handles_a_name")
        or args.get("object_handles")
    )
    if not object_ids:
        return None
    # Predicate value: default True for is_clean/is_powered_on etc.
    target_value = True
    for key in ("value", "target_value"):
        if key in args:
            try:
                target_value = bool(args[key])
            except Exception:
                target_value = True
    if predicate in ("is_powered_off", "is_empty", "is_closed"):
        target_value = False
    return StateFlipTarget(
        proposition_index=int(idx),
        predicate=predicate,
        object_ids=object_ids,
        target_value=target_value,
    )


def _compute_stages(
    n_props: int,
    dependencies: Sequence[Any],
) -> Tuple[StageSpec, ...]:
    """Topologically sort propositions into stages.

    ``dependencies`` entries are expected to expose either mapping keys or
    attributes ``proposition_indices`` (set this stage) and ``depends_on``
    (must precede).  We collapse strongly connected predecessor sets into
    a single stage: any proposition whose dependency chain has length ``k``
    lands in stage ``k``.
    """
    pred_depth: Dict[int, int] = {i: 0 for i in range(n_props)}
    if dependencies:
        # Build a predecessor map
        predecessors: Dict[int, Set[int]] = {i: set() for i in range(n_props)}
        for dep in dependencies:
            if isinstance(dep, Mapping):
                targets = dep.get("proposition_indices") or []
                preds = dep.get("depends_on") or []
                relation = str(dep.get("relation_type", "before"))
            else:
                targets = getattr(dep, "proposition_indices", []) or []
                preds = getattr(dep, "depends_on", []) or []
                relation = str(getattr(dep, "relation_type", "before"))
            if relation.lower() != "before":
                continue
            for t in targets:
                try:
                    t_int = int(t)
                except (TypeError, ValueError):
                    continue
                if t_int not in predecessors:
                    continue
                for p in preds:
                    try:
                        p_int = int(p)
                    except (TypeError, ValueError):
                        continue
                    if p_int in predecessors:
                        predecessors[t_int].add(p_int)
        # Longest path from a root gives the stage depth
        memo: Dict[int, int] = {}

        def depth(i: int, visiting: Set[int]) -> int:
            if i in memo:
                return memo[i]
            if i in visiting:
                # cycle - treat as depth 0 to avoid infinite recursion
                memo[i] = 0
                return 0
            visiting.add(i)
            d = 0
            for p in predecessors.get(i, ()):  # type: ignore[arg-type]
                d = max(d, 1 + depth(p, visiting))
            visiting.discard(i)
            memo[i] = d
            return d

        for i in range(n_props):
            pred_depth[i] = depth(i, set())

    bucket: Dict[int, List[int]] = {}
    for i, d in pred_depth.items():
        bucket.setdefault(d, []).append(i)
    return tuple(
        StageSpec(stage_index=d, proposition_indices=tuple(sorted(v)))
        for d, v in sorted(bucket.items())
    )


def build_episode_target_graph(
    episode: Any,
    proposition_tracker: Optional[Mapping[str, Any]] = None,
) -> EpisodeTargetGraph:
    """Construct an :class:`EpisodeTargetGraph` from a CollaborationEpisode.

    Parameters
    ----------
    episode:
        ``CollaborationEpisode`` (or duck-typed counterpart) carrying
        ``evaluation_propositions`` and optional ``evaluation_proposition_dependencies``.
    proposition_tracker:
        Optional runtime ``info["auto_eval_proposition_tracker"]`` snapshot.
        When provided we prefer its already-unrolled ``propositions`` list so
        templates stay consistent with what the runtime measure is scoring.
    """
    propositions: Sequence[Any] = ()
    dependencies: Sequence[Any] = ()
    if proposition_tracker and isinstance(proposition_tracker, Mapping):
        propositions = proposition_tracker.get("propositions") or ()
        dependencies = proposition_tracker.get("dependencies") or ()
    if not propositions and episode is not None:
        propositions = getattr(episode, "evaluation_propositions", ()) or ()
    if not dependencies and episode is not None:
        dependencies = (
            getattr(episode, "evaluation_proposition_dependencies", ()) or ()
        )

    placements: List[PlacementTarget] = []
    state_flips: List[StateFlipTarget] = []
    raw: List[Mapping[str, Any]] = []
    for idx, prop in enumerate(propositions):
        prop_map = _as_mapping(prop)
        raw.append(prop_map)
        placement = _extract_placement(idx, prop_map)
        if placement is not None:
            placements.append(placement)
            continue
        state_flip = _extract_state_flip(idx, prop_map)
        if state_flip is not None:
            state_flips.append(state_flip)

    stages = _compute_stages(len(propositions), dependencies)

    placement_by_object: Dict[str, Tuple[PlacementTarget, ...]] = {}
    for p in placements:
        for obj in p.object_ids:
            placement_by_object[obj] = placement_by_object.get(obj, ()) + (p,)
    state_flip_by_object: Dict[str, Tuple[StateFlipTarget, ...]] = {}
    for s in state_flips:
        for obj in s.object_ids:
            state_flip_by_object[obj] = state_flip_by_object.get(obj, ()) + (s,)
    stage_of_prop: Dict[int, int] = {}
    for st in stages:
        for i in st.proposition_indices:
            stage_of_prop[int(i)] = int(st.stage_index)

    return EpisodeTargetGraph(
        placements=tuple(placements),
        state_flips=tuple(state_flips),
        stages=stages,
        _placement_by_object=placement_by_object,
        _state_flip_by_object=state_flip_by_object,
        _stage_of_prop=stage_of_prop,
        _raw_propositions=tuple(raw),
    )


# ----------------------------------------------------------------------
# Action parsing helpers (shared with templates)
# ----------------------------------------------------------------------
_NAV_ACTIONS = {"navigate", "goto", "explore", "findagent"}
_PICK_ACTIONS = {"pick", "rearrange_pick"}
_PLACE_ACTIONS = {"place", "rearrange_place"}
_MANIP_ACTIONS = _PICK_ACTIONS | _PLACE_ACTIONS | {
    "open",
    "close",
    "clean",
    "poweron",
    "power_on",
    "poweroff",
    "power_off",
    "fill",
    "pour",
}


@dataclass(frozen=True)
class ActionFrame:
    """Semantic normalization of one high-level action tuple."""

    name: str                      # lower-case skill name, e.g. "navigate"
    primary_arg: Optional[str]     # e.g. the object id / receptacle id
    secondary_arg: Optional[str] = None
    agent_uid: Optional[int] = None
    step: int = -1
    raw: Any = None

    @property
    def is_nav(self) -> bool:
        return self.name in _NAV_ACTIONS

    @property
    def is_pick(self) -> bool:
        return self.name in _PICK_ACTIONS

    @property
    def is_place(self) -> bool:
        return self.name in _PLACE_ACTIONS

    @property
    def is_manipulation(self) -> bool:
        return self.name in _MANIP_ACTIONS


def parse_action_entry(
    action_entry: Any,
    step: int = -1,
    agent_uid: Optional[int] = None,
) -> Optional[ActionFrame]:
    """Normalise an entry from ``planner_info['high_level_actions']``.

    Handles:
    * tuples ``(name, primary, secondary?)``
    * strings like ``"Navigate[obj_1]"`` / ``"Place[obj_1, rec_2]"``
    * mappings with ``name``/``args`` keys.
    """
    if action_entry is None:
        return None
    name: Optional[str] = None
    primary: Optional[str] = None
    secondary: Optional[str] = None

    def _select_secondary(action_name: Optional[str], parts: Sequence[Any]) -> Optional[str]:
        if not parts:
            return None
        lowered = str(action_name).strip().lower() if action_name is not None else ""
        normalized = [str(p) for p in parts if p is not None]
        if not normalized:
            return None
        # Place / Rearrange typically look like:
        #   Place[obj, on, receptacle, ...]
        # so the receptacle lives in arg[2], not arg[1].
        if lowered in {"place", "rearrange", "rearrange_place"} and len(normalized) > 2:
            return normalized[2]
        if len(normalized) > 1:
            return normalized[1]
        return None

    if isinstance(action_entry, tuple) and action_entry:
        name = str(action_entry[0]) if action_entry[0] is not None else None
        if len(action_entry) > 1 and action_entry[1] is not None:
            primary = str(action_entry[1])
        secondary = _select_secondary(name, action_entry[1:])
    elif isinstance(action_entry, Mapping):
        name = action_entry.get("name") or action_entry.get("action")
        args = action_entry.get("args")
        if isinstance(args, (list, tuple)) and args:
            primary = str(args[0]) if args[0] is not None else None
            secondary = _select_secondary(name, args)
        elif isinstance(args, Mapping):
            primary = args.get("object") or args.get("object_id")
            secondary = args.get("receptacle") or args.get("receptacle_id")
    elif isinstance(action_entry, str):
        text = action_entry.strip()
        if "[" in text and text.endswith("]"):
            head, body = text[:-1].split("[", 1)
            name = head
            parts = [p.strip() for p in body.split(",") if p.strip()]
            if parts:
                primary = parts[0]
            secondary = _select_secondary(name, parts)
        else:
            name = text
    else:
        # Last-ditch fallback: stringify and try again
        return parse_action_entry(str(action_entry), step=step, agent_uid=agent_uid)

    if not name:
        return None
    return ActionFrame(
        name=str(name).strip().lower(),
        primary_arg=primary,
        secondary_arg=secondary,
        agent_uid=agent_uid,
        step=int(step),
        raw=action_entry,
    )


__all__ = [
    "PlacementTarget",
    "StateFlipTarget",
    "StageSpec",
    "EpisodeTargetGraph",
    "ActionFrame",
    "parse_action_entry",
    "build_episode_target_graph",
    "PLACEMENT_PREDICATES",
    "STATE_FLIP_PREDICATES",
]
