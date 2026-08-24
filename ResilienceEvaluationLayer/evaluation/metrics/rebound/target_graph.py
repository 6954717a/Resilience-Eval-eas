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
* A **stage** is a topological generation of the episode's
  ``TemporalConstraint``. Dependencies are only a compatibility fallback
  when the temporal constraint is unavailable.

The graph is intentionally *read-only* after construction: it is built once
per episode from the episode/runtime proposition tracker and then queried by
the templates with constant-time lookups.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------
PLACEMENT_PREDICATES: Tuple[str, ...] = (
    "is_on_top",
    "is_inside",
    "is_in_room",
    "is_next_to",
    "is_on_floor",
    "is_clustered",
)
STATE_FLIP_PREDICATES: Tuple[str, ...] = (
    "is_clean",
    "is_dirty",
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
        object_matches = str(object_id) in self.object_ids
        if self.predicate == "is_on_floor":
            receptacle = str(receptacle_id).strip().lower()
            allowed = {
                str(candidate).strip().lower()
                for candidate in self.receptacle_ids
            }
            # ``("floor",)`` is the only intentionally generic form.  When
            # dataset generation pairs ``is_on_floor`` with ``is_in_room``,
            # construction below replaces it with exact ``floor_<room>``
            # aliases so a placement on the wrong room's floor cannot count.
            if allowed == {"floor"}:
                return object_matches and (
                    receptacle == "floor" or receptacle.startswith("floor_")
                )
            return object_matches and receptacle in allowed
        return object_matches and str(receptacle_id) in self.receptacle_ids

    @property
    def is_action_matchable(self) -> bool:
        """Whether a Place/Rearrange action can directly prove this target.

        Room membership and relative/cluster relations require world-state
        context that is intentionally not carried by this lightweight graph.
        """
        return self.predicate in {"is_on_top", "is_inside", "is_on_floor"}


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
        if not self.stages:
            return tuple()
        if not proposition_satisfied_at:
            return (self.stages[0].stage_index,)
        for stage in self.stages:
            for idx in stage.proposition_indices:
                if 0 <= idx < len(proposition_satisfied_at):
                    if int(proposition_satisfied_at[idx]) < 0:
                        return (stage.stage_index,)
        return tuple()

    def open_placements(
        self, proposition_satisfied_at: Sequence[int]
    ) -> Tuple[PlacementTarget, ...]:
        active_stages = set(self.active_stage_indices(proposition_satisfied_at))
        return tuple(
            p for p in self.placements
            if 0 <= p.proposition_index < len(proposition_satisfied_at)
            and int(proposition_satisfied_at[p.proposition_index]) < 0
            and self.stage_of_proposition(p.proposition_index) in active_stages
        )

    def open_state_flips(
        self, proposition_satisfied_at: Sequence[int]
    ) -> Tuple[StateFlipTarget, ...]:
        active_stages = set(self.active_stage_indices(proposition_satisfied_at))
        return tuple(
            s for s in self.state_flips
            if 0 <= s.proposition_index < len(proposition_satisfied_at)
            and int(proposition_satisfied_at[s.proposition_index]) < 0
            and self.stage_of_proposition(s.proposition_index) in active_stages
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


def _expand_ids(
    values: Any,
    aliases: Optional[Mapping[Any, Any]] = None,
) -> Tuple[str, ...]:
    """Return stable raw-id + planner-name aliases without duplicates."""

    expanded: List[str] = []
    alias_map = aliases or {}
    for raw_value in _as_tuple(values):
        candidates = [raw_value]
        alias = alias_map.get(raw_value)
        if alias is None:
            # Region ids can be numeric in serialized/runtime records while
            # perception mappings use an integer key (or vice versa).
            try:
                alias = alias_map.get(int(raw_value))
            except (TypeError, ValueError):
                alias = None
        if alias is not None:
            candidates.append(str(alias))
        for candidate in candidates:
            if candidate not in expanded:
                expanded.append(candidate)
    return tuple(expanded)


def _extract_placement(
    idx: int,
    prop_map: Mapping[str, Any],
    entity_aliases: Optional[Mapping[Any, Any]] = None,
    room_aliases: Optional[Mapping[Any, Any]] = None,
) -> Optional[PlacementTarget]:
    predicate = str(prop_map.get("function_name") or "")
    if predicate not in PLACEMENT_PREDICATES:
        return None
    args = prop_map.get("args") or {}
    if not isinstance(args, Mapping):
        return None

    if predicate == "is_clustered":
        entity_groups = args.get("*args") or ()
        if not isinstance(entity_groups, (list, tuple)) or not entity_groups:
            return None
        object_ids = _expand_ids(entity_groups[0], entity_aliases)
        receptacle_ids = tuple(
            entity
            for group in entity_groups[1:]
            for entity in _expand_ids(group, entity_aliases)
        )
    else:
        # Accept both legacy name-based fields and dataset_generation handles.
        object_ids = _expand_ids(
            args.get("object_names")
            or args.get("entity_handles_a_name")
            or args.get("entity_handles_a")
            or args.get("object_handles"),
            entity_aliases,
        )
        raw_receptacles = (
            args.get("receptacle_names")
            or args.get("entity_handles_b_name")
            or args.get("entity_handles_b")
            or args.get("receptacle_handles")
            or args.get("room_ids")
        )
        aliases = room_aliases if predicate == "is_in_room" else entity_aliases
        receptacle_ids = _expand_ids(raw_receptacles, aliases)
        if predicate == "is_on_floor" and object_ids:
            receptacle_ids = ("floor",)

    raw_number = args.get("number", 1)
    number = int(raw_number or 1) if not isinstance(raw_number, (list, tuple)) else 1
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
    idx: int,
    prop_map: Mapping[str, Any],
    entity_aliases: Optional[Mapping[Any, Any]] = None,
) -> Optional[StateFlipTarget]:
    predicate = str(prop_map.get("function_name") or "")
    if predicate not in STATE_FLIP_PREDICATES:
        return None
    args = prop_map.get("args") or {}
    if not isinstance(args, Mapping):
        return None
    object_ids: Tuple[str, ...] = _expand_ids(
        args.get("object_names")
        or args.get("entity_handles_a_name")
        or args.get("object_handles"),
        entity_aliases,
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
    if predicate in ("is_dirty", "is_powered_off", "is_empty", "is_closed"):
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
    constraints: Sequence[Any] = (),
) -> Tuple[StageSpec, ...]:
    """Topologically sort propositions into stages.

    ``TemporalConstraint`` is authoritative. Sequential dependency relations
    are retained as a compatibility fallback for compact records that omit
    constraints.
    """
    pred_depth: Dict[int, int] = {i: 0 for i in range(n_props)}

    # TemporalConstraint is the authoritative task-order DAG. Runtime objects
    # expose get_topological_generations(); serialized dataset entries expose
    # type + args.dag_edges. Prefer either representation over dependencies,
    # which control proposition query eligibility rather than temporal order.
    for constraint in constraints:
        groups: Sequence[Sequence[int]] = ()
        if hasattr(constraint, "get_topological_generations"):
            try:
                groups = constraint.get_topological_generations() or ()
            except Exception:
                groups = ()
        elif isinstance(constraint, Mapping):
            constraint_type = str(constraint.get("type") or "")
            if constraint_type == "TemporalConstraint":
                constraint_args = constraint.get("args") or {}
                edges = (
                    constraint_args.get("dag_edges") or ()
                    if isinstance(constraint_args, Mapping)
                    else ()
                )
                predecessors: Dict[int, Set[int]] = {
                    i: set() for i in range(n_props)
                }
                for edge in edges:
                    if isinstance(edge, (list, tuple)) and len(edge) == 2:
                        src, dst = int(edge[0]), int(edge[1])
                        if src in predecessors and dst in predecessors:
                            predecessors[dst].add(src)
                groups_by_depth: Dict[int, List[int]] = {}
                memo: Dict[int, int] = {}

                def temporal_depth(i: int, visiting: Set[int]) -> int:
                    if i in memo:
                        return memo[i]
                    if i in visiting:
                        logger.warning("Cycle found in serialized TemporalConstraint")
                        return 0
                    visiting.add(i)
                    value = max(
                        (1 + temporal_depth(p, visiting) for p in predecessors[i]),
                        default=0,
                    )
                    visiting.discard(i)
                    memo[i] = value
                    return value

                for prop_idx in range(n_props):
                    depth = temporal_depth(prop_idx, set())
                    groups_by_depth.setdefault(depth, []).append(prop_idx)
                groups = tuple(groups_by_depth[d] for d in sorted(groups_by_depth))
        if groups:
            assigned: Set[int] = set()
            normalized: List[StageSpec] = []
            for stage_idx, group in enumerate(groups):
                valid = tuple(sorted(int(i) for i in group if 0 <= int(i) < n_props))
                if valid:
                    normalized.append(StageSpec(stage_idx, valid))
                    assigned.update(valid)
            missing = tuple(i for i in range(n_props) if i not in assigned)
            if missing:
                if normalized and normalized[0].stage_index == 0:
                    normalized[0] = StageSpec(
                        0, tuple(sorted(normalized[0].proposition_indices + missing))
                    )
                else:
                    normalized.insert(0, StageSpec(0, missing))
            return tuple(normalized)

        # An explicit empty TemporalConstraint means all propositions are in
        # one temporal generation.
        if (
            constraint.__class__.__name__ == "TemporalConstraint"
            or (
                isinstance(constraint, Mapping)
                and str(constraint.get("type") or "") == "TemporalConstraint"
            )
        ):
            return (StageSpec(0, tuple(range(n_props))),) if n_props else tuple()

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
            if relation.lower() not in {
                "before",
                "after_satisfied",
                "after_unsatisfied",
            }:
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
    *,
    entity_aliases: Optional[Mapping[Any, Any]] = None,
    room_aliases: Optional[Mapping[Any, Any]] = None,
) -> EpisodeTargetGraph:
    """Construct an :class:`EpisodeTargetGraph` from a CollaborationEpisode.

    Parameters
    ----------
    episode:
        ``CollaborationEpisode`` (or duck-typed counterpart) carrying
        ``evaluation_propositions`` and optional dependencies/constraints.
    proposition_tracker:
        Optional runtime ``info["auto_eval_proposition_tracker"]`` snapshot.
        When provided we prefer its already-unrolled ``propositions`` list so
        templates stay consistent with what the runtime measure is scoring;
        its unrolled constraints are preferred for temporal stages as well.
    entity_aliases, room_aliases:
        Runtime mappings from simulator handles/region ids to the semantic
        names emitted by planner actions. Both raw and semantic identifiers
        are retained so dataset and action records share one lookup surface.
    """
    propositions: Sequence[Any] = ()
    dependencies: Sequence[Any] = ()
    constraints: Sequence[Any] = ()
    if proposition_tracker and isinstance(proposition_tracker, Mapping):
        propositions = proposition_tracker.get("propositions") or ()
        dependencies = proposition_tracker.get("dependencies") or ()
        constraints = proposition_tracker.get("constraints") or ()
    if not propositions and episode is not None:
        propositions = getattr(episode, "evaluation_propositions", ()) or ()
    if not dependencies and episode is not None:
        dependencies = (
            getattr(episode, "evaluation_proposition_dependencies", ()) or ()
        )
    if not constraints and episode is not None:
        constraints = getattr(episode, "evaluation_constraints", ()) or ()

    placements: List[PlacementTarget] = []
    state_flips: List[StateFlipTarget] = []
    raw: List[Mapping[str, Any]] = []
    for idx, prop in enumerate(propositions):
        prop_map = _as_mapping(prop)
        raw.append(prop_map)
        placement = _extract_placement(
            idx,
            prop_map,
            entity_aliases=entity_aliases,
            room_aliases=room_aliases,
        )
        if placement is not None:
            placements.append(placement)
            continue
        state_flip = _extract_state_flip(
            idx,
            prop_map,
            entity_aliases=entity_aliases,
        )
        if state_flip is not None:
            state_flips.append(state_flip)

    stages = _compute_stages(len(propositions), dependencies, constraints)
    stage_by_proposition = {
        proposition_index: stage.stage_index
        for stage in stages
        for proposition_index in stage.proposition_indices
    }

    # dataset_generation explicitly links ``is_on_floor`` to its companion
    # ``is_in_room`` through a while_satisfied dependency. Prefer that edge:
    # object identity alone is ambiguous when a temporal task moves the same
    # object through different rooms in different stages.
    explicit_floor_rooms: Dict[int, Set[int]] = {}
    for dependency in dependencies:
        if isinstance(dependency, Mapping):
            relation = str(dependency.get("relation_type") or "")
            targets = dependency.get("proposition_indices") or ()
            predecessors = dependency.get("depends_on") or ()
        else:
            relation = str(getattr(dependency, "relation_type", "") or "")
            targets = getattr(dependency, "proposition_indices", ()) or ()
            predecessors = getattr(dependency, "depends_on", ()) or ()
        if relation != "while_satisfied":
            continue
        for target_index in targets:
            try:
                floor_index = int(target_index)
            except (TypeError, ValueError):
                continue
            for predecessor_index in predecessors:
                try:
                    room_index = int(predecessor_index)
                except (TypeError, ValueError):
                    continue
                explicit_floor_rooms.setdefault(floor_index, set()).add(
                    room_index
                )

    room_targets = {
        p.proposition_index: p
        for p in placements
        if p.predicate == "is_in_room"
    }
    for index, floor_target in enumerate(placements):
        if floor_target.predicate != "is_on_floor":
            continue
        explicit_indices = explicit_floor_rooms.get(
            floor_target.proposition_index,
            set(),
        )
        paired_rooms = [
            room_targets[room_index]
            for room_index in sorted(explicit_indices)
            if room_index in room_targets
        ]
        if not paired_rooms:
            # Compatibility fallback for legacy episodes without generated
            # dependencies. Only bind an unambiguous room target from the same
            # temporal generation; otherwise retain the generic floor target.
            fallback_rooms = [
                room_target
                for room_target in room_targets.values()
                if (
                    set(room_target.object_ids) == set(floor_target.object_ids)
                    and stage_by_proposition.get(room_target.proposition_index)
                    == stage_by_proposition.get(floor_target.proposition_index)
                )
            ]
            if len(fallback_rooms) == 1:
                paired_rooms = fallback_rooms
        if not paired_rooms:
            continue
        floor_ids: List[str] = []
        for room_target in paired_rooms:
            for room_id in room_target.receptacle_ids:
                floor_id = (
                    str(room_id)
                    if str(room_id).startswith("floor_")
                    else f"floor_{room_id}"
                )
                if floor_id not in floor_ids:
                    floor_ids.append(floor_id)
        if floor_ids:
            placements[index] = replace(
                floor_target,
                receptacle_ids=tuple(floor_ids),
            )

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
_PLACE_ACTIONS = {"place", "rearrange", "rearrange_place"}
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
    * runtime tuples ``(name, comma_separated_action_input, error)``
    * legacy tuples ``(name, primary, secondary?)``
    * strings such as ``"Navigate[obj_1]"`` and five-slot ``Place[...]``
    * mappings with ``name``/``args`` keys.
    """
    if action_entry is None:
        return None
    name: Optional[str] = None
    primary: Optional[str] = None
    secondary: Optional[str] = None

    def _parse_args(
        action_name: Optional[str], args: Any
    ) -> Tuple[Optional[str], Optional[str]]:
        if args is None:
            return None, None
        if isinstance(args, str):
            parts = [part.strip() for part in args.split(",")]
        elif isinstance(args, (list, tuple)):
            parts = [str(part).strip() for part in args if part is not None]
        else:
            parts = [str(args).strip()]
        if not parts:
            return None, None
        lowered = str(action_name or "").strip().lower()
        first = parts[0] or None
        if lowered in _PLACE_ACTIONS:
            # Place/Rearrange[obj, relation, receptacle, constraint, reference]
            if len(parts) >= 3:
                return first, parts[2] or None
            return first, (parts[1] or None) if len(parts) >= 2 else None
        return first, (parts[1] or None) if len(parts) >= 2 else None

    if isinstance(action_entry, tuple) and action_entry:
        name = str(action_entry[0]) if action_entry[0] is not None else None
        if len(action_entry) > 1 and action_entry[1] is not None:
            primary, secondary = _parse_args(name, action_entry[1])
        # Support legacy tuples (name, object, receptacle) while preserving the
        # runtime tuple contract (name, comma-separated action_input, error).
        if (
            len(action_entry) > 2
            and action_entry[2] is not None
            and isinstance(action_entry[1], str)
            and "," not in action_entry[1]
        ):
            secondary = str(action_entry[2])
    elif isinstance(action_entry, Mapping):
        name = action_entry.get("name") or action_entry.get("action")
        args = action_entry.get("args")
        if isinstance(args, (list, tuple)) and args:
            primary, secondary = _parse_args(name, args)
        elif isinstance(args, Mapping):
            primary = args.get("object") or args.get("object_id")
            secondary = args.get("receptacle") or args.get("receptacle_id")
        else:
            primary, secondary = _parse_args(name, args)
    elif isinstance(action_entry, str):
        text = action_entry.strip()
        if "[" in text and text.endswith("]"):
            head, body = text[:-1].split("[", 1)
            name = head
            primary, secondary = _parse_args(name, body)
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
