#!/usr/bin/env python3

r"""
Runtime perturbation injector.
==============================

Implements the world-level and prompt-level perturbation hooks used by the
Stability (:math:`\hat\beta`) experiments in §2 of the resilience design doc.

Design constraints
------------------
* The injector is **opt-in** — default evaluation paths are unaffected.
* It is layered: ``world_level`` perturbations modify the shared
  :class:`WorldGraph` snapshot that the planner receives, while
  ``prompt_level`` perturbations rewrite the instruction string that
  :class:`LLMPlanner` consumes when building prompts.
* Hooks are idempotent within a rollout.  The runner calls
  :meth:`PerturbationInjector.begin_rollout` before applying a new rollout so
  an environment/world-graph reset cannot reuse an audit from an older one.

Integration
-----------
* In :meth:`EvaluationRunner.run_instruction` (after ``reset_planners``),
  call::

      injector.apply_world_perturbation(env_interface, episode)
      instruction = injector.rewrite_instruction(instruction, episode)

Prompt rewriting is owned by ``EvaluationRunner``. The legacy planner hook is
kept as a compatibility API for external callers but must not be installed at
the same time, because that would apply the perturbation twice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .audit import PerturbationAudit, prompt_audit
from .bank import (
    DEFAULT_TEXT_DOSE_TOLERANCE,
    STRESS_FAMILY,
    FrozenPerturbationBank,
    FrozenPerturbationBankError,
    build_episode_task_contract,
)
from .paraphrase_bank import ParaphraseBank

logger = logging.getLogger(__name__)


# Canonical perturbation taxonomy (must match keys used in resilience_raw.csv).
WORLD_LEVEL_PERTURBATIONS: Tuple[str, ...] = (
    "room_shift",
    "holding_toggle",
    "object_state_toggle",
    "irrelevant_perturbation",
)
PROMPT_LEVEL_PERTURBATIONS: Tuple[str, ...] = (
    "surface_rewrite",
    "filler_injection",
    "llm_instruction_context_stress",
)
ALL_PERTURBATIONS: Tuple[str, ...] = WORLD_LEVEL_PERTURBATIONS + PROMPT_LEVEL_PERTURBATIONS

_OBJECT_STATE_PREDICATES: Mapping[str, Tuple[str, bool]] = {
    "is_clean": ("is_clean", True),
    "is_dirty": ("is_clean", False),
    "is_filled": ("is_filled", True),
    "is_empty": ("is_filled", False),
    "is_powered_on": ("is_powered_on", True),
    "is_powered_off": ("is_powered_on", False),
}


@dataclass
class PerturbationSpec:
    """Declarative description of a perturbation to apply to one rollout."""

    kind: str = "clean"          # one of ALL_PERTURBATIONS or "clean"
    seed: int = 0
    intensity: float = 1.0       # stress magnitude λ (§3 AUDC), 0.0 = no-op
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.intensity, bool) or not isinstance(
            self.intensity, (int, float)
        ):
            raise TypeError("intensity must be a real number in [0, 1]")
        intensity = float(self.intensity)
        if not math.isfinite(intensity) or not 0.0 <= intensity <= 1.0:
            raise ValueError(
                f"intensity must be finite and in [0, 1]; got {self.intensity!r}"
            )
        self.intensity = intensity
        self.seed = int(self.seed)
        self.kind = str(self.kind)
        self.extras = dict(self.extras)
        if self.kind not in ALL_PERTURBATIONS and self.kind not in (
            "clean",
            "",
            "baseline",
        ):
            raise ValueError(f"Unknown perturbation kind: {self.kind!r}")

    @property
    def is_clean(self) -> bool:
        return self.kind in ("clean", "", "baseline")

    @property
    def is_world_level(self) -> bool:
        return self.kind in WORLD_LEVEL_PERTURBATIONS

    @property
    def is_prompt_level(self) -> bool:
        return self.kind in PROMPT_LEVEL_PERTURBATIONS

    def tag(self) -> str:
        return f"{self.kind}@seed={self.seed}@lambda={self.intensity:.2f}"


@dataclass(frozen=True)
class _ObjectStateMutationPlan:
    """One episode-grounded planner-belief state corruption."""

    proposition_index: int
    predicate: str
    state_key: str
    entity_ids: Tuple[str, ...]
    goal_value: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposition_index": int(self.proposition_index),
            "predicate": self.predicate,
            "state": self.state_key,
            "entity_ids": list(self.entity_ids),
            "goal_value": bool(self.goal_value),
        }


class PerturbationInjector:
    """Opt-in runtime perturbation helper used by the E2 / E3 runners.

    Parameters
    ----------
    spec:
        :class:`PerturbationSpec` controlling this run's perturbation.
    paraphrase_bank:
        Optional :class:`ParaphraseBank`; created with defaults otherwise.
    irrelevant_object_count:
        Count of distractor objects injected when ``kind ==
        "irrelevant_perturbation"``. Scales with ``intensity``.
    """

    def __init__(
        self,
        spec: Optional[PerturbationSpec] = None,
        paraphrase_bank: Optional[ParaphraseBank] = None,
        irrelevant_object_count: int = 5,
    ) -> None:
        self.spec = spec or PerturbationSpec()
        self.paraphrase_bank = paraphrase_bank or ParaphraseBank()
        self.irrelevant_object_count = int(max(1, irrelevant_object_count))
        self._applied_key: Optional[Tuple[str, int, str]] = None
        self._latest_audit: Optional[PerturbationAudit] = None
        self._latest_world_audit: Optional[PerturbationAudit] = None
        self._latest_prompt_audit: Optional[PerturbationAudit] = None
        self._frozen_bank: Optional[FrozenPerturbationBank] = None
        self._frozen_bank_error: Optional[FrozenPerturbationBankError] = None
        if self.spec.kind == STRESS_FAMILY:
            bank_path = self.spec.extras.get("bank_path")
            if bank_path in (None, ""):
                self._frozen_bank_error = FrozenPerturbationBankError(
                    "bank_path_missing",
                    "PerturbationSpec.extras.bank_path is required",
                )
            else:
                try:
                    self._frozen_bank = FrozenPerturbationBank.from_path(
                        str(bank_path)
                    )
                except FrozenPerturbationBankError as exc:
                    self._frozen_bank_error = exc

    @property
    def latest_audit(self) -> Optional[PerturbationAudit]:
        return self._latest_audit

    @property
    def latest_world_audit(self) -> Optional[PerturbationAudit]:
        return self._latest_world_audit

    @property
    def latest_prompt_audit(self) -> Optional[PerturbationAudit]:
        return self._latest_prompt_audit

    def begin_rollout(self) -> None:
        """Start a fresh injection generation.

        ``episode_id`` and ``seed`` identify a paired experimental cell, not a
        concrete world-graph generation.  Retries may therefore rebuild the
        graph while retaining both values.  Clearing the per-generation cache
        here preserves idempotence for repeated calls *inside* one rollout but
        guarantees that every new rollout observes its own intervention and
        audit.
        """

        self._applied_key = None
        self._latest_audit = None
        self._latest_world_audit = None
        self._latest_prompt_audit = None

    # ==================================================================
    # World-level perturbations
    # ==================================================================
    def apply_world_perturbation(
        self,
        env_interface: Any,
        episode: Any = None,
    ) -> Dict[str, Any]:
        """Mutate ``env_interface`` in-place for a world-level perturbation.

        Returns a small audit dict describing what was changed. Safe to call
        when ``spec`` is clean or prompt-level (returns ``{}``).
        """
        if self.spec.is_clean or not self.spec.is_world_level:
            return {}
        return self.apply_world_perturbation_with_audit(
            env_interface,
            episode=episode,
        ).to_dict()

    def apply_world_perturbation_with_audit(
        self,
        env_interface: Any,
        episode: Any = None,
    ) -> PerturbationAudit:
        """Apply a world mutation and report its observed realized dose."""

        episode_id = self._episode_id(env_interface, episode)
        if self.spec.is_clean or not self.spec.is_world_level:
            audit = PerturbationAudit(
                kind=self.spec.kind,
                requested=self.spec.intensity,
                realized=0.0,
                valid=True,
                reason="not_world_level",
                applied=False,
                seed=self.spec.seed,
                episode_id=episode_id,
            )
            self._remember_audit(audit, channel="world")
            return audit
        if self.spec.intensity <= 0.0:
            audit = PerturbationAudit(
                kind=self.spec.kind,
                requested=0.0,
                realized=0.0,
                valid=True,
                reason="zero_intensity",
                applied=False,
                seed=self.spec.seed,
                episode_id=episode_id,
            )
            self._remember_audit(audit, channel="world")
            return audit

        key = (self.spec.kind, self.spec.seed, episode_id)
        if key == self._applied_key and self._latest_world_audit is not None:
            return self._latest_world_audit

        graph_results: List[Dict[str, Any]] = []
        mutated_graphs: List[Tuple[Any, Dict[str, Any]]] = []
        state_plan: Optional[_ObjectStateMutationPlan] = None
        state_plan_reason = ""
        try:
            world_graphs = getattr(env_interface, "world_graph", None)
            if isinstance(world_graphs, Mapping) and world_graphs:
                # Under full observability multiple agent ids can point to the
                # same WorldGraph object. Mutating once per mapping entry would
                # toggle a boolean twice and silently cancel the intervention.
                graph_items = self._unique_world_graph_items(world_graphs)
                if self.spec.kind == "object_state_toggle":
                    state_plan, state_plan_reason = self._select_object_state_plan(
                        episode=episode,
                        env_interface=env_interface,
                        graph_items=graph_items,
                        spec=self.spec,
                    )
                    if state_plan is not None and not state_plan_reason:
                        graph_items = self._group_graph_items_by_state_target(
                            graph_items,
                            state_plan,
                        )
                preflight = [
                    (
                        agent_uids,
                        wg,
                        state_plan_reason
                        or self._preflight_world_graph(
                            wg,
                            self.spec,
                            state_plan=state_plan,
                        ),
                    )
                    for agent_uids, wg in graph_items
                ]
                failed_preflight = {
                    tuple(agent_uids): reason
                    for agent_uids, _wg, reason in preflight
                    if reason
                }
                if failed_preflight:
                    for agent_uids, _wg, reason in preflight:
                        graph_results.append(
                            {
                                "agent_uid": agent_uids[0],
                                "agent_uids": list(agent_uids),
                                "applied": False,
                                "valid": False,
                                "reason": reason
                                or "aggregate_preflight_aborted",
                                "realized": 0.0,
                                "components": (
                                    {"state_plan": state_plan.to_dict()}
                                    if state_plan is not None
                                    else {}
                                ),
                            }
                        )
                else:
                    for agent_uids, wg in graph_items:
                        changes = self._mutate_world_graph(
                            wg,
                            self.spec,
                            irrelevant_object_count=self.irrelevant_object_count,
                            state_plan=state_plan,
                        )
                        graph_result = {
                            "agent_uid": agent_uids[0],
                            "agent_uids": list(agent_uids),
                            **changes,
                        }
                        graph_results.append(graph_result)
                        mutated_graphs.append((wg, graph_result))
            else:
                logger.warning(
                    "env_interface has no world_graph mapping; skipping world-level perturbation %s",
                    self.spec.kind,
                )
                graph_results.append(
                    {
                        "agent_uid": "none",
                        "applied": False,
                        "valid": False,
                        "reason": "world_graph_mapping_missing",
                        "realized": 0.0,
                        "components": {},
                    }
                )
        except Exception as exc:
            logger.error(
                "Failed to apply world perturbation %s: %s", self.spec.kind, exc
            )
            graph_results.append(
                {
                    "agent_uid": "error",
                    "applied": False,
                    "valid": False,
                    "reason": f"world_mutation_error:{type(exc).__name__}",
                    "realized": 0.0,
                    "components": {"error": str(exc)},
                }
            )

        # A multi-agent intervention is one atomic stress cell. If any graph
        # fails after preflight, restore every graph already changed so an
        # invalid cell cannot leak a partial perturbation into the rollout.
        if graph_results and not all(
            bool(result.get("valid")) and bool(result.get("applied"))
            for result in graph_results
        ):
            for world_graph, result in reversed(mutated_graphs):
                if not bool(result.get("applied")):
                    continue
                rollback_ok, rollback_reason = self._rollback_world_graph(
                    world_graph,
                    self.spec,
                    result,
                )
                components = result.setdefault("components", {})
                components["rollback_attempted"] = True
                components["rollback_succeeded"] = rollback_ok
                if rollback_ok:
                    result["applied"] = False
                    result["valid"] = False
                    result["realized"] = 0.0
                    result["reason"] = "aggregate_invalid_rolled_back"
                else:
                    result["valid"] = False
                    result["reason"] = (
                        f"aggregate_invalid_rollback_failed:{rollback_reason}"
                    )

        valid = bool(graph_results) and all(
            bool(result.get("valid")) for result in graph_results
        )
        applied = bool(graph_results) and all(
            bool(result.get("applied")) for result in graph_results
        )
        reasons = sorted(
            {
                str(result.get("reason"))
                for result in graph_results
                if result.get("reason")
            }
        )
        realized = (
            sum(float(result.get("realized", 0.0)) for result in graph_results)
            / len(graph_results)
            if graph_results
            else 0.0
        )
        audit = PerturbationAudit(
            kind=self.spec.kind,
            requested=self.spec.intensity,
            realized=realized,
            valid=valid and applied,
            reason="" if valid and applied else "|".join(reasons) or "no_world_change_realized",
            components={
                "graph_count": len(graph_results),
                "agent_view_count": sum(
                    len(result.get("agent_uids", [result.get("agent_uid")]))
                    for result in graph_results
                ),
                "applied_graph_count": sum(
                    int(bool(result.get("applied"))) for result in graph_results
                ),
                "selected_state_plan": (
                    state_plan.to_dict() if state_plan is not None else None
                ),
                "state_plan_reason": state_plan_reason,
                "per_agent": graph_results,
            },
            applied=applied,
            seed=self.spec.seed,
            episode_id=episode_id,
        )
        # Cache both success and failure. Retrying an invalid toggle with the
        # same key can otherwise reverse a partially applied graph mutation.
        self._applied_key = key
        self._remember_audit(audit, channel="world")
        return audit

    @staticmethod
    def _unique_world_graph_items(
        world_graphs: Mapping[Any, Any],
    ) -> List[Tuple[Tuple[Any, ...], Any]]:
        """Group agent views that share the exact same graph instance."""

        grouped: List[Tuple[List[Any], Any]] = []
        index_by_identity: Dict[int, int] = {}
        for agent_uid, world_graph in world_graphs.items():
            identity = id(world_graph)
            if identity in index_by_identity:
                grouped[index_by_identity[identity]][0].append(agent_uid)
                continue
            index_by_identity[identity] = len(grouped)
            grouped.append(([agent_uid], world_graph))
        return [(tuple(agent_uids), graph) for agent_uids, graph in grouped]

    @classmethod
    def _group_graph_items_by_state_target(
        cls,
        graph_items: Sequence[Tuple[Tuple[Any, ...], Any]],
        plan: _ObjectStateMutationPlan,
    ) -> List[Tuple[Tuple[Any, ...], Any]]:
        """Deduplicate graph views that share the selected Entity object."""

        grouped: List[Tuple[List[Any], Any]] = []
        index_by_target_identity: Dict[int, int] = {}
        for agent_uids, world_graph in graph_items:
            target = cls._resolve_object_state_target(world_graph, plan)
            # Selection calls this only after every graph resolved the target;
            # retaining an unresolved graph lets preflight fail closed if the
            # graph changes between selection and mutation.
            identity = id(target) if target is not None else id(world_graph)
            if identity in index_by_target_identity:
                grouped[index_by_target_identity[identity]][0].extend(agent_uids)
                continue
            index_by_target_identity[identity] = len(grouped)
            grouped.append((list(agent_uids), world_graph))
        return [(tuple(agent_uids), graph) for agent_uids, graph in grouped]

    @staticmethod
    def _state_entities(world_graph: Any) -> List[Any]:
        """Return deterministic Object + Furniture state-bearing candidates."""

        entities: List[Any] = []
        seen: set[int] = set()
        for getter_name in ("get_all_objects", "get_all_furnitures"):
            getter = _safe_attr(world_graph, getter_name, None)
            if not callable(getter):
                continue
            try:
                values = list(getter() or [])
            except Exception:
                continue
            for entity in values:
                if id(entity) in seen:
                    continue
                seen.add(id(entity))
                entities.append(entity)
        return sorted(
            entities,
            key=lambda entity: (
                str(getattr(entity, "name", "")),
                str(getattr(entity, "sim_handle", "")),
            ),
        )

    @classmethod
    def _resolve_object_state_target(
        cls,
        world_graph: Any,
        plan: _ObjectStateMutationPlan,
    ) -> Optional[Any]:
        """Resolve the exact episode entity and writable state in one graph."""

        expected_ids = set(plan.entity_ids)
        for entity in cls._state_entities(world_graph):
            entity_ids = {
                str(getattr(entity, "name", "")),
                str(getattr(entity, "sim_handle", "")),
            }
            if not expected_ids.intersection(entity_ids):
                continue
            properties = getattr(entity, "properties", None)
            state_map = (
                properties.get("states")
                if isinstance(properties, Mapping)
                else None
            )
            if (
                callable(getattr(entity, "set_state", None))
                and isinstance(state_map, Mapping)
                and isinstance(state_map.get(plan.state_key), bool)
            ):
                return entity
        return None

    @staticmethod
    def _episode_propositions(episode: Any) -> Sequence[Any]:
        if isinstance(episode, Mapping):
            return episode.get("evaluation_propositions", ()) or ()
        return getattr(episode, "evaluation_propositions", ()) or ()

    @classmethod
    def _object_state_plan_candidates(
        cls,
        episode: Any,
        env_interface: Any,
        spec: PerturbationSpec,
    ) -> List[_ObjectStateMutationPlan]:
        """Build deterministic candidates from real evaluation propositions."""

        perception = getattr(env_interface, "perception", None)
        alias_map = getattr(perception, "sim_handle_to_name", None)
        aliases = alias_map if isinstance(alias_map, Mapping) else {}
        requested_entity = str(
            spec.extras.get("object_handle")
            or spec.extras.get("object_name")
            or spec.extras.get("entity_id")
            or ""
        )
        requested_state = str(
            spec.extras.get("state")
            or spec.extras.get("state_name")
            or ""
        )

        plans: List[_ObjectStateMutationPlan] = []
        seen: set[Tuple[int, str, str]] = set()
        for proposition_index, proposition in enumerate(
            cls._episode_propositions(episode)
        ):
            if isinstance(proposition, Mapping):
                predicate = str(proposition.get("function_name") or "")
                args = proposition.get("args") or {}
            else:
                predicate = str(getattr(proposition, "function_name", "") or "")
                args = getattr(proposition, "args", {}) or {}
            state_contract = _OBJECT_STATE_PREDICATES.get(predicate)
            if state_contract is None or not isinstance(args, Mapping):
                continue
            state_key, goal_value = state_contract
            if requested_state and requested_state not in {predicate, state_key}:
                continue
            raw_entities = (
                args.get("object_handles")
                or args.get("object_names")
                or args.get("entity_handles_a")
                or args.get("entity_handles_a_name")
                or ()
            )
            if isinstance(raw_entities, str):
                raw_entities = (raw_entities,)
            for raw_entity in raw_entities:
                raw_id = str(raw_entity)
                entity_ids = [raw_id]
                alias = aliases.get(raw_entity)
                if alias is None:
                    alias = aliases.get(raw_id)
                if alias is not None and str(alias) not in entity_ids:
                    entity_ids.append(str(alias))
                if requested_entity and requested_entity not in entity_ids:
                    continue
                identity = (proposition_index, state_key, raw_id)
                if identity in seen:
                    continue
                seen.add(identity)
                plans.append(
                    _ObjectStateMutationPlan(
                        proposition_index=proposition_index,
                        predicate=predicate,
                        state_key=state_key,
                        entity_ids=tuple(entity_ids),
                        goal_value=goal_value,
                    )
                )

        if plans:
            offset = int(spec.seed) % len(plans)
            plans = plans[offset:] + plans[:offset]
        return plans

    @classmethod
    def _select_object_state_plan(
        cls,
        *,
        episode: Any,
        env_interface: Any,
        graph_items: Sequence[Tuple[Tuple[Any, ...], Any]],
        spec: PerturbationSpec,
    ) -> Tuple[Optional[_ObjectStateMutationPlan], str]:
        candidates = cls._object_state_plan_candidates(
            episode,
            env_interface,
            spec,
        )
        if not candidates:
            return None, "no_episode_object_state_target"
        # Seed selects one stable proposition/entity. Do not silently switch to
        # a different task target when observability changes across conditions.
        candidate = candidates[0]
        if bool(getattr(env_interface, "partial_obs", False)):
            return candidate, "object_state_toggle_requires_full_observation"
        if all(
            cls._resolve_object_state_target(world_graph, candidate) is not None
            for _agent_uids, world_graph in graph_items
        ):
            return candidate, ""
        return candidate, "episode_object_state_target_unavailable_in_all_graphs"

    @staticmethod
    def _preflight_world_graph(
        world_graph: Any,
        spec: PerturbationSpec,
        *,
        state_plan: Optional[_ObjectStateMutationPlan] = None,
    ) -> str:
        """Return an empty string only when the graph exposes a usable adapter."""

        kind = spec.kind
        if kind == "room_shift":
            setter = _safe_attr(
                world_graph, "set_counterfactual_allowed_regions", None
            )
            getter = _safe_attr(
                world_graph, "get_counterfactual_allowed_regions", None
            )
            if not callable(setter) or not callable(getter):
                return "unsupported_room_shift_adapter"
            try:
                getter()
            except Exception:
                return "unreadable_room_shift_adapter"
            return ""
        if kind == "holding_toggle":
            task_objects = list(_safe_attr(world_graph, "task_objects", []) or [])
            setter = _safe_attr(world_graph, "set_holding", None)
            getter = _safe_attr(world_graph, "is_holding", None)
            if not task_objects or not callable(setter) or not callable(getter):
                return "unsupported_holding_toggle_adapter"
            try:
                getter(task_objects[0])
            except Exception:
                return "unreadable_holding_toggle_adapter"
            return ""
        if kind == "object_state_toggle":
            if state_plan is None:
                return "no_episode_object_state_target"
            target = PerturbationInjector._resolve_object_state_target(
                world_graph,
                state_plan,
            )
            return (
                ""
                if target is not None
                else "episode_object_state_target_unavailable"
            )
        if kind == "irrelevant_perturbation":
            describer = _safe_attr(world_graph, "get_world_descr", None)
            if not callable(describer):
                return "planner_visible_measurement_unavailable"
            try:
                str(describer())
            except Exception:
                return "planner_visible_measurement_unavailable"
            return ""
        return f"unknown_kind:{kind}"

    @staticmethod
    def _rollback_world_graph(
        world_graph: Any,
        spec: PerturbationSpec,
        result: Mapping[str, Any],
    ) -> Tuple[bool, str]:
        """Best-effort verified rollback for an invalid multi-graph mutation."""

        components = result.get("components", {})
        if not isinstance(components, Mapping):
            return False, "rollback_components_missing"
        try:
            if spec.kind == "room_shift":
                setter = _safe_attr(
                    world_graph, "set_counterfactual_allowed_regions", None
                )
                getter = _safe_attr(
                    world_graph, "get_counterfactual_allowed_regions", None
                )
                before = list(components.get("before", []))
                setter(before)
                return list(getter() or []) == before, "room_rollback_unverified"
            if spec.kind == "holding_toggle":
                object_label = str(components.get("object", ""))
                target = next(
                    (
                        obj
                        for obj in list(
                            _safe_attr(world_graph, "task_objects", []) or []
                        )
                        if str(obj) == object_label
                    ),
                    None,
                )
                if target is None:
                    return False, "holding_rollback_target_missing"
                setter = _safe_attr(world_graph, "set_holding", None)
                getter = _safe_attr(world_graph, "is_holding", None)
                before = bool(components.get("before"))
                setter(target, before)
                return bool(getter(target)) == before, "holding_rollback_unverified"
            if spec.kind == "object_state_toggle":
                object_label = str(components.get("object", ""))
                sim_handle = str(components.get("sim_handle", ""))
                target = next(
                    (
                        entity
                        for entity in PerturbationInjector._state_entities(
                            world_graph
                        )
                        if (
                            str(getattr(entity, "name", "")) == object_label
                            or (
                                sim_handle
                                and str(getattr(entity, "sim_handle", ""))
                                == sim_handle
                            )
                        )
                    ),
                    None,
                )
                if target is None:
                    return False, "object_state_rollback_target_missing"
                state_key = str(components.get("state", ""))
                before = bool(components.get("before"))
                target.set_state({state_key: before})
                restored = target.properties["states"].get(state_key)
                return bool(restored) == before, "object_state_rollback_unverified"
            if spec.kind == "irrelevant_perturbation":
                before = list(components.get("before_distractors", []))
                setattr(world_graph, "counterfactual_distractors", before)
                restored = list(
                    _safe_attr(world_graph, "counterfactual_distractors", []) or []
                )
                return restored == before, "irrelevant_rollback_unverified"
        except Exception as exc:
            return False, f"{type(exc).__name__}:{exc}"
        return False, f"unsupported_rollback_kind:{spec.kind}"

    @staticmethod
    def _mutate_world_graph(
        world_graph: Any,
        spec: PerturbationSpec,
        *,
        irrelevant_object_count: int = 5,
        state_plan: Optional[_ObjectStateMutationPlan] = None,
    ) -> Dict[str, Any]:
        """Apply ``spec.kind`` to an individual per-agent WorldGraph instance.

        Uses best-effort attribute probing so it works both with the current
        PartnR ``WorldGraph`` and future refactors. Falls back to a stashed
        ``perturbation_log`` attribute when the graph does not expose the
        required mutators.
        """
        intensity = float(spec.intensity)
        if intensity <= 0.0:
            return {
                "applied": False,
                "valid": True,
                "reason": "zero_intensity",
                "realized": 0.0,
                "components": {},
            }
        kind = spec.kind
        applied = False
        valid = False
        reason = ""
        realized = 0.0
        components: Dict[str, Any] = {}

        if kind == "room_shift":
            # Only an explicit, observable adapter is accepted. Writing an
            # arbitrary attribute is not evidence that the planner consumed it.
            setter = _safe_attr(
                world_graph,
                "set_counterfactual_allowed_regions",
                None,
            )
            getter = _safe_attr(
                world_graph,
                "get_counterfactual_allowed_regions",
                None,
            )
            if callable(setter) and callable(getter):
                before = [str(value) for value in (getter() or [])]
                setter(["counterfactual_room"])
                after = [str(value) for value in (getter() or [])]
                applied = before != after
                valid = applied
                realized = 1.0 if applied else 0.0
                components = {"before": before, "after": after}
                reason = "" if applied else "room_adapter_no_observed_change"
            else:
                reason = "unsupported_room_shift_adapter"

        elif kind == "holding_toggle":
            # A setter without a readable state cannot prove a realized dose.
            task_objects = _safe_attr(world_graph, "task_objects", [])
            setter = _safe_attr(world_graph, "set_holding", None)
            getter = _safe_attr(world_graph, "is_holding", None)
            if task_objects and callable(setter) and callable(getter):
                first = task_objects[0]
                before = bool(getter(first))
                setter(first, not before)
                after = bool(getter(first))
                applied = before != after
                valid = applied
                realized = 1.0 if applied else 0.0
                components = {
                    "object": str(first),
                    "before": before,
                    "after": after,
                }
                reason = "" if applied else "holding_adapter_no_observed_change"
            else:
                reason = "unsupported_holding_toggle_adapter"

        elif kind == "object_state_toggle":
            target = (
                PerturbationInjector._resolve_object_state_target(
                    world_graph,
                    state_plan,
                )
                if state_plan is not None
                else None
            )
            if target is not None and state_plan is not None:
                state_map = target.properties["states"]
                state_key = state_plan.state_key
                before = bool(state_map[state_key])
                # This intervention corrupts the planner-visible belief
                # snapshot, not simulator truth. Toggling the observed value
                # guarantees a real contradiction while keeping the target
                # object/state grounded in the episode proposition.
                target.set_state({state_key: not before})
                after = bool(target.properties["states"].get(state_key))
                applied = before != after
                valid = applied
                realized = 1.0 if applied else 0.0
                components = {
                    "object": str(getattr(target, "name", target)),
                    "sim_handle": str(getattr(target, "sim_handle", "") or ""),
                    "state": str(state_key),
                    "predicate": state_plan.predicate,
                    "proposition_index": int(state_plan.proposition_index),
                    "goal_value": bool(state_plan.goal_value),
                    "entity_ids": list(state_plan.entity_ids),
                    "before": before,
                    "after": after,
                    "scope": "initial_planner_world_graph_snapshot",
                    "exposure_contract": (
                        "visible_to_initial_planner_call_before_first_env_step"
                    ),
                }
                reason = "" if applied else "object_state_no_observed_change"
            else:
                reason = (
                    "no_episode_object_state_target"
                    if state_plan is None
                    else "episode_object_state_target_unavailable"
                )

        elif kind == "irrelevant_perturbation":
            # Stash a list of distractor entities on the graph so that
            # downstream prompt builders can optionally surface them. Scaled
            # by intensity so that the AUDC sweep has a monotone stress axis.
            describer = _safe_attr(world_graph, "get_world_descr", None)
            before_description = ""
            planner_visible_measurement_available = False
            if callable(describer):
                try:
                    before_description = str(describer())
                    planner_visible_measurement_available = True
                except Exception:
                    before_description = ""
            maximum = max(1, int(irrelevant_object_count))
            count = max(1, int(math.ceil(maximum * intensity - 1.0e-12)))
            distractors = [
                f"distractor_object_seed{spec.seed}_{i}" for i in range(count)
            ]
            existing = list(_safe_attr(world_graph, "counterfactual_distractors", []) or [])
            additions = [name for name in distractors if name not in existing]
            setattr(world_graph, "counterfactual_distractors", existing + additions)
            after = list(
                _safe_attr(world_graph, "counterfactual_distractors", []) or []
            )
            after_description = ""
            if planner_visible_measurement_available:
                try:
                    after_description = str(describer())
                except Exception:
                    after_description = ""
                    planner_visible_measurement_available = False
            before_tokens = before_description.split()
            after_tokens = after_description.split()
            visible_added_tokens = max(0, len(after_tokens) - len(before_tokens))
            applied = len(after) == len(existing) + len(additions) and bool(additions)
            valid = bool(
                applied
                and len(additions) == count
                and planner_visible_measurement_available
                and visible_added_tokens > 0
            )
            realized = float(len(additions) / maximum)
            components = {
                "requested_count": count,
                "added_count": len(additions),
                "maximum_count": maximum,
                "before_distractors": existing,
                "planner_visible_added_token_count": visible_added_tokens,
                "planner_visible_token_ratio": float(
                    visible_added_tokens / max(len(before_tokens), 1)
                ),
                "planner_visible_measurement_available": bool(
                    planner_visible_measurement_available
                ),
            }
            if valid:
                reason = ""
            elif not planner_visible_measurement_available:
                reason = "planner_visible_measurement_unavailable"
            elif visible_added_tokens <= 0:
                reason = "planner_visible_change_not_observed"
            else:
                reason = "distractor_change_not_fully_realized"
        else:
            reason = f"unknown_kind:{kind}"

        result = {
            "applied": applied,
            "valid": valid,
            "reason": reason,
            "realized": realized,
            "components": components,
        }
        audit_list = list(_safe_attr(world_graph, "perturbation_log", []) or [])
        audit_list.append(
            {
                "kind": kind,
                "seed": spec.seed,
                "requested": intensity,
                **result,
            }
        )
        setattr(world_graph, "perturbation_log", audit_list)
        return result

    # ==================================================================
    # Prompt-level perturbations
    # ==================================================================
    def rewrite_instruction(
        self,
        instruction: str,
        episode: Any = None,
    ) -> str:
        """Return a (possibly rewritten) instruction string.

        Safe to call for any spec — only prompt-level kinds mutate the
        instruction; everything else returns the input unchanged.
        """
        rewritten, _audit = self.rewrite_instruction_with_audit(
            instruction,
            episode=episode,
        )
        return rewritten

    def rewrite_instruction_with_audit(
        self,
        instruction: str,
        episode: Any = None,
    ) -> Tuple[str, PerturbationAudit]:
        """Rewrite once and return the observed prompt-dose audit."""

        anchor_id = self._episode_id(None, episode)
        if not self.spec.is_prompt_level:
            audit = PerturbationAudit(
                kind=self.spec.kind,
                requested=self.spec.intensity,
                realized=0.0,
                valid=True,
                reason="not_prompt_level",
                applied=False,
                seed=self.spec.seed,
                episode_id=anchor_id,
            )
            self._remember_audit(audit, channel="prompt")
            return instruction, audit

        if self.spec.kind == STRESS_FAMILY:
            rewritten, audit = self._rewrite_frozen_stress_instruction(
                instruction,
                episode_id=anchor_id,
                episode=episode,
            )
            self._remember_audit(audit, channel="prompt")
            return rewritten, audit

        rewritten, metadata = self.paraphrase_bank.rewrite_with_metadata(
            instruction,
            mode=self.spec.kind,
            anchor_id=anchor_id,
            seed=self.spec.seed,
            intensity=self.spec.intensity,
        )
        audit = prompt_audit(
            kind=self.spec.kind,
            requested=self.spec.intensity,
            original=instruction,
            perturbed=rewritten,
            metadata=metadata,
            seed=self.spec.seed,
            episode_id=anchor_id,
        )
        self._remember_audit(audit, channel="prompt")
        return rewritten, audit

    def _rewrite_frozen_stress_instruction(
        self,
        instruction: str,
        *,
        episode_id: str,
        episode: Any,
    ) -> Tuple[str, PerturbationAudit]:
        """Resolve one frozen cumulative stress level without an API call."""

        requested = float(self.spec.intensity)
        variant = str(self.spec.extras.get("variant") or "")
        entry = None
        failure_reason = ""
        failure_detail = ""
        if self._frozen_bank_error is not None:
            failure_reason = self._frozen_bank_error.reason
            failure_detail = self._frozen_bank_error.detail
        elif self._frozen_bank is None:
            failure_reason = "bank_not_loaded"
        elif not variant:
            failure_reason = "bank_variant_missing"
        else:
            try:
                current_task_contract = build_episode_task_contract(episode)
                entry = self._frozen_bank.resolve(
                    episode_id=episode_id,
                    family=STRESS_FAMILY,
                    lambda_value=requested,
                    variant=variant,
                    current_task_contract=current_task_contract,
                )
            except FrozenPerturbationBankError as exc:
                failure_reason = exc.reason
                failure_detail = exc.detail
            except ValueError as exc:
                failure_reason = "runtime_task_contract_unavailable"
                failure_detail = str(exc)

        if entry is None:
            audit = PerturbationAudit(
                kind=self.spec.kind,
                requested=requested,
                realized=0.0,
                valid=False,
                reason=failure_reason or "bank_entry_invalid",
                components={
                    "stress_family": STRESS_FAMILY,
                    "variant": variant,
                    "bank_path": str(self.spec.extras.get("bank_path") or ""),
                    "error": failure_detail,
                    "operation_count": 0,
                    "max_operations": 5,
                },
                applied=False,
                seed=self.spec.seed,
                episode_id=episode_id,
            )
            return instruction, audit

        metadata = dict(entry.components)
        metadata.update(
            {
                "stress_family": entry.family,
                "variant": entry.variant,
                "level_index": entry.level_index,
                "complete_unit_count": entry.complete_unit_count,
                "stress_realized": entry.stress_realized,
                "text_dose": entry.text_dose,
                "bank_alignment_valid": entry.alignment_valid,
                "bank_path": entry.bank_path,
                "unit_budgets": list(entry.unit_budgets),
            }
        )
        observed = prompt_audit(
            kind=self.spec.kind,
            requested=requested,
            original=instruction,
            perturbed=entry.policy_instruction,
            metadata=metadata,
            seed=self.spec.seed,
            episode_id=episode_id,
        )
        try:
            tolerance = float(
                self.spec.extras.get(
                    "text_dose_tolerance",
                    metadata.get(
                        "text_dose_tolerance",
                        DEFAULT_TEXT_DOSE_TOLERANCE,
                    ),
                )
            )
        except (TypeError, ValueError):
            tolerance = DEFAULT_TEXT_DOSE_TOLERANCE
        instruction_matches = entry.canonical_instruction == instruction
        lambda_matches = math.isclose(
            entry.stress_realized,
            requested,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        text_dose_matches = abs(entry.text_dose - requested) <= tolerance + 1.0e-12
        valid = bool(
            observed.valid
            and entry.alignment_valid
            and instruction_matches
            and lambda_matches
            and text_dose_matches
        )
        if not instruction_matches:
            reason = "bank_instruction_mismatch"
        elif not entry.alignment_valid:
            reason = entry.reason or "bank_not_approved"
        elif not lambda_matches:
            reason = "bank_lambda_mismatch"
        elif not text_dose_matches:
            reason = "bank_dose_alignment_failed"
        elif not observed.valid:
            reason = observed.reason
        else:
            reason = ""
        metadata.update(
            {
                "configured_text_dose_tolerance": tolerance,
                "instruction_matches": instruction_matches,
                "lambda_matches": lambda_matches,
                "text_dose_matches": text_dose_matches,
                # This is the runtime alignment decision consumed by formal
                # Stability/GE. Keep the sidecar-only decision separately as
                # ``bank_alignment_valid`` for diagnostics.
                "alignment_valid": valid,
            }
        )
        applied = bool(valid and observed.applied)
        audit = PerturbationAudit(
            kind=self.spec.kind,
            requested=requested,
            realized=entry.stress_realized if applied or requested <= 0.0 else 0.0,
            valid=valid,
            reason=reason,
            components=metadata,
            applied=applied,
            seed=self.spec.seed,
            episode_id=episode_id,
            original_hash=observed.original_hash,
            perturbed_hash=(
                observed.perturbed_hash if applied else observed.original_hash
            ),
        )
        return entry.policy_instruction if applied else instruction, audit

    def prompt_hook(
        self,
        seed: Optional[int] = None,
        anchor_id: Optional[str] = None,
    ) -> Callable[[str], str]:
        """Return a closure suitable for ``LLMPlanner._prompt_perturbation_hook``.

        The closure receives the raw instruction and returns the rewritten
        one. It is a no-op for non-prompt-level specs.
        """
        effective_seed = int(seed) if seed is not None else int(self.spec.seed)
        effective_anchor = anchor_id

        def _hook(instruction: str, _episode: Any = None) -> str:
            if not self.spec.is_prompt_level:
                return instruction
            if self.spec.kind == STRESS_FAMILY:
                episode = _episode or {"episode_id": effective_anchor or "unknown"}
                return self.rewrite_instruction(instruction, episode=episode)
            return self.paraphrase_bank.rewrite(
                instruction,
                mode=self.spec.kind,
                anchor_id=effective_anchor,
                seed=effective_seed,
                intensity=self.spec.intensity,
            )

        return _hook

    @staticmethod
    def clear_prompt_hooks(planners: Any) -> None:
        """Remove legacy planner hooks so runner-side rewriting occurs once."""

        candidates = planners.values() if isinstance(planners, Mapping) else [planners]
        for planner in candidates:
            if planner is not None and hasattr(planner, "_prompt_perturbation_hook"):
                setattr(planner, "_prompt_perturbation_hook", None)

    def _remember_audit(self, audit: PerturbationAudit, *, channel: str) -> None:
        self._latest_audit = audit
        if channel == "world":
            self._latest_world_audit = audit
        elif channel == "prompt":
            self._latest_prompt_audit = audit

    # ==================================================================
    # Utilities
    # ==================================================================
    @staticmethod
    def _episode_id(env_interface: Any, episode: Any) -> str:
        if episode is not None:
            ep_id = getattr(episode, "episode_id", None)
            if ep_id is not None:
                return str(ep_id)
            if isinstance(episode, Mapping):
                return str(episode.get("episode_id", "unknown"))
        if env_interface is not None:
            try:
                return str(env_interface.env.env.env._env.current_episode.episode_id)
            except Exception:
                pass
        return "unknown"

    # ------------------------------------------------------------------
    # Convenience factory: produce one injector per (anchor, spec, seed).
    # ------------------------------------------------------------------
    @classmethod
    def iter_specs(
        cls,
        kinds: Iterable[str],
        seeds: Sequence[int],
        intensities: Sequence[float] = (1.0,),
    ) -> List[PerturbationSpec]:
        out: List[PerturbationSpec] = []
        for kind in kinds:
            if kind not in ALL_PERTURBATIONS and kind not in ("clean", "baseline"):
                raise ValueError(f"Unknown perturbation kind: {kind!r}")
            for seed in seeds:
                for intensity in intensities:
                    out.append(
                        PerturbationSpec(
                            kind=kind,
                            seed=int(seed),
                            intensity=float(intensity),
                        )
                    )
        return out


def _safe_attr(obj: Any, name: str, default: Any) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def derive_run_seed(experiment_seed: int, repetition_seed: int) -> int:
    """Derive a paired execution seed independent of perturbation family.

    Calling this with the same ``repetition_seed`` for clean and every
    perturbation family yields the same simulator/planner RNG seed.
    """

    payload = f"{int(experiment_seed)}::{int(repetition_seed)}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value % (2**31 - 1)


def spec_artifact_key(spec: PerturbationSpec) -> str:
    """Return a collision-resistant directory key for one perturbation spec."""

    kind_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", spec.kind).strip("-") or "clean"
    extras_payload = json.dumps(
        spec.extras,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    )
    canonical = (
        f"{spec.kind}\0{spec.seed}\0{spec.intensity.hex()}\0{extras_payload}"
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]
    return (
        f"spec_{kind_slug}_seed{spec.seed}_lambda{spec.intensity:.6f}_{digest}"
    )


def build_perturbation_spec_grid(
    cfg: Mapping[str, Any],
) -> List[PerturbationSpec]:
    """Build a deduplicated, seed-major perturbation grid.

    Clean and baseline are reference conditions, not stress families. They
    therefore must not inherit the global intensity sweep and masquerade as
    positive-dose interventions.  Seed is the outer scheduling axis so all
    conditions for one seed are completed before the next seed is admitted.
    """

    specs: List[PerturbationSpec] = []
    seen: set[str] = set()
    configured_kinds = [str(value).strip() for value in cfg["perturbations"]]
    has_clean_reference = any(
        value.lower() in {"", "clean", "baseline"}
        for value in configured_kinds
    )
    extras_by_kind = cfg.get("perturbation_extras", {}) or {}
    if not isinstance(extras_by_kind, Mapping):
        raise TypeError("resilience.perturbation_extras must be a mapping")
    for raw_seed in cfg["seeds"]:
        seed = int(raw_seed)
        for raw_kind in configured_kinds:
            kind = str(raw_kind).strip()
            is_clean = kind.lower() in {"", "clean", "baseline"}
            canonical_kind = "clean" if is_clean else kind
            intensities = (0.0,) if is_clean else cfg["intensities"]
            if not is_clean and has_clean_reference:
                intensities = tuple(
                    value
                    for value in intensities
                    if float(value) > 0.0
                )
            raw_extras = extras_by_kind.get(canonical_kind, {}) or {}
            if (
                is_clean
                and not raw_extras
                and STRESS_FAMILY in configured_kinds
            ):
                # A single-grid formal run declares clean and the frozen
                # family together. Bind its lambda=0 row to the same path;
                # clean-only phases declare this explicitly in their patch.
                raw_extras = extras_by_kind.get(STRESS_FAMILY, {}) or {}
            if not isinstance(raw_extras, Mapping):
                raise TypeError(
                    "resilience.perturbation_extras."
                    f"{canonical_kind} must be a mapping"
                )
            spec_extras = dict(raw_extras)
            bank_path = spec_extras.get("bank_path")
            if bank_path not in (None, ""):
                spec_extras["bank_path"] = str(
                    Path(str(bank_path)).expanduser().resolve()
                )
            variant_by_seed = spec_extras.get("variant_by_seed", {}) or {}
            if isinstance(variant_by_seed, Mapping):
                variant = variant_by_seed.get(seed)
                if variant is None:
                    variant = variant_by_seed.get(str(seed))
                if variant not in (None, ""):
                    spec_extras["variant"] = str(variant)
            for raw_intensity in intensities:
                spec = PerturbationSpec(
                    kind=canonical_kind,
                    seed=seed,
                    intensity=float(raw_intensity),
                    extras=spec_extras,
                )
                identity = spec_artifact_key(spec)
                if identity in seen:
                    continue
                seen.add(identity)
                specs.append(spec)
    return specs
