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
* Hooks are idempotent: calling :meth:`PerturbationInjector.apply_world_perturbation`
  multiple times on the same ``env_interface`` only mutates state once per
  rollout (indexed by ``episode_id`` + ``seed``).

Integration
-----------
* In :meth:`EvaluationRunner.run_instruction` (after ``reset_planners``),
  call::

      injector.apply_world_perturbation(env_interface, episode)
      instruction = injector.rewrite_instruction(instruction, episode)

* In :class:`LLMPlanner`, register the instruction-rewriter by setting::

      planner._prompt_perturbation_hook = injector.prompt_hook(seed=...)

  The planner only needs to call the hook on the resolved instruction right
  before ``prepare_prompt``; no structural changes to the planner are
  required. The hook defaults to a no-op when the injector is disabled.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
)
ALL_PERTURBATIONS: Tuple[str, ...] = WORLD_LEVEL_PERTURBATIONS + PROMPT_LEVEL_PERTURBATIONS


@dataclass
class PerturbationSpec:
    """Declarative description of a perturbation to apply to one rollout."""

    kind: str = "clean"          # one of ALL_PERTURBATIONS or "clean"
    seed: int = 0
    intensity: float = 1.0       # stress magnitude λ (§3 AUDC), 0.0 = no-op
    extras: Dict[str, Any] = field(default_factory=dict)

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
        episode_id = self._episode_id(env_interface, episode)
        key = (self.spec.kind, self.spec.seed, episode_id)
        if key == self._applied_key:
            return {"skipped": "already_applied"}

        audit: Dict[str, Any] = {"kind": self.spec.kind, "episode_id": episode_id}
        try:
            world_graphs = getattr(env_interface, "world_graph", None)
            if isinstance(world_graphs, Mapping) and world_graphs:
                for agent_uid, wg in world_graphs.items():
                    changes = self._mutate_world_graph(wg, self.spec)
                    if changes:
                        audit.setdefault("world_graph_changes", []).append(
                            {"agent_uid": agent_uid, **changes}
                        )
            else:
                logger.warning(
                    "env_interface has no world_graph mapping; skipping world-level perturbation %s",
                    self.spec.kind,
                )
        except Exception as exc:
            logger.error(
                "Failed to apply world perturbation %s: %s", self.spec.kind, exc
            )
            audit["error"] = str(exc)
        self._applied_key = key
        return audit

    @staticmethod
    def _mutate_world_graph(world_graph: Any, spec: PerturbationSpec) -> Dict[str, Any]:
        """Apply ``spec.kind`` to an individual per-agent WorldGraph instance.

        Uses best-effort attribute probing so it works both with the current
        PartnR ``WorldGraph`` and future refactors. Falls back to a stashed
        ``perturbation_log`` attribute when the graph does not expose the
        required mutators.
        """
        intensity = max(0.0, float(spec.intensity))
        if intensity <= 0.0:
            return {"no_op_reason": "zero_intensity"}
        log: Dict[str, Any] = {"intensity": intensity}
        kind = spec.kind

        if kind == "room_shift":
            # Shift the "task-allowed regions" to a counterfactual room.
            rooms = _safe_attr(world_graph, "room_entities", [])
            if rooms:
                fake_room = "counterfactual_room"
                setattr(world_graph, "counterfactual_allowed_regions", [fake_room])
            log["applied"] = "counterfactual_allowed_regions"

        elif kind == "holding_toggle":
            # Flip a ``holding`` flag for the first task object if the graph
            # exposes a ``set_holding`` mutator.
            task_objects = _safe_attr(world_graph, "task_objects", [])
            if task_objects and hasattr(world_graph, "set_holding"):
                first = task_objects[0]
                try:
                    world_graph.set_holding(first, True)
                    log["applied"] = f"set_holding({first}, True)"
                except Exception as exc:  # pragma: no cover
                    log["error"] = str(exc)

        elif kind == "object_state_toggle":
            # Toggle the first boolean state on the first task object.
            task_objects = _safe_attr(world_graph, "task_objects", [])
            if task_objects:
                first = task_objects[0]
                states = _safe_attr(world_graph, "object_states", {}) or {}
                affordance_map = states.get(first) if isinstance(states, Mapping) else None
                if isinstance(affordance_map, Mapping) and affordance_map:
                    key = next(iter(affordance_map.keys()))
                    try:
                        affordance_map[key] = not bool(affordance_map[key])
                        log["applied"] = f"toggle({first}.{key})"
                    except Exception as exc:  # pragma: no cover
                        log["error"] = str(exc)

        elif kind == "irrelevant_perturbation":
            # Stash a list of distractor entities on the graph so that
            # downstream prompt builders can optionally surface them. Scaled
            # by intensity so that the AUDC sweep has a monotone stress axis.
            count = max(1, int(round(5 * intensity)))
            distractors = [f"distractor_object_{i}" for i in range(count)]
            existing = list(_safe_attr(world_graph, "counterfactual_distractors", []) or [])
            existing.extend(distractors)
            setattr(world_graph, "counterfactual_distractors", existing)
            log["applied"] = f"added {count} distractors"
        else:
            log["no_op_reason"] = f"unknown_kind:{kind}"

        # Always stash a perturbation audit on the graph for downstream
        # logging / debugging; never raise on failure.
        audit_list = list(_safe_attr(world_graph, "perturbation_log", []) or [])
        audit_list.append({"kind": kind, "seed": spec.seed, **log})
        setattr(world_graph, "perturbation_log", audit_list)
        return log

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
        if not self.spec.is_prompt_level:
            return instruction
        anchor_id = self._episode_id(None, episode)
        return self.paraphrase_bank.rewrite(
            instruction,
            mode=self.spec.kind,
            anchor_id=anchor_id,
            seed=self.spec.seed,
        )

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
            return self.paraphrase_bank.rewrite(
                instruction,
                mode=self.spec.kind,
                anchor_id=effective_anchor,
                seed=effective_seed,
            )

        return _hook

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
