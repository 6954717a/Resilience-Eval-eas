#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from habitat.tasks.rearrange.utils import coll_name_matches
from hydra.utils import instantiate
from omegaconf import OmegaConf

from habitat_llm.context import CompressionConfig, ContextCompressor
from habitat_llm.llm.instruct.utils import (
    get_objects_descr,
    get_rearranged_objects_descr,
    get_world_descr,
)
from habitat_llm.planner.planner import Planner
from habitat_llm.utils.grammar import (
    FREE_TEXT,
    FURNITURE,
    NAV_TARGET,
    OBJECT,
    OBJECT_OR_FURNITURE,
    ROOM,
    SPATIAL_CONSTRAINT,
    SPATIAL_RELATION,
)
from habitat_llm.planner.perception_connector import PerceptionConnector
from habitat_llm.planner.rebound.rebound_manager import ReboundManager
from habitat_llm.planner.rebound.planner_integration import (
    apply_rebound_termination_guard,
    handle_rebound_side_effects,
    try_execute_recovery,
)
from habitat_llm.planner.connector.prompt_context import PromptBuilder
from habitat_llm.planner.replan.phase_planner import PhasePlanningHelper
from habitat_llm.planner.replan.phase_based_pipeline import run_phase_based_replan

from habitat_llm.planner.copal import (
    ErrorTranslator,
    ActionHistoryManager,
    CoPALContextBuilder,
    CoPALContextBuilder,
    build_copal_context,
)
from habitat_llm.planner.copal.planner_integration import (
    process_copal_step,
    generate_and_inject_copal_context,
)
from habitat_llm.planner.evolve.types import EvolutionPolicyOverlay

# CycleVLA imports (lazy loaded in replan to avoid circular imports)

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from habitat_llm.agent.agent import Agent
    from habitat_llm.agent.env import EnvironmentInterface
    from habitat_llm.planner.rag import RAG
    from habitat_llm.world_model.world_graph import WorldGraph


class LLMPlanner(Planner):
    """
    High level planner policy used by agents to decide high level actions
    given task description and state of the world
    """

    def __init__(
        self, plan_config: "DictConfig", env_interface: "EnvironmentInterface"
    ):
        """
        Initialize the LLMPlanner.

        :param plan_config: The planner configuration.
        :param env_interface: The environment interface.
        """
        # Set the planner config
        super().__init__(plan_config, env_interface)
        # Initialize LLM
        self.__initialize_llm()

        # Initialize Framework Components
        self.perception_connector = PerceptionConnector(
            llm_client=getattr(self.llm, "client", None) if hasattr(self.llm, "client") else None, # Try to get client from LLM wrapper
            api_key_filename="./habitat_llm/planner/api_key.txt", # Explicitly pass the key path
            perception_sim=env_interface.perception if hasattr(env_interface, "perception") else None
        )
        # If llm wrapper doesn't expose client, PhaseManager might fail. 
        # But we assume moonshot/openai client is accessible or we might need to pass api_key.
        # For now, let's assume standard setup.

        self.rebound_managers: Dict[int, ReboundManager] = {}
        # Get rebound config from plan_config if available
        self.rebound_config = plan_config.get("rebound", {})
        # Check if rebound is enabled (default to False if not specified)
        self.rebound_enabled = self.rebound_config.get("enabled", False)
        self.context_compression_config = self._cfg_to_plain_dict(
            self.rebound_config.get("context_compression", {})
        )
        self.context_compressor = self._initialize_context_compressor()
        self._context_compression_request_reason: Optional[str] = None
        # Note: ReboundManagers are initialized in the agents setter since self._agents is empty here

        self.phase_planner = PhasePlanningHelper(
            self.perception_connector,
            self._agents
        )

        self.use_prompt_builder = True
        self.prompt_builder = PromptBuilder(plan_config, env_interface)
        self.pending_context_updates: List[Tuple[str, str]] = []
        self.pending_perception_scans = set()
        self.evolution_context: Optional[Tuple[str, str]] = None
        self.evolution_manager: Optional[Any] = None
        self.current_evolution_overlay = EvolutionPolicyOverlay()
        self.current_instruction: str = ""
        self.current_scene_graph_snapshot: Dict[str, Any] = {}

        # Optional instruction-rewrite hook used by the Stability (β̂) and
        # Graceful-Extensibility (M_f) experiments. The hook, when set,
        # receives the raw instruction and returns a (possibly) paraphrased
        # version. Defaults to ``None`` so baseline planner behavior is
        # unchanged; set via
        # :class:`~habitat_llm.evaluation.perturbation.PerturbationInjector`.
        self._prompt_perturbation_hook: Optional[Any] = None

        # Initialize a variable to indicate if replanning is required
        self.replan_required: bool = True

        # Initialize a variable to count number of llm calls
        self.replanning_count: int = 0

        # Initialize container to store entire prompt and current object states
        self.curr_prompt: str = ""
        self.curr_obj_states: str = ""

        # Initialize container to store rollout without
        # any other material in prompt
        self.trace: str = ""
        self.rag: Optional["RAG"] = None

        self.reset()

        # Build RAG dataset if we want to use RAG
        if self.enable_rag:
            from habitat_llm.planner.rag import RAG

            self.rag = RAG(
                plan_config.example_type,
                plan_config.rag_dataset_dir,
                plan_config.rag_data_source_name,
                plan_config.llm,
            )
        
        # Read task decomposition flag
        self.use_task_decomposition = plan_config.get("use_task_decomposition", False)
        
        # CycleVLA flag for proactive self-correction
        self.use_cycle_vla = plan_config.get("use_cycle_vla", False)
        
        # CoPAL (Corrective Planning) configuration
        self.use_copal = plan_config.get("use_copal", False)
        self.copal_config = plan_config.get("copal", {})
        
        # Initialize CoPAL components if enabled
        if self.use_copal:
            from habitat_llm.planner.copal.planner_integration import initialize_copal
            initialize_copal(self)
        else:
            self.copal_context_builder = None
            self.error_translator = None
            self.copal_recovery_metrics = None
        
        # Inner Monologue initialization
        inner_mono_config = plan_config.get("inner_monologue", {})
        self.inner_monologue_enabled = inner_mono_config.get("enabled", False)
        self.feedback_generator = None
        self._last_feedback_was_failure = False  # Track for correction effectiveness
        if self.inner_monologue_enabled:
            try:
                from habitat_llm.planner.inno_mono import FeedbackGenerator
                self.feedback_generator = FeedbackGenerator(
                    config=inner_mono_config,
                    env_interface=env_interface
                )
            except Exception as e:
                print(f"[LLMPlanner] Failed to initialize Inner Monologue: {e}")
                self.inner_monologue_enabled = False
        
        # SayCan initialization
        saycan_config = plan_config.get("saycan", {})
        self.saycan_enabled = saycan_config.get("enabled", False)
        self.saycan_analyzer = None
        if self.saycan_enabled:
            try:
                from habitat_llm.evaluation.methods.saycan.saycan_analyzer import SayCanAnalyzer
                self.saycan_analyzer = SayCanAnalyzer(config=saycan_config)
                print("[LLMPlanner] SayCan analyzer initialized (integration mode)")
            except Exception as e:
                print(f"[LLMPlanner] Failed to initialize SayCan analyzer: {e}")
                self.saycan_enabled = False

    def reset(self):
        """
        Reset the planner state.
        """
        self.last_high_level_actions: Dict[int, Tuple[str, str, str]] = {}
        self.replan_required: bool = True
        self.replanning_count: int = 0
        self.is_done: bool = False
        self._consecutive_all_wait_count: int = 0
        self._last_joint_action_signature: Optional[Tuple[Tuple[int, str, str], ...]] = None
        self._consecutive_identical_joint_action_count: int = 0
        self._last_wait_explore_scan_signature: Optional[
            Tuple[Tuple[str, str], ...]
        ] = None
        self._consecutive_wait_explore_scan_count: int = 0

        # save agent observations to get feedback on skill execution
        self.latest_agent_response: Dict[int, str] = {}
        self.curr_prompt: str = ""
        self.trace: str = ""
        self.curr_obj_states: str = ""
        self.params: Dict[str, Any] = {}
        self.pending_context_updates = []
        self.pending_perception_scans = set()
        self._context_compression_request_reason = None
        self.current_instruction = ""
        self.current_scene_graph_snapshot = {}
        self.current_evolution_overlay = EvolutionPolicyOverlay()
        if self.use_prompt_builder and self.prompt_builder:
            self.prompt_builder.reset()
        
        # Reset Inner Monologue statistics
        if hasattr(self, 'inner_monologue_enabled') and self.inner_monologue_enabled and self.feedback_generator:
            self.feedback_generator.reset_statistics()
            self._last_feedback_was_failure = False
        
        # Reset CycleVLA state if it was initialized
        if hasattr(self, '_cycle_initialized') and self._cycle_initialized:
            from habitat_llm.planner.cyclevla import reset_cycle_state
            reset_cycle_state(self)

        # Reset SayCan analyzer if available
        if hasattr(self, 'saycan_analyzer') and self.saycan_analyzer:
            self.saycan_analyzer.reset()
        
        # Reset CoPAL state if enabled
        if hasattr(self, 'copal_context_builder') and self.copal_context_builder:
            self.copal_context_builder.reset()
        if hasattr(self, 'copal_recovery_metrics') and self.copal_recovery_metrics:
            self.copal_recovery_metrics.reset()
            
        # Clean up ReboundManager objects if rebound is disabled
        # if not self.rebound_enabled:
        #     self.rebound_managers.clear()

        # Reset agents
        for agent in self._agents:
            agent.reset()

    def _assistant_turn_start(self) -> str:
        if self.planner_config.planning_mode.lower() == "cot":
            return f"{self.planner_config.llm.assistant_tag}Thought:"
        return f"{self.planner_config.llm.assistant_tag}"

    def _refresh_prompt(self) -> None:
        if self.use_prompt_builder and self.prompt_builder:
            self.curr_prompt = self.prompt_builder.build_prompt(
                self._assistant_turn_start()
            )

    def _is_wait_action(self, action: Tuple) -> bool:
        return bool(
            action
            and action[0] is not None
            and str(action[0]).strip().lower() == "wait"
        )

    def _complete_missing_actions_as_wait(
        self, actions: Optional[Dict[int, Tuple]]
    ) -> Tuple[Dict[int, Tuple], Dict[int, str]]:
        complete_actions = dict(actions) if actions else {}
        autofill_msgs: Dict[int, str] = {}
        for agent in self.agents:
            if agent.uid not in complete_actions:
                msg = (
                    "Auto-filled Wait[] because planner output omitted this "
                    f"agent action (agent_id={agent.uid})."
                )
                complete_actions[agent.uid] = ("Wait", None, msg)
                autofill_msgs[agent.uid] = msg
        return complete_actions, autofill_msgs

    def _is_all_wait(self, actions: Dict[int, Tuple]) -> bool:
        """Check whether *every* agent action is Wait.

        Requires every controlled agent to have a valid Wait action, avoiding
        the ``all([]) == True`` Python gotcha and not masking parser errors.
        """
        if not actions or len(actions) < len(self.agents):
            return False
        return len(self.agents) > 0 and all(
            self._is_wait_action(actions.get(agent.uid)) for agent in self.agents
        )

    def _joint_action_signature(
        self, actions: Dict[int, Tuple]
    ) -> Tuple[Tuple[int, str, str], ...]:
        signature: List[Tuple[int, str, str]] = []
        for agent in self.agents:
            action = actions.get(agent.uid)
            if not action:
                signature.append((int(agent.uid), "", ""))
                continue
            action_name = "" if action[0] is None else str(action[0]).strip().lower()
            action_arg = ""
            if len(action) > 1 and action[1] is not None:
                action_arg = str(action[1]).strip().lower()
            signature.append((int(agent.uid), action_name, action_arg))
        return tuple(signature)

    def _wait_explore_scan_signature(
        self, actions: Dict[int, Tuple]
    ) -> Optional[Tuple[Tuple[str, str], ...]]:
        """Detect no-progress Wait/Explore cycles, ignoring agent swaps and scans."""
        if not actions:
            return None

        rebound_noop_tools = {
            "perceptionscantool",
            "contextsummarizetool",
            "cognitiverollbacktool",
            "taskreflecttool",
        }
        productive: List[Tuple[str, str]] = []
        saw_rebound_noop = False

        for agent in self.agents:
            action = actions.get(agent.uid)
            if not action or action[0] is None:
                continue
            action_name = str(action[0]).strip().lower()
            action_arg = ""
            if len(action) > 1 and action[1] is not None:
                action_arg = str(action[1]).strip().lower()

            if action_name == "wait":
                continue
            if action_name in rebound_noop_tools:
                saw_rebound_noop = True
                continue
            if action_name == "explore":
                productive.append(("explore", action_arg))
                continue
            return None

        if productive:
            return tuple(sorted(productive))
        if saw_rebound_noop:
            return self._last_wait_explore_scan_signature or (
                ("rebound_noop", ""),
            )
        return None

    def _update_wait_explore_scan_cycle_count(
        self, actions: Dict[int, Tuple]
    ) -> int:
        signature = self._wait_explore_scan_signature(actions)
        if signature is None:
            self._reset_wait_explore_scan_cycle_count()
            return 0
        if signature == self._last_wait_explore_scan_signature:
            self._consecutive_wait_explore_scan_count += 1
        else:
            self._last_wait_explore_scan_signature = signature
            self._consecutive_wait_explore_scan_count = 1
        return self._consecutive_wait_explore_scan_count

    def _reset_wait_explore_scan_cycle_count(self) -> None:
        self._last_wait_explore_scan_signature = None
        self._consecutive_wait_explore_scan_count = 0

    def _update_identical_joint_action_count(
        self, actions: Dict[int, Tuple]
    ) -> int:
        signature = self._joint_action_signature(actions)
        if signature == self._last_joint_action_signature:
            self._consecutive_identical_joint_action_count += 1
        else:
            self._last_joint_action_signature = signature
            self._consecutive_identical_joint_action_count = 1
        return self._consecutive_identical_joint_action_count

    def _reset_identical_joint_action_count(self) -> None:
        self._last_joint_action_signature = None
        self._consecutive_identical_joint_action_count = 0

    def _inject_stall_observations(self, msg: str) -> str:
        """Inject a stall-recovery message as standard Agent_X_Observation
        lines, ensuring the LLM always receives the expected prompt format.

        Returns the print_str fragment.
        """
        responses = {agent.uid: msg for agent in self.agents}
        return self._add_responses_to_prompt(responses, is_replanning_step=True)

    def enqueue_context_update(self, title: str, content: str) -> None:
        if not content:
            return
        self.trace += f"\n[{title}]\n{content}\n"
        if not (self.use_prompt_builder and self.prompt_builder):
            return
        self.pending_context_updates.append((title, content))

    def set_evolution_context(self, title: str, content: str) -> None:
        if not content:
            self.evolution_context = None
            return
        if not title:
            title = "Evolution Context"
        self.evolution_context = (title, content.strip())

    def set_evolution_manager(self, manager: Any) -> None:
        self.evolution_manager = manager

    def refresh_evolution_overlay(
        self,
        instruction: str,
        fault_type: str = "",
        inject_context: bool = True,
    ) -> EvolutionPolicyOverlay:
        self.current_instruction = instruction or self.current_instruction
        if self.evolution_manager is None or not hasattr(self.evolution_manager, "retrieve_overlay"):
            self.current_evolution_overlay = EvolutionPolicyOverlay()
            return self.current_evolution_overlay

        world_state = {}
        if hasattr(self.env_interface, "get_world_state_dict"):
            try:
                world_state = self.env_interface.get_world_state_dict(
                    instruction=self.current_instruction,
                    task_context={"active_skills": self.tool_list},
                )
            except Exception:
                world_state = {}
        if not world_state:
            try:
                world_state = self.perception_connector.extract_world_state(self.env_interface)
            except Exception:
                world_state = {}

        scene_graph_snapshot = world_state.get("scene_graph_snapshot", {})
        self.current_scene_graph_snapshot = scene_graph_snapshot
        try:
            overlay = self.evolution_manager.retrieve_overlay(
                instruction=self.current_instruction,
                world_state=world_state,
                graph_snapshot=scene_graph_snapshot,
                fault_type=fault_type,
            )
        except TypeError:
            overlay = self.evolution_manager.retrieve_overlay(
                instruction=self.current_instruction,
                world_state=world_state,
                graph_snapshot=scene_graph_snapshot,
            )

        if not isinstance(overlay, EvolutionPolicyOverlay):
            overlay = EvolutionPolicyOverlay(
                scene_id=str(overlay.get("scene_id", "")),
                task_signature=str(overlay.get("task_signature", "")),
                graph_snapshot_hash=str(overlay.get("graph_snapshot_hash", "")),
                current_task_guidance=list(overlay.get("current_task_guidance", []) or []),
                candidate_risks=list(overlay.get("candidate_risks", []) or []),
                preferred_recovery=list(overlay.get("preferred_recovery", []) or []),
                preferred_skill_order=list(overlay.get("preferred_skill_order", []) or []),
                lesson_refs=list(overlay.get("lesson_refs", []) or []),
                confidence=float(overlay.get("confidence", 0.0) or 0.0),
                source=str(overlay.get("source", "memory")),
            )

        self.current_evolution_overlay = overlay
        if inject_context and not overlay.is_empty():
            self.enqueue_context_update("Evolution Overlay", overlay.to_prompt_context())
        return overlay

    def _consume_context_updates(self) -> List[Tuple[str, str]]:
        updates = list(self.pending_context_updates)
        self.pending_context_updates = []
        return updates

    def truncate_prompt_history(self, keep_last_turns: int) -> None:
        if self.use_prompt_builder and self.prompt_builder:
            self.prompt_builder.truncate_history(keep_last_turns)
            self._refresh_prompt()

    def request_context_compression(self, reason: str = "context_overflow") -> None:
        self._context_compression_request_reason = reason or "context_overflow"

    def _cfg_to_plain_dict(self, cfg: Any) -> Dict[str, Any]:
        if cfg is None:
            return {}
        if isinstance(cfg, dict):
            return dict(cfg)
        try:
            resolved = OmegaConf.to_container(cfg, resolve=True)
            return dict(resolved) if isinstance(resolved, dict) else {}
        except Exception:
            return {}

    def _initialize_context_compressor(self) -> Optional[ContextCompressor]:
        if not self.context_compression_config:
            return None
        if not bool(self.context_compression_config.get("enabled", False)):
            return None
        try:
            compression_cfg = CompressionConfig.from_mapping(
                self.context_compression_config
            )
            llm_client = None if compression_cfg.llm_base_url else self.llm
            return ContextCompressor(
                llm_client=llm_client,
                config=compression_cfg,
            )
        except Exception as exc:
            print(f"[LLMPlanner] Failed to initialize context compressor: {exc}")
            return None

    def _iter_named_values(self, payload: Any) -> List[str]:
        names: List[str] = []
        if isinstance(payload, dict):
            for key, value in payload.items():
                if isinstance(key, str) and key in {
                    "name",
                    "object_name",
                    "furniture_name",
                    "room_name",
                    "entity_name",
                } and isinstance(value, str):
                    names.append(value)
                else:
                    names.extend(self._iter_named_values(value))
        elif isinstance(payload, list):
            for item in payload:
                names.extend(self._iter_named_values(item))
        return names

    def _extract_protected_entities(self) -> List[str]:
        config = self.context_compression_config or {}
        sources = set(config.get("protected_entity_sources", []))
        names: List[str] = []

        if "world_graph" in sources and getattr(self.env_interface, "full_world_graph", None):
            full_graph = self.env_interface.full_world_graph
            for getter_name in ("get_all_objects", "get_all_furnitures", "get_all_rooms"):
                getter = getattr(full_graph, getter_name, None)
                if getter is None:
                    continue
                try:
                    for node in getter():
                        name = getattr(node, "name", None)
                        if isinstance(name, str) and name.strip():
                            names.append(name.strip())
                except Exception:
                    continue

        if "scene_graph" in sources and self.current_scene_graph_snapshot:
            names.extend(self._iter_named_values(self.current_scene_graph_snapshot))

        if "episode" in sources:
            try:
                episode = self.env_interface.env.env.env._env.current_episode
            except Exception:
                episode = None
            if episode is not None:
                names.extend(self._iter_named_values(getattr(episode, "__dict__", {})))

        if "task" in sources:
            task_text = (self.current_instruction or self.params.get("input", "")).strip()
            if task_text:
                lowered_task = task_text.lower()
                for candidate in list(names):
                    if candidate.lower() in lowered_task:
                        names.append(candidate)

        if "agent_descriptions" in sources:
            agent_text = str(self.params.get("agent_descriptions", "")).strip()
            if agent_text:
                names.extend(
                    re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*(?:_[0-9]+)?\b", agent_text)
                )

        # Keep termination/action markers (e.g., Done[], Wait[]) from prompt rules
        # so LLM-based compression does not drop completion constraints.
        if "termination_rules" in sources:
            prompt_text = "\n".join(
                [
                    str(getattr(self, "prompt", "") or ""),
                    str(self.curr_prompt or ""),
                ]
            )
            names.extend(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\[\]", prompt_text))

        deduped = []
        seen = set()
        for name in sorted(names, key=lambda item: (-len(item), item.lower())):
            lowered = name.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            deduped.append(name)
        return deduped

    def _build_compressed_prompt(
        self,
        *,
        assistant_turn_start: str,
        context_updates: Optional[List[Tuple[str, str]]] = None,
        force: bool = False,
    ) -> Optional[str]:
        if not (self.use_prompt_builder and self.prompt_builder and self.context_compressor):
            return None

        max_prompt_chars = int(
            self.context_compression_config.get("max_prompt_chars", 12000)
        )
        preserve_last_turns = int(
            self.context_compression_config.get("preserve_last_turns", 6)
        )

        base_prompt = self.curr_prompt
        if context_updates:
            base_prompt = self.prompt_builder.build_prompt_with_context(
                current_prompt=self.curr_prompt,
                assistant_turn_start=assistant_turn_start,
                context_blocks=context_updates,
            )
        if not force and len(base_prompt) <= max_prompt_chars:
            return None

        history = list(self.prompt_builder.prompt_history)
        if not history:
            return None
        if len(history) <= preserve_last_turns and not force:
            return None

        compression = self.context_compressor.compress_prompt_state(
            task_instruction=self.current_instruction or self.params.get("input", ""),
            prompt_history=history,
            trajectory_summary=self.trace,
            world_description=str(self.params.get("world_description", "")),
            agent_description=str(self.params.get("agent_descriptions", "")),
            protected_keywords=self._extract_protected_entities(),
            preserve_last_turns=preserve_last_turns,
        )
        if not compression:
            return None

        recent_history = history[-preserve_last_turns:] if preserve_last_turns > 0 else []
        compressed_history = list(compression.get("summary_turns") or []) + recent_history

        compressed_params = dict(self.params)
        if compressed_params.get("world_description") is not None:
            compressed_params["world_description"] = (
                compression.get("compressed_world_description")
                or compressed_params.get("world_description", "")
            )
        if compressed_params.get("agent_descriptions") is not None:
            compressed_params["agent_descriptions"] = (
                compression.get("compressed_agent_description")
                or compressed_params.get("agent_descriptions", "")
            )

        try:
            compressed_base_prompt = self.prompt.format(**compressed_params)
        except Exception as exc:
            print(f"[LLMPlanner] Context compression prompt rebuild failed: {exc}")
            return None

        compressed_prompt = self.prompt_builder.build_prompt_from_parts(
            base_prompt=compressed_base_prompt,
            history=compressed_history,
            assistant_turn_start=assistant_turn_start,
        )
        if context_updates:
            compressed_prompt = self.prompt_builder.build_prompt_with_context(
                current_prompt=compressed_prompt,
                assistant_turn_start=assistant_turn_start,
                context_blocks=context_updates,
            )

        reason = self._context_compression_request_reason or "prompt_budget"
        self.trace += (
            f"\n[Context Compression]\nreason={reason}; "
            f"history_turns={len(history)}->{len(compressed_history)}\n"
        )
        print(
            f"[LLMPlanner] Applied context compression ({reason}) "
            f"{len(base_prompt)} -> {len(compressed_prompt)} chars."
        )
        self._context_compression_request_reason = None

        # ── Unified trigger bridge: stamp t_c so OnlineReboundTracker ──
        if not hasattr(self, "_pending_compression_events"):
            self._pending_compression_events: list = []
        self._pending_compression_events.append(
            {"reason": reason, "chars_before": len(base_prompt), "chars_after": len(compressed_prompt)}
        )

        return compressed_prompt


    def build_tool_grammar(self, world_graph: "WorldGraph") -> str:
        """
        This method builds a grammar that accepts all valid tool calls based a world graph
        The grammar is specified in the EBNF grammar description format
        see https://github.com/epfl-dlab/transformers-CFG for details and examples

        :param world_graph: The world graph.
        """
        tool_grammar = {}
        objects = world_graph.get_all_objects()
        rules = []
        # we cannot include rules which have objects when there are no objects
        if len(objects) != 0:
            object_expansion = " | ".join(
                (f'"{x.name}"' for x in world_graph.get_all_objects())
            )
            object_rule = f"{OBJECT} ::= " + object_expansion
            nav_target_rule = f"{NAV_TARGET} ::= ({FURNITURE} | {ROOM} | {OBJECT})"
            object_of_furniture_rule = (
                f"{OBJECT_OR_FURNITURE} ::= ({FURNITURE} | {OBJECT})"
            )
            rules.append(nav_target_rule)
            rules.append(object_rule)
            rules.append(object_of_furniture_rule)
        else:
            object_of_furniture_rule = f"{OBJECT_OR_FURNITURE} ::= {FURNITURE}"
            nav_target_rule = f"{NAV_TARGET} ::= ({FURNITURE} | {ROOM})"
            rules.append(nav_target_rule)
            rules.append(object_of_furniture_rule)
        for agent in self.agents:
            for tool_name, tool in agent.tools.items():
                if tool_name not in tool_grammar:
                    # skip tools that require objects when there are no objects
                    if OBJECT in tool.argument_types and len(objects) == 0:
                        continue
                    tool_grammar[tool_name] = tool.grammar()
        tool_grammar["Done"] = '"Done[]"'
        grammar_str = "tool_call ::= " + " | ".join(tool_grammar.keys()) + "\n"
        for tool_name, tool_grammar_str in tool_grammar.items():
            grammar_str += f"{tool_name} ::= {tool_grammar_str}\n"

        # build rules for each of the argument types
        furniture_rule = f"{FURNITURE} ::= " + " | ".join(
            (f'"{x.name}"' for x in world_graph.get_all_furnitures())
        )
        room_rule = f"{ROOM} ::= " + " | ".join(
            (f'"{x.name}"' for x in world_graph.get_all_rooms())
        )
        spatial_constraint_rule = f'{SPATIAL_CONSTRAINT} ::= "next_to"'
        spatial_relation_rule = f'{SPATIAL_RELATION} ::= "on" | "within"'
        free_text_rule = f"{FREE_TEXT} ::= [ \"'.:,!a-zA-Z_0-9]*"
        white_space_rule = "WS ::= [ ]*"
        rules.append(furniture_rule)
        rules.append(room_rule)
        rules.append(spatial_constraint_rule)
        rules.append(spatial_relation_rule)
        rules.append(free_text_rule)
        rules.append(white_space_rule)
        grammar_str += "\n".join(rules)
        return grammar_str

    def build_response_grammar(self, world_graph: "WorldGraph") -> str:
        """
        Build a grammar that accepts all valid responses based on a world graph.

        :param world_graph: The world graph.
        """
        delimiter = "\\n"
        tool_rules = self.build_tool_grammar(world_graph)

        action_rules = []
        for i, agent in enumerate(self.agents):
            agent_id = agent.uid
            action_rule = f'action_{i} ::= "Agent_{agent_id}_Action: " tool_call'
            action_rules.append(action_rule)

        combined_action_rule = (
            "action ::= "
            + f' "{delimiter}" '.join(f"action_{i}" for i in range(len(self.agents)))
            + f' "{delimiter}Assigned!"'
        )
        # termination_rule = 'termination ::= "Final Thought: Exit!"'
        termination_rule = f'termination ::= "{self.end_expression}"'
        root_role = f'root ::= {FREE_TEXT} "{delimiter}" (action | termination)'

        return "\n".join(
            [root_role, combined_action_rule]
            + action_rules
            + [termination_rule, tool_rules]
        )

    def __initialize_llm(self):
        """
        This method instantiates LLM as defined in the config
        """
        # Instantiate LLM from the Hydra config
        llm_conf = self.planner_config.llm
        self.llm = instantiate(llm_conf.llm)
        self.llm = self.llm(llm_conf)

        # Setup the LLM parameters
        # self.instruct = self.planner_config.llm.instruct
        self.instruct = self.planner_config.instruct
        self.prompt = self.instruct.prompt
        self.stopword = self.instruct.stopword
        self.end_expression = self.instruct.end_expression
        self.actions_parser = instantiate(self.instruct.actions_parser)

        # save agent observations to get feedback on skill execution
        self.latest_agent_response = {}

    def prepare_prompt(
        self, input_instruction: str, world_graph: "WorldGraph", **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Prepare the prompt for the LLM.

        :param input_instruction: The input instruction.
        :param world_graph: The world graph.
        :return: The prepared prompt and parameters.
        """
        params = {
            "input": input_instruction,
            "tool_list": self.tool_list,
            "world_graph": world_graph,
            "id": self.agents[0].uid,
        }

        # We modify the prompt if we want to use RAG and the prompt has not
        # been modified
        if "{rag_examples}" in self.prompt:
            if self.rag is not None:
                _, index = self.rag.retrieve_top_k_given_query(
                    input_instruction, top_k=1, agent_id=self._agents[0].uid
                )
                index = index[0]

                example_str = (
                    f"{self.planner_config.llm.user_tag}Below are some example solutions from different settings:\nExample 1:\n"
                    + self.rag.data_dict[index]["trace"]
                    + "\n"
                )
                params["rag_examples"] = example_str
            else:
                params["rag_examples"] = ""
        if "{tool_descriptions}" in self.prompt:
            params["tool_descriptions"] = self.agents[0].tool_descriptions
        if "{agent_descriptions}" in self.prompt:
            params["agent_descriptions"] = self.agent_descriptions
        if "{tool_list}" in self.prompt:
            params["tool_list"] = self.tool_list
        if "{system_tag}" in self.prompt:
            params["system_tag"] = self.planner_config.llm.system_tag
        if "{user_tag}" in self.prompt:
            params["user_tag"] = self.planner_config.llm.user_tag
        if "{assistant_tag}" in self.prompt:
            params["assistant_tag"] = self.planner_config.llm.assistant_tag
        if "{eot_tag}" in self.prompt:
            params["eot_tag"] = self.planner_config.llm.eot_tag
        if "{agent_role_description}" in self.prompt:
            # only support agent role description when planning for a single agent
            assert len(self.agents) == 1
            if str(self.agents[0].uid) == "1":
                agent_role_description = '\nYou are playing the role of the task giver. This means if the instruction says something like "You should move the object and I will wash it", then the other agent should be moving the object, and you should washing the it.\n'
            else:
                agent_role_description = '\nYou are playing the role of the task receiver. This means if the instruction says something like "You should move the object and I will wash it", then you should move the object and the other agent should wash it.\n'
            params["agent_role_description"] = agent_role_description
        if "{world_description}" in self.prompt:
            # only designed for the decentralized setting
            assert len(self.agents) == 1
            world_description = get_world_descr(
                world_graph,
                agent_uid=self.agents[0].uid,
                add_state_info=self.planner_config.objects_response_include_states,
                include_room_name=True,
                centralized=self.planner_config.centralized,
            )
            params["world_description"] = world_description

        if "should_format" in kwargs and not kwargs["should_format"]:
            # In some cases a subclass may want to fill the extra arguments here, so we don't format
            # because those arguments would be missing.
            output_prompt = ""
        else:
            output_prompt = self.prompt.format(**params)
            assistant_turn_start = self._assistant_turn_start()
            if self.use_prompt_builder and self.prompt_builder:
                self.prompt_builder.set_base_prompt(
                    output_prompt, assistant_turn_start
                )
                if self.evolution_context:
                    title, content = self.evolution_context
                    self.prompt_builder.add_user_turn(content, title=title)
                    self.evolution_context = None
                output_prompt = self.prompt_builder.build_prompt(
                    assistant_turn_start
                )
            elif self.evolution_context:
                title, content = self.evolution_context
                base_prompt = output_prompt.rstrip()
                if assistant_turn_start and base_prompt.endswith(assistant_turn_start):
                    base_prompt = base_prompt[: -len(assistant_turn_start)].rstrip()
                content = content.strip()
                if title:
                    content = f"{title}:\n{content}"
                output_prompt = (
                    f"{base_prompt}\n{self.planner_config.llm.user_tag}{content}\n{self.planner_config.llm.eot_tag}"
                )
                if assistant_turn_start:
                    output_prompt = f"{output_prompt}\n{assistant_turn_start}"
                self.evolution_context = None
        return output_prompt, params

    @property
    def tool_list(self) -> List[str]:
        """
        Returns a string listing the agents tools
        :return: A sorted list of tool names.
        """
        tool_set = set()
        for agent in self.agents:
            for tool_name in agent.tools:
                tool_set.add(tool_name)

        return sorted(tool_set)

    @property
    def agents(self) -> List["Agent"]:
        """
        Get the list of agents associated with this planner.

        :return: A list of Agent objects.
        """
        return self._agents

    @agents.setter
    def agents(self, agents: List["Agent"]) -> None:
        """
        Set the list of agents for this planner.

        :param agents: A list of Agent objects to be associated with this planner.
        """
        self._agents = agents
        # Pass on respective LLM instance into agent tools
        for agent in self._agents:
            agent.pass_llm_to_tools(self.llm)
            
            # Initialize ReboundManager for the agent if not already present (only if enabled)
            # If rebound_enabled is False, we initialize in 'monitoring_only' mode via config
            if agent.uid not in self.rebound_managers:
                 mgr_config = dict(self.rebound_config) if self.rebound_config else {}
                 if not self.rebound_enabled:
                     mgr_config["monitoring_only"] = True
                 
                 self.rebound_managers[agent.uid] = ReboundManager(agent.uid, config=mgr_config)

            # Inject ReboundManager into CognitiveRollbackTool if present
            if "CognitiveRollbackTool" in agent.tools:
                agent.tools["CognitiveRollbackTool"].rebound_manager = self.rebound_managers[agent.uid]

    def get_last_agent_states(self) -> Dict[int, str]:
        """
        Get the last state descriptions for all agents.

        :return: A dictionary mapping agent UIDs to their last state descriptions.
        """
        # Container to store state descriptions
        agent_states = {}

        # Loop through the agents and populate state descriptions
        for agent in self._agents:
            agent_states[agent.uid] = agent.get_last_state_description()

        return agent_states

    # TODO: @zephirefaith implement agent's room affiliations in the world graph
    # and edit this function to read from it
    def get_last_agent_positions(self) -> Dict[str, Any]:
        """
        Get the last positions for all agents.

        :return: A dictionary mapping agent names to their positions.
        """
        # Container to store agent positions
        agent_positions = {}

        # get agent nodes
        agents = self.env_interface.full_world_graph.get_agents()

        # Loop through the agents and populate nodes
        for agent in agents:
            agent_positions[agent.name] = agent.get_property("translation")

        return agent_positions

    def get_agent_collisions(self) -> Dict[int, bool]:
        """
        Check if the agents are colliding.

        :return: A dictionary mapping agent UIDs to collision status.
        """
        # set collision to false
        collision = False

        # Get list of agent ids
        agent_ids = [
            articulated_agent.sim_obj.object_id
            for articulated_agent in self.env_interface.sim.agents_mgr.articulated_agents_iter
        ]

        # Return false if only one agent is in the scene
        if len(agent_ids) == 2:
            # Perform collision check
            self.env_interface.sim.perform_discrete_collision_detection()
            contact_points = self.env_interface.sim.get_physics_contact_points()

            for cp in contact_points:
                if coll_name_matches(cp, agent_ids[0]) and coll_name_matches(
                    cp, agent_ids[1]
                ):
                    collision = True

        # Declare output container
        out = {}

        # update the output
        for agent in self._agents:
            out[agent.uid] = collision

        return out

    def format_response(
        self, response: str, end_expression: Union[str, List[str]]
    ) -> str:
        """
        Format the LLM response by trimming it up to the first appearance of end_expression.

        :param response: The LLM response to format.
        :param end_expression: The end expression(s) to look for.
        :return: The formatted response.
        """
        response = response.rstrip("\n")
        if type(end_expression) == str:
            index = response.find(end_expression)
            target_end_expression = end_expression
        else:
            # end_expression is a list of string
            index = -1
            target_end_expression = ""
            for _end_expression in end_expression:
                _index = response.find(_end_expression)
                if _index != -1:
                    if index == -1 or _index < index:
                        index = _index
                        target_end_expression = _end_expression
        return (
            response[: index + len(target_end_expression)] if index != -1 else response
        )

    def parse_thought(self, input_string: str) -> str:
        """
        Extract thought from the LLM response.

        :param input_string: The input string to parse.
        :return: The extracted thought.
        """
        # Define the patterns for Agent actions
        pattern = r"\n|Final Thought"

        # Search for the pattern in the input string
        match = re.search(pattern, input_string)

        if match:
            # Extract the text before the pattern
            return input_string[: match.start()].strip()
        else:
            # If no pattern is found, return the whole string
            return ""

    def _add_responses_to_prompt(self, responses: Dict[int, str], is_replanning_step: bool = False) -> str:
        """
        Add agent responses to the prompt.

        :param responses: A dictionary of agent responses.
        :param is_replanning_step: Whether this is a replanning step (Planning Step) vs simulation step.
        :return: The updated print string.
        """
        print_str = ""
        observation_lines: List[str] = []
        add_object_update = False
        for agent_uid in sorted(responses.keys()):
            # If the response for a given agent is valid, add to the prompt and printout
            if responses[agent_uid]:
                line = f"Agent_{agent_uid}_Observation:{responses[agent_uid]}"
                print_str += f"{line}\n"
                observation_lines.append(line)
                self.trace += f"{line}\n"

            # If the response is empty then indicate the action is still in progress
            # only when replanning was required
            elif self.replan_required:
                responses[
                    agent_uid
                ] = f"Action {self.last_high_level_actions[agent_uid][0]}[{self.last_high_level_actions[agent_uid][1]}] is still in progress."

                line = f"Agent_{agent_uid}_Observation:{responses[agent_uid]}"
                print_str += f"{line}\n"
                observation_lines.append(line)
                self.trace += f"{line}\n"
                add_object_update = True

            # save agent observations to get feedback on skill execution
            self.latest_agent_response[agent_uid] = responses[agent_uid]
        
        if observation_lines:
            if getattr(self, "pending_perception_scans", None):
                self.pending_perception_scans.clear()
            context_sections = ["Observations:\n" + "\n".join(observation_lines)]
            result = ""
            if (
                self.planner_config.objects_response
                and add_object_update
                and self.planner_config.centralized
            ):
                world_graph = self.env_interface.world_graph[agent_uid]
                objects = get_objects_descr(
                    world_graph,
                    agent_uid,
                    include_room_name=True,
                    add_state_info=self.planner_config.objects_response_include_states,
                    centralized=self.planner_config.centralized,
                )
                if self.planner_config.prompt_w_updatedobjects_only:
                    # add details on what changed in the world.
                    # TODO: this currently assumes symmetric world graph,
                    # extend for decentralized/asymmetric WG
                    updated_objects = get_rearranged_objects_descr(
                        obj_descr_t=objects, obj_descr_t_1=self.curr_obj_states
                    )
                    self.curr_obj_states = objects
                    if updated_objects != "":
                        result = (
                            "Newly found objects/updates on known objects: "
                            f"{updated_objects}"
                        )
                    else:
                        result = (
                            "No new objects or updates on known objects were found."
                        )
                else:
                    result = f"Objects: {objects}"
                self.curr_obj_states = objects
                if result:
                    context_sections.append(f"World Update:\n{result}")
                    self.trace += f"{result}\n"
                    print_str += f"World Update:\n{result}\n"

            if self.use_prompt_builder and self.prompt_builder:
                context_content = "\n\n".join(context_sections)
                self.prompt_builder.add_user_turn(
                    context_content, title="Context Update"
                )
                self._refresh_prompt()
            else:
                prompt_addition = "\n".join(observation_lines) + "\n"
                self.curr_prompt += self.planner_config.llm.user_tag + prompt_addition
                if result:
                    self.curr_prompt += result + "\n"
                self.curr_prompt += self.planner_config.llm.eot_tag

        # Force add thought after every observation
        # Enhanced: Also force Thought if Inner Monologue is enabled
        if (self.planner_config.planning_mode.lower() == "cot" or 
            (self.inner_monologue_enabled and self.feedback_generator)):
            for agent_uid in sorted(responses.keys()):
                if responses[agent_uid]:
                    print_str += "Thought:"
                    if not (self.use_prompt_builder and self.prompt_builder):
                        prompt_addition = (
                            f"{self.planner_config.llm.assistant_tag}Thought:"
                        )
                        self.curr_prompt += prompt_addition
                    self.trace += "Thought:"
                    break
        return print_str

    def generate_action_response(self, prompt_override: str = None) -> str:
        """
        Generate an action response using the LLM.
        Factored out to allow pipelines to invoke generation with modified prompts.
        """
        if prompt_override is not None:
            prompt = prompt_override
        else:
            prompt = self.curr_prompt
            if self.use_prompt_builder and self.prompt_builder:
                context_updates = self._consume_context_updates()
                assistant_turn_start = self._assistant_turn_start()
                compressed_prompt = self._build_compressed_prompt(
                    assistant_turn_start=assistant_turn_start,
                    context_updates=context_updates,
                    force=bool(self._context_compression_request_reason),
                )
                if compressed_prompt is not None:
                    prompt = compressed_prompt
                elif context_updates:
                    prompt = self.prompt_builder.build_prompt_with_context(
                        current_prompt=self.curr_prompt,
                        assistant_turn_start=assistant_turn_start,
                        context_blocks=context_updates,
                    )
        
        if self.planner_config.get("constrained_generation", False):
            # We need world_graph for grammar. Assuming self._agents[0] context.
            world_graph = self.env_interface.world_graph[self._agents[0].uid]
            llm_response = self.llm.generate(
                prompt,
                self.stopword,
                generation_args={
                    "grammar_definition": self.build_response_grammar(world_graph)
                },
            )
        else:
            llm_response = self.llm.generate(prompt, self.stopword)
        
        # CoPAL: Record estimated tokens for RTR (Recovery Token Ratio)
        if hasattr(self, "copal_recovery_metrics") and self.copal_recovery_metrics:
            # Estimate tokens: ~4 chars per token
            est_tokens = (len(prompt) + len(llm_response)) // 4
            self.copal_recovery_metrics.record_tokens(
                tokens=est_tokens,
                is_recovery=self.copal_recovery_metrics.is_in_recovery()
            )
        
        return llm_response

    def replan(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
    ):
        """
        Replan a high level action using the LLM/VLM, Phase-Based Pipeline, or CycleVLA.
        """
        if self.use_cycle_vla:
            # Use CycleVLA proactive self-correction pipeline
            from habitat_llm.planner.cyclevla import run_cycle_pipeline, get_cycle_statistics
            llm_response = run_cycle_pipeline(
                self,
                instruction,
                self.env_interface,
                self.planner_config.llm
            )
            # Fetch CycleVLA statistics for logging
            cycle_stats = get_cycle_statistics(self)
        elif self.use_task_decomposition:
            # Call the phase-based pipeline
            llm_response = run_phase_based_replan(
                self,
                instruction,
                self.env_interface, # Pass env_interface for world state extraction
                self.planner_config.llm # Pass config
            )
            cycle_stats = {}
        else:
            # Fallback to direct generation
            llm_response = self.generate_action_response(prompt_override=None)
            cycle_stats = {}

        # Format the response
        end_expressions = []
        if isinstance(self.end_expression, list):
            end_expressions.extend(self.end_expression)
        else:
            end_expressions.append(self.end_expression)

        if isinstance(self.stopword, list):
            end_expressions.extend(self.stopword)
        else:
            end_expressions.append(self.stopword)

        llm_response = self.format_response(llm_response, end_expressions)

        info = {"llm_response": llm_response}
        info.update(cycle_stats)
        return info


    def get_next_action(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Dict[int, "WorldGraph"],
        verbose: bool = False,
    ) -> Tuple[Dict[int, Any], Dict[str, Any], bool]:
        """
        Get the next low-level action to execute.

        :param instruction: The instruction for the task.
        :param observations: The current observations.
        :param world_graph: The world graph for each agent.
        :param verbose: Whether to print verbose output. Defaults to False.
        :return: A tuple containing:
                 - The low-level actions for each agent
                 - Planner information
                 - Whether the planner is done
        """
        planner_info: Dict[str, Union[Any, str]] = {}
        # Apply optional prompt-level perturbation (no-op by default).
        if self._prompt_perturbation_hook is not None:
            try:
                rewritten = self._prompt_perturbation_hook(instruction)
                if rewritten and isinstance(rewritten, str) and rewritten != instruction:
                    instruction = rewritten
            except Exception:
                # Never let a perturbation hook break planning.
                pass
        self.current_instruction = instruction
        # Early return if planner is already done
        if self.is_done:
            planner_info = {
                "prompts": {agent.uid: self.curr_prompt for agent in self.agents},
                "traces": {agent.uid: self.trace for agent in self.agents},
                "replanning_count": {
                    agent.uid: self.replanning_count for agent in self.agents
                },
                "replanned": {agent.uid: False for agent in self.agents},
                "replan_required": {
                    agent.uid: self.replan_required for agent in self.agents
                },
                "is_done": {agent.uid: self.is_done for agent in self.agents},
            }
            return {}, planner_info, self.is_done

        # Log state for Rebound (enabled or monitoring only)
        for agent in self.agents:
            mgr = self.rebound_managers.get(agent.uid)
            if mgr:
                # Get standardized world state from PerceptionConnector
                world_state_dict = self.perception_connector.extract_world_state(self.env_interface)
                
                mgr.log_planner_state(
                    step=self.replanning_count,
                    action=str(self.last_high_level_actions.get(agent.uid, "Wait")),
                    world_graph=world_graph.get(agent.uid),
                    progress=mgr.current_progress, # Use mgr's independently updated progress
                    prompt=self.curr_prompt,
                    trace=self.trace,
                    world_state_dict=world_state_dict
                )

        if self.curr_prompt == "":
            # Prepare prompts
            self.curr_prompt, self.params = self.prepare_prompt(
                instruction, world_graph[self._agents[0].uid], observations=observations
            )
            self.curr_obj_states = get_objects_descr(
                world_graph[self._agents[0].uid],
                self._agents[0].uid,
                include_room_name=True,
                add_state_info=self.planner_config.objects_response_include_states,
                centralized=self.planner_config.centralized,
            )

        if self.trace == "":
            self.trace += f"Task: {instruction}\nThought: "

        print_str = ""
        self.is_done = False

        # Track if this is a replanning step, should only record steps at Planning Steps, not Simulation Steps
        is_replanning_step = self.replan_required

        if self.replan_required:
            try:
                replanning_threshold = int(self.planner_config.replanning_threshold)
            except (TypeError, ValueError):
                replanning_threshold = 100
            if self.replanning_count >= replanning_threshold:
                self.is_done = True
                planner_info = {
                    "print": print_str,
                    "prompts": {agent.uid: self.curr_prompt for agent in self.agents},
                    "traces": {agent.uid: self.trace for agent in self.agents},
                    "replanning_count": {
                        agent.uid: self.replanning_count for agent in self.agents
                    },
                    "replan_required": {
                        agent.uid: self.replan_required for agent in self.agents
                    },
                    "replanned": {agent.uid: True for agent in self.agents},
                    "is_done": {agent.uid: True for agent in self.agents},
                    "thought": {agent.uid: None for agent in self.agents},
                    "high_level_actions": {
                        agent.uid: ("Done", None, None) for agent in self.agents
                    },
                    "termination_reason": "replanning_threshold",
                }
                return {}, planner_info, True

            self.refresh_evolution_overlay(instruction)
            # Simplified Recovery Logic, in high-level analysis
            recovery_result = try_execute_recovery(self, observations, planner_info, print_str)
            if recovery_result:
                return recovery_result

            # Replan
            planner_info["replanned"] = {agent.uid: True for agent in self.agents}
            if verbose:
                # calculate the total time of response generation
                start_time = time.time()

            response_info = self.replan(instruction, observations, world_graph)
            llm_response = response_info["llm_response"]
            # parse thought from the response
            thought = self.parse_thought(llm_response)

            if verbose:
                total_time = time.time() - start_time
                print(
                    f"Time taken for LLM response generation: {total_time}; replanning_count: {self.replanning_count}"
                )

            # Update prompt with the first response
            suffix = f"\n{self.stopword}"
            stopwords_tuple = (
                tuple(self.stopword)
                if isinstance(self.stopword, list)
                else (self.stopword,)
            )

            if llm_response.strip().endswith(stopwords_tuple):
                suffix = ""

            print_str += f"""{llm_response}{suffix}\n"""
            prompt_addition = f"""{llm_response}{suffix}"""
            if self.use_prompt_builder and self.prompt_builder:
                self.prompt_builder.add_assistant_turn(prompt_addition)
                self._refresh_prompt()
                self.trace += f"{prompt_addition}{self.planner_config.llm.eot_tag}"
            else:
                prompt_addition = (
                    f"""{prompt_addition}{self.planner_config.llm.eot_tag}"""
                )
                self.curr_prompt += prompt_addition
                self.trace += prompt_addition
            
            # Track Thought generation for Inner Monologue metrics
            if self.inner_monologue_enabled and self.feedback_generator:
                if "Thought:" in llm_response:
                    self.feedback_generator.record_thought_generation()

            # Check if the planner should stop
            # Stop if the replanning count exceed a certain threshold
            # or end expression is found in llm response
            # This is helpful to break infinite planning loop.
            self.is_done = (self.check_if_agent_done(llm_response)) or (
                self.replanning_count >= replanning_threshold
            )
            
            # Double check termination validity using Rebound (context-only guard).
            apply_rebound_termination_guard(self, llm_response)

            # Increment the llm call counter on every replan
            # the first required plan
            self.replanning_count += 1

            # Track replan sim step for reward shaper (used by critic)
            # This allows LLM evaluation to align with planning steps
            current_sim_step = planner_info.get("sim_step_count", 0)
            for agent in self.agents:
                mgr = self.rebound_managers.get(agent.uid)
                if mgr:
                    mgr.last_replan_sim_step = current_sim_step

            # Early return if stop is required (simplified)
            if self.is_done:
                planner_info = {
                    "print": print_str,
                    "prompts": {agent.uid: self.curr_prompt for agent in self.agents},
                    "traces": {agent.uid: self.trace for agent in self.agents},
                    "replanning_count": {agent.uid: self.replanning_count for agent in self.agents},
                    "replan_required": {agent.uid: self.replan_required for agent in self.agents},
                    "replanned": {agent.uid: True for agent in self.agents},
                    "is_done": {agent.uid: self.is_done for agent in self.agents},
                    "thought": {agent.uid: thought for agent in self.agents},
                    "high_level_actions": {agent.uid: ("Done", None, None) for agent in self.agents},
                }
                return {}, planner_info, self.is_done

            # Parse high level action directives from llm response
            high_level_actions = self.actions_parser(
                self.agents, llm_response, self.params
            )
            if high_level_actions:
                high_level_actions_for_diagnostics, _ = (
                    self._complete_missing_actions_as_wait(high_level_actions)
                )
            else:
                high_level_actions_for_diagnostics = high_level_actions

            # ---- All-Wait interception (replan branch) ----
            # If LLM assigned Wait to every agent, skip execution entirely.
            # Inject a stall observation in the standard Agent_X_Observation
            # format and return immediately so the next get_next_action call
            # triggers a replan with the corrective feedback visible.
            if self._is_all_wait(high_level_actions_for_diagnostics):
                self._consecutive_all_wait_count += 1
                self._reset_identical_joint_action_count()
                self._reset_wait_explore_scan_cycle_count()
                max_all_wait = self.planner_config.get("max_consecutive_all_wait", 3)
                print(
                    f"[WARN] All-Wait intercepted at replan "
                    f"(consecutive={self._consecutive_all_wait_count}/{max_all_wait})"
                )
                if self._consecutive_all_wait_count >= max_all_wait:
                    print("[WARN] All-Wait limit reached — forcing task completion.")
                    self.is_done = True
                    planner_info = {
                        "print": print_str,
                        "prompts": {a.uid: self.curr_prompt for a in self.agents},
                        "traces": {a.uid: self.trace for a in self.agents},
                        "replanning_count": {a.uid: self.replanning_count for a in self.agents},
                        "replan_required": {a.uid: self.replan_required for a in self.agents},
                        "replanned": {a.uid: True for a in self.agents},
                        "is_done": {a.uid: True for a in self.agents},
                        "thought": {a.uid: thought for a in self.agents},
                        "high_level_actions": {a.uid: ("Done", None, None) for a in self.agents},
                    }
                    return {}, planner_info, True

                stall_msg = (
                    "Wait rejected — all agents idle, no progress made. Assign a productive action (Navigate, Pick, Place, Open, Close) immediately."
                )
                print_str += self._inject_stall_observations(stall_msg)
                self.replan_required = True
                self.last_high_level_actions = high_level_actions_for_diagnostics
                planner_info["replanned"] = {a.uid: True for a in self.agents}
                planner_info["replan_required"] = {a.uid: True for a in self.agents}
                planner_info["responses"] = {a.uid: stall_msg for a in self.agents}
                planner_info["thought"] = {a.uid: thought for a in self.agents}
                planner_info["is_done"] = {a.uid: False for a in self.agents}
                planner_info["print"] = print_str
                planner_info["high_level_actions"] = dict(high_level_actions_for_diagnostics)
                planner_info["prompts"] = {a.uid: self.curr_prompt for a in self.agents}
                planner_info["traces"] = {a.uid: self.trace for a in self.agents}
                planner_info["replanning_count"] = {a.uid: self.replanning_count for a in self.agents}
                return {}, planner_info, False
            else:
                self._consecutive_all_wait_count = 0

            if high_level_actions_for_diagnostics:
                cycle_count = self._update_wait_explore_scan_cycle_count(
                    high_level_actions_for_diagnostics
                )
                try:
                    max_wait_explore_scan = int(
                        self.planner_config.get(
                            "max_consecutive_wait_explore_scan_cycles", 4
                        )
                    )
                except (TypeError, ValueError):
                    max_wait_explore_scan = 4
                if cycle_count > 1 and max_wait_explore_scan > 0:
                    print(
                        "[WARN] Wait/Explore/scan no-progress cycle detected "
                        f"(consecutive={cycle_count}/{max_wait_explore_scan})"
                    )
                    if cycle_count >= max_wait_explore_scan:
                        print(
                            "[WARN] Wait/Explore/scan cycle limit reached, "
                            "forcing task completion."
                        )
                        self.is_done = True
                        planner_info = {
                            "print": print_str,
                            "prompts": {a.uid: self.curr_prompt for a in self.agents},
                            "traces": {a.uid: self.trace for a in self.agents},
                            "replanning_count": {
                                a.uid: self.replanning_count for a in self.agents
                            },
                            "replan_required": {
                                a.uid: self.replan_required for a in self.agents
                            },
                            "replanned": {a.uid: True for a in self.agents},
                            "is_done": {a.uid: True for a in self.agents},
                            "thought": {a.uid: thought for a in self.agents},
                            "high_level_actions": {
                                a.uid: ("Done", None, None) for a in self.agents
                            },
                            "termination_reason": (
                                "max_consecutive_wait_explore_scan_cycles"
                            ),
                        }
                        return {}, planner_info, True

                    stall_msg = (
                        "Repeated Wait/Explore/PerceptionScan pattern rejected. "
                        "Do not swap agents or scan the same area again. Choose a "
                        "different room or execute a concrete Navigate/Pick/Place/Open/Close action."
                    )
                    print_str += self._inject_stall_observations(stall_msg)
                    self.replan_required = True
                    self.last_high_level_actions = high_level_actions_for_diagnostics
                    planner_info["replanned"] = {a.uid: True for a in self.agents}
                    planner_info["replan_required"] = {
                        a.uid: True for a in self.agents
                    }
                    planner_info["responses"] = {
                        a.uid: stall_msg for a in self.agents
                    }
                    planner_info["thought"] = {a.uid: thought for a in self.agents}
                    planner_info["is_done"] = {a.uid: False for a in self.agents}
                    planner_info["print"] = print_str
                    planner_info["high_level_actions"] = dict(
                        high_level_actions_for_diagnostics
                    )
                    planner_info["prompts"] = {
                        a.uid: self.curr_prompt for a in self.agents
                    }
                    planner_info["traces"] = {a.uid: self.trace for a in self.agents}
                    planner_info["replanning_count"] = {
                        a.uid: self.replanning_count for a in self.agents
                    }
                    planner_info["stall_reason"] = (
                        "wait_explore_scan_no_progress_cycle"
                    )
                    return {}, planner_info, False

                identical_count = self._update_identical_joint_action_count(
                    high_level_actions_for_diagnostics
                )
                try:
                    max_identical = int(
                        self.planner_config.get(
                            "max_consecutive_identical_joint_actions", 5
                        )
                    )
                except (TypeError, ValueError):
                    max_identical = 3
                if identical_count > 1:
                    if max_identical <= 0:
                        pass
                    else:
                        print(
                            "[WARN] Repeated identical joint action detected "
                            f"(consecutive={identical_count}/{max_identical})"
                        )
                    if max_identical <= 0:
                        pass
                    elif identical_count >= max_identical:
                        print(
                            "[WARN] Identical joint-action limit reached, "
                            "forcing task completion."
                        )
                        self.is_done = True
                        planner_info = {
                            "print": print_str,
                            "prompts": {a.uid: self.curr_prompt for a in self.agents},
                            "traces": {a.uid: self.trace for a in self.agents},
                            "replanning_count": {
                                a.uid: self.replanning_count for a in self.agents
                            },
                            "replan_required": {
                                a.uid: self.replan_required for a in self.agents
                            },
                            "replanned": {a.uid: True for a in self.agents},
                            "is_done": {a.uid: True for a in self.agents},
                            "thought": {a.uid: thought for a in self.agents},
                            "high_level_actions": {
                                a.uid: ("Done", None, None) for a in self.agents
                            },
                            "termination_reason": (
                                "max_consecutive_identical_joint_actions"
                            ),
                        }
                        return {}, planner_info, True
                    elif max_identical > 2: # Here we give 2 retry by LLM
                        stall_msg = (
                            "Repeated identical joint action rejected. The same agent assignments were proposed again without a new strategy. Choose a different target, or finish only if the task is complete."
                        )
                        print_str += self._inject_stall_observations(stall_msg)
                        self.replan_required = True
                        self.last_high_level_actions = high_level_actions_for_diagnostics
                        planner_info["replanned"] = {a.uid: True for a in self.agents}
                        planner_info["replan_required"] = {
                            a.uid: True for a in self.agents
                        }
                        planner_info["responses"] = {
                            a.uid: stall_msg for a in self.agents
                        }
                        planner_info["thought"] = {a.uid: thought for a in self.agents}
                        planner_info["is_done"] = {a.uid: False for a in self.agents}
                        planner_info["print"] = print_str
                        planner_info["high_level_actions"] = dict(
                            high_level_actions_for_diagnostics
                        )
                        planner_info["prompts"] = {
                            a.uid: self.curr_prompt for a in self.agents
                        }
                        planner_info["traces"] = {a.uid: self.trace for a in self.agents}
                        planner_info["replanning_count"] = {
                            a.uid: self.replanning_count for a in self.agents
                        }
                        planner_info["stall_reason"] = (
                            "repeated_identical_joint_action"
                        )
                        return {}, planner_info, False

            print(f"\n\n[DEBUG] Now Executing: {high_level_actions}\n\n")

            # Get low level actions and/or responses
            low_level_actions, responses = self.process_high_level_actions(
                high_level_actions, observations
            )
            responses = handle_rebound_side_effects(self, responses)

            if hasattr(self, 'saycan_analyzer') and self.saycan_analyzer:
                pass

            # Store last executed high level action
            self.last_high_level_actions = high_level_actions
            
            # ========== Inner Monologue Feedback Injection (Replan Step Only) ==========
            # Generate feedback ONLY after a replan step, not at every simulation step.
            # This aligns with evaluation metrics which measure action correctness at the
            # Planning Step level (when actions are decided), not Simulation Step level.
            # Reference: offline_analyzer.py uses "Number of Planning Steps" not Simulation Steps.
            if self.inner_monologue_enabled and self.feedback_generator:
                try:
                    # Get world state from PerceptionConnector
                    world_state = self.perception_connector.extract_world_state(self.env_interface)
                    
                    # If world_state is empty, try to build a minimal one from env_interface
                    if not world_state or (isinstance(world_state, dict) and len(world_state) == 0):
                        world_state = self._build_minimal_world_state()
                    
                    # Get Inner Monologue config
                    inner_mono_config = self.planner_config.get("inner_monologue", {})
                    
                    # Get Critic feedback (optional)
                    critic_feedback = None
                    if inner_mono_config.get("use_critic_feedback", False):
                        # Try to get Critic from evaluation runner
                        # Note: This requires eval_runner to be accessible
                        # For now, we skip this integration point
                        pass
                    
                    # Get Rebound feedback (optional)
                    rebound_feedback = None
                    if inner_mono_config.get("use_rebound_feedback", False) and self.rebound_enabled:
                        # Extract Rebound feedback if available
                        # This can be enhanced later
                        pass
                    
                    # Generate feedback based on responses from the replan step
                    feedback = self.feedback_generator.generate_feedback(
                        agent_responses=responses,
                        last_actions=self.last_high_level_actions,
                        world_state=world_state,
                        critic_feedback=critic_feedback,
                        rebound_feedback=rebound_feedback
                    )
                    
                    # Format and inject feedback
                    if feedback:
                        feedback_text = self._format_inner_monologue_feedback(feedback)
                        if feedback_text and feedback_text.strip():
                            self.curr_prompt += feedback_text
                            self.trace += f"\n[Feedback]\n{feedback_text}\n"
                        
                        # Track if this feedback indicates failure (for correction tracking)
                        has_failure = any(
                            not info.get("success", True)
                            for info in feedback.get("success_detection", {}).values()
                        )
                        if has_failure:
                            # Mark that we need to track correction effectiveness
                            # Record correction attempt (will be marked as success if next feedback shows success)
                            if self.feedback_generator:
                                self.feedback_generator.record_correction_attempt(success=False)
                            self._last_feedback_was_failure = True
                        else:
                            # If last feedback was failure and now we have success, record correction success
                            if getattr(self, "_last_feedback_was_failure", False):
                                if self.feedback_generator:
                                    # Mark the pending correction as successful
                                    self.feedback_generator.record_correction_attempt(success=True)
                                self._last_feedback_was_failure = False
                except Exception as e:
                    # Fail gracefully - don't break the planning loop
                    print(f"[LLMPlanner] Inner Monologue feedback generation failed: {e}")
            # =========================================================
        else:
            planner_info["replanned"] = {agent.uid: False for agent in self.agents}
            thought = None

            low_level_actions, responses = self.process_high_level_actions(
                self.last_high_level_actions, observations
            )
            responses = handle_rebound_side_effects(self, responses)

            process_copal_step(self, responses, is_replanning_step=is_replanning_step)

        # Log if replanning was done or not before overwriting the value
        planner_info["replan_required"] = {
            agent.uid: self.replan_required for agent in self.agents
        }

        # Determine whether replanning is needed based on execution responses.
        replan_triggers = []
        for agent_id, response in responses.items():
            if not response:
                continue
            last_action = self.last_high_level_actions.get(agent_id, ("Wait",))
            if last_action[0] == "Wait" and "Successful execution" in response:
                continue
            if "still in progress" in response:
                continue
            replan_triggers.append(response)

        # All-Wait detection via action tuples (not response content).
        # Uses _is_all_wait which guards against the all([]) == True gotcha.
        _actions_all_wait = self._is_all_wait(self.last_high_level_actions)

        if replan_triggers:
            self.replan_required = True
            if _actions_all_wait:
                self._consecutive_all_wait_count = 0
        elif _actions_all_wait:
            self._consecutive_all_wait_count += 1
            self.replan_required = True
            max_all_wait = self.planner_config.get("max_consecutive_all_wait", 3)
            print(
                f"[WARN] All-Wait detected in post-processing "
                f"(consecutive={self._consecutive_all_wait_count}/{max_all_wait})"
            )
            if self._consecutive_all_wait_count >= max_all_wait:
                print("[WARN] Consecutive all-Wait limit reached, forcing done.")
                self.is_done = True
            else:
                stall_msg = (
                    "Wait rejected — all agents idle, no progress made. Assign a productive action immediately."
                )
                print_str += self._inject_stall_observations(stall_msg)
        else:
            self._consecutive_all_wait_count = 0
            self.replan_required = False

        if self.replan_required and replan_triggers:
            print_str += self._add_responses_to_prompt(responses, is_replanning_step=is_replanning_step)
            generate_and_inject_copal_context(self, responses)
        # Update planner info
        planner_info["responses"] = responses
        planner_info["thought"] = {agent.uid: thought for agent in self.agents}
        planner_info["is_done"] = {agent.uid: self.is_done for agent in self.agents}
        planner_info["print"] = print_str
        # planner_info["print_no_tags"] = print_str_no_tags
        
        # Ensure all agents have entries in high_level_actions (fix for missing
        # agent_id issue). Missing directives are normalized to a no-op Wait
        # with an explicit diagnostic message.
        complete_high_level_actions, autofill_msgs = (
            self._complete_missing_actions_as_wait(self.last_high_level_actions)
        )
        for agent_uid, msg in autofill_msgs.items():
            # Ensure action-history logging has a non-empty response for the
            # auto-filled placeholder action.
            if not responses.get(agent_uid):
                responses[agent_uid] = msg
        planner_info["high_level_actions"] = complete_high_level_actions
        if autofill_msgs:
            planner_info["autofill_actions"] = autofill_msgs
        
        planner_info["prompts"] = {agent.uid: self.curr_prompt for agent in self.agents}
        planner_info["traces"] = {agent.uid: self.trace for agent in self.agents}
        planner_info["replanning_count"] = {
            agent.uid: self.replanning_count for agent in self.agents
        }
        planner_info["agent_states"] = self.get_last_agent_states()
        planner_info["agent_positions"] = self.get_last_agent_positions()
        planner_info["agent_collisions"] = self.get_agent_collisions()

        # Add SayCan data if analyzer is available
        if hasattr(self, 'saycan_analyzer') and self.saycan_analyzer:
            saycan_stats = self.saycan_analyzer.get_statistics()
            planner_info["saycan_data"] = saycan_stats
            
        # # Retrieve and log Rebound Metrics (MTTR, MTBF) (only if enabled)
        # if self.rebound_enabled:
        #     for agent in self.agents:
        #         mgr = self.rebound_managers.get(agent.uid)
        #         if mgr:
        #             metrics = mgr.get_metrics(self.replanning_count)
        #             if metrics:  # Check if metrics is not None
        #                 for k, v in metrics.items():
        #                     # Flatten into planner_info for single agent, or prefix for multi-agent
        #                     # For compatibility with demo script scalar extraction:
        #                     key = f"{k}_{agent.uid}" if len(self.agents) > 1 else k
        #                     planner_info[key] = v

        return low_level_actions, planner_info, self.is_done

    def check_if_agent_done(self, llm_response: str) -> bool:
        """
        Check if the agent is done based on the LLM response.

        :param llm_response: The LLM response to check.
        :return: True if the agent is done, False otherwise.
        """
        return self.end_expression in llm_response

    def _format_inner_monologue_feedback(self, feedback: Dict[str, Any]) -> str:
        """
        Format Inner Monologue feedback dictionary as natural language text.
        
        This method formats the structured feedback from FeedbackGenerator
        into a text string ready for injection into the prompt.
        
        :param feedback: Structured feedback dictionary from FeedbackGenerator.generate_feedback()
        :return: Formatted feedback text
        """
        if not self.feedback_generator:
            return ""
        
        return self.feedback_generator.format_feedback_as_text(feedback)

    def _build_minimal_world_state(self) -> Dict[str, Any]:
        """
        Build a minimal world state dictionary from env_interface.
        
        This is a fallback method when PerceptionConnector.extract_world_state()
        returns empty. It extracts basic information from env_interface.
        
        :return: Minimal world state dictionary
        """
        world_state = {
            "agent_poses": {},
            "object_positions": {},
            "furniture_positions": {},
            "agent_holdings": {},
        }
        
        try:
            # Extract agent poses from simulator
            sim = self.env_interface.sim
            for agent in self.agents:
                agent_id = agent.uid
                try:
                    agent_pos = sim.agents_mgr[agent_id].articulated_agent.base_pos
                    world_state["agent_poses"][agent_id] = {
                        "position": [float(agent_pos[0]), float(agent_pos[1]), float(agent_pos[2])],
                        "rotation": [0.0, 0.0, 0.0, 1.0],  # Placeholder
                        "yaw": 0.0
                    }
                except Exception:
                    pass
            
            # Extract object positions from world graph
            for agent_id, wg in self.env_interface.world_graph.items():
                try:
                    objects = wg.get_all_objects() if hasattr(wg, 'get_all_objects') else []
                    for obj in objects:
                        if hasattr(obj, 'name') and hasattr(obj, 'properties'):
                            obj_name = obj.name
                            parent = obj.properties.get('parent_receptacle', 'unknown')
                            world_state["object_positions"][obj_name] = {
                                "position": [0.0, 0.0, 0.0],  # Placeholder
                                "parent": parent
                            }
                    
                    # Extract held objects
                    for agent in self.agents:
                        agent_id = agent.uid
                        # Try to get held object from world graph
                        # This is a simplified version
                        world_state["agent_holdings"][agent_id] = []
                    
                    # Only need to extract from one agent in centralized setting
                    break
                except Exception:
                    pass
        except Exception:
            # Return minimal structure even if extraction fails
            pass
        
        return world_state
