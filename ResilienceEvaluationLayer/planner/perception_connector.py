"""
PerceptionConnector module - bridges perception/world state and planning utilities.
"""
from typing import Dict, List, Any, Optional, Tuple, Union, TYPE_CHECKING
from pathlib import Path

import numpy as np
import openai

from habitat_llm.planner.connector.phase_manager import PhaseManager
from habitat_llm.planner.connector.prompt_context import PromptContextBuilder
from habitat_llm.planner.connector.task_decomposer import TaskDecomposer
from habitat_llm.world_model import build_world_state_from_graphs
# from habitat_llm.perception.world_state import extract_world_state_from_graph
# from habitat_llm.perception.perception_utils import find_target_position

if TYPE_CHECKING:
    from habitat_llm.agent.env import EnvironmentInterface
    # from habitat_llm.perception.perception_sim import PerceptionSim # Assuming this exists or not used in MVP

def extract_world_state_from_graph(
    world_graph,
    env_interface: Optional["EnvironmentInterface"] = None,
    instruction: str = "",
    task_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Extracts a simplified world state dictionary from the WorldGraph.
    """
    return build_world_state_from_graphs(
        world_graph=world_graph,
        env_interface=env_interface,
        instruction=instruction,
        task_context=task_context,
    )

def find_target_position(target_name: str, world_state: Dict[str, Any]) -> Optional[List[float]]:
    """
    Finds the position of a target object/furniture from the world state.
    """
    # Check objects
    if 'object_positions' in world_state:
        for name, info in world_state['object_positions'].items():
            if name == target_name and 'position' in info:
                return info['position']
    
    # Check furniture (if available in world_state)
    # The PromptContextBuilder builds agent_status_prompt from object_positions and agent_poses.
    # It seems world_state usually contains 'object_positions', 'agent_poses', and maybe 'furniture_positions'.
    
    return None


class PerceptionConnector:
    """
    Facade that connects perception/world-state with planning utilities.
    Refactored to remove MIQP dependencies.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        api_key_filename: Optional[str] = None,
        llm_base_url: Optional[str] = "https://api.moonshot.cn/v1",
        perception_sim: Optional[Any] = None,
    ):
        """
        Initialize PerceptionConnector.
        """
        self.last_world_state: Dict[str, Any] = {}
        self.llm_client = llm_client
        self.perception_sim = perception_sim
        self._world_state_provider: Optional[Any] = None
        api_key_filename = api_key_filename or "api_key"

        # Task sequencing state
        self.task_dependency_graph: Dict[str, List[str]] = {}
        self.completed_tasks: List[str] = []
        self.active_tasks: List[str] = []

        self._init_llm_client(api_key_filename, llm_base_url)

        # Sub-modules (MIQP modules removed)
        self.phase_manager = PhaseManager(self.llm_client)
        self.prompt_context = PromptContextBuilder()
        self.task_decomposer = TaskDecomposer(self.llm_client, self.prompt_context)

    def reset(self) -> None:
        """
        Reset the state of the PerceptionConnector for a new episode.
        """
        self.last_world_state = {}
        self.task_dependency_graph = {}
        self.completed_tasks = []
        self.active_tasks = []

        # Re-initialize phase manager to clear state.
        if getattr(self, "phase_manager", None) and self.phase_manager.llm_client:
            self.phase_manager = PhaseManager(self.phase_manager.llm_client)
        else:
            self.phase_manager = PhaseManager(self.llm_client)

    def set_world_state_provider(self, provider: Any) -> None:
        """
        Bind a world-state provider that implements get_world_state_dict().
        """
        self._world_state_provider = provider

    def _resolve_world_state_provider(
        self,
        env_interface: Optional["EnvironmentInterface"],
        perception_sim: Optional[Any],
    ) -> Optional[Any]:
        candidates: List[Any] = [
            perception_sim,
            self.perception_sim,
            self._world_state_provider,
        ]
        if env_interface is not None:
            if getattr(env_interface, "perception", None) is not None:
                candidates.append(env_interface.perception)
            if hasattr(env_interface, "get_world_state_dict"):
                candidates.append(env_interface)
        for candidate in candidates:
            if candidate is not None and hasattr(candidate, "get_world_state_dict"):
                return candidate
        return None

    def extract_world_state(
        self,
        env_interface: Optional["EnvironmentInterface"] = None,
        perception_sim: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Extract world state from a perception provider or fall back to WorldGraph.
        """
        provider = self._resolve_world_state_provider(env_interface, perception_sim)
        if provider is not None:
            world_state = provider.get_world_state_dict()
            self.last_world_state = world_state
            return world_state

        if env_interface is None:
            # raise ValueError("Either env_interface or perception_sim must be provided.")
            return {}

        # Fallback: direct WorldGraph access if needed, or just return empty for now
        # Ideally we implement extract_world_state_from_graph properly
        world_state = extract_world_state_from_graph(
            getattr(env_interface, "world_graph", None) or env_interface.full_world_graph,
            env_interface=env_interface,
        )
        self.last_world_state = world_state
        return world_state

    def structured_decompose_task_with_sequencing(
        self,
        instruction: str,
        env_interface: "EnvironmentInterface",
        llm_config: Dict[str, Any],
        max_agents: int = 2,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Decompose instruction into structured subtasks and organize phases.
        """
        current_world_state = self.last_world_state or self.extract_world_state(
            env_interface
        )

        enhanced_subtasks, execution_phases, dependency_graph = (
            self.task_decomposer.decompose_with_sequencing(
                instruction,
                env_interface,
                current_world_state,
                self.phase_manager,
                llm_config,
                max_agents,
            )
        )

        self.task_dependency_graph = dependency_graph
        print(
            f"DEBUG: Task decomposed into {len(enhanced_subtasks)} subtasks across "
            f"{len(execution_phases)} phases"
        )
        for i, phase in enumerate(execution_phases):
            task_summaries = [f"{t['task_type']}->{t['target']}" for t in phase["tasks"]]
            print(
                f"  Phase {i + 1}: {task_summaries} "
                f"(max_parallel: {phase['max_parallel_tasks']})"
            )

        return enhanced_subtasks, execution_phases

    def get_current_phase_tasks(self) -> Optional[Dict[str, Any]]:
        """
        Return current phase tasks from PhaseManager.
        """
        return self.phase_manager.get_current_phase_tasks()

    def get_enriched_current_phase(
        self, world_state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Enrich current phase tasks with target positions.
        """
        current_phase = self.phase_manager.get_current_phase_tasks()
        if not current_phase:
            return None

        current_phase_tasks = current_phase.get("tasks", [])
        enriched_tasks = []

        for task in current_phase_tasks:
            new_task = task.copy()
            target_name = new_task.get("target")
            if target_name:
                target_pos_data = find_target_position(target_name, world_state)

                if target_pos_data is not None:
                    new_task["target_pos"] = target_pos_data
                else:
                    # print(
                    #     f"  [Enrichment] WARNING: Could not find position for target "
                    #     f"'{target_name}'"
                    # )
                    pass
            enriched_tasks.append(new_task)

        enriched_phase = current_phase.copy()
        enriched_phase["tasks"] = enriched_tasks
        return enriched_phase

    def advance_to_next_phase(self) -> bool:
        """
        Advance to the next execution phase.
        """
        return self.phase_manager.advance_to_next_phase()

    def is_current_phase_complete(self, agent_statuses: Dict[int, str]) -> bool:
        """
        Check whether current phase is complete.
        """
        return self.phase_manager.is_current_phase_complete(agent_statuses)

    def structured_decompose_task(
        self,
        instruction: str,
        env_interface: "EnvironmentInterface",
        llm_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to decompose instruction into structured subtasks.
        """
        current_world_state = self.last_world_state or self.extract_world_state(
            env_interface
        )
        try:
            structured_tasks = self.task_decomposer.decompose(
                instruction, env_interface, current_world_state, llm_config
            )
            print(f"DEBUG: Successfully parsed {len(structured_tasks)} subtasks")
            return structured_tasks
        except Exception as e:
            print(f"Error calling LLM for structured task decomposition: {e}")
            raise

    def _init_llm_client(
        self,
        api_key_filename: Optional[str],
        llm_base_url: Optional[str],
    ) -> None:
        """
        Initialize the LLM client if it was not provided explicitly.
        """
        if self.llm_client is not None:
            return

        if not api_key_filename:
            print(
                "PerceptionConnector: LLM client not provided and api_key_filename "
                "not specified. Task decomposition will not be available."
            )
            return

        try:
            api_key_path = Path(api_key_filename + ".txt")
            if not api_key_path.exists():
                api_key_path = Path(api_key_filename)

            if api_key_path.exists():
                api_key = api_key_path.read_text().strip()
                self.llm_client = openai.OpenAI(
                    api_key=api_key,
                    base_url=llm_base_url,
                )
                if getattr(self, "task_decomposer", None) is not None:
                    self.task_decomposer.llm_client = self.llm_client
                print(
                    "PerceptionConnector: LLM client initialized using API key from "
                    f"{api_key_path} and base URL {llm_base_url}"
                )
            else:
                print(f"API key file not found at {api_key_path}")
        except Exception as e:
            print(f"Error initializing LLM client in PerceptionConnector: {e}")

