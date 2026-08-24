"""
Phase planning helper for replan pipelines.
"""
from typing import Any, Dict, List, Optional


class PhasePlanningHelper:
    """
    Handles task decomposition, phase creation, and current phase selection.
    """

    def __init__(
        self,
        perception_connector: Any,
        agents: List[Any],
        llm_decompose_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize PhasePlanningHelper.
        """
        self.perception_connector = perception_connector
        self.agents = agents
        
        # Default LLM decomposition configuration
        self.llm_decompose_config = llm_decompose_config or {
            "gpt_version": "gpt-4o-2024-11-20",
            "max_tokens": 2000,
        }

    def ensure_current_phase(
        self,
        instruction: str,
        env_interface: Any,
        world_state: Dict[str, Any],
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Ensure phases are available and return the enriched current phase.
        """
        # Use provided config or fall back to instance config
        config_to_use = llm_config if llm_config is not None else self.llm_decompose_config
        phase_manager = self.perception_connector.phase_manager
        
        if not phase_manager.task_execution_phases:
            if not self._initialize_phases(instruction, env_interface, config_to_use):
                return None
        else:
            current_phase = self.perception_connector.get_current_phase_tasks()
            if current_phase:
                idx = phase_manager.current_phase_index + 1
                total = len(phase_manager.task_execution_phases)
                print(f"[DEBUG] Continuing execution: Phase {idx}/{total}")
            else:
                print("[DEBUG] All phases completed!")
                return None

        if phase_manager.task_execution_phases and (
            phase_manager.current_phase_index >= len(phase_manager.task_execution_phases)
        ):
            print("[INFO] All phases completed. Mission successful. Signaling exit.")
            return None

        current_phase = self.perception_connector.get_enriched_current_phase(world_state)
        if not current_phase:
            print("[INFO] No more phases to execute - task completed!")
            return None

        self._log_current_phase(current_phase)
        return current_phase

    def _initialize_phases(
        self,
        instruction: str,
        env_interface: Any,
        llm_config: Dict[str, Any],
    ) -> bool:
        try:
            structured_subtasks, execution_phases = (
                self.perception_connector.structured_decompose_task_with_sequencing(
                    instruction,
                    env_interface,
                    llm_config,
                    max_agents=len(self.agents),
                )
            )
            print(
                f"[DEBUG] Task decomposed into {len(structured_subtasks)} subtasks "
                f"across {len(execution_phases)} phases"
            )
            return True
        except Exception as e:
            print(f"[ERROR] Sequenced task decomposition failed: {e}")
            # In MVP, we might fail here if decomposition fails. 
            # Original had fallback logic. We can add simple fallback if needed.
            return False

    def _log_current_phase(self, current_phase: Dict[str, Any]) -> None:
        phase_id = current_phase.get("phase_id", 0)
        tasks = current_phase.get("tasks", [])
        print(f"[DEBUG] Current phase {phase_id} has {len(tasks)} tasks:")
        for task in tasks:
            task_info = f"  - {task.get('task_type')} -> {task.get('target')}"
            if "target_pos" in task:
                pos = task["target_pos"]
                task_info += f" (Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}])"
            print(task_info)

