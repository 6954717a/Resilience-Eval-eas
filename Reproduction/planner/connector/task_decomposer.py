"""
Task decomposition helpers for LLM-driven planning.
"""
from typing import Any, Dict, List, Optional, Tuple
import json

from habitat_llm.planner.prompting import TaskDecompositionPrompt
from habitat_llm.planner.connector.dependency_enhancer import TaskDependencyEnhancer
from habitat_llm.planner.connector.planner_utils import (
    extract_json_from_text,
    get_llm_config,
    normalize_task_list,
)
from habitat_llm.planner.connector.prompt_context import PromptContextBuilder


class TaskDecomposer:
    """
    Encapsulates LLM-based task decomposition and phase structuring.
    """

    def __init__(self, llm_client: Optional[Any], prompt_context: PromptContextBuilder):
        self.llm_client = llm_client
        self.prompt_context = prompt_context

    def decompose(
        self,
        instruction: str,
        env_interface: Any,
        world_state: Dict[str, Any],
        llm_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Use LLM to decompose instruction into structured subtasks.
        """
        if not self.llm_client:
            raise ValueError("LLM client not initialized. Cannot decompose task.")

        world_state_desc = self.prompt_context.build_task_decomposition_context(
            env_interface, world_state
        )
        prompt_template = TaskDecompositionPrompt("task_decomposition", get_llm_config())
        structured_prompt = prompt_template(instruction, world_state_desc)

        messages = [
            {
                "role": "system",
                "content": "You are a precise task decomposition assistant. Return only valid JSON.",
            },
            {"role": "user", "content": structured_prompt},
        ]
        api_params = {
            "model": llm_config.get("gpt_version", "moonshot-v1-32k"),
            "messages": messages,
            "max_tokens": llm_config.get("max_tokens", 1500),
            "temperature": 0.0,
        }

        response = self.llm_client.chat.completions.create(**api_params)
        decomposed_text = response.choices[0].message.content.strip()
        structured_tasks = self._parse_decomposition_response(decomposed_text)

        if structured_tasks:
            return structured_tasks
        return self._simple_task_decomposition(instruction)

    def decompose_with_sequencing(
        self,
        instruction: str,
        env_interface: Any,
        world_state: Dict[str, Any],
        phase_manager: Any,
        llm_config: Dict[str, Any],
        max_agents: int,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[str]]]:
        """
        Decompose tasks and organize them into execution phases.
        """
        if not phase_manager or not phase_manager.llm_client:
            raise ValueError(
                "PhaseManager or its LLM client not initialized. Cannot decompose task."
            )

        world_desc_string, agent_info_string = self.prompt_context.build_sequencing_context(
            env_interface, world_state
        )

        structured_subtasks = phase_manager.decompose_and_initialize_phases(
            instruction,
            world_desc_string,
            agent_info_string,
            llm_config,
        )

        enhancer = TaskDependencyEnhancer()
        enhanced_subtasks, execution_phases, dependency_graph = (
            enhancer.structure_and_phase(structured_subtasks, max_agents)
        )
        phase_manager.set_execution_phases(execution_phases)

        return enhanced_subtasks, execution_phases, dependency_graph

    def _parse_decomposition_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response into validated structured tasks.
        """
        try:
            json_text = extract_json_from_text(response_text, list)
            if not json_text:
                return []

            structured_tasks = json.loads(json_text)
            return normalize_task_list(
                structured_tasks,
                required_fields=["task_type", "target", "description"],
                defaults={
                    "priority": 3,
                    "estimated_duration": 5.0,
                    "preferred_agent": None,
                    "prerequisites": [],
                },
            )
        except (json.JSONDecodeError, TypeError):
            return []

    def _simple_task_decomposition(self, instruction: str) -> List[Dict[str, Any]]:
        """
        Fallback rule-based decomposition when LLM parsing fails.
        """
        tasks: List[Dict[str, Any]] = []
        if "pick" in instruction.lower() and "place" in instruction.lower():
            tasks.extend(
                [
                    {
                        "task_type": "Explore",
                        "target": "environment",
                        "description": "Explore to find targets",
                        "priority": 2,
                    },
                    {
                        "task_type": "Navigate",
                        "target": "object_location",
                        "description": "Navigate to object",
                        "priority": 3,
                    },
                    {
                        "task_type": "Pick",
                        "target": "target_object",
                        "description": "Pick up object",
                        "priority": 4,
                    },
                    {
                        "task_type": "Navigate",
                        "target": "receptacle_location",
                        "description": "Navigate to receptacle",
                        "priority": 3,
                    },
                    {
                        "task_type": "Place",
                        "target": "target_receptacle",
                        "description": "Place object",
                        "priority": 4,
                    },
                ]
            )
        else:
            tasks.append(
                {
                    "task_type": "Explore",
                    "target": "environment",
                    "description": instruction,
                    "priority": 3,
                }
            )

        for task in tasks:
            task.setdefault("estimated_duration", 10.0)
            task.setdefault("preferred_agent", None)
            task.setdefault("prerequisites", [])
        return tasks

