"""
Shared prompt utilities for planners.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import json

from habitat_llm.llm.instruct.utils import get_world_descr


class BasePrompt:
    """
    Base class for prompt templates.
    """
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.prompt_template = ""


class TaskDecompositionPrompt(BasePrompt):
    """
    Prompt for standard task decomposition.
    """
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.prompt_template = """Your goal is to decompose a high-level instruction into a sequence of sub-tasks for two robots.
The robots can perform the following actions: {tool_list}.

Decompose the user's instruction into a JSON list of dictionaries, where each dictionary represents a sub-task.

Instruction: "{instruction}"

Environment State:
{world_state_description}

Provide the output in the following JSON format:
[
  {{
    "task_type": "Navigate",
    "target": "kitchen",
    "description": "Navigate to the kitchen to find items.",
    "priority": 3,
    "estimated_duration": 15.0,
    "preferred_agent": "Agent_0"
  }},
  {{
    "task_type": "Pick",
    "target": "apple",
    "description": "Pick up the apple from the table.",
    "priority": 5,
    "prerequisites": ["task_0"]
  }}
]
"""

    def __call__(self, instruction: str, world_state_description: str) -> str:
        # tool_list will be injected by the planner's context
        return self.prompt_template.format(
            instruction=instruction,
            world_state_description=world_state_description,
            tool_list="" # Placeholder, to be filled by planner
        )


class MIQPAnalysisPrompt(BasePrompt):
    """
    Prompt for analyzing task constraints and generating MIQP matrices.
    """
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.prompt_template = """Analyze the provided sub-tasks and world state to generate guidance for an MIQP solver.
Sub-tasks: {subtasks}
World State: {world_state}

Return a JSON object with:
- "task_complexity": Ratings (0.0-1.0) for each task type.
- "agent_suitability": Ratings for each agent on key capabilities (movement, manipulation).
- "constraints": Any special conditions.
- "critical_tasks": A list of the most important task types.

Example format:
{{
  "task_complexity": {{"navigate": 0.6, "pick": 0.8}},
  "agent_suitability": {{"agent_0": {{"movement": 0.9}}, "agent_1": {{...}}}},
  "constraints": {{"conservative_factors": {{"safety": 0.9}}}},
  "critical_tasks": ["Pick", "Navigate"]
}}
"""

    def __call__(self, subtasks: List[Dict[str, Any]], world_state: Dict[str, Any]) -> str:
        return self.prompt_template.format(
            subtasks=json.dumps(subtasks, indent=2),
            world_state=json.dumps(world_state, indent=2)
        )


class SequencingDecompositionPrompt(BasePrompt):
    """
    Prompt for task decomposition with dependency sequencing.
    """
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.prompt_template = """Decompose the instruction into a JSON array of subtasks with dependencies, focusing on carrying-state logic.

INSTRUCTION: {instruction}

WORLD STATE:
{objects_info}

AGENT STATUS:
{agent_info}

AGENT CAPABILITIES:
- Agent 0: Navigate, Explore, Pick, Place, Open, Close, Rearrange, Wait
- Agent 1: All Agent 0 skills + Clean, Fill, Pour, PowerOn, PowerOff

DEPENDENCY RULES:
- A "Navigate" task for placing items is a "carrying navigation" and MUST depend on ALL relevant "Pick" tasks.
- Example: For "Put A and B on C", the "Navigate to C" task must have prerequisites pointing to "Pick A" and "Pick B".
- A "Place" task must depend on its corresponding "Pick" and "carrying navigation" tasks.

JSON OUTPUT FORMAT (JSON only, no comments):
[
  {{
    "task_id": "task_0",
    "task_type": "Navigate|Pick|Place|etc.",
    "target": "object_or_location",
    "description": "description of the subtask",
    "priority": 1-5,
    "prerequisites": ["task_id_1"],
    "can_parallel": false
  }}
]
"""

    def __call__(self, instruction: str, objects_info: str, agent_info: str) -> str:
        return self.prompt_template.format(
            instruction=instruction,
            objects_info=objects_info,
            agent_info=agent_info
        )


SEQUENCING_SYSTEM_PROMPT = """You are an expert robotic task planner. Your goal is to decompose a high-level instruction into a sequence of executable subtasks for one or more robots.

**INPUT:**
1.  **Instruction:** The user's command (e.g., "put the apple on the table").
2.  **World Description:** A list of all known rooms, furniture, and objects with their current locations (e.g., "apple is on counter_1").
3.  **Agent Status:** The current state of each robot, including their position.

**CRITICAL RULES:**
1.  **Navigate Before Interaction:** This is the most important rule. Before any `Pick` or `Open` action, there MUST be a preceding `Navigate` task to the object's parent furniture. Do not assume the robot is already there. The only exception is if another task in the same phase has already navigated to that exact location.
2.  **Handling Unknown Objects:** If the instruction mentions an object NOT in the "World Description", your first step MUST be an `Explore` task. The `Explore` target should be the most likely `ROOM` or `FURNITURE`. Do not explore for the object name itself.
3.  **Prioritize Exploration:** If you are uncertain about any aspect of the environment, start with an `Explore` task.
4.  **Output Format:** Your response MUST be a valid JSON list of task objects.

**EXAMPLE of Good vs. Bad Plans:**
-   **Instruction:** "Pick up the apple."
-   **World State:** `agent_0` is in `living_room`, `apple` is on `kitchen_counter`.
-   **BAD Plan (X):** `[{"task_type": "Pick", "target": "apple", ...}]`
-   **GOOD Plan (✓):** `[{"task_type": "Navigate", "target": "kitchen_counter", ...}, {"task_type": "Pick", "target": "apple", ...}]`

Follow these rules strictly. A plan without a necessary `Navigate` before `Pick` is an invalid plan."""


def get_sequencing_system_prompt() -> str:
    return SEQUENCING_SYSTEM_PROMPT


class MatrixGenerationPrompt(BasePrompt):
    """
    Prompt for generating MIQP matrices directly with an LLM.
    """
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.prompt_template = """
You are an expert AI assistant for a robotic task planner. Your goal is to generate optimal parameters for a Mixed-Integer Quadratic Program (MIQP) solver that assigns tasks to two robots.

Based on the mission goal and decomposed subtasks, you must generate three matrices in a single JSON object:
1.  **Task-Capability Matrix (T)**: How much each task requires each capability.
2.  **Agent-Capability Matrix (A)**: How proficient each agent is at each capability.
3.  **Capability Weights (ws)**: How important each capability is for the overall mission.

**Mission Context:**
- **Subtasks**: {subtasks}
- **Agent Descriptions**: {agent_descriptions}
- **Capabilities**: {capability_names}
- **Task Types**: {task_type_names}

**Instructions for Generating JSON:**

1.  **`task_capability_matrix` (T)**:
    - A list of lists with shape (13 tasks x 5 capabilities).
    - Each value should be between 0.0 (not required) and 1.0 (highly required).
    - Rows correspond to Task Types, columns correspond to Capabilities in the order provided above.
    - A task should only have non-zero values for capabilities it actually uses. For example, 'Navigate' should only require 'Movement'. 'Pick' should only require 'Object Manipulation'.

2.  **`agent_capability_matrix` (A)**:
    - A list of lists with shape (5 capabilities x 2 agents).
    - Each value represents proficiency, from 0.0 (incapable) to 1.0 (highly proficient).
    - Rows correspond to Capabilities, columns correspond to Agents (Agent 0, Agent 1).
    - Values should reflect the agent descriptions. If an agent cannot perform a task, its score for that capability must be 0.0. A score of 1.0 is considered baseline proficiency.

3.  **`capability_weights` (ws)**:
    - A list of 5 numbers, one for each capability.
    - Each value represents the importance of that capability for this specific mission, from 1.0 (less important) to 5.0 (critically important).
    - Consider which capabilities are central to completing the given subtasks.

4.  **`reasoning`**:
    - A brief text string explaining your choices for the weights and any non-obvious matrix values.

**Output Format (JSON only):**
```json
{{
  "reasoning": "Brief explanation here.",
  "task_capability_matrix": [],
  "agent_capability_matrix": [],
  "capability_weights": []
}}
```
"""
    def __call__(
        self,
        structured_subtasks: List[Dict[str, Any]],
        agent_descriptions: Dict[str, str],
        capability_names: List[str],
        task_type_names: List[str]
    ) -> str:
        return self.prompt_template.format(
            subtasks=json.dumps(structured_subtasks, indent=2),
            agent_descriptions=json.dumps(agent_descriptions, indent=2),
            capability_names=json.dumps(capability_names, indent=2),
            task_type_names=json.dumps(task_type_names, indent=2)
        )


_PROMPT_TYPES = {
    "task_decomposition": TaskDecompositionPrompt,
    "miqp_analysis": MIQPAnalysisPrompt,
    "sequencing_decomposition": SequencingDecompositionPrompt,
    "matrix_generation": MatrixGenerationPrompt
}


def get_miqp_prompt(name: str, config: Dict[str, Any] = None) -> Union[TaskDecompositionPrompt, MIQPAnalysisPrompt, SequencingDecompositionPrompt, MatrixGenerationPrompt]:
    """
    Return a prompt instance by name.
    """
    if config is None:
        config = {}
    if name not in _PROMPT_TYPES:
        raise ValueError(f"Unknown prompt type: {name}")
    return _PROMPT_TYPES[name](name, config)
