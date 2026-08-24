from typing import Dict, List, Any, Optional, Tuple, Union
import json
from habitat_llm.llm.instruct.utils import get_world_descr

class PromptContextBuilder:
    """
    Builds consistent prompt context strings from the environment and world state.
    """

    def build_world_description(self, env_interface: Any) -> str:
        """
        Generate a detailed world description string for LLM prompts.
        """
        full_graph = env_interface.full_world_graph
        return get_world_descr(
            full_graph,
            agent_uid=0,
            include_room_name=True,
            add_state_info=True,
            centralized=True,
        )

    def build_objects_prompt(self, env_interface: Any) -> str:
        """
        Format objects/furniture list for task decomposition prompts.
        """
        full_graph = env_interface.full_world_graph
        objects_for_prompt = []

        for obj_node in full_graph.get_all_objects():
            try:
                category = obj_node.category
            except AttributeError:
                category = "Unknown"
            objects_for_prompt.append(
                {"name": obj_node.name, "category": category or "Unknown"}
            )

        for furn_node in full_graph.get_all_furnitures():
            try:
                category = furn_node.category
            except AttributeError:
                category = "Furniture"
            objects_for_prompt.append(
                {"name": furn_node.name, "category": category or "Furniture"}
            )

        return f"objects = {json.dumps(objects_for_prompt)}"

    def build_agent_status_prompt(self, world_state: Dict[str, Any]) -> str:
        """
        Format agent status text from the world-state dictionary.
        """
        if not world_state:
            return "No agent status available"

        agent_status_lines = []
        agent_poses = world_state.get("agent_poses", {})
        held_objects_by_agent = {agent: [] for agent in agent_poses}

        for obj_name, obj_info in world_state.get("object_positions", {}).items():
            if obj_info and obj_info.get("parent") in agent_poses:
                held_objects_by_agent[obj_info["parent"]].append(obj_name)

        for agent_name, pose_info in agent_poses.items():
            pos_str = "Position unknown"
            if pose_info and "position" in pose_info:
                pos = pose_info["position"]
                pos_str = f"Position [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"

            held_str = (
                ", holding: " + ", ".join(held_objects_by_agent[agent_name])
                if held_objects_by_agent.get(agent_name)
                else ", hands free"
            )
            agent_status_lines.append(f"- {agent_name}: {pos_str}{held_str}")

        return "\n".join(agent_status_lines) if agent_status_lines else "No agents found."

    def build_task_decomposition_context(
        self, env_interface: Any, world_state: Dict[str, Any]
    ) -> str:
        """
        Build a combined context block for task decomposition prompts.
        """
        objects_prompt_string = self.build_objects_prompt(env_interface)
        agent_info_string = self.build_agent_status_prompt(world_state)
        return (
            f"Environment State:\n{objects_prompt_string}\n\n"
            f"Agent Status:\n{agent_info_string}"
        )

    def build_sequencing_context(
        self, env_interface: Any, world_state: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Build world and agent context strings for sequencing prompts.
        """
        world_desc_string = self.build_world_description(env_interface)
        agent_info_string = self.build_agent_status_prompt(world_state)
        return world_desc_string, agent_info_string


class PromptBuilder:
    def __init__(self, planner_config, env_interface):
        self.planner_config = planner_config
        self.env_interface = env_interface
        self.world_graph = self.env_interface.full_world_graph
        # Add tags for convenience
        self.assistant_tag = self.planner_config.llm.assistant_tag
        self.eot_tag = self.planner_config.llm.eot_tag
        self.user_tag = self.planner_config.llm.user_tag
        self.base_prompt = ""
        self.cached_prompts = {}
        self.prompt_history = []

    def reset(self):
        """
        Reset the PromptBuilder state.
        """
        self.base_prompt = ""
        self.cached_prompts = {}
        self.prompt_history = []
        print("[DEBUG] PromptBuilder reset completed")

    def _strip_trailing_assistant(self, prompt: str, assistant_turn_start: str) -> str:
        if not prompt:
            return ""
        stripped = prompt.rstrip()
        if stripped.endswith(assistant_turn_start):
            return stripped[: -len(assistant_turn_start)].rstrip()
        return stripped

    def _format_turn(self, role_tag: str, content: str) -> str:
        content = content.rstrip()
        return f"{role_tag}{content}\n{self.eot_tag}"

    def set_base_prompt(self, prompt: str, assistant_turn_start: str) -> None:
        """
        Cache the base prompt without a trailing assistant turn.
        """
        self.base_prompt = self._strip_trailing_assistant(prompt, assistant_turn_start)

    def add_user_turn(self, content: str, title: Optional[str] = None) -> None:
        """
        Append a user turn to the prompt history.
        """
        text = content.strip()
        if title:
            text = f"{title}:\n{text}"
        self.prompt_history.append({"role": "user", "content": text})

    def add_assistant_turn(self, content: str) -> None:
        """
        Append an assistant turn to the prompt history.
        """
        self.prompt_history.append({"role": "assistant", "content": content.rstrip()})

    def truncate_history(self, keep_last_turns: int) -> None:
        """
        Keep only the last N turns in history.
        """
        if keep_last_turns is None or keep_last_turns <= 0:
            return
        if len(self.prompt_history) > keep_last_turns:
            self.prompt_history = self.prompt_history[-keep_last_turns:]

    def build_prompt(self, assistant_turn_start: str) -> str:
        """
        Build the full prompt from base prompt and history.
        """
        parts = []
        if self.base_prompt:
            parts.append(self.base_prompt.rstrip())
        for turn in self.prompt_history:
            role = turn.get("role")
            content = turn.get("content", "")
            if role == "assistant":
                parts.append(self._format_turn(self.assistant_tag, content))
            else:
                parts.append(self._format_turn(self.user_tag, content))
        prompt = "\n".join(p for p in parts if p)
        if assistant_turn_start:
            if not prompt.endswith(assistant_turn_start):
                prompt = f"{prompt}\n{assistant_turn_start}" if prompt else assistant_turn_start
        return prompt

    def build_prompt_with_context(
        self,
        current_prompt: str,
        assistant_turn_start: str,
        context_blocks: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """
        Inject context blocks as distinct user turns without mutating history.
        """
        prompt_for_llm = self._strip_trailing_assistant(
            current_prompt, assistant_turn_start
        )

        if context_blocks:
            for title, content in context_blocks:
                if not content:
                    continue
                text = content.strip()
                if title:
                    text = f"{title}:\n{text}"
                guidance_injection = (
                    f"\n{self.user_tag}{text}\n{self.eot_tag}"
                )
                prompt_for_llm += guidance_injection

        if assistant_turn_start and not prompt_for_llm.endswith(assistant_turn_start):
            prompt_for_llm += f"\n{assistant_turn_start}"

        return prompt_for_llm

    def prepare_llm_prompt(
        self, current_prompt: str, miqp_guidance: Optional[str]
    ) -> str:
        """
        Prepare the final prompt for the LLM by injecting guidance as a distinct
        system message. This maintains the integrity of the dialogue history.
        """
        assistant_turn_start = f"{self.assistant_tag}Thought:"
        context_blocks = []
        if miqp_guidance:
            context_blocks.append(("System (Guidance)", miqp_guidance))
        return self.build_prompt_with_context(
            current_prompt=current_prompt,
            assistant_turn_start=assistant_turn_start,
            context_blocks=context_blocks,
        )

    def _format_task_description(self, task: Dict[str, Any]) -> str:
        """Format a single task dictionary into a human-readable string."""
        task_type = task.get("task_type", "Unknown")
        target = task.get("target", "None")

        if target and target.lower() != "none":
            if task_type == "Pick":
                return f"pick up the {target}"
            if task_type == "Place":
                return f"place the item at {target}"
            if task_type == "Navigate":
                return f"go to the {target}"
            if task_type == "Explore":
                return f"explore the {target}"
            return f"execute {task_type} on {target}"

        if task_type == "Navigate":
            return "move to a strategic location"
        if task_type == "Explore":
            return "explore the area"
        return f"perform {task_type}"
