from typing import Dict, Any, Optional, Tuple
from habitat_llm.tools.tool import Tool

class TaskReflectTool(Tool):
    """
    Tool to trigger LLM self-reflection.
    Used when the agent is stuck or looping to induce meta-cognitive analysis.
    """
    def __init__(self, agent_uid: int = None, max_thought_chars: int = 1200):
        super().__init__("TaskReflectTool", agent_uid)
        self._description = "Triggers a self-reflection phase to analyze failure causes."
        self.max_thought_chars = max_thought_chars

    @staticmethod
    def _sanitize_max_thought_chars(
        value: Any, default_value: int = 1200, min_value: int = 200, max_value: int = 8000
    ) -> int:
        try:
            resolved = int(value)
        except Exception:
            resolved = int(default_value)
        return max(min_value, min(max_value, resolved))

    @classmethod
    def _resolve_max_thought_chars(
        cls, planner: Any, explicit_value: Optional[int] = None
    ) -> int:
        if explicit_value is not None:
            return cls._sanitize_max_thought_chars(explicit_value)

        fallback = getattr(planner, "rebound_config", {}) if planner is not None else {}
        if isinstance(fallback, dict):
            reflection_cfg = fallback.get("reflection", {})
            if isinstance(reflection_cfg, dict) and "max_thought_chars" in reflection_cfg:
                return cls._sanitize_max_thought_chars(
                    reflection_cfg.get("max_thought_chars")
                )
            if "max_thought_chars" in fallback:
                return cls._sanitize_max_thought_chars(fallback.get("max_thought_chars"))

        return cls._sanitize_max_thought_chars(1200)

    @property
    def description(self) -> str:
        return self._description

    @property
    def argument_types(self):
        return []

    def process_high_level_action(
        self, 
        last_action: Any, 
        observations: Dict[str, Any]
    ) -> Tuple[Any, str]:
        """
        Execute reflection trigger.
        """
        # No physical action, just a mental state change signal
        max_thought_chars = self._sanitize_max_thought_chars(self.max_thought_chars)
        if isinstance(last_action, dict):
            max_thought_chars = self._sanitize_max_thought_chars(
                last_action.get("max_thought_chars", max_thought_chars),
                default_value=max_thought_chars,
            )
        return None, f"ReflectionTriggered: max_thought_chars={max_thought_chars}"

    @staticmethod
    def apply_reflection(planner, agent_id: int, max_thought_chars: Optional[int] = None) -> str:
        """
        Injects a reflection prompt into the agent's stream.
        """
        try:
            print(f"[LLMPlanner] Triggering Self-Reflection for Agent {agent_id}")
            resolved_chars = TaskReflectTool._resolve_max_thought_chars(
                planner, explicit_value=max_thought_chars
            )
            
            reflection_prompt = (
                "\n\n[System]: You appear to be stuck or making repeated errors. "
                "Stop and analyze the situation. "
                "1. Why is the current plan failing? "
                "2. Are there inconsistencies in your world knowledge? "
                "3. What alternative approaches exist? "
                f"\nProvide a concise but complete 'Thought' ({resolved_chars} chars max) before your next action."
            )
            
            # Inject into prompt and trace
            # Assuming planner handles per-agent prompts correctly or we are modifying the shared prompt context
            # If centralized, we append to the shared prompt.
            planner.curr_prompt += reflection_prompt
            planner.trace += reflection_prompt
            
            return f" [Reflection Injected; max_thought_chars={resolved_chars}]"
        except Exception as e:
            print(f"[LLMPlanner] Error during reflection injection: {e}")
            return f" [Reflection Failed: {e}]"
