from typing import Dict, Any, Tuple
from habitat_llm.tools.tool import Tool

class ContextSummarizeTool(Tool):
    """
    Tool to summarize/compress the current interaction context.
    Used by ReboundManager when context overflow is detected.
    """
    def __init__(
        self,
        agent_uid: int = None,
        keep_last_n: int = 5,
        max_prompt_chars: int = 8000,
        max_trace_chars: int = 6000
    ):
        super().__init__("ContextSummarizeTool", agent_uid)
        self._description = "Compresses the execution history to free up cognitive space."
        self.keep_last_n = keep_last_n
        self.max_prompt_chars = max_prompt_chars
        self.max_trace_chars = max_trace_chars

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
        Execute context summarization.
        This tool doesn't perform a physical action but returns a signal
        for the Planner to prune its prompt/trace.
        """
        # In a real system, this might call an LLM to summarize the trace.
        # For now, we return a signal that the Planner interprets.
        
        # We can pass instructions on HOW to prune via the response
        keep_last_n = self.keep_last_n
        max_prompt_chars = self.max_prompt_chars
        max_trace_chars = self.max_trace_chars
        if isinstance(last_action, dict):
            keep_last_n = int(last_action.get("keep_last_n", 5))
            max_prompt_chars = int(last_action.get("max_prompt_chars", max_prompt_chars))
            max_trace_chars = int(last_action.get("max_trace_chars", max_trace_chars))
            
        return (
            None,
            "ContextPruned: "
            f"keep_last={keep_last_n} "
            f"max_prompt={max_prompt_chars} "
            f"max_trace={max_trace_chars}"
        )

    @staticmethod
    def apply_context_pruning(
        planner,
        agent_id: int,
        keep_last_n: int = 5,
        max_prompt_chars: int = 8000,
        max_trace_chars: int = 6000
    ) -> str:
        """
        Static helper to apply the side effects of context pruning on the planner instance.
        """
        try:
            print("[LLMPlanner] Scheduling Context Compression")
            if hasattr(planner, "request_context_compression"):
                planner.request_context_compression("tool_requested")
                return " [Context Compression Requested]"

            # Legacy fallback when the planner does not support LLM compression.
            keep_last_n = max(1, keep_last_n)
            max_prompt_chars = max(1000, max_prompt_chars)
            max_trace_chars = max(1000, max_trace_chars)

            if len(planner.curr_prompt) > max_prompt_chars:
                marker = "\n...[Context Pruned]...\n"
                head_len = min(1200, max_prompt_chars // 4)
                tail_len = max_prompt_chars - head_len - len(marker)
                if tail_len > 0:
                    planner.curr_prompt = (
                        planner.curr_prompt[:head_len] + marker + planner.curr_prompt[-tail_len:]
                    )
                else:
                    planner.curr_prompt = planner.curr_prompt[:max_prompt_chars]

            trace_lines = planner.trace.split("\n")
            if len(trace_lines) > keep_last_n * 3:
                task_line = trace_lines[0] if trace_lines else ""
                recent_lines = trace_lines[-(keep_last_n * 3):]
                planner.trace = task_line + "\n...[Context Pruned]...\n" + "\n".join(recent_lines)

            if len(planner.trace) > max_trace_chars:
                planner.trace = planner.trace[-max_trace_chars:]
                planner.trace = "[Trace Pruned]...\n" + planner.trace

            return " [Context Pruned]"
        except Exception as e:
            print(f"[LLMPlanner] Error during context pruning: {e}")
            return f" [Context Pruning Failed: {e}]"
