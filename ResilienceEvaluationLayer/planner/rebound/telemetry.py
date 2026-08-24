import copy
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

class TelemetryLogger:
    """
    Logs agent actions, planner states, and world graph snapshots.
    Used for fault detection and cognitive rollback.
    """

    def __init__(self, agent_id: int, max_history: int = 50):
        self.agent_id = agent_id
        self.max_history = max_history
        
        # Ring buffers for history
        self.action_history = deque(maxlen=max_history)
        self.planner_history = deque(maxlen=max_history)
        self.world_graph_snapshots = deque(maxlen=max_history)
        
        # Progress tracking
        self.percent_complete_history = deque(maxlen=max_history)

    def log_action(self, tool_name: str, args: Any, result: str, duration: float, error: Optional[str] = None):
        """
        Log a tool/skill execution.
        """
        entry = {
            "timestamp": time.time(),
            "tool": tool_name,
            "args": args,
            "result": result,
            "duration": duration,
            "error": error
        }
        self.action_history.append(entry)

    def log_planner_state(
        self, 
        step_count: int, 
        high_level_action: str, 
        world_graph: Any, 
        percent_complete: float = 0.0,
        prompt: str = "",
        trace: str = "",
        world_state_dict: Optional[Dict[str, Any]] = None
    ):
        """
        Log planner state and take a snapshot of the world graph belief.
        Also captures Mental State (prompt, trace) and Perception State (world_state_dict).
        This is critical for Cognitive Rollback.
        """
        # Snapshot world graph using the specialized method if available, else fallback
        try:
            if hasattr(world_graph, "create_rebound_snapshot"):
                wg_snapshot = world_graph.create_rebound_snapshot()
            else:
                wg_snapshot = copy.deepcopy(world_graph)
        except Exception as e:
            print(f"[Telemetry] Warning: Failed to snapshot world graph for agent {self.agent_id}: {e}")
            wg_snapshot = None

        snapshot_entry = {
            "timestamp": time.time(),
            "step": step_count,
            "high_level_action": high_level_action,
            "world_graph": wg_snapshot,
            "percent_complete": percent_complete,
            "prompt": prompt,
            "trace": trace,
            "world_state_dict": world_state_dict or {}
        }
        
        self.world_graph_snapshots.append(snapshot_entry)
        self.percent_complete_history.append(percent_complete)

    def get_last_n_actions(self, n: int) -> List[Dict[str, Any]]:
        """Return the last n actions."""
        return list(self.action_history)[-n:]

    def get_snapshot_at_step(self, step: int) -> Optional[Dict[str, Any]]:
        """Retrieve a snapshot from a specific step."""
        # Search backwards as it's likely recent
        for snapshot in reversed(self.world_graph_snapshots):
            if snapshot["step"] == step:
                return snapshot
        return None

    def get_last_snapshot(self) -> Optional[Dict[str, Any]]:
        """Get the most recent snapshot."""
        if self.world_graph_snapshots:
            return self.world_graph_snapshots[-1]
        return None

    def get_progress_trend(self, window: int = 5) -> List[float]:
        """Get the progress (percent complete) over the last window steps."""
        return list(self.percent_complete_history)[-window:]

