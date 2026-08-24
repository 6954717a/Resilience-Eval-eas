from typing import Any, Dict, Optional, Tuple

import numpy as np
from habitat_baselines.utils.common import get_num_actions

from habitat_llm.tools.tool import Tool


class PerceptionScanTool(Tool):
    """
    Requests a perception scan of the environment.

    The tool itself never mutates the world graph. It emits a legal no-op scan
    action and lets the next standard observation/update pass refresh the world
    model through EnvironmentInterface.
    """
    def __init__(self):
        super().__init__("PerceptionScanTool")
        self._description = (
            "Requests a perception scan of the surroundings and lets the "
            "standard observation pipeline refresh the world graph."
        )
        # Set by :meth:`set_environment` when the tool is registered under
        # ``tools.rebound`` (see :class:`habitat_llm.agent.agent.Agent`).
        self.env_interface: Optional[Any] = None

    def set_environment(self, env_interface: Any) -> None:
        """Receive the env so we can emit a valid noop low-level action vector."""
        self.env_interface = env_interface

    def _noop_low_level_action(self) -> np.ndarray:
        """Return a zero joint-action vector in CPU numpy format.

        Motor skills ultimately feed env.step with numpy arrays. Returning the
        same type here avoids mixed numpy/cuda tensor arithmetic in the
        EnvironmentInterface action merger.
        """
        if self.env_interface is not None:
            space = getattr(self.env_interface, "action_space", None)
            if space is not None:
                n = int(get_num_actions(space))
                return np.zeros(n, dtype=np.float32)
        return np.zeros(1, dtype=np.float32)

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
        Execute scan.
        """
        # In a real robot, this might rotate the head/base.
        # In Habitat-LLM, we might trigger a global graph update from the simulator truth
        # or just "wait" and let the standard perception hook run.
        
        # For MVP: We return a "Wait" low-level action, but with a response 
        # that indicates "Scanning Complete". 
        # The PerceptionConnector in the planner usually updates the graph every step anyway.
        # So this tool primarily acts as a 'pause and look' deliberate action.
        
        target = "area"
        if isinstance(last_action, dict):
            target = last_action.get("target", "area")

        # Physical side: no-op control (scan semantics are in planner-side
        # ``apply_scan`` / graph refresh). Do **not** return a dict here.
        return self._noop_low_level_action(), f"PerceptionScanned: target={target}"

    @staticmethod
    def apply_scan(planner, agent_id: int) -> str:
        """
        Apply planner-side scan bookkeeping without touching the world graph.

        The actual graph refresh must happen via the normal observation ->
        parse_observations -> update_world_graphs path on the next env step.
        """
        try:
            print(f"[LLMPlanner] Registered PerceptionScan request for Agent {agent_id}")
            if not hasattr(planner, "pending_perception_scans"):
                planner.pending_perception_scans = set()
            planner.pending_perception_scans.add(agent_id)

            # Invalidate planner-side caches so the next observation refresh is
            # forced through the standard world-graph update path.
            planner.curr_obj_states = ""
            if hasattr(planner, "current_scene_graph_snapshot"):
                planner.current_scene_graph_snapshot = {}
            if hasattr(planner, "perception_connector") and hasattr(
                planner.perception_connector, "clear_cache"
            ):
                planner.perception_connector.clear_cache()

            return " [Scan Requested; world graph refresh deferred to observation update]"
        except Exception as e:
            print(f"[LLMPlanner] Error during scan application: {e}")
            return f" [Scan Failed: {e}]"

