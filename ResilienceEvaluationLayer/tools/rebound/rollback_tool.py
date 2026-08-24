from typing import Dict, Any, Tuple
from habitat_llm.tools.tool import Tool
from habitat_llm.planner.rebound.rebound_manager import ReboundManager

class CognitiveRollbackTool(Tool):
    """
    Tool to perform Cognitive Rollback.
    Restores the agent's world graph belief to a previous state.
    """
    def __init__(self, rebound_manager: ReboundManager = None):
        super().__init__("CognitiveRollbackTool")
        self.rebound_manager = rebound_manager
        self._description = "Restores the agent's belief state to a previous timestamp."

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
        Execute the rollback.
        """
        # Parse args
        steps = 1
        if isinstance(last_action, dict):
            steps = int(last_action.get("steps", 1))
        
        # Access telemetry via ReboundManager
        telemetry = self.rebound_manager.telemetry
        
        # Find the target snapshot (current - steps)
        # Assuming last_action is triggered at step T, we want T-steps
        # We need the current step count. Telemetry has history.
        last_snapshot = telemetry.get_last_snapshot()
        if not last_snapshot:
             return None, "Rollback failed: No history available."
             
        target_step = last_snapshot["step"] - steps
        target_snapshot = telemetry.get_snapshot_at_step(target_step)
        
        if not target_snapshot:
             # Fallback to oldest if we overshoot
             target_snapshot = telemetry.world_graph_snapshots[0] if telemetry.world_graph_snapshots else None

        if target_snapshot:
            # RESTORE!
            # In a real system, we'd need a way to push this back into the Agent/Planner.
            # For MVP, we'll return a special string that the ExecutionManager intercepts,
            # or update the rebound_manager's pending state restoration queue.
            
            # Since tools usually return (low_level_action, response),
            # and this is a cognitive action, we return a Null action with a specific response.
            
            # We can't easily modify the *Planner's* active world graph directly from here 
            # without a reference to the Planner or WorldGraph instance.
            # Assuming 'rebound_manager' is shared or has access.
            
            # Ideally, the Planner should poll the ReboundManager for "state overrides" at the start of a step.
            # OR, we pass the planner/agent into this Tool during init.
            
            # MVP Hack: The response string contains the serialized state (or reference ID)
            # which the Planner parses.
            return None, f"RollbackSuccess: step={target_snapshot['step']}"
            
    @staticmethod
    def apply_rollback(planner, agent_id: int, target_step: int, preserve_action_intent: bool = True) -> str:
        """
        Static helper to apply the side effects of a rollback on the planner instance.
        This encapsulates the restoration logic.
        """
        try:
            mgr = planner.rebound_managers.get(agent_id)
            if not mgr:
                return f"[Restoration Failed: No ReboundManager for {agent_id}]"

            snapshot = mgr.telemetry.get_snapshot_at_step(target_step)
            if not snapshot:
                return f"[Restoration Failed: Snapshot {target_step} not found]"

            print(f"[LLMPlanner] Performing Cognitive Rollback for Agent {agent_id} to step {target_step}")
            
            # 1. Restore Belief State (World Graph)
            if snapshot.get('world_graph'):
                planner.env_interface.world_graph[agent_id].restore_from_snapshot(snapshot['world_graph'])
            
            # 2. Restore Mental State (Prompt/Trace)
            planner.curr_prompt = snapshot.get('prompt', "")
            planner.trace = snapshot.get('trace', "")
            
            # 3. Restore Perception State (if tracked by connector)
            if snapshot.get('world_state_dict') and hasattr(planner, 'perception_connector'):
                planner.perception_connector.last_world_state = snapshot['world_state_dict']
                
            # Adjust internal counters
            # planner.replanning_count = target_step  # Commented out to preserve global progress count
            
            # Restore intent if available
            if preserve_action_intent and mgr.pending_high_level_action:
                print(f"[LLMPlanner] Restoring pending intent: {mgr.pending_high_level_action}")
                if not planner.last_high_level_actions:
                    planner.last_high_level_actions = {}
                planner.last_high_level_actions[agent_id] = mgr.pending_high_level_action
            
            return " [Restoration Complete]"
            
        except Exception as e:
            print(f"[LLMPlanner] Error during restoration: {e}")
            return f" [Restoration Failed: {e}]"

