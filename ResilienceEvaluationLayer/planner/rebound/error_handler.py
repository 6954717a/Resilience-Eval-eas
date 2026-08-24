from typing import Dict, List, Any, Optional, Tuple
import time

class ErrorHandler:
    """
    Handles error detection and recovery during the planning process.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.error_history = []
        self.recovery_attempts = {}
        self.persistent_failures = set()
        
        # Initialize Rebound Managers for each potential agent
        # We assume a max number of agents or dynamic initialization
        # For MVP, we'll lazy-load them or just support agent 0 and 1
        self.config = config or {}
        self.rebound_managers: Dict[int, Any] = {}
        
        # Import here to avoid circular dependencies if any
        from habitat_llm.planner.rebound.rebound_manager import ReboundManager
        self.ReboundManagerClass = ReboundManager

    def get_rebound_manager(self, agent_id: int) -> Any:
        if agent_id not in self.rebound_managers:
            self.rebound_managers[agent_id] = self.ReboundManagerClass(agent_id, self.config)
        return self.rebound_managers[agent_id]

    def reset(self):
        """
        Reset the ErrorHandler state.
        Clear error history and recovery tracking.
        """
        self.error_history = []
        self.recovery_attempts = {}
        self.persistent_failures = set()
        # print("[DEBUG] ErrorHandler reset completed")

    def recover_and_log_assignments(
        self,
        agent_task_assignments: Dict[int, List[Dict[str, Any]]],
        last_high_level_actions: Dict[int, Tuple[str, str, str]],
        latest_agent_response: Dict[int, str],
        current_step: int = 0
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Applies intelligent error recovery and logs the outcome.
        This method checks for action failures and may add recovery tasks.
        It prints the changes if any recovery actions were taken.
        current_step is expected to be the planning step (replanning_count).
        """
        try:
            recovered_assignments = self.apply_intelligent_error_recovery(
                agent_task_assignments,
                last_high_level_actions,
                latest_agent_response,
                current_step=current_step
            )

            # Log the results if changes were made
            if recovered_assignments != agent_task_assignments:
                print(f"[DEBUG] **RECOVERY** Updated task assignments after error recovery:")
                for agent_id, tasks in recovered_assignments.items():
                    if tasks:
                        task_summaries = []
                        for t in tasks:
                            recovery_marker = " [RECOVERY]" if t.get('is_recovery', False) else ""
                            task_summaries.append(f"{t['task_type']}->{t['target']}{recovery_marker}")
                        print(f"  Agent {agent_id}: {task_summaries}")
                    else:
                        print(f"  Agent {agent_id}: []")
            else:
                print(f"[DEBUG] **RECOVERY** No error recovery needed")
            
            return recovered_assignments

        except Exception as e:
            print(f"[ERROR] Error recovery failed: {e}")
            return agent_task_assignments

    def apply_intelligent_error_recovery(
        self,
        agent_task_assignments: Dict[int, List[Dict[str, Any]]],
        last_high_level_actions: Dict[int, Tuple[str, str, str]],
        latest_agent_response: Dict[int, str],
        current_step: int = 0
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Apply intelligent error recovery using ReboundManager (planning-step based)."""
        try:
            if not latest_agent_response or not last_high_level_actions:
                return agent_task_assignments

            updated_assignments = agent_task_assignments.copy()

            for agent_id, response in latest_agent_response.items():
                if not response:
                    continue
                
                # Check for "success" response or empty errors to skip
                # (Simple check, actual logic might depend on response format)
                if "success" in response.lower() and "fail" not in response.lower():
                    continue

                current_action = last_high_level_actions.get(agent_id)
                # Note: current_action might be None if no action was taken, but we still have a response (e.g. from background monitoring)
                
                # Use ReboundManager
                mgr = self.get_rebound_manager(agent_id)
                
                # Log execution to telemetry (assuming this is called AFTER execution)
                # In a real loop, logging might happen elsewhere. MVP: Log here to drive the detector.
                # duration is unknown here, defaulting to 1.0
                action_name = current_action[0] if current_action else "Unknown"
                action_args = current_action[1] if current_action else ""
                mgr.log_execution(action_name, action_args, response, 1.0, response if "fail" in response.lower() else None)
                
                # Ask for recovery suggestions
                recovery_tasks = mgr.suggest_recovery(response, current_step=current_step)

                if recovery_tasks:
                    if agent_id not in updated_assignments:
                        updated_assignments[agent_id] = []
                    
                    # Prepend recovery tasks
                    # We reverse them to insert them in correct order at index 0
                    for task in reversed(recovery_tasks):
                        # Ensure required fields for the planner
                        task['priority'] = 10 # High priority
                        task['phase_group'] = 'recovery'
                        task['is_recovery'] = True
                        task['preferred_agent'] = agent_id
                        
                        updated_assignments[agent_id].insert(0, task)

            return updated_assignments

        except Exception as e:
            print(f"[ERROR] Intelligent error recovery failed: {e}")
            return agent_task_assignments

    def analyze_failure_and_suggest_recovery(
        self,
        agent_id: int,
        status: str,
        current_action: Tuple[str, str, Optional[str]]
    ) -> Optional[Tuple[str, str, Optional[str]]]:
        """Analyze agent failure status and suggest recovery actions."""
        try:
            if not status or not current_action:
                return None

            action_name, args, target = current_action
            status_lower = status.lower()

            if action_name == "Pick" and "not close enough" in status_lower:
                return ("Navigate", target, target)
            elif action_name == "Pick" and ("object not found" in status_lower or "not present in the graph" in status_lower):
                return ("Explore", "environment", "environment")
            elif action_name == "Place" and "not close enough" in status_lower:
                return ("Navigate", target, target)
            elif "navigation failed" in status_lower or "path not found" in status_lower:
                return ("Explore", "environment", "environment")
            elif "collision" in status_lower or "blocked" in status_lower:
                return ("Wait", "", "")

            return None

        except Exception:
            return None

