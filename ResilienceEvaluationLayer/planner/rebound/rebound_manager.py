from typing import List, Dict, Any, Optional, Tuple
from .telemetry import TelemetryLogger
from .detectors import FaultDetector

class ReboundManager:
    """
    Orchestrates the agent's self-recovery (Rebound) process.
    Acts as a 'sidecar' to the planner.

    IMPORTANT - Step Counting Convention:
    All step counts in this class refer to PLANNING STEPS (replanning_count from LLMPlanner),
    NOT simulation steps. This ensures consistent timing across:
    - startup_grace_period: Number of planning steps before stagnation detection activates (default: 15)
    - Cooldown periods (all in planning steps):
        - retry: 5 planning steps
        - replan: 20 planning steps
        - rollback: 15 planning steps
        - context_summarize: 10 planning steps
        - retry_with_scan: 8 planning steps
    - MTTR (Mean Time To Recovery): Measured in planning steps
    - MTBF (Mean Time Between Failures): Measured in planning steps
    - failure_history: Records (start_step, end_step, strategy) tuples in planning steps
    """

    def __init__(self, agent_id: int, config: Optional[Dict[str, Any]] = None):
        self.agent_id = agent_id
        self.config = config or {}
        
        self.telemetry = TelemetryLogger(agent_id)
        self.detector = FaultDetector(self.config.get("detectors", {}))
        
        # Strategy weights/priorities
        self.max_retries = self.config.get("max_retries", 3)
        self.startup_grace_period = self.config.get("startup_grace_period", 15)
        self.current_progress = 0.0

        # Metrics Tracking (MTTR, MTBF)
        # (start_step, end_step, strategy), List
        self.failure_history: List[Tuple[int, int, str]] = [] 
        # Start step of the currently active failure, None if healthy
        self.active_failure_start: Optional[int] = None
        
        # RR (Recovery Ratio) Tracking - Eq.12 in Detailed_Metrics.tex
        self.errors_occurred: int = 0
        self.errors_fixed: int = 0
        self._progress_before_recovery: float = 0.0
        
        # B_epi tracking with retry/backtrack distinction
        self.retry_count: int = 0
        self.backtrack_count: int = 0

        # Cooldown Tracking
        self._recovery_cooldown: Dict[str, int] = {}
        # Fault cooldown tracking (planning steps)
        self._last_fault_step: Dict[str, int] = {}
        self.fault_cooldown_steps = self.config.get("fault_cooldown_steps", 2)
        self.fault_cooldowns = self.config.get("fault_cooldowns", {})
        # Pending action for intent preservation
        self.pending_high_level_action: Optional[Tuple[str, str, str]] = None
        
        # Termination Guard Tracking
        self.termination_block_count: int = 0
        self.last_progress_at_termination: float = -1.0
        
        self.monitoring_only = self.config.get("monitoring_only", False)
        
        # Immediate Error Tolerance Tracking (Issue 2)
        # Only trigger context modification after N consecutive same-type failures
        self.immediate_error_window = self.config.get("immediate_error_window", 3)
        self.immediate_error_buffer: List[Tuple[int, str, str]] = []  # (step, action_type, error_msg)

    def update_progress(self, progress: float, source: str = "runtime"):
        """
        Update current progress perception from external evaluator.
        
        IMPORTANT: This method MUST be called during both runtime and offline analysis
        to ensure Rebound metrics (especially termination_guard) work correctly.
        
        Args:
            progress: Task completion percentage (0.0 to 1.0)
            source: Source of the progress update ("runtime" or "offline_analysis")
        """
        self.current_progress = progress

    def log_execution(self, tool_name: str, args: Any, result: str, duration: float, error: Optional[str] = None):
        """Pass-through to telemetry."""
        self.telemetry.log_action(tool_name, args, result, duration, error)

    def log_planner_state(
        self, 
        step: int, 
        action: str, 
        world_graph: Any, 
        progress: float,
        prompt: str = "",
        trace: str = "",
        world_state_dict: Optional[Dict[str, Any]] = None
    ):
        """Pass-through to telemetry."""
        self.telemetry.log_planner_state(
            step, 
            action, 
            world_graph, 
            progress,
            prompt=prompt,
            trace=trace,
            world_state_dict=world_state_dict
        )

    def _check_cooldown(self, strategy: str, current_step: int) -> bool:
        """Check if a strategy is in cooldown."""
        last_trigger = self._recovery_cooldown.get(strategy, -999)
        cooldown_period = self.config.get("cooldown_periods", {}).get(strategy, 0)
        
        # Defaults if not in config
        if cooldown_period == 0:
            defaults = {
                "retry": 5,
                "replan": 20,
                "rollback": 15,
                "context_summarize": 10,
                "retry_with_scan": 8
            }
            cooldown_period = defaults.get(strategy, 0)
            
        if current_step - last_trigger < cooldown_period:
            return False # In cooldown
        return True # Ready

    def filter_faults(self, faults: List[str], current_step: int) -> List[str]:
        """
        Filter detected faults based on planning-step rules (grace period + fault cooldowns).
        """
        if not faults:
            return []

        filtered = []
        for fault in faults:
            if fault == "progress_stagnation" and current_step < self.startup_grace_period:
                continue

            cooldown = self.fault_cooldowns.get(fault, self.fault_cooldown_steps)
            last_step = self._last_fault_step.get(fault)
            if cooldown and last_step is not None and current_step - last_step < cooldown:
                continue

            filtered.append(fault)
            self._last_fault_step[fault] = current_step

        return filtered

    def update_failure_metrics(self, faults: List[str], current_step: int):
        """
        Update failure metrics based on detected faults.
        Call this when faults are detected externally or via detection-only mode.
        """
        if faults:
            # Fault Detected
            if self.active_failure_start is None:
                # New failure begins
                self.active_failure_start = current_step
        else:
            # No Fault Detected
            if self.active_failure_start is not None:
                # Failure Resolved
                start_step = self.active_failure_start
                end_step = current_step
                
                # Filter out transient "resolved_context" events (duration < 2 steps)
                # unless a critical fault strategy was used (not tracked here yet).
                # This addresses "resolved_context" appearing too frequently.
                if end_step - start_step >= 2:
                    self.failure_history.append((start_step, end_step, "resolved_context"))
                
                self.active_failure_start = None

    def log_termination_intervention(self, current_step: int):
        """
        Log an intervention by the Termination Guard.
        This counts as a single-step Rebound event.
        """
        self.termination_block_count += 1
        # Log as a 1-step failure event (start=current, end=current+1) 
        # but mark as 'termination_guard' so it's distinct
        self.failure_history.append((current_step, current_step + 1, "termination_guard"))

    def record_immediate_error(self, step: int, action_type: str, error_msg: str) -> bool:
        """
        Record an immediate error and check if intervention is warranted.
        
        Trial-and-error tolerance: Only returns True (trigger intervention) after
        N consecutive failures of the same action type.
        
        Args:
            step: Current planning step
            action_type: Type of action that failed (e.g., 'Pick', 'Place', 'Navigate')
            error_msg: Error message from the failure
            
        Returns:
            True if intervention should be triggered, False otherwise (allow retry)
        """
        self.immediate_error_buffer.append((step, action_type, error_msg))
        
        # Keep only the most recent N entries
        if len(self.immediate_error_buffer) > self.immediate_error_window:
            self.immediate_error_buffer = self.immediate_error_buffer[-self.immediate_error_window:]
        
        # Not enough consecutive failures yet
        if len(self.immediate_error_buffer) < self.immediate_error_window:
            return False
        
        # Check if all failures in window are the same action type
        action_types = [entry[1] for entry in self.immediate_error_buffer]
        if len(set(action_types)) == 1:
            # Same action failed N times consecutively - trigger intervention
            # Save start_step before clearing buffer
            start_step = self.immediate_error_buffer[0][0]
            # Log as a failure event for metrics
            self.failure_history.append((start_step, step, "immediate_error_burst"))
            # Clear buffer to avoid repeated triggers
            self.immediate_error_buffer = []
            return True
        
        return False

    def clear_immediate_error_buffer(self):
        """Clear immediate error buffer (e.g., after successful action)."""
        self.immediate_error_buffer = []

    def suggest_recovery(self, current_error: str, current_step: int = 0) -> Optional[List[Dict[str, Any]]]:
        """
        Analyze the current situation and suggest a list of RECOVERY ACTIONS (Tool Calls).
        Returns None if no rebound is needed/possible (e.g., let the planner fail naturally).
        Also tracks failure intervals for MTTR/MTBF calculation.
        """
        # 1. Detect Faults
        faults = self.detector.detect_faults(self.telemetry)
        
        # Add the immediate error to consideration if not caught by detectors
        if not faults and current_error:
            faults.append("immediate_error")
        faults = self.filter_faults(faults, current_step)

        # --- Metrics Tracking Logic ---
        if faults:
            # Fault Detected
            if self.active_failure_start is None:
                # New failure begins
                self.active_failure_start = current_step
        else:
            # No Fault Detected
            if self.active_failure_start is not None:
                # Failure Resolved
                start_step = self.active_failure_start
                end_step = current_step
                # We don't know the exact strategy that fixed it here without more context, 
                # but we can log 'resolved' or infer from last action.
                # For now, just logging the interval is sufficient for MTTR.
                self.failure_history.append((start_step, end_step, "resolved"))
                self.active_failure_start = None
        # -----------------------------

        if self.monitoring_only:
             # Stop here if we are only monitoring, do NOT suggest interventions
             return None

        if not faults:
            return None

        print(f"[ReboundManager] Agent {self.agent_id} detected faults: {faults}")

        # 2. Select Strategy
        strategy = self._select_strategy(faults, current_step)
        
        # 3. Generate Tool Calls (Reflex Executor)
        if strategy:
            # Update cooldown
            self._recovery_cooldown[strategy] = current_step
            print(f"[ReboundManager] Selected strategy: {strategy}")
            return self._generate_tool_calls(strategy, current_error)
            
        return None

    def get_metrics(self, current_step: int) -> Dict[str, float]:
        """
        Calculate MTTR-A (Mean Time To Recovery - Agent) and MTBF (Mean Time Between Failures).
        Returns a dictionary of metrics.
        """
        resolved_failures = self.failure_history
        num_failures = len(resolved_failures)
        
        # Calculate MTTR (Average duration of resolved failures)
        if num_failures > 0:
            total_downtime = sum(end - start for start, end, _ in resolved_failures)
            mttr = total_downtime / num_failures
        else:
            mttr = 0.0

        # Calculate MTBF (Average duration between failure starts)
        # MTBF is typically Operating Time / Number of Failures
        # Operating Time = Total Time - Total Downtime
        # Here we approximate using steps.
        
        # If currently failing, include current unfinished downtime in "total downtime" for MTBF calculation purposes?
        # Standard MTBF usually considers completed cycles. Let's stick to completed intervals.
        
        if num_failures > 0:
            # Time from first failure start to last failure end
            # Or total uptime / failures
            # Let's use: (Current Step - Total Downtime) / Failure Count
            # This represents average uptime duration per failure event.
            
            total_downtime = sum(end - start for start, end, _ in resolved_failures)
            if self.active_failure_start is not None:
                total_downtime += (current_step - self.active_failure_start)
            
            total_time = current_step # Assuming start at 0
            operating_time = max(0, total_time - total_downtime)
            mtbf = operating_time / (num_failures + (1 if self.active_failure_start else 0))
        else:
            # If no failures, MTBF is essentially the current duration
            mtbf = float(current_step)

        # Build stats dictionary for detailed analysis
        fault_counts = {}
        for _, _, strategy_name in resolved_failures:
             # Strategy loosely maps to fault types handled. 
             # Ideally we would log fault types in history too.
             # For now, counting by resolution strategy is a proxy.
             fault_counts[strategy_name] = fault_counts.get(strategy_name, 0) + 1

        # RR = |Err_fix| / (|Err_occur| + ε)
        recovery_ratio = self.get_recovery_ratio()
        
        # B_epi with γ weight (default γ=1.5)
        gamma = self.config.get("gamma_backtrack", 1.5)
        b_epi = self.retry_count + gamma * self.backtrack_count
        
        return {
            "rebound_mttr": mttr,
            "rebound_mtbf": mtbf,
            "rebound_count": float(num_failures),
            "fault_distribution": fault_counts,
            "recovery_ratio": recovery_ratio,
            "b_epi": b_epi,
            "retry_count": self.retry_count,
            "backtrack_count": self.backtrack_count,
        }

    def _select_strategy(self, faults: List[str], current_step: int) -> Optional[str]:
        """
        Decide which reflex to trigger based on fault pattern.
        """
        strategy = None
        if "context_overflow" in faults:
            # High priority: Compress memory before LLM context window breaks
            strategy = "context_summarize"

        elif "graph_inconsistency" in faults:
            # High priority: Fix world model belief
            strategy = "retry_with_scan"
            
        elif "tool_failure_burst" in faults:
            # Burst failure -> Try something different (Replan or Rollback)
            # For MVP, let's try Rollback if we have history, else Replan
            strategy = "rollback"
            
        elif "progress_stagnation" in faults:
            # Stuck -> Replan
            strategy = "replan"
            
        elif "immediate_error" in faults:
            # Single error -> Simple Retry
            # Check if we have retried recently
            recent = self.telemetry.get_last_n_actions(5) # Look back 5 steps
            retry_count = 0
            for action in recent:
                if "retry" in str(action.get("tool", "")).lower() or "wait" in str(action.get("tool", "")).lower(): # Wait is often used as retry in MVP
                     retry_count += 1
            
            if retry_count >= 3:
                 # Already retried too many times, escalate to replan/reflect
                 strategy = "replan"
            else:
                strategy = "retry"

        if strategy and not self._check_cooldown(strategy, current_step):
            # If preferred strategy is in cooldown, maybe fallback?
            # For now, just suppress.
            print(f"[ReboundManager] Strategy {strategy} suppressed by cooldown.")
            return None

        return strategy

    def _generate_tool_calls(self, strategy: str, context: str) -> List[Dict[str, Any]]:
        """
        Translate strategy into executable Tool Calls.
        """
        recovery_tasks = []

        if strategy == "context_summarize":
            # 1. Trigger context compression
            recovery_tasks.append({
                "task_id": f"rebound_{self.agent_id}_context_prune",
                "task_type": "ContextSummarizeTool",
                "description": "Pruning context to prevent overflow",
                "keep_last_n": 5 # Keep last 5 interactions
            })

        elif strategy == "retry_with_scan":
            # 1. Scan/Perceive
            recovery_tasks.append({
                "task_id": f"rebound_{self.agent_id}_scan",
                "task_type": "PerceptionScanTool",
                "description": "Scanning to fix graph inconsistency",
                "target": "local_area" # MVP simplification
            })
            # 2. The retry happens naturally as the planner loop continues? 
            # Or we can explicitly queue the failed action again?
            # For MVP, we'll just insert the fix actions, assuming planner will re-attempt the goal.

        elif strategy == "retry":
            # Simple retry, maybe with a small wait or nav adjustment
            # MVP: Just Wait (to let physics settle) then let Planner retry
            recovery_tasks.append({
                "task_id": f"rebound_{self.agent_id}_wait",
                "task_type": "Wait",
                "description": "Waiting before retry",
                "target": "10" # 10 steps
            })

        elif strategy == "rollback":
            # Rollback K steps
            steps_to_rollback = 3 # Configurable
            recovery_tasks.append({
                "task_id": f"rebound_{self.agent_id}_rollback",
                "task_type": "CognitiveRollbackTool",
                "description": f"Rolling back {steps_to_rollback} steps",
                "steps": steps_to_rollback
            })
            # Optional: Undo action (not implemented in MVP yet)

        elif strategy == "replan":
            # Force replan via TaskReflectTool to encourage analysis
            recovery_tasks.append({
                "task_id": f"rebound_{self.agent_id}_replan_trigger",
                "task_type": "TaskReflectTool", 
                "description": "Triggering Reflection due to stagnation",
            })
            
        return recovery_tasks

    # ========== RR (Recovery Ratio) Methods ==========
    
    def record_error_occurred(self, fault_type: str):
        """
        Called when FaultDetector detects a fault.
        """
        self.errors_occurred += 1
        self._progress_before_recovery = self.current_progress
    
    def record_error_fixed(self):
        """
        Called when progress increases after recovery attempt.
        """
        if self.current_progress > self._progress_before_recovery:
            self.errors_fixed += 1
    
    def get_recovery_ratio(self, epsilon: float = 1e-6) -> float:
        """
        Compute the Recovery Ratio (RR), Eq. 12 in Detailed_Metrics.tex.
        
        RR = |Err_fix| / (|Err_occur| + ε)
        
        Returns:
            RR value in [0, 1], higher is better
        """
        return self.errors_fixed / (self.errors_occurred + epsilon)
    
    def record_retry(self):
        """Record a retry operation for B_epi."""
        self.retry_count += 1
    
    def record_backtrack(self):
        """Record a backtrack operation for B_epi with a higher weight."""
        self.backtrack_count += 1
