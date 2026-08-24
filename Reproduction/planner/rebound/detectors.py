from typing import List, Dict, Any, Optional
from .telemetry import TelemetryLogger

class FaultDetector:
    """
    Analyzes telemetry data to detect cognitive or execution faults.

    IMPORTANT - Step Counting Convention:
    All thresholds and windows in this class are measured in PLANNING STEPS
    (replanning_count from LLMPlanner), NOT simulation steps:
    - stagnation_window: Number of planning steps to check for progress (default: 25)
    - burst_threshold: Number of consecutive planning steps with failures (default: 5)
    - context_length_threshold: Character limit before triggering overflow (default: 60000)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # Thresholds (all measured in planning steps, not simulation steps)
        self.stagnation_window = self.config.get("stagnation_window", 25)
        self.stagnation_delta = self.config.get("stagnation_delta", 0.05)
        self.burst_threshold = self.config.get("burst_threshold", 5)
        self.context_length_threshold = self.config.get("context_length_threshold", 60000)

    def _is_enabled(self, detector_name: str) -> bool:
        enabled = self.config.get(detector_name, True)
        return enabled if isinstance(enabled, bool) else True

    def detect_faults(self, telemetry: TelemetryLogger) -> List[str]:
        """
        Run all detectors and return a list of fault tags.
        """
        faults = []
        
        if self._is_enabled("progress_stagnation") and self._check_progress_stagnation(telemetry):
            faults.append("progress_stagnation")
            
        if self._is_enabled("tool_failure_burst") and self._check_tool_failure_burst(telemetry):
            faults.append("tool_failure_burst")
            
        if self._is_enabled("graph_inconsistency") and self._check_graph_inconsistency(telemetry):
            faults.append("graph_inconsistency")
            
        if self._is_enabled("context_overflow") and self._check_context_length(telemetry):
            faults.append("context_overflow")
            
        return faults
    
    def _check_context_length(self, telemetry: TelemetryLogger) -> bool:
        """
        Check if the context (prompt/trace) is getting too long.
        """
        # Get the latest snapshot to check trace/prompt length
        snapshot = telemetry.get_last_snapshot()
        if not snapshot:
            return False
            
        # Simplified length check based on string length
        # In production, this should count tokens
        prompt_len = len(snapshot.get("prompt", ""))
        trace_len = len(snapshot.get("trace", ""))
        total_len = prompt_len + trace_len
        
        # Threshold (e.g., 20000 chars approx 5k tokens as a rough heuristic)
        # TODO: Move to config
        return total_len > self.context_length_threshold

    def _check_progress_stagnation(self, telemetry: TelemetryLogger) -> bool:
        """
        Check if progress has stalled for N steps.
        """
        history = telemetry.get_progress_trend(self.stagnation_window)
        if len(history) < self.stagnation_window:
            return False
            
        # Check if the difference between oldest and newest in window is negligible
        delta = history[-1] - history[0]
        return delta < self.stagnation_delta

    def _check_tool_failure_burst(self, telemetry: TelemetryLogger) -> bool:
        """
        Check for consecutive failures of the same tool.
        """
        actions = telemetry.get_last_n_actions(self.burst_threshold)
        if len(actions) < self.burst_threshold:
            return False
            
        # Check if all recent actions failed
        all_failed = all(a.get("error") is not None or "failure" in str(a.get("result", "")).lower() for a in actions)
        
        # Check if they are the same tool
        if not actions:
            return False
        first_tool = actions[0]["tool"]
        same_tool = all(a["tool"] == first_tool for a in actions)
        
        return all_failed and same_tool

    def _check_graph_inconsistency(self, telemetry: TelemetryLogger) -> bool:
        """
        MVP: Check for explicit inconsistency flags in recent logs.
        Refined to be less sensitive to transient 'not found' errors.
        """
        # Check last 3 actions to see if there's a pattern
        actions = telemetry.get_last_n_actions(3)
        if not actions:
            return False
            
        def is_inconsistent(action):
            err = str(action.get("error", "")).lower()
            return "inconsistent" in err or "not in graph" in err

        def is_not_found(action):
            err = str(action.get("error", "")).lower()
            return "not found" in err

        # Immediate trigger for explicit inconsistency
        if is_inconsistent(actions[-1]):
            return True
            
        # For "not found", require persistence (2 consecutive)
        if len(actions) >= 2 and is_not_found(actions[-1]) and is_not_found(actions[-2]):
            return True
            
        return False
