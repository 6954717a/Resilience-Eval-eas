"""
Success Detector for Inner Monologue

% Different from the proposal completion checker, this is about the traj stat detector.
Detects whether actions succeeded or failed based on agent responses and state changes.
Generates natural language feedback for the LLM.
"""

from typing import Dict, Any, Tuple, Optional


class SuccessDetector:
    """
    Detects action execution success/failure and generates feedback messages.
    
    This component analyzes agent responses and optionally checks state changes
    to determine if an action succeeded or failed, then generates natural language
    feedback for Inner Monologue.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SuccessDetector.
        
        Args:
            config: Configuration dictionary with optional settings:
                - check_state_changes: Whether to verify state changes (default: False)
                - verbose_feedback: Whether to include detailed reasons (default: True)
        """
        self.config = config or {}
        self.check_state_changes = self.config.get("check_state_changes", False)
        self.verbose_feedback = self.config.get("verbose_feedback", True)

    def detect_success(
        self,
        agent_id: int,
        action: Tuple[str, str, str],
        response: str,
        world_state: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect if an action succeeded or failed.
        
        Args:
            agent_id: Agent identifier
            action: Tuple of (action_name, arg1, arg2)
            response: Agent response string from process_high_level_action
            world_state: Optional world state dict for state change verification
        
        Returns:
            Dictionary with keys:
                - success: bool - Whether action succeeded
                - message: str - Natural language feedback message
                - reason: str - Brief reason for success/failure
        """
        action_name, arg1, arg2 = action
        
        # Parse response to determine success/failure
        # Common success patterns: "Successful execution!", "Success", etc.
        # Common failure patterns: "fail", "error", "cannot", "unable", etc.
        response_lower = response.lower() if response else ""
        
        is_success = self._parse_response_success(response_lower)
        
        # Generate feedback message
        if is_success:
            message = self._generate_success_message(agent_id, action_name, arg1, arg2, response)
            reason = "Action completed successfully"
        else:
            message = self._generate_failure_message(agent_id, action_name, arg1, arg2, response)
            reason = self._extract_failure_reason(response)
        
        # Optional: Verify state changes for certain actions
        if self.check_state_changes and world_state:
            state_verified = self._verify_state_change(action_name, arg1, world_state)
            if not state_verified and is_success:
                # State didn't change but response said success - might be a false positive
                message += " (Note: State change verification inconclusive)"
        
        return {
            "success": is_success,
            "message": message,
            "reason": reason,
            "action": action_name,
            "response": response
        }

    def _parse_response_success(self, response_lower: str) -> bool:
        """
        Parse agent response to determine success/failure.
        
        Args:
            response_lower: Lowercase agent response
        
        Returns:
            True if action succeeded, False otherwise
        """
        if not response_lower:
            # Empty response usually means action is still in progress
            return False
        
        # Success indicators
        success_patterns = [
            "successful execution",
            "success",
            "completed",
            "done",
            "finished",
            "succeeded"
        ]
        
        # Failure indicators (check these first to avoid false positives)
        failure_patterns = [
            "fail",
            "error",
            "cannot",
            "unable",
            "impossible",
            "not found",
            "not available",
            "blocked",
            "collision",
            "out of reach"
        ]
        
        # Check for explicit failure first
        for pattern in failure_patterns:
            if pattern in response_lower:
                return False
        
        # Check for explicit success
        for pattern in success_patterns:
            if pattern in response_lower:
                return True
        
        # If response contains "still in progress", it's not a failure yet
        if "still in progress" in response_lower or "in progress" in response_lower:
            return False
        
        # Default: if response exists and doesn't indicate failure, assume success
        # (This is conservative - empty responses are handled above)
        return len(response_lower.strip()) > 0

    def _generate_success_message(
        self,
        agent_id: int,
        action_name: str,
        arg1: str,
        arg2: str,
        response: str
    ) -> str:
        """Generate natural language success message."""
        # Build action description
        action_desc = self._format_action(action_name, arg1, arg2)
        
        if self.verbose_feedback:
            # Include response details if available
            if response and len(response) < 100:
                return f"Agent {agent_id} successfully executed {action_desc}. {response}"
            else:
                return f"Agent {agent_id} successfully executed {action_desc}."
        else:
            return f"Agent {agent_id}: Success - {action_desc} completed."

    def _generate_failure_message(
        self,
        agent_id: int,
        action_name: str,
        arg1: str,
        arg2: str,
        response: str
    ) -> str:
        """Generate natural language failure message."""
        action_desc = self._format_action(action_name, arg1, arg2)
        
        if self.verbose_feedback:
            # Extract key failure reason from response
            failure_reason = self._extract_failure_reason(response)
            if failure_reason:
                return f"Agent {agent_id} failed to execute {action_desc}. {failure_reason}"
            else:
                return f"Agent {agent_id} failed to execute {action_desc}."
        else:
            failure_reason = self._extract_failure_reason(response) or "Action failed"
            return f"Agent {agent_id}: Failure - {action_desc} failed. {failure_reason}"

    def _format_action(self, action_name: str, arg1: str, arg2: str) -> str:
        """Format action as natural language."""
        if not arg1 and not arg2:
            return f"{action_name}[]"
        elif arg1 and not arg2:
            return f"{action_name}[{arg1}]"
        else:
            return f"{action_name}[{arg1}, {arg2}]"

    def _extract_failure_reason(self, response: str) -> str:
        """
        Extract failure reason from response.
        
        Args:
            response: Agent response string
        
        Returns:
            Brief failure reason, or empty string if not found
        """
        if not response:
            return "No response received"
        
        response_lower = response.lower()
        
        # Common failure reasons
        if "not found" in response_lower:
            return "Target object or location not found"
        elif "out of reach" in response_lower or "too far" in response_lower:
            return "Target is out of reach"
        elif "blocked" in response_lower or "obstacle" in response_lower:
            return "Path is blocked by obstacle"
        elif "collision" in response_lower:
            return "Collision detected"
        elif "cannot" in response_lower or "unable" in response_lower:
            # Extract the reason after "cannot" or "unable"
            parts = response_lower.split("cannot")
            if len(parts) > 1:
                reason = parts[1].strip()
                if len(reason) < 80:
                    return f"Cannot {reason}"
            parts = response_lower.split("unable")
            if len(parts) > 1:
                reason = parts[1].strip()
                if len(reason) < 80:
                    return f"Unable {reason}"
        elif "error" in response_lower:
            # Try to extract error message
            parts = response_lower.split("error")
            if len(parts) > 1:
                error_msg = parts[1].strip()
                if len(error_msg) < 80:
                    return f"Error: {error_msg}"
        
        # Default: return a cleaned version of the response
        if len(response) < 100:
            return response
        else:
            return "Action execution failed"

    def _verify_state_change(
        self,
        action_name: str,
        arg1: str,
        world_state: Dict[str, Any]
    ) -> bool:
        """
        Verify that state changed as expected for the action.
        
        This is an optional verification step. For now, we return True
        as a placeholder. In a full implementation, this would check:
        - For Pick actions: object should be in agent's hand
        - For Place actions: object should be at target location
        - For Navigate actions: agent position should have changed
        
        Args:
            action_name: Name of the action
            arg1: First argument (usually target object/location)
            world_state: Current world state dict
        
        Returns:
            True if state change verified, False otherwise
        """
        # Placeholder implementation
        # In a full implementation, this would compare current state
        # with previous state to verify expected changes
        return True
