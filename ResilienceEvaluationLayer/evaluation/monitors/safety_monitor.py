#!/usr/bin/env python3

"""
Safety Monitor Module

This module implements the SafetyMonitor class, which is responsible for tracking
safety-related constraints and metrics during episode execution.
It calculates Control Barrier Function (CBF) penalties and tracks safety violations.
"""

from typing import Any, Dict, List, Mapping, Sequence

class SafetyMonitor:
    """
    Monitors execution safety and calculates CBF (Control Barrier Function) metrics.
    
    Metrics tracked:
    - P_cbf: Execution Safety penalty (sum of constraint violations)
    - Safety Score: Distribution safety aggregated over time
    """
    
    monitor_name = "safety"

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SafetyMonitor.
        
        :param config: Configuration dictionary containing thresholds and penalties.
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self.collision_penalty = config.get("collision_penalty", 1.0)
        self.min_dist_threshold = config.get("min_dist_threshold", 0.5)
        self.unsafe_state_penalty = config.get("unsafe_state_penalty", 0.5)
        raw_required = config.get(
            "required_channels", ("agent_pair_collision",)
        )
        if isinstance(raw_required, str):
            raw_required = [raw_required]
        if not isinstance(raw_required, Sequence):
            raise TypeError("safety_monitor.required_channels must be a sequence")
        self.required_channels = tuple(
            (
                "agent_pair_collision"
                if str(channel).strip().lower() == "collision"
                else str(channel).strip().lower()
            )
            for channel in raw_required
            if str(channel).strip()
        )
        unsupported = set(self.required_channels).difference(
            {"agent_pair_collision", "distance"}
        )
        if unsupported:
            raise ValueError(
                "Unsupported safety channel(s): " + ", ".join(sorted(unsupported))
            )
        
        self.accumulated_cbf_penalty = 0.0
        self.violation_history: List[float] = []
        self.collision_history: List[bool] = []
        self.distance_observation_count = 0
        self.distance_missing_count = 0
        self.collision_observation_count = 0
        self.collision_missing_count = 0
        self.step_count = 0
        
    def reset(self):
        """Reset monitor state for new episode."""
        self.accumulated_cbf_penalty = 0.0
        self.violation_history = []
        self.collision_history = []
        self.distance_observation_count = 0
        self.distance_missing_count = 0
        self.collision_observation_count = 0
        self.collision_missing_count = 0
        self.step_count = 0

    def update(self, step_data: Mapping[str, Any]) -> Dict[str, Any]:
        """
        Check safety constraints for the current step.
        
        :param step_data: Dictionary containing current step data (collisions, positions, etc.)
        :return: Dictionary containing safety metrics for this step.
        """
        if not self.enabled:
            return {
                "cbf_penalty": None,
                "is_safe": None,
                "safety_valid": False,
                "safety_missing_reason": "monitor_disabled",
            }

        self.step_count += 1
        step_penalty = 0.0
        
        # 1. Collision Check
        # Expecting step_data["collisions"] to be a dict of agent_id -> bool
        collisions = step_data.get("agent_collisions")
        collision_scope = str(step_data.get("collision_scope") or "").strip().lower()
        collision_observed = (
            isinstance(collisions, Mapping)
            and bool(collisions)
            and collision_scope == "agent_pair_collision"
        )
        if collision_observed:
            self.collision_observation_count += 1
            is_collision = any(bool(value) for value in collisions.values())
            if is_collision:
                step_penalty += self.collision_penalty
            self.collision_history.append(is_collision)
        else:
            self.collision_missing_count += 1

        # 2. Minimum Distance Constraint (Example CBF)
        # h(x) = dist(agent, hazard) - min_dist >= 0
        # Violation if h(x) < 0. Penalty = max(0, -h(x))
        # Here we approximate by checking if any agent is too close to obstacles if data provided
        # For now, we use a placeholder heuristic if 'min_obstacle_dist' is passed
        min_obstacle_dist = step_data.get("min_obstacle_dist")
        if min_obstacle_dist is None:
            self.distance_missing_count += 1
        else:
            self.distance_observation_count += 1
            if min_obstacle_dist < self.min_dist_threshold:
                h_x = min_obstacle_dist - self.min_dist_threshold
                step_penalty += abs(h_x) * self.unsafe_state_penalty

        # Accumulate
        self.accumulated_cbf_penalty += step_penalty
        self.violation_history.append(step_penalty)

        return {
            "cbf_penalty": step_penalty,
            "accumulated_cbf_penalty": self.accumulated_cbf_penalty,
            "is_safe": 1.0 if step_penalty == 0 else 0.0
        }

    def check_safety(self, step_data: Mapping[str, Any]) -> Dict[str, Any]:
        """Backward-compatible alias for :meth:`update`."""
        return self.update(step_data)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the episode."""
        total_steps = int(self.step_count)
        base = {
            "safety_required_channels": "+".join(self.required_channels),
            "collision_observation_count": self.collision_observation_count,
            "collision_missing_count": self.collision_missing_count,
            "distance_constraint_available": self.distance_observation_count > 0,
            "distance_observation_count": self.distance_observation_count,
            "distance_missing_count": self.distance_missing_count,
            "distance_constraint_missing_reason": (
                "" if self.distance_observation_count > 0 else "sensor_not_supplied"
            ),
        }
        if not self.enabled:
            return {
                **base,
                "total_cbf_penalty": None,
                "safety_score": None,
                "unsafe_rate": None,
                "collision_rate": None,
                "safety_valid": False,
                "safety_scope": "none",
                "safety_missing_reason": "monitor_disabled",
            }
        if total_steps == 0:
            return {
                **base,
                "total_cbf_penalty": None,
                "safety_score": None,
                "unsafe_rate": None,
                "collision_rate": None,
                "safety_valid": False,
                "safety_scope": "none",
                "safety_missing_reason": "no_step_observations",
                "distance_constraint_missing_reason": "no_step_observations",
            }

        complete_channels = {
            channel
            for channel, observed in (
                ("agent_pair_collision", self.collision_observation_count),
                ("distance", self.distance_observation_count),
            )
            if observed == total_steps
        }
        missing_required = [
            channel for channel in self.required_channels if channel not in complete_channels
        ]
        safety_valid = not missing_required
        safety_scope = "+".join(sorted(complete_channels)) or "none"
        missing_reason = (
            ""
            if safety_valid
            else "missing_required_safety_channels:" + ",".join(missing_required)
        )
            
        unsafe_steps = sum(1 for v in self.violation_history if v > 0)
        collision_steps = sum(1 for c in self.collision_history if c)
        
        return {
            **base,
            "total_cbf_penalty": (
                self.accumulated_cbf_penalty if safety_valid else None
            ),
            "safety_score": (
                1.0 - (unsafe_steps / total_steps) if safety_valid else None
            ),
            "unsafe_rate": unsafe_steps / total_steps if safety_valid else None,
            "collision_rate": (
                collision_steps / self.collision_observation_count
                if self.collision_observation_count > 0
                else None
            ),
            "safety_valid": safety_valid,
            "safety_scope": safety_scope,
            "safety_missing_reason": missing_reason,
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Backward-compatible alias used by stability collection code."""
        return self.get_summary()
