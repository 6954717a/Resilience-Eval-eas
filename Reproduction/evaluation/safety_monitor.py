#!/usr/bin/env python3

"""
Safety Monitor Module

This module implements the SafetyMonitor class, which is responsible for tracking
safety-related constraints and metrics during episode execution.
It calculates Control Barrier Function (CBF) penalties and tracks safety violations.
"""

from typing import Dict, Any, List, Optional
import numpy as np

class SafetyMonitor:
    """
    Monitors execution safety and calculates CBF (Control Barrier Function) metrics.
    
    Metrics tracked:
    - P_cbf: Execution Safety penalty (sum of constraint violations)
    - Safety Score: Distribution safety aggregated over time
    """
    
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
        
        self.accumulated_cbf_penalty = 0.0
        self.violation_history: List[float] = []
        self.collision_history: List[bool] = []
        
    def reset(self):
        """Reset monitor state for new episode."""
        self.accumulated_cbf_penalty = 0.0
        self.violation_history = []
        self.collision_history = []

    def check_safety(self, step_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Check safety constraints for the current step.
        
        :param step_data: Dictionary containing current step data (collisions, positions, etc.)
        :return: Dictionary containing safety metrics for this step.
        """
        if not self.enabled:
            return {"cbf_penalty": 0.0, "is_safe": 1.0}

        step_penalty = 0.0
        
        # 1. Collision Check
        # Expecting step_data["collisions"] to be a dict of agent_id -> bool
        collisions = step_data.get("agent_collisions", {})
        is_collision = any(collisions.values())
        if is_collision:
            step_penalty += self.collision_penalty
        self.collision_history.append(is_collision)

        # 2. Minimum Distance Constraint (Example CBF)
        # h(x) = dist(agent, hazard) - min_dist >= 0
        # Violation if h(x) < 0. Penalty = max(0, -h(x))
        # Here we approximate by checking if any agent is too close to obstacles if data provided
        # For now, we use a placeholder heuristic if 'min_obstacle_dist' is passed
        min_obstacle_dist = step_data.get("min_obstacle_dist", float('inf'))
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

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics for the episode."""
        total_steps = len(self.violation_history)
        if total_steps == 0:
            return {
                "total_cbf_penalty": 0.0,
                "safety_score": 1.0, # Perfect safety
                "collision_rate": 0.0
            }
            
        unsafe_steps = sum(1 for v in self.violation_history if v > 0)
        collision_steps = sum(1 for c in self.collision_history if c)
        
        return {
            "total_cbf_penalty": self.accumulated_cbf_penalty,
            "safety_score": 1.0 - (unsafe_steps / total_steps), # S_F approximation for single episode
            "collision_rate": collision_steps / total_steps
        }
