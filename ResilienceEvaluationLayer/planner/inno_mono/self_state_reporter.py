"""
Self State Reporter for Inner Monologue

Reports agent's own state (position, holdings, etc.) in natural language.
"""

from typing import Dict, Any, Optional, List


class SelfStateReporter:
    """
    Reports agent's own state for Inner Monologue feedback.
    
    This component generates natural language descriptions of the agent's
    current state, including position, orientation, and what it's holding.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SelfStateReporter.
        
        Args:
            config: Configuration dictionary with optional settings:
                - include_position: Whether to include position (default: True)
                - include_rotation: Whether to include rotation/yaw (default: False)
                - include_holdings: Whether to include held objects (default: True)
                - position_precision: Decimal places for position (default: 1)
        """
        self.config = config or {}
        self.include_position = self.config.get("include_position", True)
        self.include_rotation = self.config.get("include_rotation", False)
        self.include_holdings = self.config.get("include_holdings", True)
        self.position_precision = self.config.get("position_precision", 1)

    def report_state(
        self,
        agent_id: int,
        world_state: Dict[str, Any]
    ) -> str:
        """
        Generate self-state report for an agent.
        
        Args:
            agent_id: Agent identifier
            world_state: World state dictionary containing agent_poses and agent_holdings
        
        Returns:
            Natural language description of agent's state
        """
        if not world_state:
            return f"Agent {agent_id} state unavailable."
        
        state_parts = []
        
        # Get agent pose
        agent_poses = world_state.get("agent_poses", {})
        pose_info = agent_poses.get(agent_id)
        
        # Position
        if self.include_position and pose_info:
            position = pose_info.get("position")
            if position:
                pos_str = self._format_position(position)
                state_parts.append(f"at position {pos_str}")
        
        # Rotation/Yaw (optional)
        if self.include_rotation and pose_info:
            yaw = pose_info.get("yaw")
            if yaw is not None:
                # Convert radians to degrees for readability
                yaw_deg = yaw * 180 / 3.14159
                state_parts.append(f"facing {yaw_deg:.0f}°")
        
        # Holdings
        if self.include_holdings:
            agent_holdings = world_state.get("agent_holdings", {})
            holdings = agent_holdings.get(agent_id, [])
            
            if holdings:
                if isinstance(holdings, list):
                    if len(holdings) == 1:
                        state_parts.append(f"holding {holdings[0]}")
                    else:
                        holdings_str = ", ".join(holdings)
                        state_parts.append(f"holding {holdings_str}")
                else:
                    state_parts.append(f"holding {holdings}")
            else:
                state_parts.append("hands free")
        
        if state_parts:
            return f"Agent {agent_id} is {' and '.join(state_parts)}."
        else:
            return f"Agent {agent_id} state: active."

    def report_state_for_all_agents(
        self,
        world_state: Dict[str, Any],
        agent_ids: Optional[List[int]] = None
    ) -> Dict[int, str]:
        """
        Generate state reports for multiple agents.
        
        Args:
            world_state: World state dictionary
            agent_ids: List of agent IDs to report (if None, reports all)
        
        Returns:
            Dictionary mapping agent_id to state description
        """
        if agent_ids is None:
            agent_poses = world_state.get("agent_poses", {})
            agent_ids = list(agent_poses.keys())
        
        reports = {}
        for agent_id in agent_ids:
            reports[agent_id] = self.report_state(agent_id, world_state)
        
        return reports

    def _format_position(self, position: List[float]) -> str:
        """Format position as readable string."""
        if not position or len(position) < 3:
            return "unknown"
        
        precision = self.position_precision
        return f"[{position[0]:.{precision}f}, {position[1]:.{precision}f}, {position[2]:.{precision}f}]"

