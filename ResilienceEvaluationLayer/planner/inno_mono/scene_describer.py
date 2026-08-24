"""
Scene Describer for Inner Monologue

Generates concise scene descriptions based on current world state.
Reuses existing world description utilities.
"""

from typing import Dict, Any, Optional
from habitat_llm.planner.connector.prompt_context import PromptContextBuilder
from habitat_llm.llm.instruct.utils import get_world_descr


class SceneDescriber:
    """
    Generates scene description feedback for Inner Monologue.
    
    This component creates natural language descriptions of the current scene
    state, focusing on relevant objects, their locations, and key changes.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SceneDescriber.
        
        Args:
            config: Configuration dictionary with optional settings:
                - max_length: Maximum length of scene description (default: 200)
                - include_rooms: Whether to include room information (default: True)
                - include_furniture: Whether to include furniture (default: True)
                - focus_on_changes: Whether to focus on recent changes (default: False)
        """
        self.config = config or {}
        self.max_length = self.config.get("max_length", 200)
        self.include_rooms = self.config.get("include_rooms", True)
        self.include_furniture = self.config.get("include_furniture", True)
        self.focus_on_changes = self.config.get("focus_on_changes", False)
        
        self.prompt_context_builder = PromptContextBuilder()
        self.last_scene_state: Optional[Dict[str, Any]] = None

    def describe_scene(
        self,
        world_state: Dict[str, Any],
        env_interface: Any
    ) -> str:
        """
        Generate scene description text.
        
        Args:
            world_state: World state dictionary with agent_poses, object_positions, etc.
            env_interface: Environment interface for accessing world graph
        
        Returns:
            Concise scene description text
        """
        if not world_state:
            return "Scene state unavailable."
        
        description_parts = []
        
        # 1. Describe object locations (most relevant for tasks)
        object_desc = self._describe_objects(world_state)
        if object_desc:
            description_parts.append(object_desc)
        
        # 2. Describe agent positions and holdings
        agent_desc = self._describe_agents(world_state)
        if agent_desc:
            description_parts.append(agent_desc)
        
        # 3. Optionally describe furniture/rooms
        if self.include_rooms or self.include_furniture:
            furniture_desc = self._describe_furniture(world_state, env_interface)
            if furniture_desc:
                description_parts.append(furniture_desc)
        
        # Combine and truncate if needed
        full_description = ". ".join(description_parts)
        
        if len(full_description) > self.max_length:
            # Truncate intelligently (prefer keeping the beginning)
            full_description = full_description[:self.max_length - 3] + "..."
        
        return full_description if full_description else "Scene state observed."

    def _describe_objects(self, world_state: Dict[str, Any]) -> str:
        """Describe object positions."""
        object_positions = world_state.get("object_positions", {})
        if not object_positions:
            return ""
        
        descriptions = []
        for obj_name, obj_info in object_positions.items():
            if not obj_info:
                continue
            
            parent = obj_info.get("parent", "unknown")
            if parent and parent != "unknown":
                descriptions.append(f"{obj_name} is on {parent}")
            else:
                pos = obj_info.get("position")
                if pos:
                    descriptions.append(f"{obj_name} is at position {self._format_position(pos)}")
        
        if descriptions:
            return "Objects: " + ", ".join(descriptions[:5])  # Limit to 5 objects
        return ""

    def _describe_agents(self, world_state: Dict[str, Any]) -> str:
        """Describe agent states (positions and holdings)."""
        agent_poses = world_state.get("agent_poses", {})
        agent_holdings = world_state.get("agent_holdings", {})
        
        if not agent_poses:
            return ""
        
        descriptions = []
        for agent_id, pose_info in agent_poses.items():
            agent_desc_parts = []
            
            # Position
            if pose_info and "position" in pose_info:
                pos = pose_info["position"]
                agent_desc_parts.append(f"at position {self._format_position(pos)}")
            
            # Holdings
            holdings = agent_holdings.get(agent_id, [])
            if holdings:
                if isinstance(holdings, list):
                    holdings_str = ", ".join(holdings)
                else:
                    holdings_str = str(holdings)
                agent_desc_parts.append(f"holding {holdings_str}")
            else:
                agent_desc_parts.append("hands free")
            
            if agent_desc_parts:
                descriptions.append(f"Agent {agent_id} is {' and '.join(agent_desc_parts)}")
        
        if descriptions:
            return "Agents: " + "; ".join(descriptions)
        return ""

    def _describe_furniture(self, world_state: Dict[str, Any], env_interface: Any) -> str:
        """Describe furniture locations (optional, can be verbose)."""
        if not self.include_furniture:
            return ""
        
        # Use furniture positions from world_state if available
        furniture_positions = world_state.get("furniture_positions", {})
        if furniture_positions:
            furn_names = list(furniture_positions.keys())[:5]  # Limit to 5
            return f"Furniture present: {', '.join(furn_names)}"
        
        # Fallback: try to get from env_interface if world_state doesn't have it
        if env_interface:
            try:
                # Get a concise world description (but limit length)
                world_desc = self.prompt_context_builder.build_world_description(env_interface)
                if len(world_desc) > 150:
                    # Too verbose, skip furniture description
                    return ""
                return f"Environment: {world_desc[:100]}"
            except Exception:
                pass
        
        return ""

    def _format_position(self, position: list) -> str:
        """Format position as readable string."""
        if not position or len(position) < 3:
            return "unknown"
        return f"[{position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f}]"

    def describe_scene_changes(
        self,
        current_state: Dict[str, Any],
        previous_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Describe changes in scene state (optional feature).
        
        Args:
            current_state: Current world state
            previous_state: Previous world state for comparison
        
        Returns:
            Description of what changed
        """
        if not previous_state:
            return self.describe_scene(current_state, None)
        
        changes = []
        
        # Check object position changes
        current_objects = current_state.get("object_positions", {})
        prev_objects = previous_state.get("object_positions", {})
        
        for obj_name, obj_info in current_objects.items():
            prev_info = prev_objects.get(obj_name)
            if prev_info:
                prev_parent = prev_info.get("parent")
                curr_parent = obj_info.get("parent")
                if prev_parent != curr_parent:
                    changes.append(f"{obj_name} moved from {prev_parent} to {curr_parent}")
        
        # Check agent holdings changes
        current_holdings = current_state.get("agent_holdings", {})
        prev_holdings = previous_state.get("agent_holdings", {})
        
        for agent_id, holdings in current_holdings.items():
            prev_agent_holdings = prev_holdings.get(agent_id, [])
            if holdings != prev_agent_holdings:
                if holdings:
                    changes.append(f"Agent {agent_id} picked up {holdings}")
                else:
                    changes.append(f"Agent {agent_id} released objects")
        
        if changes:
            return "Changes: " + "; ".join(changes[:3])  # Limit to 3 changes
        else:
            return "No significant changes observed."

