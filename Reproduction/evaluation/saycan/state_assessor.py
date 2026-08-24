"""
State Assessor for SayCan

Assesses world state to provide context for Affordance calculation.
Extracts agent positions, object positions, visibility, and distance information.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class StateAssessor:
    """
    Assesses world state for SayCan Affordance calculation.
    
    Provides structured state information that can be used by AffordanceModel
    and other SayCan components.
    """

    def __init__(self, env_interface: Any, config: Optional[Dict[str, Any]] = None):
        """
        Initialize StateAssessor.
        
        Args:
            env_interface: Environment interface
            config: Optional configuration dictionary
        """
        self.env_interface = env_interface
        self.config = config or {}
        self.sim = env_interface.sim if hasattr(env_interface, "sim") else None

    def assess_world_state(self, agent_id: int) -> Dict[str, Any]:
        """
        Assess world state for a given agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary with state information:
                - agent_position: Agent position [x, y, z]
                - agent_rotation: Agent rotation quaternion
                - objects: List of object information
                - furniture: List of furniture information
                - rooms: List of room information
        """
        state = {
            "agent_position": None,
            "agent_rotation": None,
            "objects": [],
            "furniture": [],
            "rooms": [],
        }
        
        try:
            # Get agent position
            if self.sim and hasattr(self.sim, 'agents_mgr'):
                try:
                    agent = self.sim.agents_mgr[agent_id].articulated_agent
                    state["agent_position"] = list(agent.base_pos)
                    state["agent_rotation"] = [
                        agent.base_rot.x,
                        agent.base_rot.y,
                        agent.base_rot.z,
                        agent.base_rot.w,
                    ]
                except Exception as e:
                    logger.debug(f"Error getting agent position: {e}")
            
            # Get world graph information
            if hasattr(self.env_interface, 'world_graph') and agent_id in self.env_interface.world_graph:
                wg = self.env_interface.world_graph[agent_id]
                
                # Get objects
                try:
                    objects = wg.get_all_objects() if hasattr(wg, 'get_all_objects') else []
                    for obj in objects:
                        if hasattr(obj, 'name') and hasattr(obj, 'properties'):
                            obj_info = {
                                "name": obj.name,
                                "position": list(obj.properties.get('translation', [0, 0, 0])),
                                "parent": obj.properties.get('parent_receptacle', 'unknown'),
                            }
                            state["objects"].append(obj_info)
                except Exception as e:
                    logger.debug(f"Error getting objects: {e}")
                
                # Get furniture
                try:
                    furniture = wg.get_all_furniture() if hasattr(wg, 'get_all_furniture') else []
                    for furn in furniture:
                        if hasattr(furn, 'name') and hasattr(furn, 'properties'):
                            furn_info = {
                                "name": furn.name,
                                "position": list(furn.properties.get('translation', [0, 0, 0])),
                            }
                            state["furniture"].append(furn_info)
                except Exception as e:
                    logger.debug(f"Error getting furniture: {e}")
                
                # Get rooms
                try:
                    rooms = wg.get_all_rooms() if hasattr(wg, 'get_all_rooms') else []
                    for room in rooms:
                        if hasattr(room, 'name'):
                            state["rooms"].append({"name": room.name})
                except Exception as e:
                    logger.debug(f"Error getting rooms: {e}")
        
        except Exception as e:
            logger.debug(f"Error assessing world state: {e}")
        
        return state

    def check_visibility(
        self, target_name: str, agent_id: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if target is visible from agent position.
        
        Args:
            target_name: Name of target object/location
            agent_id: ID of the agent
            
        Returns:
            Tuple of (is_visible, details)
        """
        try:
            # Get agent position
            if not self.sim or not hasattr(self.sim, 'agents_mgr'):
                return True, {"visibility_check": "sim_unavailable"}
            
            agent_pos = self.sim.agents_mgr[agent_id].articulated_agent.base_pos
            
            # Get target position
            target_pos = None
            if hasattr(self.env_interface, 'world_graph') and agent_id in self.env_interface.world_graph:
                wg = self.env_interface.world_graph[agent_id]
                try:
                    node = wg.get_node_from_name(target_name)
                    if node and hasattr(node, 'properties'):
                        pos = node.properties.get('translation', node.properties.get('position'))
                        if pos is not None:
                            target_pos = np.array(pos)
                except Exception:
                    pass
            
            if target_pos is None:
                return True, {"visibility_check": "target_position_unknown"}
            
            # Use occlusion check if available
            try:
                from habitat.datasets.rearrange.navmesh_utils import snap_point_is_occluded
                import magnum as mn
                
                target_point = mn.Vector3(target_pos[0], target_pos[1], target_pos[2])
                agent_point = mn.Vector3(agent_pos[0], agent_pos[1], agent_pos[2])
                
                # Get agent object IDs to ignore
                agent_object_ids = []
                for agent in self.sim.agents_mgr.articulated_agents_iter:
                    agent_object_ids.append(agent.sim_obj.object_id)
                
                is_occluded = snap_point_is_occluded(
                    target=target_point,
                    snap_point=agent_point,
                    height=1.3,
                    sim=self.sim,
                    target_object_ids=[],
                    ignore_object_ids=agent_object_ids,
                )
                
                return not is_occluded, {
                    "visibility_check": "raycast",
                    "is_visible": not is_occluded,
                }
            except ImportError:
                return True, {"visibility_check": "occlusion_check_unavailable"}
            except Exception as e:
                logger.debug(f"Error checking visibility: {e}")
                return True, {"visibility_check": "error", "error": str(e)}
        
        except Exception as e:
            logger.debug(f"Error in check_visibility: {e}")
            return True, {"visibility_check": "error", "error": str(e)}
