"""
Enhanced Affordance Model for SayCan

Calculates P(success | state, skill) using heuristics and simulation state.
Enhanced version that returns detailed calculation information for analysis.
"""

import logging
import numpy as np
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Import occlusion checking utility
try:
    from habitat.datasets.rearrange.navmesh_utils import snap_point_is_occluded
    _HAS_OCCLUSION_CHECK = True
except ImportError:
    _HAS_OCCLUSION_CHECK = False
    logger.warning("snap_point_is_occluded not available, visibility checks will be limited")


class AffordanceModel:
    """
    Affordance Model for SayCan.
    
    Calculates P(success | state, skill) using heuristics and simulation state.
    Enhanced to return detailed calculation information for analysis.
    
    Core Formula: P(Can) = P(success | state, skill)
    """

    def __init__(self, env_interface: Any, config: Optional[Dict[str, Any]] = None):
        """
        Initialize AffordanceModel.
        
        Args:
            env_interface: Environment interface with sim and world_graph
            config: Configuration dictionary with affordance parameters
        """
        self.env_interface = env_interface
        self.sim = env_interface.sim if hasattr(env_interface, "sim") else None
        self.config = config or {}
        
        # Heuristic parameters
        self.max_nav_distance = self.config.get("max_nav_distance", 20.0)
        self.pick_distance_thresh = self.config.get("pick_distance_thresh", 2.0)
        self.visibility_check_enabled = self.config.get("visibility_check_enabled", True)
        
        # Optional: Load a learned value function (Critic) if available
        self.critic = None
        # Integration point: if A2C Critic is initialized elsewhere, we could link it here

    def get_affordance(
        self, skill_name: str, target_name: str, agent_id: int
    ) -> float:
        """
        Calculate affordance score [0, 1] for a given skill and target.
        
        Backward-compatible method that returns only the score.
        For detailed information, use get_affordance_with_details().
        
        Args:
            skill_name: Name of the skill (e.g., "Navigate", "Pick")
            target_name: Name of the target object/location
            agent_id: ID of the agent executing the skill
            
        Returns:
            Float probability of success (0.0 to 1.0)
        """
        score, _ = self.get_affordance_with_details(skill_name, target_name, agent_id)
        return score

    def get_affordance_with_details(
        self, skill_name: str, target_name: str, agent_id: int
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate affordance score with detailed calculation information.
        
        Args:
            skill_name: Name of the skill (e.g., "Navigate", "Pick")
            target_name: Name of the target object/location
            agent_id: ID of the agent executing the skill
            
        Returns:
            Tuple of (score, details):
                - score: Float probability of success (0.0 to 1.0)
                - details: Dict with calculation details:
                    - distance: Distance to target (if applicable)
                    - visibility: Whether target is visible (if applicable)
                    - reachability: Whether target is reachable (if applicable)
                    - failure_reason: Reason for low score (if applicable)
                    - skill_type: Normalized skill type
        """
        skill_type = skill_name.lower()
        details = {
            "skill_type": skill_type,
            "target_name": target_name,
            "agent_id": agent_id,
        }
        
        # Normalize skill names and route to appropriate scorer
        if "navigate" in skill_type:
            score, skill_details = self._navigate_score_with_details(target_name, agent_id)
        elif "pick" in skill_type:
            score, skill_details = self._pick_score_with_details(target_name, agent_id)
        elif "place" in skill_type:
            score, skill_details = self._place_score_with_details(target_name, agent_id)
        elif "open" in skill_type or "close" in skill_type:
            score, skill_details = self._interaction_score_with_details(target_name, agent_id)
        elif "wait" in skill_type or "done" in skill_type:
            score = 1.0
            skill_details = {"reason": "Always possible to wait"}
        else:
            score = 0.5
            skill_details = {"reason": "Unknown skill type, using default"}
        
        details.update(skill_details)
        
        # Add failure reason if score is low
        if score < 0.3:
            if "distance" in details and details.get("distance", 0) > self.pick_distance_thresh:
                details["failure_reason"] = "target_too_far"
            elif "visibility" in details and not details.get("visibility", True):
                details["failure_reason"] = "target_occluded"
            elif "reachability" in details and not details.get("reachability", True):
                details["failure_reason"] = "target_unreachable"
            elif "target_unknown" in details and details.get("target_unknown", False):
                details["failure_reason"] = "target_unknown"
            else:
                details["failure_reason"] = "low_confidence"
        
        return score, details

    def _get_agent_pos(self, agent_id: int) -> Optional[np.ndarray]:
        """Get agent position from simulator."""
        try:
            return self.sim.agents_mgr[agent_id].articulated_agent.base_pos
        except Exception:
            return None

    def _get_target_pos(self, target_name: str, agent_id: int) -> Optional[np.ndarray]:
        """
        Resolve target position from WorldGraph.
        
        Returns:
            Target position as numpy array, or None if not found
        """
        if not target_name or target_name.lower() == "none":
            return None
            
        try:
            # Try getting from agent's world graph first
            if hasattr(self.env_interface, 'world_graph') and agent_id in self.env_interface.world_graph:
                wg = self.env_interface.world_graph[agent_id]
                node = wg.get_node_from_name(target_name)
                if node and hasattr(node, 'properties'):
                    pos = node.properties.get('translation', node.properties.get('position'))
                    if pos is not None:
                        return np.array(pos)
                        
            # Fallback to full world graph if available
            if hasattr(self.env_interface, 'full_world_graph'):
                wg = self.env_interface.full_world_graph
                node = wg.get_node_from_name(target_name)
                if node:
                    pos = node.properties.get('translation')
                    if pos is not None:
                        return np.array(pos)
                        
        except Exception as e:
            logger.debug(f"Error resolving target {target_name}: {e}")
            
        return None

    def _check_visibility(
        self, target_pos: np.ndarray, agent_pos: np.ndarray, target_name: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if target is visible from agent position using raycast.
        
        Args:
            target_pos: Target position
            agent_pos: Agent position
            target_name: Target name (for object ID lookup)
            
        Returns:
            Tuple of (is_visible, details)
        """
        if not self.visibility_check_enabled or not _HAS_OCCLUSION_CHECK:
            return True, {"visibility_check": "disabled"}
        
        if self.sim is None:
            return True, {"visibility_check": "sim_unavailable"}
        
        try:
            # Get target object ID
            target_obj_id = None
            if hasattr(self.env_interface, 'world_graph'):
                for agent_id, wg in self.env_interface.world_graph.items():
                    try:
                        node = wg.get_node_from_name(target_name)
                        if node and hasattr(node, 'sim_obj_id'):
                            target_obj_id = node.sim_obj_id
                            break
                    except Exception:
                        pass
            
            if target_obj_id is None:
                # Fallback: try to get from full_world_graph
                if hasattr(self.env_interface, 'full_world_graph'):
                    try:
                        node = self.env_interface.full_world_graph.get_node_from_name(target_name)
                        if node and hasattr(node, 'sim_obj_id'):
                            target_obj_id = node.sim_obj_id
                    except Exception:
                        pass
            
            if target_obj_id is None:
                return True, {"visibility_check": "object_id_not_found"}
            
            # Check occlusion using snap_point_is_occluded
            # This checks if there are obstacles between agent and target
            import magnum as mn
            target_point = mn.Vector3(target_pos[0], target_pos[1], target_pos[2])
            agent_point = mn.Vector3(agent_pos[0], agent_pos[1], agent_pos[2])
            
            # Get agent object IDs to ignore
            agent_object_ids = []
            if hasattr(self.sim, 'agents_mgr'):
                for agent in self.sim.agents_mgr.articulated_agents_iter:
                    agent_object_ids.append(agent.sim_obj.object_id)
                    if hasattr(agent.sim_obj, 'link_object_ids'):
                        agent_object_ids.extend(agent.sim_obj.link_object_ids.keys())
            
            is_occluded = snap_point_is_occluded(
                target=target_point,
                snap_point=agent_point,
                height=1.3,  # Agent eye height
                sim=self.sim,
                target_object_ids=[target_obj_id],
                ignore_object_ids=agent_object_ids,
            )
            
            return not is_occluded, {
                "visibility_check": "raycast",
                "is_visible": not is_occluded,
                "target_obj_id": target_obj_id,
            }
        except Exception as e:
            logger.debug(f"Error checking visibility: {e}")
            return True, {"visibility_check": "error", "error": str(e)}

    def _navigate_score_with_details(
        self, target_name: str, agent_id: int
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score for Navigation with details.
        
        High if: Target exists, is reachable, and close.
        
        Returns:
            Tuple of (score, details)
        """
        agent_pos = self._get_agent_pos(agent_id)
        target_pos = self._get_target_pos(target_name, agent_id)
        
        details = {}
        
        if agent_pos is None or target_pos is None:
            details["target_unknown"] = True
            details["failure_reason"] = "target_or_agent_position_unknown"
            return 0.1, details
        
        # 1. Distance calculation
        euclidean_dist = np.linalg.norm(agent_pos - target_pos)
        details["euclidean_distance"] = float(euclidean_dist)
        
        # 2. Path distance (NavMesh)
        path_dist = euclidean_dist
        path_valid = True
        if hasattr(self.sim, "pathfinder") and self.sim.pathfinder is not None:
            try:
                path = self.sim.pathfinder.find_path(agent_pos, target_pos)
                if path.is_valid:
                    path_dist = path.length
                    details["path_distance"] = float(path_dist)
                    details["reachability"] = True
                else:
                    path_valid = False
                    details["reachability"] = False
                    details["failure_reason"] = "no_valid_path"
                    return 0.0, details
            except Exception as e:
                logger.debug(f"Error computing path: {e}")
                details["path_computation_error"] = str(e)
        
        # 3. Distance-based score (exponential decay)
        # 0m -> 1.0, 10m -> ~0.36, 20m -> ~0.13
        dist_score = np.exp(-0.1 * path_dist)
        details["distance"] = float(path_dist)
        details["distance_score"] = float(dist_score)
        
        # 4. Check if within max distance
        if path_dist > self.max_nav_distance:
            details["failure_reason"] = "exceeds_max_distance"
            return 0.01, details
        
        final_score = max(0.01, dist_score)
        return final_score, details

    def _pick_score_with_details(
        self, target_name: str, agent_id: int
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score for Pick with details.
        
        High if: Agent is close to object, hands free, object visible.
        
        Returns:
            Tuple of (score, details)
        """
        agent_pos = self._get_agent_pos(agent_id)
        target_pos = self._get_target_pos(target_name, agent_id)
        
        details = {}
        
        if agent_pos is None or target_pos is None:
            details["target_unknown"] = True
            details["failure_reason"] = "target_or_agent_position_unknown"
            return 0.1, details
        
        # 1. Proximity check
        dist = np.linalg.norm(agent_pos - target_pos)
        details["distance"] = float(dist)
        
        if dist > self.pick_distance_thresh:
            details["failure_reason"] = "target_too_far"
            details["reachability"] = False
            return 0.1, details
        
        details["reachability"] = True
        
        # 2. Visibility check
        is_visible, visibility_details = self._check_visibility(target_pos, agent_pos, target_name)
        details.update(visibility_details)
        
        if not is_visible:
            details["failure_reason"] = "target_occluded"
            return 0.1, details
        
        # 3. Hands check (simplified)
        # Check if agent is already holding something
        try:
            wg = self.env_interface.world_graph[agent_id]
            # Try to get agent node
            agent_node = None
            if hasattr(wg, 'get_spot_robot'):
                try:
                    agent_node = wg.get_spot_robot()
                except Exception:
                    pass
            elif hasattr(wg, 'get_human'):
                try:
                    agent_node = wg.get_human()
                except Exception:
                    pass
            
            if agent_node and hasattr(agent_node, 'properties'):
                held_obj = agent_node.properties.get("last_held_object")
                if held_obj is not None:
                    details["hands_busy"] = True
                    details["failure_reason"] = "agent_already_holding_object"
                    return 0.2, details  # Lower but not zero (could drop first)
            
            details["hands_busy"] = False
        except Exception as e:
            logger.debug(f"Error checking hands: {e}")
            details["hands_check_error"] = str(e)
        
        # All checks passed
        return 1.0, details

    def _place_score_with_details(
        self, target_name: str, agent_id: int
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score for Place with details.
        
        High if: Agent is holding something, target is close.
        
        Returns:
            Tuple of (score, details)
        """
        agent_pos = self._get_agent_pos(agent_id)
        target_pos = self._get_target_pos(target_name, agent_id)
        
        details = {}
        
        if agent_pos is None or target_pos is None:
            details["target_unknown"] = True
            details["failure_reason"] = "target_or_agent_position_unknown"
            return 0.1, details
        
        # 1. Proximity check
        dist = np.linalg.norm(agent_pos - target_pos)
        details["distance"] = float(dist)
        
        if dist > self.pick_distance_thresh:
            details["failure_reason"] = "target_too_far"
            details["reachability"] = False
            return 0.1, details
        
        details["reachability"] = True
        
        # 2. Check if agent is holding something
        try:
            wg = self.env_interface.world_graph[agent_id]
            agent_node = None
            if hasattr(wg, 'get_spot_robot'):
                try:
                    agent_node = wg.get_spot_robot()
                except Exception:
                    pass
            elif hasattr(wg, 'get_human'):
                try:
                    agent_node = wg.get_human()
                except Exception:
                    pass
            
            if agent_node and hasattr(agent_node, 'properties'):
                held_obj = agent_node.properties.get("last_held_object")
                if held_obj is None:
                    details["hands_empty"] = True
                    details["failure_reason"] = "agent_not_holding_anything"
                    return 0.1, details
                
                details["hands_empty"] = False
                details["held_object"] = str(held_obj) if hasattr(held_obj, 'name') else str(held_obj)
        except Exception as e:
            logger.debug(f"Error checking held object: {e}")
            details["hands_check_error"] = str(e)
        
        # 3. Visibility check (optional, less critical for Place)
        is_visible, visibility_details = self._check_visibility(target_pos, agent_pos, target_name)
        details.update(visibility_details)
        
        if not is_visible:
            # For Place, visibility is less critical, just reduce score slightly
            return 0.7, details
        
        return 1.0, details

    def _interaction_score_with_details(
        self, target_name: str, agent_id: int
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score for Open/Close interactions with details.
        
        Similar to Pick: requires proximity and visibility.
        
        Returns:
            Tuple of (score, details)
        """
        # Reuse Pick scoring logic
        score, details = self._pick_score_with_details(target_name, agent_id)
        details["skill_type"] = "interaction"
        return score, details
