"""
State Assessor for Inner Monologue

Provides state assessment capabilities using StateEncoder.
This is an optional component for advanced state analysis.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class StateAssessor:
    """
    Assesses world state using StateEncoder.
    
    This component provides state assessment capabilities that can be used
    for generating more sophisticated feedback. It wraps StateEncoder functionality
    and provides additional analysis.
    """

    def __init__(
        self,
        state_encoder: Any,  # StateEncoder
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize StateAssessor.
        
        Args:
            state_encoder: StateEncoder instance
            config: Configuration dictionary
        """
        self.state_encoder = state_encoder
        self.config = config or {}

    def assess_state(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess world state and return assessment results.
        
        Args:
            world_state: World state dictionary
        
        Returns:
            Dictionary with assessment results:
                - state_vector: Encoded state vector
                - state_description: Text description of state
                - key_features: Key features extracted
        """
        assessment = {}
        
        try:
            # Encode state
            if self.state_encoder:
                state_vec = self.state_encoder.encode(world_state)
                assessment["state_vector"] = state_vec
                
                # Generate state description
                if hasattr(self.state_encoder, '_generate_state_description'):
                    state_desc = self.state_encoder._generate_state_description(world_state)
                    assessment["state_description"] = state_desc
        except Exception as e:
            logger.debug(f"Error assessing state: {e}")
            assessment["error"] = str(e)
        
        return assessment

    def compare_states(
        self,
        state1: Dict[str, Any],
        state2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare two states and identify differences.
        
        Args:
            state1: First world state
            state2: Second world state
        
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            "differences": [],
            "similarity": 0.0
        }
        
        try:
            # Compare object positions
            obj_pos1 = state1.get("object_positions", {})
            obj_pos2 = state2.get("object_positions", {})
            
            # Find objects that moved
            for obj_name in set(list(obj_pos1.keys()) + list(obj_pos2.keys())):
                pos1 = obj_pos1.get(obj_name, {}).get("parent") if obj_name in obj_pos1 else None
                pos2 = obj_pos2.get(obj_name, {}).get("parent") if obj_name in obj_pos2 else None
                
                if pos1 != pos2:
                    if pos1 and pos2:
                        comparison["differences"].append(f"{obj_name} moved from {pos1} to {pos2}")
                    elif pos2:
                        comparison["differences"].append(f"{obj_name} appeared at {pos2}")
                    elif pos1:
                        comparison["differences"].append(f"{obj_name} disappeared from {pos1}")
            
            # Compare agent holdings
            holdings1 = state1.get("agent_holdings", {})
            holdings2 = state2.get("agent_holdings", {})
            
            for agent_id in set(list(holdings1.keys()) + list(holdings2.keys())):
                h1 = holdings1.get(agent_id, [])
                h2 = holdings2.get(agent_id, [])
                if h1 != h2:
                    if h2 and not h1:
                        comparison["differences"].append(f"Agent {agent_id} picked up {h2}")
                    elif h1 and not h2:
                        comparison["differences"].append(f"Agent {agent_id} released {h1}")
                    else:
                        comparison["differences"].append(f"Agent {agent_id} holdings changed")
        
        except Exception as e:
            logger.debug(f"Error comparing states: {e}")
            comparison["error"] = str(e)
        
        return comparison

