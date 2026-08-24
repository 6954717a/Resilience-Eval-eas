"""
MBR (Minimum Bayes Risk) Action Sampler for CycleVLA.

This module implements the MBR decoding strategy for robust action selection
after backtracking. It samples multiple action traces and selects the one
with minimum average distance to all others (consensus trajectory).

Reference: CycleVLA Paper Section IV-B
- "We apply test-time scaling by sampling multiple action chunk hypotheses
   and selecting a consensus one via MBR decoding."
- "MBR selects the hypothesis that minimizes the expected risk under the
   policy distribution."
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class MBRActionSampler:
    """
    Minimum Bayes Risk sampler for LLM-based planner.
    
    Unlike CycleVLA's original MBR which operates on continuous VLA action
    space, this implementation adapts MBR to discrete LLM action sequences:
    
    1. Sample N action traces from the LLM with temperature > 0
    2. Extract action sequences from each trace
    3. Compute pairwise distances between action sequences
    4. Select the trace with minimum average distance (consensus)
    
    This improves retry success rate after backtracking by selecting
    the most "typical" action sequence.
    """
    
    def __init__(self, planner: Any, config: Dict[str, Any]):
        """
        Initialize MBR Sampler.
        
        Args:
            planner: LLMPlanner instance
            config: Configuration dictionary:
                - mbr_num_samples: Number of samples (default: 5)
                - mbr_temperature: Sampling temperature (default: 0.7)
                - distance_metric: "edit" or "jaccard" (default: "edit")
        """
        self.planner = planner
        self.config = config
        self.num_samples = config.get("mbr_num_samples", 5)
        self.temperature = config.get("mbr_temperature", 0.7)
        self.distance_metric = config.get("distance_metric", "edit")
        
        # Statistics
        self.sample_count = 0
        self.avg_consensus_distance = 0.0
    
    def sample_and_select(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Any,
        current_subtask_instruction: Optional[str] = None
    ) -> Tuple[str, List[Tuple[str, str, str]]]:
        """
        Sample N traces and select consensus via MBR.
        
        This implements the MBR decoding from CycleVLA, adapted for
        discrete action sequences from LLM.
        
        Args:
            instruction: Task instruction
            observations: Current observations
            world_graph: Current world graph
            current_subtask_instruction: Optional focused subtask instruction
        
        Returns:
            Tuple of:
                - selected_response: The selected LLM response string
                - selected_actions: List of (tool_name, args, agent) tuples
        """
        self.sample_count += 1
        
        samples = []
        
        # Save original planner state
        original_prompt = self.planner.curr_prompt
        original_trace = self.planner.trace
        
        # Sample N responses
        for i in range(self.num_samples):
            try:
                response = self._generate_sample(instruction, observations, world_graph)
                actions = self._extract_actions(response)
                
                samples.append({
                    "response": response,
                    "actions": actions,
                    "action_str": self._actions_to_string(actions)
                })
                
            except Exception as e:
                logger.warning(f"Sample {i} generation failed: {e}")
                continue
        
        # Restore planner state
        self.planner.curr_prompt = original_prompt
        self.planner.trace = original_trace
        
        # Handle edge cases
        if not samples:
            logger.error("All MBR samples failed, returning empty")
            return "", []
        
        if len(samples) == 1:
            return samples[0]["response"], samples[0]["actions"]
        
        # Compute pairwise distance matrix
        n = len(samples)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                d = self._compute_distance(
                    samples[i]["action_str"],
                    samples[j]["action_str"]
                )
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        
        # Compute average distance for each sample
        avg_distances = dist_matrix.sum(axis=1) / (n - 1)
        
        # Select minimum average distance (MBR criterion)
        selected_idx = int(np.argmin(avg_distances))
        
        # Update statistics
        self.avg_consensus_distance = float(avg_distances[selected_idx])
        
        logger.info(
            f"MBR selected sample {selected_idx}/{n}, "
            f"avg_distance={self.avg_consensus_distance:.3f}"
        )
        
        return samples[selected_idx]["response"], samples[selected_idx]["actions"]
    
    def _generate_sample(
        self,
        instruction: str,
        observations: Dict[str, Any],
        world_graph: Any
    ) -> str:
        """Generate one LLM sample with temperature."""
        
        # Get LLM wrapper
        llm = self.planner.llm
        
        # Store original temperature if accessible
        original_temp = None
        
        # Try to set temperature
        if hasattr(llm, 'temperature'):
            original_temp = llm.temperature
            llm.temperature = self.temperature
        elif hasattr(llm, 'config') and hasattr(llm.config, 'temperature'):
            original_temp = llm.config.temperature
            llm.config.temperature = self.temperature
        
        try:
            # Generate response using planner's method
            response = self.planner.generate_action_response()
            
            # Format response
            end_expressions = []
            if hasattr(self.planner, 'end_expression'):
                if isinstance(self.planner.end_expression, list):
                    end_expressions.extend(self.planner.end_expression)
                else:
                    end_expressions.append(self.planner.end_expression)
            
            if hasattr(self.planner, 'stopword'):
                if isinstance(self.planner.stopword, list):
                    end_expressions.extend(self.planner.stopword)
                else:
                    end_expressions.append(self.planner.stopword)
            
            if end_expressions:
                response = self.planner.format_response(response, end_expressions)
            
            return response
            
        finally:
            # Restore original temperature
            if original_temp is not None:
                if hasattr(llm, 'temperature'):
                    llm.temperature = original_temp
                elif hasattr(llm, 'config') and hasattr(llm.config, 'temperature'):
                    llm.config.temperature = original_temp
    
    def _extract_actions(self, response: str) -> List[Tuple[str, str, str]]:
        """
        Extract action tuples from LLM response.
        
        Returns list of (tool_name, args, agent_id) tuples.
        """
        actions = []
        
        # Pattern: Agent_X_Action: ToolName[args]
        pattern = r'Agent_(\d+)_Action:\s*(\w+)\[([^\]]*)\]'
        matches = re.findall(pattern, response)
        
        for agent_id, tool_name, args in matches:
            actions.append((tool_name, args, agent_id))
        
        return actions
    
    def _actions_to_string(self, actions: List[Tuple[str, str, str]]) -> str:
        """Convert actions to string for distance computation."""
        return " ".join(f"{tool}[{args}]" for tool, args, _ in actions)
    
    def _compute_distance(self, s1: str, s2: str) -> float:
        """
        Compute distance between two action strings.
        
        Supports:
        - "edit": Normalized Levenshtein edit distance
        - "jaccard": 1 - Jaccard similarity of action sets
        """
        if self.distance_metric == "jaccard":
            return self._jaccard_distance(s1, s2)
        else:
            return self._edit_distance(s1, s2)
    
    def _edit_distance(self, s1: str, s2: str) -> float:
        """
        Compute normalized Levenshtein edit distance.
        
        Returns value in [0, 1] where 0 = identical, 1 = completely different.
        """
        if s1 == s2:
            return 0.0
        
        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 1.0
        
        # Use tokens instead of characters for efficiency
        tokens1 = s1.split()
        tokens2 = s2.split()
        
        n1, n2 = len(tokens1), len(tokens2)
        if n1 == 0 or n2 == 0:
            return 1.0
        
        # DP table
        dp = [[0] * (n2 + 1) for _ in range(n1 + 1)]
        
        for i in range(n1 + 1):
            dp[i][0] = i
        for j in range(n2 + 1):
            dp[0][j] = j
        
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                if tokens1[i - 1] == tokens2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # delete
                        dp[i][j - 1],      # insert
                        dp[i - 1][j - 1]   # replace
                    )
        
        # Normalize by max length
        return dp[n1][n2] / max(n1, n2)
    
    def _jaccard_distance(self, s1: str, s2: str) -> float:
        """
        Compute Jaccard distance (1 - Jaccard similarity).
        
        Treats action strings as sets of tokens.
        """
        set1 = set(s1.split())
        set2 = set(s2.split())
        
        if not set1 and not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return 1.0 - (intersection / union) if union > 0 else 1.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get sampling statistics."""
        return {
            "sample_count": self.sample_count,
            "num_samples_per_call": self.num_samples,
            "temperature": self.temperature,
            "avg_consensus_distance": self.avg_consensus_distance,
        }
    
    def reset_statistics(self):
        """Reset statistics for new episode."""
        self.sample_count = 0
        self.avg_consensus_distance = 0.0
