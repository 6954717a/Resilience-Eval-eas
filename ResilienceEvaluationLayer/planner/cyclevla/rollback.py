"""
Cognitive Rollback Tool for CycleVLA.

This module implements context-level backtracking as an alternative to
CycleVLA's physical rollback. Since Habitat-LLM uses LLM-based planning
(not continuous VLA actions), we cannot reverse-execute delta actions.

Instead, we implement cognitive backtracking:
1. Save planner context (prompt, trace) at subtask boundaries
2. On backtrack, restore context to the target subtask's checkpoint
3. Inject guidance to help the LLM avoid previous mistakes

Reference: CycleVLA Paper Section IV-B
- "When backtracking is triggered, we restore the robot state to the
   beginning of the target subtask by reverse-executing recorded delta actions"
- Our adaptation: Restore planner context instead of physical state
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class CheckpointData:
    """
    Snapshot of planner state at subtask start.
    
    Used for cognitive backtracking - restoring the planner's
    context to a previous state when backtracking.
    """
    prompt: str
    trace: str
    replanning_count: int
    world_state_summary: Dict[str, Any] = field(default_factory=dict)
    graph_snapshot_hash: str = ""
    subtask_idx: int = 0
    step_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize checkpoint."""
        return {
            "prompt_length": len(self.prompt),
            "trace_length": len(self.trace),
            "replanning_count": self.replanning_count,
            "graph_snapshot_hash": self.graph_snapshot_hash,
            "subtask_idx": self.subtask_idx,
            "step_count": self.step_count,
        }


class CognitiveRollbackTool:
    """
    Cognitive backtracking tool for CycleVLA.
    
    This implements context-level backtracking as an alternative to
    physical state rollback. When a backtrack is triggered:
    
    1. Restore planner's prompt and trace to the checkpoint state
    2. Inject "Backtrack Guidance" explaining what went wrong
    3. Let the LLM re-plan from that point with awareness of the failure
    
    Key difference from CycleVLA:
    - CycleVLA: Physical rollback (reverse delta actions, actual state reset)
    - Our approach: Cognitive rollback (prompt/trace reset, guidance injection)
    
    The environment's physical state is NOT restored. The LLM must adapt
    to the current physical state while being informed of the context reset.
    """
    
    # Backtrack guidance template
    BACKTRACK_GUIDANCE_TEMPLATE = '''[CycleVLA Backtrack Notice]
A potential failure was detected in a later subtask. The planning context has been 
restored to Subtask {target_idx}: "{target_instruction}".

Failure Context:
- Backtracked from Subtask {from_idx}
- Reason: {reason}
- VLM confidence: {confidence}

Recovery Guidance:
Please reconsider the approach from this point. Consider:
1. What might have gone wrong after this subtask?
2. Are there alternative methods to achieve the same goal?
3. Should preconditions be verified more carefully?

Continue planning from Subtask {target_idx}.'''
    
    def __init__(self, planner: Any):
        """
        Initialize CognitiveRollbackTool.
        
        Args:
            planner: LLMPlanner instance to manage checkpoints for
        """
        self.planner = planner
        self.checkpoints: Dict[int, CheckpointData] = {}
        
        # Statistics
        self.rollback_count = 0
        self.checkpoint_count = 0
    
    def save_checkpoint(
        self, 
        subtask_idx: int, 
        world_state: Dict[str, Any],
        step_count: int = 0
    ):
        """
        Save checkpoint at subtask start.
        
        Called when entering a new subtask to save the planner's
        current context for potential future rollback.
        
        Args:
            subtask_idx: Index of the subtask starting
            world_state: Current world state dictionary
            step_count: Current step count
        """
        self.checkpoints[subtask_idx] = CheckpointData(
            prompt=copy.deepcopy(self.planner.curr_prompt),
            trace=copy.deepcopy(self.planner.trace),
            replanning_count=self.planner.replanning_count,
            world_state_summary=self._summarize_world_state(world_state),
            graph_snapshot_hash=str(world_state.get("graph_snapshot_hash", "")),
            subtask_idx=subtask_idx,
            step_count=step_count,
        )
        self.checkpoint_count += 1
        
        logger.debug(
            f"Saved checkpoint for subtask {subtask_idx}, "
            f"prompt_len={len(self.planner.curr_prompt)}, "
            f"trace_len={len(self.planner.trace)}"
        )
    
    def _summarize_world_state(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create a compact summary of world state for checkpoint."""
        summary = {}
        
        if world_state.get("agent_holding"):
            summary["agent_holding"] = world_state["agent_holding"]
        if world_state.get("graph_snapshot_hash"):
            summary["graph_snapshot_hash"] = world_state["graph_snapshot_hash"]
        
        if world_state.get("objects"):
            # Just store object names and locations
            summary["object_locations"] = {
                name: info.get("location", "unknown")
                for name, info in list(world_state.get("objects", {}).items())[:10]
            }
        
        return summary
    
    def rollback_to(
        self, 
        subtask_idx: int,
        from_subtask_idx: Optional[int] = None,
        reason: str = "VLM predicted failure",
        confidence: str = "medium",
        subtask_instruction: str = ""
    ) -> bool:
        """
        Rollback to specified subtask's checkpoint.
        
        This is the core cognitive backtracking method:
        1. Restore prompt and trace from checkpoint
        2. Inject backtrack guidance
        
        Args:
            subtask_idx: Target subtask index to rollback to
            from_subtask_idx: Subtask where failure was detected (for logging)
            reason: Reason for backtrack
            confidence: VLM confidence level
            subtask_instruction: Instruction of target subtask (for guidance)
        
        Returns:
            True if rollback successful, False if checkpoint not found
        """
        if subtask_idx not in self.checkpoints:
            logger.warning(f"No checkpoint found for subtask {subtask_idx}")
            return False
        
        cp = self.checkpoints[subtask_idx]
        
        # Restore planner state
        self.planner.curr_prompt = cp.prompt
        self.planner.trace = cp.trace
        
        # Build and inject backtrack guidance
        guidance = self.BACKTRACK_GUIDANCE_TEMPLATE.format(
            target_idx=subtask_idx,
            target_instruction=subtask_instruction or f"Subtask {subtask_idx}",
            from_idx=from_subtask_idx or "later",
            reason=reason,
            confidence=confidence,
        )
        
        self.planner.enqueue_context_update("CycleVLA Backtrack", guidance)
        if not hasattr(self.planner, "cyclevla_events"):
            self.planner.cyclevla_events = []
        self.planner.cyclevla_events.append(
            {
                "event": "rollback",
                "target_subtask_idx": subtask_idx,
                "from_subtask_idx": from_subtask_idx,
                "reason": reason,
                "confidence": confidence,
                "graph_snapshot_hash": cp.graph_snapshot_hash,
            }
        )

        # Update statistics
        self.rollback_count += 1
        
        logger.info(
            f"Cognitive rollback to subtask {subtask_idx} from {from_subtask_idx}, "
            f"reason: {reason}"
        )
        
        return True
    
    def clear_checkpoints_after(self, subtask_idx: int):
        """
        Remove checkpoints after given index.
        
        Called after rollback to clean up invalid future checkpoints.
        
        Args:
            subtask_idx: Keep checkpoints up to and including this index
        """
        to_remove = [k for k in self.checkpoints if k > subtask_idx]
        for k in to_remove:
            del self.checkpoints[k]
        
        if to_remove:
            logger.debug(f"Cleared {len(to_remove)} checkpoints after subtask {subtask_idx}")
    
    def has_checkpoint(self, subtask_idx: int) -> bool:
        """Check if checkpoint exists for subtask."""
        return subtask_idx in self.checkpoints
    
    def get_checkpoint_info(self, subtask_idx: int) -> Optional[Dict[str, Any]]:
        """Get info about a checkpoint without restoring it."""
        if subtask_idx not in self.checkpoints:
            return None
        return self.checkpoints[subtask_idx].to_dict()
    
    def get_all_checkpoints(self) -> Dict[int, Dict[str, Any]]:
        """Get info about all checkpoints."""
        return {
            idx: cp.to_dict() 
            for idx, cp in self.checkpoints.items()
        }
    
    def reset(self):
        """Reset all checkpoints for new episode."""
        self.checkpoints.clear()
        self.rollback_count = 0
        self.checkpoint_count = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rollback statistics."""
        return {
            "checkpoint_count": self.checkpoint_count,
            "rollback_count": self.rollback_count,
            "active_checkpoints": len(self.checkpoints),
        }
