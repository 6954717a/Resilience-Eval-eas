"""
Expansion Controller for CLARE.

Controls when to create new adapters based on OOD detection patterns.
Uses cooldown periods to prevent excessive expansion.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING
from datetime import datetime

import numpy as np

if TYPE_CHECKING:
    from habitat_llm.planner.clare.router import RouterResult
    from habitat_llm.planner.clare.adapter_manager import AdapterManager

logger = logging.getLogger(__name__)


@dataclass
class ExpansionEvent:
    """Record of an expansion event."""
    task_id: str
    instruction: str
    step: int
    timestamp: str


class ExpansionController:
    """
    Controls adapter expansion (creation of new adapters).
    
    Tracks consecutive OOD occurrences and applies cooldown periods
    to prevent excessive expansion.
    Pattern reused from: ReboundManager.filter_faults()
    """
    
    def __init__(
        self,
        min_ood_count: int = 3,
        cooldown_steps: int = 10,
        expand_threshold: float = 0.3,
        max_adapters: int = 20,
    ) -> None:
        """
        Initialize ExpansionController.
        
        Args:
            min_ood_count: Minimum consecutive OOD detections to trigger expansion
            cooldown_steps: Steps to wait after expansion before allowing another
            expand_threshold: Confidence threshold below which to consider for expansion
            max_adapters: Maximum adapters (expansion stops when reached)
        """
        self.min_ood_count = min_ood_count
        self.cooldown_steps = cooldown_steps
        self.expand_threshold = expand_threshold
        self.max_adapters = max_adapters
        
        self._ood_buffer: List["RouterResult"] = []
        self._last_expansion_step: int = -self.cooldown_steps  # Allow immediate first expansion
        self._current_step: int = 0
        self._expansion_history: List[ExpansionEvent] = []
        self._pending_instruction: Optional[str] = None
        self._pending_embedding: Optional[np.ndarray] = None
    
    def should_expand(self, result: "RouterResult") -> bool:
        """
        Check if expansion should be triggered.
        
        Args:
            result: RouterResult from TaskRouter
            
        Returns:
            True if a new adapter should be created
        """
        if not result.is_ood:
            # Not OOD, clear buffer
            self._ood_buffer.clear()
            return False
        
        # Add to OOD buffer
        self._ood_buffer.append(result)
        
        # Check cooldown
        steps_since_expansion = self._current_step - self._last_expansion_step
        if steps_since_expansion < self.cooldown_steps:
            logger.debug(
                f"[CLARE Expansion] In cooldown ({steps_since_expansion}/{self.cooldown_steps})"
            )
            return False
        
        # Check consecutive OOD count
        if len(self._ood_buffer) < self.min_ood_count:
            logger.debug(
                f"[CLARE Expansion] OOD buffer: {len(self._ood_buffer)}/{self.min_ood_count}"
            )
            return False
        
        # Check confidence threshold
        avg_confidence = sum(r.confidence for r in self._ood_buffer) / len(self._ood_buffer)
        if avg_confidence > self.expand_threshold:
            logger.debug(
                f"[CLARE Expansion] Avg confidence {avg_confidence:.3f} > threshold"
            )
            return False
        
        return True
    
    def trigger_expansion(
        self,
        adapter_manager: "AdapterManager",
        instruction: str,
        embedding: Optional[np.ndarray],
    ) -> str:
        """
        Trigger creation of a new adapter.
        
        Args:
            adapter_manager: AdapterManager to create adapter in
            instruction: Instruction text for the new task
            embedding: Embedding vector for the task
            
        Returns:
            New task_id
        """
        # Check adapter limit
        if len(adapter_manager.adapters) >= self.max_adapters:
            logger.warning("[CLARE Expansion] Max adapters reached, triggering eviction")
        
        # Generate task ID
        task_id = self._generate_task_id(instruction)
        
        # Create adapter
        description = self._extract_description(instruction)
        adapter_manager.create_adapter(
            task_id=task_id,
            description=description,
            embedding=embedding,
        )
        
        # Record expansion
        self._last_expansion_step = self._current_step
        self._ood_buffer.clear()
        self._expansion_history.append(ExpansionEvent(
            task_id=task_id,
            instruction=instruction[:200],
            step=self._current_step,
            timestamp=datetime.now().isoformat(),
        ))
        
        logger.info(f"[CLARE Expansion] Created adapter: {task_id}")
        return task_id
    
    def step(self) -> None:
        """Increment step counter."""
        self._current_step += 1
    
    def reset(self) -> None:
        """Reset controller state for new episode."""
        self._ood_buffer.clear()
    
    def set_pending(self, instruction: str, embedding: Optional[np.ndarray]) -> None:
        """Set pending task for offline expansion."""
        self._pending_instruction = instruction
        self._pending_embedding = embedding
    
    def get_pending(self) -> Optional[tuple]:
        """Get and clear pending task."""
        if self._pending_instruction is None:
            return None
        result = (self._pending_instruction, self._pending_embedding)
        self._pending_instruction = None
        self._pending_embedding = None
        return result
    
    def get_expansion_stats(self) -> dict:
        """Get expansion statistics."""
        return {
            "total_expansions": len(self._expansion_history),
            "current_step": self._current_step,
            "ood_buffer_size": len(self._ood_buffer),
            "cooldown_remaining": max(
                0, self.cooldown_steps - (self._current_step - self._last_expansion_step)
            ),
        }
    
    def _generate_task_id(self, instruction: str) -> str:
        """Generate a unique task ID from instruction."""
        import hashlib
        # Take first few words + hash
        words = instruction.split()[:3]
        prefix = "_".join(w.lower()[:8] for w in words)
        suffix = hashlib.md5(instruction.encode()).hexdigest()[:6]
        return f"{prefix}_{suffix}"
    
    def _extract_description(self, instruction: str) -> str:
        """Extract a short description from instruction."""
        # Take first sentence or first 100 chars
        sentences = instruction.split(".")
        if sentences:
            return sentences[0][:100]
        return instruction[:100]
