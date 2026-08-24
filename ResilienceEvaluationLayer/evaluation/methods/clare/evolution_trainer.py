"""
CLARE Evolution Trainer.

Performs offline training of Context Adapters after batch completion.
Pattern reused from: EvolutionContextManager.evolve_batch()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from habitat_llm.planner.clare.adapter_manager import AdapterManager
    from habitat_llm.planner.clare.router import TaskRouter

logger = logging.getLogger(__name__)


@dataclass
class EvolutionResult:
    """Result of adapter evolution/training."""
    task_id: str
    success: bool
    n_episodes_used: int
    context_template: str
    is_new_adapter: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "n_episodes_used": self.n_episodes_used,
            "context_template": self.context_template[:200],
            "is_new_adapter": self.is_new_adapter,
        }


class CLAREEvolutionTrainer:
    """
    Offline trainer for Context Adapters.
    
    Collects episode data during evaluation and trains adapters
    at batch boundaries (offline training).
    
    Pattern reused from: EvolutionContextManager.evolve_batch()
    """
    
    def __init__(
        self,
        adapter_manager: "AdapterManager",
        router: "TaskRouter",
        min_episodes_for_training: int = 3,
        max_patterns_per_adapter: int = 10,
    ) -> None:
        """
        Initialize trainer.
        
        Args:
            adapter_manager: AdapterManager instance
            router: TaskRouter instance
            min_episodes_for_training: Minimum episodes before training
            max_patterns_per_adapter: Max success/failure patterns to keep
        """
        self.adapter_manager = adapter_manager
        self.router = router
        self.min_episodes = min_episodes_for_training
        self.max_patterns = max_patterns_per_adapter
        
        self._pending_episodes: List[Dict] = []
        self._task_groups: Dict[str, List[Dict]] = {}
    
    def record_episode(
        self,
        instruction: str,
        task_id: Optional[str],
        info: Dict[str, Any],
        trajectory: List[Dict],
    ) -> None:
        """
        Record episode for training.
        
        Args:
            instruction: Task instruction
            task_id: Matched task ID (None for OOD)
            info: Episode info dict
            trajectory: Action trajectory
        """
        episode_data = {
            "instruction": instruction,
            "task_id": task_id,
            "info": info,
            "trajectory": trajectory,
            "success": info.get("task_state_success", False),
            "completion": info.get("task_percent_complete", 0.0),
            "timestamp": datetime.now().isoformat(),
        }
        
        self._pending_episodes.append(episode_data)
        
        # Group by instruction similarity
        group_id = self._get_group_id(instruction, task_id)
        if group_id not in self._task_groups:
            self._task_groups[group_id] = []
        self._task_groups[group_id].append(episode_data)
    
    def evolve_batch(self, episode_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Perform offline training after batch completion.
        
        Pattern reused from: EvolutionContextManager.evolve_batch()
        
        Args:
            episode_ids: Episode IDs in the batch
            
        Returns:
            List of evolution results
        """
        results = []
        
        logger.info(
            f"[CLARE Trainer] Evolving batch with {len(self._task_groups)} task groups"
        )
        
        for group_id, episodes in self._task_groups.items():
            if len(episodes) < self.min_episodes:
                logger.debug(
                    f"[CLARE Trainer] Skipping {group_id}: only {len(episodes)} episodes"
                )
                continue
            
            try:
                result = self._train_group(group_id, episodes)
                results.append(result.to_dict())
                logger.info(
                    f"[CLARE Trainer] Trained {group_id}: {result.n_episodes_used} episodes"
                )
            except Exception as e:
                logger.warning(f"[CLARE Trainer] Failed to train {group_id}: {e}")
        
        # Clear pending data
        self._pending_episodes.clear()
        self._task_groups.clear()
        
        return results
    
    def _train_group(
        self,
        group_id: str,
        episodes: List[Dict],
    ) -> EvolutionResult:
        """
        Train a single task group.
        
        Args:
            group_id: Task group identifier
            episodes: Episodes for this group
            
        Returns:
            EvolutionResult
        """
        # Separate successes and failures
        successes = [e for e in episodes if e["success"]]
        failures = [e for e in episodes if not e["success"]]
        
        # Check if adapter exists
        adapter = self.adapter_manager.get_adapter(group_id)
        is_new = adapter is None
        
        if is_new:
            # Create new adapter
            instruction = episodes[0]["instruction"]
            embedding = None
            if self.router.encoder is not None:
                embedding = self.router._compute_embedding(instruction)
            
            adapter = self.adapter_manager.create_adapter(
                task_id=group_id,
                description=self._extract_description(instruction),
                embedding=embedding,
            )
        
        # Extract patterns
        success_patterns = self._extract_success_patterns(successes)
        failure_patterns = self._extract_failure_patterns(failures)
        
        # Generate context template
        context_template = self._generate_context_template(
            episodes[0]["instruction"],
            success_patterns,
            failure_patterns,
            len(successes),
            len(failures),
        )
        
        # Update adapter
        self.adapter_manager.update_adapter(
            task_id=group_id,
            success_patterns=success_patterns,
            failure_patterns=failure_patterns,
            context_template=context_template,
        )
        
        # Register embedding with router
        if is_new and adapter.embedding:
            import numpy as np
            self.router.register_task(group_id, np.array(adapter.embedding))
        
        return EvolutionResult(
            task_id=group_id,
            success=True,
            n_episodes_used=len(episodes),
            context_template=context_template,
            is_new_adapter=is_new,
        )
    
    def _get_group_id(
        self,
        instruction: str,
        existing_task_id: Optional[str],
    ) -> str:
        """
        Get or generate group ID for instruction.
        
        If task_id is known, use it.
        Otherwise, generate from instruction.
        """
        if existing_task_id:
            return existing_task_id
        
        # Generate from instruction
        import hashlib
        words = instruction.lower().split()[:3]
        prefix = "_".join(w[:8] for w in words)
        suffix = hashlib.md5(instruction.encode()).hexdigest()[:6]
        return f"{prefix}_{suffix}"
    
    def _extract_description(self, instruction: str) -> str:
        """Extract short description from instruction."""
        sentences = instruction.split(".")
        return sentences[0][:100] if sentences else instruction[:100]
    
    def _extract_success_patterns(self, successes: List[Dict]) -> List[str]:
        """Extract patterns from successful episodes."""
        patterns = []
        
        for ep in successes[:self.max_patterns]:
            pattern_parts = []
            
            # Add completion info
            completion = ep.get("completion", 0.0)
            pattern_parts.append(f"Completed with {completion:.0%} progress")
            
            # Add trajectory summary
            trajectory = ep.get("trajectory", [])
            if trajectory:
                actions = [t.get("action", "")[:30] for t in trajectory[-3:]]
                if actions:
                    pattern_parts.append(f"Final actions: {', '.join(actions)}")
            
            # Add replanning info
            replanning = ep.get("info", {}).get("replanning_count", 0)
            if replanning:
                pattern_parts.append(f"Used {replanning} replanning steps")
            
            if pattern_parts:
                patterns.append(" | ".join(pattern_parts))
        
        return patterns
    
    def _extract_failure_patterns(self, failures: List[Dict]) -> List[str]:
        """Extract patterns from failed episodes."""
        patterns = []
        
        for ep in failures[:self.max_patterns // 2]:
            pattern_parts = []
            
            # Add completion info
            completion = ep.get("completion", 0.0)
            pattern_parts.append(f"Stopped at {completion:.0%} progress")
            
            # Add last action if available
            trajectory = ep.get("trajectory", [])
            if trajectory:
                last_action = trajectory[-1].get("action", "")[:50]
                last_response = trajectory[-1].get("response", "")[:50]
                if last_action:
                    pattern_parts.append(f"Last action: {last_action}")
                if "fail" in last_response.lower() or "error" in last_response.lower():
                    pattern_parts.append(f"Error: {last_response}")
            
            if pattern_parts:
                patterns.append(" | ".join(pattern_parts))
        
        return patterns
    
    def _generate_context_template(
        self,
        instruction: str,
        success_patterns: List[str],
        failure_patterns: List[str],
        n_successes: int,
        n_failures: int,
    ) -> str:
        """
        Generate context template for adapter.
        
        This can be enhanced with LLM summarization.
        """
        parts = []
        
        # Task description
        parts.append(f"Task Type: {self._extract_description(instruction)}")
        
        # Success rate
        total = n_successes + n_failures
        if total > 0:
            rate = n_successes / total
            parts.append(f"Historical Success Rate: {rate:.0%} ({n_successes}/{total})")
        
        # Success patterns
        if success_patterns:
            parts.append("\nSuccessful Approaches:")
            for pattern in success_patterns[:3]:
                parts.append(f"  • {pattern}")
        
        # Failure patterns
        if failure_patterns:
            parts.append("\nCommon Pitfalls to Avoid:")
            for pattern in failure_patterns[:2]:
                parts.append(f"  ✗ {pattern}")
        
        # Guidance
        if n_successes > 0:
            parts.append("\nRecommendation: Follow the successful approaches above.")
        elif n_failures > 0:
            parts.append("\nRecommendation: Avoid the failure patterns listed above.")
        
        return "\n".join(parts)
