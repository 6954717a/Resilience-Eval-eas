"""
Context-based Adapter Manager for CLARE.

Manages a library of Context Adapters, each representing learned task knowledge
as prompt templates rather than LoRA weights.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ContextAdapter:
    """
    Context-based Adapter: stores task knowledge as prompt templates.
    
    Unlike LoRA adapters that modify model weights, Context Adapters
    inject learned patterns directly into the prompt.
    """
    task_id: str
    description: str
    success_patterns: List[str] = field(default_factory=list)
    failure_patterns: List[str] = field(default_factory=list)
    context_template: str = ""
    embedding: Optional[List[float]] = None
    performance: Dict[str, float] = field(default_factory=lambda: {
        "success_rate": 0.0,
        "total_episodes": 0,
        "successful_episodes": 0,
    })
    created_at: str = ""
    updated_at: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextAdapter":
        """Create from dictionary."""
        return cls(**data)


class AdapterManager:
    """
    Manages Context-based Adapter library.
    
    Provides CRUD operations for adapters and persistent storage.
    Pattern reused from: EvolutionContextManager._save()/_load()
    """
    
    def __init__(
        self,
        output_dir: Path,
        max_adapters: int = 20,
    ) -> None:
        """
        Initialize AdapterManager.
        
        Args:
            output_dir: Directory for adapter storage
            max_adapters: Maximum number of adapters to maintain
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_adapters = max_adapters
        self.adapters: Dict[str, ContextAdapter] = {}
        self._registry_path = self.output_dir / "clare_adapters.json"
        self._load_registry()
    
    # === Core API ===
    
    def get_adapter(self, task_id: str) -> Optional[ContextAdapter]:
        """Get adapter by task ID."""
        return self.adapters.get(task_id)
    
    def create_adapter(
        self,
        task_id: str,
        description: str,
        embedding: Optional[np.ndarray] = None,
    ) -> ContextAdapter:
        """
        Create a new adapter.
        
        Args:
            task_id: Unique identifier for the task
            description: Human-readable task description
            embedding: Optional embedding vector for routing
            
        Returns:
            Newly created ContextAdapter
        """
        if len(self.adapters) >= self.max_adapters:
            # Remove oldest adapter with lowest success rate
            self._evict_adapter()
        
        now = datetime.now().isoformat()
        adapter = ContextAdapter(
            task_id=task_id,
            description=description,
            embedding=embedding.tolist() if embedding is not None else None,
            created_at=now,
            updated_at=now,
        )
        self.adapters[task_id] = adapter
        self._save_registry()
        logger.info(f"[CLARE] Created adapter: {task_id}")
        return adapter
    
    def update_adapter(
        self,
        task_id: str,
        success_patterns: Optional[List[str]] = None,
        failure_patterns: Optional[List[str]] = None,
        context_template: Optional[str] = None,
        episode_success: Optional[bool] = None,
    ) -> Optional[ContextAdapter]:
        """
        Update an existing adapter.
        
        Args:
            task_id: Adapter to update
            success_patterns: New success patterns to add
            failure_patterns: New failure patterns to add
            context_template: New context template
            episode_success: Record episode outcome
            
        Returns:
            Updated adapter or None if not found
        """
        adapter = self.adapters.get(task_id)
        if not adapter:
            return None
        
        if success_patterns:
            adapter.success_patterns.extend(success_patterns)
            # Keep only recent patterns
            adapter.success_patterns = adapter.success_patterns[-10:]
        
        if failure_patterns:
            adapter.failure_patterns.extend(failure_patterns)
            adapter.failure_patterns = adapter.failure_patterns[-5:]
        
        if context_template is not None:
            adapter.context_template = context_template
        
        if episode_success is not None:
            adapter.performance["total_episodes"] += 1
            if episode_success:
                adapter.performance["successful_episodes"] += 1
            total = adapter.performance["total_episodes"]
            success = adapter.performance["successful_episodes"]
            adapter.performance["success_rate"] = success / total if total > 0 else 0.0
        
        adapter.updated_at = datetime.now().isoformat()
        self._save_registry()
        return adapter
    
    def get_context_for_prompt(self, task_id: str) -> str:
        """
        Get context string to inject into prompt.
        
        Args:
            task_id: Adapter task ID
            
        Returns:
            Formatted context string for prompt injection
        """
        adapter = self.adapters.get(task_id)
        if not adapter:
            return ""
        
        # Use custom template if available
        if adapter.context_template:
            return adapter.context_template
        
        # Generate from patterns
        parts = []
        if adapter.description:
            parts.append(f"Task Type: {adapter.description}")
        
        if adapter.success_patterns:
            parts.append("Successful approaches for similar tasks:")
            for pattern in adapter.success_patterns[-3:]:
                parts.append(f"  - {pattern}")
        
        if adapter.failure_patterns:
            parts.append("Common pitfalls to avoid:")
            for pattern in adapter.failure_patterns[-2:]:
                parts.append(f"  - {pattern}")
        
        if adapter.performance["total_episodes"] > 0:
            rate = adapter.performance["success_rate"]
            parts.append(f"Historical success rate: {rate:.1%}")
        
        return "\n".join(parts)
    
    def list_adapters(self) -> List[str]:
        """List all adapter task IDs."""
        return list(self.adapters.keys())
    
    def get_adapter_embeddings(self) -> Dict[str, np.ndarray]:
        """Get embeddings for all adapters with embeddings."""
        result = {}
        for task_id, adapter in self.adapters.items():
            if adapter.embedding:
                result[task_id] = np.array(adapter.embedding)
        return result
    
    # === Persistence (reused from EvolutionContextManager) ===
    
    def _load_registry(self) -> None:
        """Load adapters from disk."""
        if not self._registry_path.exists():
            return
        
        try:
            with open(self._registry_path, "r") as f:
                data = json.load(f)
            
            for task_id, adapter_data in data.get("adapters", {}).items():
                self.adapters[task_id] = ContextAdapter.from_dict(adapter_data)
            
            logger.info(f"[CLARE] Loaded {len(self.adapters)} adapters")
        except Exception as e:
            logger.warning(f"[CLARE] Failed to load adapters: {e}")
    
    def _save_registry(self) -> None:
        """Save adapters to disk."""
        data = {
            "adapters": {
                task_id: adapter.to_dict()
                for task_id, adapter in self.adapters.items()
            },
            "updated_at": datetime.now().isoformat(),
        }
        
        try:
            with open(self._registry_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"[CLARE] Failed to save adapters: {e}")
    
    def _evict_adapter(self) -> None:
        """Remove the adapter with lowest utility (lowest success rate + oldest)."""
        if not self.adapters:
            return
        
        # Score by success rate (higher is better) and recency
        def score(adapter: ContextAdapter) -> float:
            rate = adapter.performance.get("success_rate", 0.0)
            episodes = adapter.performance.get("total_episodes", 0)
            # Prefer adapters with more data and higher success
            return rate * 0.7 + min(episodes / 10, 1.0) * 0.3
        
        worst_id = min(self.adapters.keys(), key=lambda k: score(self.adapters[k]))
        del self.adapters[worst_id]
        logger.info(f"[CLARE] Evicted adapter: {worst_id}")
