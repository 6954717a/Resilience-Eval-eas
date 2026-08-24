"""
Task Router for CLARE.

Routes incoming tasks to appropriate adapters based on semantic similarity.
Detects Out-of-Distribution (OOD) tasks that require new adapters.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from habitat_llm.planner.clare.adapter_manager import AdapterManager

logger = logging.getLogger(__name__)


@dataclass
class RouterResult:
    """Result of task routing."""
    matched_task_id: Optional[str]  # Matched adapter ID (None if OOD)
    confidence: float               # Similarity score [0, 1]
    is_ood: bool                   # True if Out-of-Distribution
    embedding: Optional[np.ndarray] = None  # Task embedding


class TaskRouter:
    """
    Routes tasks to appropriate adapters.
    
    Uses SentenceTransformer embeddings (reused from StateEncoder)
    to compute semantic similarity between tasks.
    """
    
    def __init__(
        self,
        adapter_manager: "AdapterManager",
        ood_threshold: float = 0.5,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        """
        Initialize TaskRouter.
        
        Args:
            adapter_manager: Reference to AdapterManager
            ood_threshold: Similarity threshold below which task is OOD
            embedding_model: SentenceTransformer model name
        """
        self.adapter_manager = adapter_manager
        self.ood_threshold = ood_threshold
        self.embedding_model = embedding_model
        self._encoder = None  # Lazy initialization
        self._routing_history: List[Dict] = []
    
    @property
    def encoder(self):
        """Lazy load SentenceTransformer to avoid import overhead."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.embedding_model)
                logger.info(f"[CLARE Router] Loaded encoder: {self.embedding_model}")
            except ImportError:
                logger.warning("[CLARE Router] sentence-transformers not available")
                self._encoder = None
        return self._encoder
    
    def route(self, instruction: str) -> RouterResult:
        """
        Route a task to the best matching adapter.
        
        Args:
            instruction: Task instruction text
            
        Returns:
            RouterResult with match info and OOD status
        """
        # Compute embedding
        embedding = self._compute_embedding(instruction)
        
        if embedding is None:
            # Fallback: no encoder available
            return RouterResult(
                matched_task_id=None,
                confidence=0.0,
                is_ood=True,
                embedding=None,
            )
        
        # Find best match
        matched_id, confidence = self._find_best_match(embedding)
        
        # Determine if OOD
        is_ood = confidence < self.ood_threshold or matched_id is None
        
        result = RouterResult(
            matched_task_id=matched_id if not is_ood else None,
            confidence=confidence,
            is_ood=is_ood,
            embedding=embedding,
        )
        
        # Log routing
        self._routing_history.append({
            "instruction": instruction[:100],
            "matched_id": matched_id,
            "confidence": confidence,
            "is_ood": is_ood,
        })
        
        if is_ood:
            logger.debug(f"[CLARE Router] OOD task (conf={confidence:.3f}): {instruction[:50]}...")
        else:
            logger.debug(f"[CLARE Router] Matched {matched_id} (conf={confidence:.3f})")
        
        return result
    
    def register_task(self, task_id: str, embedding: np.ndarray) -> None:
        """
        Register a new task embedding.
        
        Args:
            task_id: Adapter task ID
            embedding: Task embedding vector
        """
        adapter = self.adapter_manager.get_adapter(task_id)
        if adapter:
            adapter.embedding = embedding.tolist()
            self.adapter_manager._save_registry()
    
    def get_routing_stats(self) -> Dict[str, float]:
        """Get routing statistics."""
        if not self._routing_history:
            return {
                "total_routings": 0,
                "ood_rate": 0.0,
                "avg_confidence": 0.0,
            }
        
        total = len(self._routing_history)
        ood_count = sum(1 for r in self._routing_history if r["is_ood"])
        avg_conf = sum(r["confidence"] for r in self._routing_history) / total
        
        return {
            "total_routings": total,
            "ood_rate": ood_count / total,
            "avg_confidence": avg_conf,
        }
    
    def _compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """Compute text embedding using SentenceTransformer."""
        if self.encoder is None:
            return None
        
        try:
            embedding = self.encoder.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.warning(f"[CLARE Router] Embedding failed: {e}")
            return None
    
    def _find_best_match(self, query_emb: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Find the adapter with highest similarity.
        
        Args:
            query_emb: Query embedding vector
            
        Returns:
            (task_id, similarity) tuple, or (None, 0.0) if no matches
        """
        adapter_embeddings = self.adapter_manager.get_adapter_embeddings()
        
        if not adapter_embeddings:
            return None, 0.0
        
        best_id = None
        best_sim = 0.0
        
        for task_id, emb in adapter_embeddings.items():
            sim = self._cosine_similarity(query_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_id = task_id
        
        return best_id, best_sim
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
