import random
from typing import Dict, List, Optional, Set, Any
import copy

class DynamicEpisodeScheduler:
    """
    Manages the scheduling of episodes for dynamic context evolution.
    It maintains pools of pending, active, and completed episodes, and determines
    the next batch based on success/failure feedback and a retry policy.
    """

    def __init__(
        self,
        episodes: List[Any],
        batch_size: int = 15,
        retry_ratio: float = 0.5,
        max_retries: int = 3,
        seed: int = 42,
    ):
        """
        Initialize the scheduler.

        :param episodes: List of all episode objects (must have `episode_id` attribute).
        :param batch_size: Number of episodes per batch.
        :param retry_ratio: Target ratio of retried episodes in a batch (0.0 to 1.0).
        :param max_retries: Maximum number of times an episode can be retried.
        :param seed: Random seed for shuffling.
        """
        self.batch_size = batch_size
        self.retry_ratio = retry_ratio
        self.max_retries = max_retries
        self.rng = random.Random(seed)

        # Store full episode objects mapped by ID
        self.episode_map: Dict[str, Any] = {str(ep.episode_id): ep for ep in episodes}
        
        # Queues (storing episode IDs)
        self.pending_ids: List[str] = list(self.episode_map.keys())
        self.retry_queue: List[str] = []
        self.active_batch_ids: Set[str] = set()
        self.completed_ids: Set[str] = set()
        self.failed_ids: Set[str] = set() # Permanently failed after max retries

        # Tracking state
        self.retry_counts: Dict[str, int] = {eid: 0 for eid in self.pending_ids}
        
        # Shuffle initial pending list
        self.rng.shuffle(self.pending_ids)

    def has_pending_work(self) -> bool:
        """Check if there are any episodes left to process."""
        return bool(self.pending_ids or self.retry_queue or self.active_batch_ids)

    def get_next_batch(self) -> List[Any]:
        """
        Construct the next batch of episodes.
        Mixes retries and new episodes based on retry_ratio.
        """
        # If there's an active batch not yet updated, we clear it (assuming clean start or crash recovery)
        # In normal flow, update_results should be called before get_next_batch
        self.active_batch_ids.clear()

        batch_ids = []
        
        # 1. Fill with retries up to ratio
        num_retries = int(self.batch_size * self.retry_ratio)
        # If we don't have enough pending, fill more with retries
        if len(self.pending_ids) < (self.batch_size - num_retries):
             num_retries = self.batch_size - len(self.pending_ids)
        
        # Take from retry queue
        while len(batch_ids) < num_retries and self.retry_queue:
            eid = self.retry_queue.pop(0)
            batch_ids.append(eid)

        # 2. Fill remainder with new episodes
        while len(batch_ids) < self.batch_size and self.pending_ids:
            eid = self.pending_ids.pop(0)
            batch_ids.append(eid)

        # 3. If still space and we have more retries, use them
        while len(batch_ids) < self.batch_size and self.retry_queue:
            eid = self.retry_queue.pop(0)
            batch_ids.append(eid)

        self.active_batch_ids = set(batch_ids)
        
        # Return actual episode objects
        return [self.episode_map[eid] for eid in batch_ids]

    def update_results(self, results: Dict[str, bool]) -> None:
        """
        Update the status of episodes based on execution results.
        
        :param results: Dictionary mapping episode_id (str) to success (bool).
        """
        for eid, success in results.items():
            eid = str(eid)
            if eid not in self.episode_map:
                continue
                
            if success:
                self.completed_ids.add(eid)
            else:
                self.retry_counts[eid] += 1
                if self.retry_counts[eid] < self.max_retries:
                    self.retry_queue.append(eid)
                else:
                    self.failed_ids.add(eid)
            
            # Remove from active set
            if eid in self.active_batch_ids:
                self.active_batch_ids.remove(eid)

    def get_stats(self) -> Dict[str, int]:
        """Return current scheduling statistics."""
        return {
            "pending": len(self.pending_ids),
            "retry_queue": len(self.retry_queue),
            "active": len(self.active_batch_ids),
            "completed": len(self.completed_ids),
            "failed": len(self.failed_ids),
            "total": len(self.episode_map)
        }
