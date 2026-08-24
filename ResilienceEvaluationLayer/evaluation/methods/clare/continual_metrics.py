"""
Continual Learning Metrics Tracker for CLARE.

Tracks BWT (Backward Transfer), FWT (Forward Transfer), and routing accuracy
for measuring continual learning performance.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContinualMetricsTracker:
    """
    Tracks continual learning metrics.
    
    Metrics:
    - BWT (Backward Transfer): Performance change on old tasks after learning new ones
    - FWT (Forward Transfer): Zero-shot performance on new tasks
    - Routing Accuracy: Correct task identification rate
    - Sample Efficiency: Episodes needed to learn new tasks
    
    Pattern reused from: eval_logging.log_resilience_summary()
    """
    
    # Task performance history: {task_id: [success_rate_per_epoch]}
    task_performance: Dict[str, List[float]] = field(default_factory=dict)
    
    # Routing log: list of routing events
    routing_log: List[Dict] = field(default_factory=list)
    
    # Expansion log: list of adapter creations
    expansion_log: List[Dict] = field(default_factory=list)
    
    # Task first-attempt log for FWT
    task_first_attempts: Dict[str, float] = field(default_factory=dict)
    
    # Ground truth tracking (if available)
    ground_truth_matches: int = 0
    total_with_ground_truth: int = 0
    
    # === Recording API ===
    
    def record_task_result(self, task_id: str, success: bool) -> None:
        """
        Record task execution result.
        
        Args:
            task_id: Task/adapter identifier
            success: Whether task succeeded
        """
        if task_id not in self.task_performance:
            self.task_performance[task_id] = []
            # Record first attempt for FWT
            self.task_first_attempts[task_id] = 1.0 if success else 0.0
        
        # Update running success rate
        current = self.task_performance[task_id]
        if current:
            # Weighted update
            new_rate = current[-1] * 0.8 + (1.0 if success else 0.0) * 0.2
        else:
            new_rate = 1.0 if success else 0.0
        
        self.task_performance[task_id].append(new_rate)
    
    def record_routing(
        self,
        instruction: str,
        matched_id: Optional[str],
        is_ood: bool,
        confidence: float,
        ground_truth_id: Optional[str] = None,
    ) -> None:
        """
        Record a routing event.
        
        Args:
            instruction: Task instruction
            matched_id: Matched adapter ID (None if OOD)
            is_ood: Whether detected as OOD
            confidence: Routing confidence score
            ground_truth_id: Optional ground truth task ID
        """
        self.routing_log.append({
            "instruction": instruction[:100],
            "matched_id": matched_id,
            "is_ood": is_ood,
            "confidence": confidence,
            "ground_truth_id": ground_truth_id,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Track accuracy if ground truth available
        if ground_truth_id is not None:
            self.total_with_ground_truth += 1
            if matched_id == ground_truth_id:
                self.ground_truth_matches += 1
    
    def record_expansion(self, task_id: str, n_demos: int) -> None:
        """
        Record adapter expansion event.
        
        Args:
            task_id: New adapter task ID
            n_demos: Number of demonstrations used
        """
        self.expansion_log.append({
            "task_id": task_id,
            "n_demos": n_demos,
            "timestamp": datetime.now().isoformat(),
        })
    
    # === Metric Computation ===
    
    def compute_bwt(self) -> float:
        """
        Compute Backward Transfer (BWT).
        
        BWT measures how learning new tasks affects old task performance.
        Positive BWT = learning new tasks helps old tasks
        Negative BWT = catastrophic forgetting
        
        BWT = (1 / (T-1)) * Σ (R_{T,i} - R_{i,i})
        where R_{T,i} is performance on task i after learning all T tasks
        """
        if len(self.task_performance) < 2:
            return 0.0
        
        bwt_sum = 0.0
        count = 0
        
        for task_id, perfs in self.task_performance.items():
            if len(perfs) >= 2:
                # Compare final performance to initial
                initial = perfs[0]
                final = perfs[-1]
                bwt_sum += final - initial
                count += 1
        
        return bwt_sum / count if count > 0 else 0.0
    
    def compute_fwt(self) -> float:
        """
        Compute Forward Transfer (FWT).
        
        FWT measures zero-shot performance on new tasks.
        Higher FWT = better generalization to new tasks.
        
        FWT = (1 / T) * Σ (first_attempt_i)
        """
        if not self.task_first_attempts:
            return 0.0
        
        return sum(self.task_first_attempts.values()) / len(self.task_first_attempts)
    
    def compute_routing_accuracy(self) -> float:
        """
        Compute routing accuracy.
        
        For routings with ground truth, measures correct matches.
        For routings without ground truth, measures non-OOD confidence.
        """
        if self.total_with_ground_truth > 0:
            return self.ground_truth_matches / self.total_with_ground_truth
        
        # Fallback: use confidence-weighted accuracy
        if not self.routing_log:
            return 0.0
        
        # Count non-OOD routings with high confidence
        high_conf_correct = sum(
            1 for r in self.routing_log
            if not r["is_ood"] and r["confidence"] >= 0.5
        )
        return high_conf_correct / len(self.routing_log)
    
    def compute_sample_efficiency(self) -> Dict[str, float]:
        """
        Compute sample efficiency per task.
        
        Returns demos needed to reach 80% success rate.
        """
        efficiency = {}
        
        for task_id, perfs in self.task_performance.items():
            # Find first epoch where performance >= 0.8
            for epoch, perf in enumerate(perfs):
                if perf >= 0.8:
                    efficiency[task_id] = epoch + 1
                    break
            else:
                # Never reached threshold
                efficiency[task_id] = len(perfs) if perfs else float("inf")
        
        return efficiency
    
    def compute_average_performance(self) -> float:
        """Compute average final performance across all tasks."""
        if not self.task_performance:
            return 0.0
        
        final_perfs = [
            perfs[-1] for perfs in self.task_performance.values() if perfs
        ]
        return sum(final_perfs) / len(final_perfs) if final_perfs else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics summary.
        
        Returns dictionary suitable for adding to info dict.
        """
        return {
            "clare_bwt": self.compute_bwt(),
            "clare_fwt": self.compute_fwt(),
            "clare_routing_accuracy": self.compute_routing_accuracy(),
            "clare_avg_performance": self.compute_average_performance(),
            "clare_adapter_count": len(self.task_performance),
            "clare_total_routings": len(self.routing_log),
            "clare_total_expansions": len(self.expansion_log),
            "clare_ood_rate": self._compute_ood_rate(),
        }
    
    def _compute_ood_rate(self) -> float:
        """Compute OOD detection rate."""
        if not self.routing_log:
            return 0.0
        ood_count = sum(1 for r in self.routing_log if r["is_ood"])
        return ood_count / len(self.routing_log)


def log_clare_metrics(
    info: Dict[str, Any],
    metrics_tracker: ContinualMetricsTracker,
    output_dir: str,
    episode_filename: str,
    env_interface: Any = None,
) -> str:
    """
    Log CLARE metrics to file.
    
    Pattern reused from: eval_logging.log_resilience_summary()
    
    Args:
        info: Episode info dict (metrics added here)
        metrics_tracker: ContinualMetricsTracker instance
        output_dir: Output directory for logs
        episode_filename: Episode filename for log naming
        env_interface: Optional environment interface
        
    Returns:
        Path to saved file
    """
    # Get summary and add to info
    summary = metrics_tracker.get_summary()
    info.update(summary)
    
    # Build detailed log
    sample_eff = metrics_tracker.compute_sample_efficiency()
    
    log_data = {
        "episode_filename": episode_filename,
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "sample_efficiency": sample_eff,
        "task_performance": {
            task_id: {
                "history": perfs[-10:],  # Last 10 entries
                "current": perfs[-1] if perfs else 0.0,
            }
            for task_id, perfs in metrics_tracker.task_performance.items()
        },
        "routing_log": metrics_tracker.routing_log[-50:],  # Last 50 routings
        "expansion_log": metrics_tracker.expansion_log,
    }
    
    # Get episode ID if available
    if env_interface:
        try:
            episode_id = env_interface.env.env.env._env.current_episode.episode_id
            log_data["episode_id"] = episode_id
        except Exception:
            pass
    
    # Save to file
    clare_dir = Path(output_dir) / "analyses" / "clare"
    clare_dir.mkdir(parents=True, exist_ok=True)
    filepath = clare_dir / f"clare_metrics_{episode_filename}.json"
    
    try:
        with open(filepath, "w") as f:
            json.dump(log_data, f, indent=2, default=str)
        logger.info(f"[CLARE] Metrics saved: {filepath}")
    except Exception as e:
        logger.warning(f"[CLARE] Failed to save metrics: {e}")
    
    return str(filepath)
