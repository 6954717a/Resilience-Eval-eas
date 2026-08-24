#!/usr/bin/env python3

"""
Offline Metrics Aggregator

This script reads episode logs (planner info, critic stats) and aggregates them
to compute Level 2 (Task-Family) and Level 3 (Evolve-Level) Resilience Metrics.
"""

import json
import os
import glob
import numpy as np
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class ResilienceMetricsAggregator:
    def __init__(self, analysis_dir: str):
        self.analysis_dir = analysis_dir
        self.episode_metrics: List[Dict[str, Any]] = []

    def load_logs(self):
        """Load all available logs from the analysis directory."""
        # Pattern matching for planner info or general analysis files
        # Assuming evaluation_runner saves log file as *trace*.json or similar
        # For this example, we scan for files containing resilience metrics
        
        # In reality, logs are usually "episode_X_trace.json" or similar.
        # Let's assume standard habitat-llm log structure or the ones we just enriched.
        log_files = glob.glob(os.path.join(self.analysis_dir, "**", "*.json"), recursive=True)
        
        for file_path in log_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Check if this file contains the metrics we need
                    # Usually 'planner_infos' is a list of steps
                    if "planner_infos" in data:
                        self.process_episode(data)
                    elif "rebound_analysis" in data:
                        self.process_rebound_data(data)
            except Exception as e:
                logger.debug(f"Skipping file {file_path}: {e}")

    def process_rebound_data(self, data: Dict[str, Any]):
        """Extract Rebound metrics from rebound_analysis log."""
        # data["rebound_analysis"] is a dict of agent_uid -> metrics
        analysis = data.get("rebound_analysis", {})
        if "status" in analysis: return
        
        for agent_id, content in analysis.items():
            if not isinstance(content, dict): continue
            
            # Extract failure history: list of [start, end, strategy]
            history = content.get("failure_history", [])
            for entry in history:
                if len(entry) >= 2:
                    start, end = entry[0], entry[1]
                    t_rec = end - start
                    
                    # Store T_rec for Level 2 aggregation
                    # We need to associate this with an episode struct or just a global list
                    # Since we aggregate mainly distributions, adding to a list is fine.
                    # But to keep structure consistent, let's create a dummy ep_stat or extend existing?
                    # Simpler: Self-contained list in class
                    if not hasattr(self, "rebound_t_rec_list"):
                         self.rebound_t_rec_list = []
                    self.rebound_t_rec_list.append(t_rec)

    def process_episode(self, data: Dict[str, Any]):
        """Extract Level 1 metrics from a single episode log."""
        steps = data.get("planner_infos", [])
        if not steps:
            return

        # Initialize episode stats
        ep_stat = {
            "cbf_penalties": [],
            "is_safe": True,
            "auc_loss": 0.0,
            "cliff_detected": False,
            "value_variance": []
        }

        for step in steps:
            # Rebound: T_rec (if recorded in step)
            if "rebound_metrics" in step:
                 # Assuming ReboundManager logs T_rec here
                 # This requires ReboundManager to populate 'rebound_metrics'
                 pass
            
            # Safety: CBF
            if "cbf_penalty" in step:
                val = step["cbf_penalty"]
                ep_stat["cbf_penalties"].append(val)
                if val > 0:
                    ep_stat["is_safe"] = False
            
            # Degradation: AUC, Cliff
            if "auc_loss" in step:
                ep_stat["auc_loss"] = max(ep_stat["auc_loss"], step["auc_loss"])
            if "cliff_detected" in step and step["cliff_detected"] > 0.5:
                ep_stat["cliff_detected"] = True

            # Stability: Value Variance
            if "value_function_variance" in step:
                ep_stat["value_variance"].append(step["value_function_variance"])

        self.episode_metrics.append(ep_stat)

    def compute_level_2_metrics(self) -> Dict[str, float]:
        """Compute Level 2: Task-Family Metrics."""
        if not self.episode_metrics:
            return {}

        N = len(self.episode_metrics)

        # 1. Safety Score (S_F): Proportion of safe episodes
        safe_count = sum(1 for e in self.episode_metrics if e["is_safe"])
        safety_score = safe_count / N

        # 2. MTTR-A
        # Flatten all T_rec values from all episodes including standalone rebound logs
        all_t_rec = getattr(self, "rebound_t_rec_list", [])
        # Also check episode_metrics if any (though we moved logic to rebound processing)
        for e in self.episode_metrics:
            all_t_rec.extend(e.get("T_rec", []))
        
        mttr_a = np.mean(all_t_rec) if all_t_rec else 0.0

        # 3. MTBF-A (Mean Time Between Failures)
        # We can approximate this as (Total Steps - Total T_rec) / Total Failures
        total_steps = sum(e.get("total_steps", 0) for e in self.episode_metrics) # Need to ensure total_steps is captured
        total_failure_time = sum(all_t_rec)
        total_failures = len(all_t_rec)
        
        if total_failures > 0:
            mtbf_a = (total_steps - total_failure_time) / total_failures
        else:
            mtbf_a = float(total_steps) # No failures

        # 4. Avg Value Variance (Stability)
        all_vars = [np.mean(e["value_variance"]) for e in self.episode_metrics if e["value_variance"]]
        stability_score = np.mean(all_vars) if all_vars else 0.0

        return {
            "L2_Safety_Score": safety_score,
            "L2_Stability_Var": stability_score,
            "L2_MTTR": mttr_a,
            "L2_MTBF": mtbf_a,
            "Episode_Count": N
        }

    def compute_level_3_metrics(self, history_data: List[Dict[str, float]], current_version_perf: float = 0.0, initial_version_perf: float = 0.0) -> Dict[str, float]:
        """
        Compute Level 3: Evolve metrics.
        
        :param history_data: List of L2 metrics from previous versions.
        :param current_version_perf: Performance scalar of current agent.
        :param initial_version_perf: Performance scalar of initial agent (v0).
        """
        current_l2 = self.compute_level_2_metrics()
        
        # 1. Normalized Resilience (NRR)
        # NRR = MTBF / (MTBF + MTTR)
        mtbf = current_l2.get("L2_MTBF", 0.0)
        mttr = current_l2.get("L2_MTTR", 0.0)
        nrr = 0.0
        if mtbf + mttr > 0:
            nrr = mtbf / (mtbf + mttr)
            
        # 2. Evolve Lower Bound (L_bd)
        # Simple check: Perf(k) >= Perf(0) - epsilon
        # We return the margin
        lower_bound_margin = current_version_perf - initial_version_perf
        
        return {
            "L3_NRR": nrr,
            "L3_Evolve_Margin": lower_bound_margin,
            "Current_L2": current_l2
        }

if __name__ == "__main__":
    # Example usage
    agg = ResilienceMetricsAggregator("./analyses")
    agg.load_logs()
    print("Level 2 Metrics:", agg.compute_level_2_metrics())
