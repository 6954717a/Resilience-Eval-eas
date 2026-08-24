#!/usr/bin/env python3

"""
Resilience Metrics Logging

Comprehensive logging for resilience metrics from Rebound, CycleVLA, and Inner Monologue.
This module collects and logs all resilience-related metrics in a unified format,
aligning with evaluation metrics that measure at Planning Step level.

Reference:
- Rebound: MTTR, MTBF, Recovery Ratio (RR), B_epi
- CycleVLA: Backtrack count/rate, MBR usage, failure prediction accuracy
- Inner Monologue: Feedback success rate, thought generation rate, correction effectiveness
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """JSON serializer for non-serializable objects."""
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return obj.tolist()
        except Exception:
            pass
    return str(obj)


def _write_json(filepath: str, data: Dict[str, Any]) -> None:
    """Safely write JSON file with atomic write."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp_", suffix=".json", dir=os.path.dirname(filepath)
    )
    try:
        with os.fdopen(tmp_fd, "w") as file:
            json.dump(data, file, indent=2, default=_json_default)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_path, filepath)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def collect_rebound_metrics(
    planner: Any,
    total_planning_steps: int
) -> Dict[str, Any]:
    """
    Collect Rebound resilience metrics.
    
    Metrics:
    - rebound_mttr: Mean Time To Recovery (planning steps)
    - rebound_mtbf: Mean Time Between Failures (planning steps)
    - rebound_count: Total number of rebound events
    - recovery_ratio: Recovery Ratio (RR) = errors_fixed / errors_occurred
    - b_epi: Backtrack penalty = retry_count + gamma * backtrack_count
    - fault_distribution: Distribution of fault types
    
    Args:
        planner: LLMPlanner instance with rebound_managers
        total_planning_steps: Total number of planning steps in episode
    
    Returns:
        Dictionary with rebound metrics aggregated across all agents
    """
    rebound_metrics = {
        "rebound_mttr": 0.0,
        "rebound_mtbf": 0.0,
        "rebound_count": 0.0,
        "recovery_ratio": 0.0,
        "b_epi": 0.0,
        "retry_count": 0,
        "backtrack_count": 0,
        "fault_distribution": {},
        "per_agent": {}
    }
    
    if not hasattr(planner, "rebound_managers") or not planner.rebound_managers:
        return rebound_metrics
    
    num_agents = 0
    total_mttr = 0.0
    total_mtbf = 0.0
    total_count = 0.0
    total_rr = 0.0
    total_b_epi = 0.0
    
    for agent_uid, mgr in planner.rebound_managers.items():
        if not mgr:
            continue
        
        # Get metrics from ReboundManager
        metrics = mgr.get_metrics(total_planning_steps)
        
        # Aggregate per-agent metrics
        agent_metrics = {
            "rebound_mttr": metrics.get("rebound_mttr", 0.0),
            "rebound_mtbf": metrics.get("rebound_mtbf", 0.0),
            "rebound_count": metrics.get("rebound_count", 0.0),
            "recovery_ratio": metrics.get("recovery_ratio", 0.0),
            "b_epi": metrics.get("b_epi", 0.0),
            "retry_count": metrics.get("retry_count", 0),
            "backtrack_count": metrics.get("backtrack_count", 0),
            "fault_distribution": metrics.get("fault_distribution", {}),
            "failure_history": [
                {
                    "start_step": start,
                    "end_step": end,
                    "strategy": strategy
                }
                for start, end, strategy in mgr.failure_history
            ],
            "active_failure_start": mgr.active_failure_start,
            "errors_occurred": mgr.errors_occurred,
            "errors_fixed": mgr.errors_fixed,
        }
        
        rebound_metrics["per_agent"][str(agent_uid)] = agent_metrics
        
        # Aggregate for summary
        total_mttr += float(metrics.get("rebound_mttr", 0.0))
        total_mtbf += float(metrics.get("rebound_mtbf", 0.0))
        total_count += float(metrics.get("rebound_count", 0.0))
        total_rr += float(metrics.get("recovery_ratio", 0.0))
        total_b_epi += float(metrics.get("b_epi", 0.0))
        rebound_metrics["retry_count"] += metrics.get("retry_count", 0)
        rebound_metrics["backtrack_count"] += metrics.get("backtrack_count", 0)
        
        # Merge fault distributions
        fault_dist = metrics.get("fault_distribution", {})
        for fault_type, count in fault_dist.items():
            rebound_metrics["fault_distribution"][fault_type] = (
                rebound_metrics["fault_distribution"].get(fault_type, 0) + count
            )
        
        num_agents += 1
    
    # Average across agents
    if num_agents > 0:
        rebound_metrics["rebound_mttr"] = total_mttr / num_agents
        rebound_metrics["rebound_mtbf"] = total_mtbf / num_agents
        rebound_metrics["rebound_count"] = total_count
        rebound_metrics["recovery_ratio"] = total_rr / num_agents
        rebound_metrics["b_epi"] = total_b_epi / num_agents
    
    return rebound_metrics


def collect_cyclevla_metrics(
    planner: Any,
    total_planning_steps: int
) -> Dict[str, Any]:
    """
    Collect CycleVLA resilience metrics.
    
    Metrics:
    - vlm_predictions: Total number of VLM predictions made
    - vlm_backtracks: Number of backtrack decisions
    - vlm_backtrack_rate: Backtrack rate (backtracks / predictions)
    - mbr_sample_count: Number of MBR sampling operations
    - mbr_consensus_dist: Average consensus distance in MBR
    - subtask_completion_rate: Percentage of subtasks completed
    - cognitive_backtrack_events: List of backtrack events with details
    
    Args:
        planner: LLMPlanner instance with CycleVLA components
        total_planning_steps: Total number of planning steps in episode
    
    Returns:
        Dictionary with CycleVLA metrics
    """
    cyclevla_metrics = {
        "vlm_predictions": 0,
        "vlm_backtracks": 0,
        "vlm_backtrack_rate": 0.0,
        "mbr_sample_count": 0,
        "mbr_consensus_dist": 0.0,
        "subtask_completion_rate": 0.0,
        "subtask_total": 0,
        "subtask_completed": 0,
        "subtask_failed": 0,
        "cognitive_backtrack_events": [],
    }
    
    # Check if CycleVLA is enabled
    if not hasattr(planner, "use_cycle_vla") or not planner.use_cycle_vla:
        return cyclevla_metrics
    
    # Collect VLM Predictor metrics
    if hasattr(planner, "_cycle_vlm") and planner._cycle_vlm:
        try:
            vlm_stats = planner._cycle_vlm.get_statistics()
            cyclevla_metrics["vlm_predictions"] = vlm_stats.get("prediction_count", 0)
            cyclevla_metrics["vlm_backtracks"] = vlm_stats.get("backtrack_count", 0)
            cyclevla_metrics["vlm_backtrack_rate"] = vlm_stats.get("backtrack_rate", 0.0)
        except Exception as e:
            logger.debug(f"Could not collect VLM predictor stats: {e}")
    
    # Collect MBR Sampler metrics
    if hasattr(planner, "_cycle_mbr") and planner._cycle_mbr:
        try:
            mbr_stats = planner._cycle_mbr.get_statistics()
            cyclevla_metrics["mbr_sample_count"] = mbr_stats.get("sample_count", 0)
            cyclevla_metrics["mbr_consensus_dist"] = mbr_stats.get("avg_consensus_distance", 0.0)
        except Exception as e:
            logger.debug(f"Could not collect MBR sampler stats: {e}")
    
    # Collect SubtaskManager metrics
    if hasattr(planner, "_cycle_subtask_mgr") and planner._cycle_subtask_mgr:
        try:
            subtask_mgr = planner._cycle_subtask_mgr
            total_subtasks = len(subtask_mgr.subtasks)
            completed = sum(1 for st in subtask_mgr.subtasks if st.status == "completed")
            failed = sum(1 for st in subtask_mgr.subtasks if st.status == "failed")
            
            cyclevla_metrics["subtask_total"] = total_subtasks
            cyclevla_metrics["subtask_completed"] = completed
            cyclevla_metrics["subtask_failed"] = failed
            
            if total_subtasks > 0:
                cyclevla_metrics["subtask_completion_rate"] = completed / total_subtasks
            
            # Collect subtask details
            cyclevla_metrics["subtask_details"] = [
                {
                    "idx": st.idx,
                    "instruction": st.instruction,
                    "status": st.status,
                    "start_step": st.start_step,
                    "end_step": st.end_step,
                }
                for st in subtask_mgr.subtasks
            ]
        except Exception as e:
            logger.debug(f"Could not collect subtask manager stats: {e}")
    
    # Collect backtrack events (if tracked)
    # Note: This would need to be implemented in cycle_planner.py to track events
    # For now, we can infer from VLM backtrack count
    
    return cyclevla_metrics


def collect_inner_monologue_metrics(
    planner: Any,
    planner_infos: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Collect Inner Monologue resilience metrics.
    
    Metrics:
    - feedback_generation_count: Number of feedback messages generated
    - feedback_success_rate: Percentage of actions with successful feedback
    - thought_generation_rate: Percentage of replan steps with Thought generation
    - correction_effectiveness: Success rate after receiving failure feedback
    - feedback_sources_used: Distribution of feedback sources used
    
    Args:
        planner: LLMPlanner instance with Inner Monologue components
        planner_infos: List of planner_info dictionaries from episode
    
    Returns:
        Dictionary with Inner Monologue metrics
    """
    inno_mono_metrics = {
        "feedback_generation_count": 0,
        "feedback_success_count": 0,
        "feedback_failure_count": 0,
        "feedback_success_rate": 0.0,
        "thought_generation_count": 0,
        "replan_steps": 0,
        "thought_generation_rate": 0.0,
        "correction_attempts": 0,
        "correction_successes": 0,
        "correction_effectiveness": 0.0,
        "feedback_sources_used": {
            "success_detection": 0,
            "scene_description": 0,
            "self_state": 0,
            "critic_feedback": 0,
            "rebound_feedback": 0,
        },
    }
    
    # Check if Inner Monologue is enabled
    if not hasattr(planner, "inner_monologue_enabled") or not planner.inner_monologue_enabled:
        return inno_mono_metrics
    
    if not hasattr(planner, "feedback_generator") or planner.feedback_generator is None:
        return inno_mono_metrics
    
    # Get statistics directly from FeedbackGenerator if available
    try:
        stats = planner.feedback_generator.get_statistics()
        inno_mono_metrics.update({
            "feedback_generation_count": stats.get("feedback_count", 0),
            "feedback_success_count": stats.get("success_count", 0),
            "feedback_failure_count": stats.get("failure_count", 0),
            "feedback_success_rate": stats.get("feedback_success_rate", 0.0),
            "thought_generation_count": stats.get("thought_generation_count", 0),
            "thought_generation_rate": stats.get("thought_generation_rate", 0.0),
            "correction_attempts": stats.get("correction_attempts", 0),
            "correction_successes": stats.get("correction_successes", 0),
            "correction_effectiveness": stats.get("correction_effectiveness", 0.0),
        })
    except Exception as e:
        logger.debug(f"Could not get FeedbackGenerator statistics: {e}")
        # Fallback to analyzing planner_infos
    
    # Count replan steps for rate calculation
    for planner_info in planner_infos:
        replanned = planner_info.get("replanned", {})
        if isinstance(replanned, dict):
            any_replanned = any(replanned.values())
        else:
            any_replanned = bool(replanned)
        
        if any_replanned:
            inno_mono_metrics["replan_steps"] += 1
    
    # Recalculate thought_generation_rate if we have replan_steps
    if inno_mono_metrics["replan_steps"] > 0 and inno_mono_metrics["thought_generation_count"] > 0:
        inno_mono_metrics["thought_generation_rate"] = (
            inno_mono_metrics["thought_generation_count"] / inno_mono_metrics["replan_steps"]
        )
    
    # Analyze feedback sources from FeedbackGenerator history if available
    try:
        if hasattr(planner.feedback_generator, "feedback_history"):
            for feedback_record in planner.feedback_generator.feedback_history:
                if feedback_record.get("has_scene_desc"):
                    inno_mono_metrics["feedback_sources_used"]["scene_description"] += 1
                if feedback_record.get("has_self_state"):
                    inno_mono_metrics["feedback_sources_used"]["self_state"] += 1
                if feedback_record.get("has_critic"):
                    inno_mono_metrics["feedback_sources_used"]["critic_feedback"] += 1
                if feedback_record.get("has_rebound"):
                    inno_mono_metrics["feedback_sources_used"]["rebound_feedback"] += 1
                
                # Success detection is always present if feedback was generated
                if feedback_record.get("success_detection"):
                    inno_mono_metrics["feedback_sources_used"]["success_detection"] += 1
    except Exception as e:
        logger.debug(f"Could not analyze feedback history: {e}")
    
    return inno_mono_metrics


def collect_saycan_metrics(
    planner: Any,
    planner_infos: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Collect SayCan resilience metrics.
    
    Metrics:
    - saycan_stability_score: Overall stability score (1.0 - violation_rate)
    - saycan_filter_effectiveness: How often SayCan filtered low-confidence actions
    - mean_say_score: Mean LLM confidence score
    - mean_can_score: Mean Affordance feasibility score
    - mean_total_score: Mean fused score (Say × Can)
    - stability_violations: Number of times stability threshold was violated
    - affordance_accuracy: Affordance prediction accuracy (if ground truth available)
    
    Args:
        planner: LLMPlanner instance (may be SayCanPlanner)
        planner_infos: List of planner_info dictionaries from episode
    
    Returns:
        Dictionary with SayCan metrics
    """
    saycan_metrics = {
        "saycan_stability_score": 0.0,
        "saycan_filter_effectiveness": 0.0,
        "mean_say_score": 0.0,
        "mean_can_score": 0.0,
        "mean_total_score": 0.0,
        "stability_violations": 0,
        "stability_violation_rate": 0.0,
        "affordance_by_skill": {},
        "score_variance": 0.0,
    }
    
    # Check if planner has SayCan analyzer
    if not hasattr(planner, "saycan_analyzer") or not planner.saycan_analyzer:
        return saycan_metrics
    
    try:
        # Get analysis from analyzer
        analysis = planner.saycan_analyzer.analyze_episode()
        summary = analysis.get("summary", {})
        
        # Extract metrics
        saycan_metrics["mean_say_score"] = summary.get("mean_say_score", 0.0)
        saycan_metrics["mean_can_score"] = summary.get("mean_can_score", 0.0)
        saycan_metrics["mean_total_score"] = summary.get("mean_total_score", 0.0)
        saycan_metrics["stability_violations"] = summary.get("stability_violations", 0)
        saycan_metrics["stability_violation_rate"] = summary.get("stability_violation_rate", 0.0)
        saycan_metrics["score_variance"] = summary.get("std_total_score", 0.0)
        
        # Stability score: inverse of violation rate
        saycan_metrics["saycan_stability_score"] = max(0.0, 1.0 - saycan_metrics["stability_violation_rate"])
        
        # Filter effectiveness: percentage of steps where SayCan filtered (stability triggered)
        total_steps = summary.get("total_steps", 0)
        if total_steps > 0:
            saycan_metrics["saycan_filter_effectiveness"] = (
                saycan_metrics["stability_violations"] / total_steps
            )
        
        # Affordance by skill
        affordance_analysis = analysis.get("affordance_analysis", {})
        saycan_metrics["affordance_by_skill"] = affordance_analysis.get("by_skill", {})
        
        # Get stability metrics from StabilityMonitor if available
        if hasattr(planner, "stability_monitor") and planner.stability_monitor:
            stability_stats = planner.stability_monitor.get_stability_metrics()
            saycan_metrics["score_variance"] = stability_stats.get("score_variance", 0.0)
            saycan_metrics["recent_variance"] = stability_stats.get("recent_variance", 0.0)
    
    except Exception as e:
        logger.debug(f"Error collecting SayCan metrics: {e}")
    
    return saycan_metrics


def collect_copal_metrics(
    planner: Any,
    total_planning_steps: int
) -> Dict[str, Any]:
    """
    Collect CoPAL (Corrective Planning) recovery metrics.
    
    Metrics aligned with Rebound metrics:
    - copal_mttr: Mean Time To Recovery (planning steps) - equivalent to RSTC
    - copal_mtbf: Mean Time Between Failures (planning steps)
    - copal_rsr: Recovery Success Rate - equivalent to recovery_ratio
    - copal_rtr: Recovery Token Ratio (CoPAL-specific)
    - copal_efficiency: Recovery Efficiency score (CoPAL-specific)
    - copal_failure_count: Total number of failed actions (equivalent to rebound_count)
    
    Args:
        planner: LLMPlanner instance with CoPAL components
        total_planning_steps: Total number of planning steps in episode
    
    Returns:
        Dictionary with CoPAL metrics aggregated
    """
    copal_metrics = {
        # Aligned with Rebound metrics (same semantics)
        "copal_mttr": 0.0,           # Same as RSTC, aligned with rebound_mttr
        "copal_mtbf": 0.0,           # Aligned with rebound_mtbf
        "copal_rsr": 1.0,            # Aligned with recovery_ratio
        "copal_failure_count": 0.0,  # Aligned with rebound_count
        
        # CoPAL-specific metrics
        "copal_rtr": 0.0,            # Recovery Token Ratio
        "copal_efficiency": 1.0,     # Recovery Efficiency
        
        # Raw counts
        "copal_total_attempts": 0,
        "copal_successful_recoveries": 0,
        "copal_failed_recoveries": 0,
        "copal_total_steps": 0,
        "copal_recovery_steps": 0,
        "copal_total_tokens": 0,
        "copal_recovery_tokens": 0,
        
        # Breakdown
        "copal_error_breakdown": {},
        "copal_action_history": [],
    }
    
    # Check if CoPAL is enabled
    if not hasattr(planner, "use_copal") or not planner.use_copal:
        return copal_metrics
    
    # Method 1: Get from RecoveryMetrics if available
    if hasattr(planner, "copal_recovery_metrics") and planner.copal_recovery_metrics:
        try:
            metrics = planner.copal_recovery_metrics.get_metrics()
            copal_metrics.update({
                "copal_mttr": metrics.get("copal_rstc", 0.0),  # RSTC is the same as MTTR
                "copal_rsr": metrics.get("copal_rsr", 1.0),
                "copal_rtr": metrics.get("copal_rtr", 0.0),
                "copal_efficiency": metrics.get("copal_efficiency", 1.0),
                "copal_failure_count": float(metrics.get("copal_total_attempts", 0)),
                "copal_total_attempts": metrics.get("copal_total_attempts", 0),
                "copal_successful_recoveries": metrics.get("copal_successful_recoveries", 0),
                "copal_failed_recoveries": metrics.get("copal_failed_recoveries", 0),
                "copal_total_steps": metrics.get("copal_total_steps", 0),
                "copal_recovery_steps": metrics.get("copal_recovery_steps", 0),
                "copal_total_tokens": metrics.get("copal_total_tokens", 0),
                "copal_recovery_tokens": metrics.get("copal_recovery_tokens", 0),
                "copal_error_breakdown": metrics.get("copal_error_breakdown", {}),
            })
            
            # Calculate MTBF from recovery metrics
            total_attempts = metrics.get("copal_total_attempts", 0)
            if total_attempts > 0 and total_planning_steps > 0:
                copal_metrics["copal_mtbf"] = total_planning_steps / total_attempts
            else:
                copal_metrics["copal_mtbf"] = float(total_planning_steps)
                
        except Exception as e:
            logger.debug(f"Could not collect RecoveryMetrics: {e}")
    
    # Method 2: Get from CoPALContextBuilder history (fallback)
    if hasattr(planner, "copal_context_builder") and planner.copal_context_builder:
        try:
            builder = planner.copal_context_builder
            stats = builder.get_statistics()
            
            total_steps = stats.get("total_steps", 0)
            failure_count = stats.get("failure_count", 0)
            success_rate = stats.get("success_rate", 1.0)
            
            # Update if not already set from RecoveryMetrics
            if copal_metrics["copal_total_steps"] == 0:
                copal_metrics["copal_total_steps"] = total_steps
            if copal_metrics["copal_failure_count"] == 0.0:
                copal_metrics["copal_failure_count"] = float(failure_count)
            if copal_metrics["copal_rsr"] == 1.0 and failure_count > 0:
                copal_metrics["copal_rsr"] = success_rate
            
            # Calculate MTBF from history if not set
            if copal_metrics["copal_mtbf"] == 0.0 and failure_count > 0:
                copal_metrics["copal_mtbf"] = total_steps / max(1, failure_count)
            elif copal_metrics["copal_mtbf"] == 0.0:
                copal_metrics["copal_mtbf"] = float(total_planning_steps)
            
            # Get action history for detailed analysis
            if hasattr(builder, "history_manager") and hasattr(builder.history_manager, "_history"):
                history_items = []
                
                # Analyze failure-recovery patterns for MTTR calculation
                failure_starts = []
                recovery_steps = []
                last_failure_idx = -1
                
                for i, step in enumerate(builder.history_manager._history):
                    history_items.append({
                        "step_id": step.step_id,
                        "action": step.action,
                        "action_type": step.action_type,
                        "success": step.success,
                        "observation": step.observation[:100] if step.observation else "",
                    })
                    
                    # Track failure-recovery pattern
                    if not step.success:
                        if last_failure_idx < 0:
                            last_failure_idx = i
                    elif last_failure_idx >= 0:
                        # Recovery achieved
                        recovery_steps.append(i - last_failure_idx)
                        failure_starts.append(last_failure_idx)
                        last_failure_idx = -1
                
                copal_metrics["copal_action_history"] = history_items[-15:]  # Last 15 items
                
                # Calculate MTTR from recovery patterns
                if recovery_steps and copal_metrics["copal_mttr"] == 0.0:
                    copal_metrics["copal_mttr"] = sum(recovery_steps) / len(recovery_steps)
                
        except Exception as e:
            logger.debug(f"Could not collect CoPALContextBuilder stats: {e}")
    
    return copal_metrics


def collect_clare_metrics(
    planner: Any,
    total_planning_steps: int
) -> Dict[str, Any]:
    """
    Collect CLARE (Continual Learning) resilience metrics.
    
    Metrics:
    - clare_bwt: Backward Transfer (stability after learning new tasks)
    - clare_fwt: Forward Transfer (zero-shot performance on new tasks)
    - clare_routing_accuracy: Task routing correctness
    - clare_adapter_count: Number of learned adapters
    - clare_ood_rate: Out-of-Distribution detection rate
    - clare_avg_performance: Average task completion across adapters
    
    Args:
        planner: LLMPlanner instance with CLARE components
        total_planning_steps: Total number of planning steps in episode
    
    Returns:
        Dictionary with CLARE metrics
    """
    clare_metrics = {
        "clare_bwt": 0.0,
        "clare_fwt": 0.0,
        "clare_routing_accuracy": 0.0,
        "clare_avg_performance": 0.0,
        "clare_adapter_count": 0,
        "clare_total_routings": 0,
        "clare_total_expansions": 0,
        "clare_ood_rate": 0.0,
        "routing_stats": {},
        "expansion_stats": {},
    }
    
    # Check if CLARE is enabled
    if not getattr(planner, "clare_enabled", False):
        return clare_metrics
    
    components = getattr(planner, "clare_components", None)
    if not components:
        return clare_metrics
    
    try:
        # Get metrics from ContinualMetricsTracker
        metrics_tracker = components.get("metrics_tracker")
        if metrics_tracker:
            summary = metrics_tracker.get_summary()
            clare_metrics.update({
                "clare_bwt": summary.get("clare_bwt", 0.0),
                "clare_fwt": summary.get("clare_fwt", 0.0),
                "clare_routing_accuracy": summary.get("clare_routing_accuracy", 0.0),
                "clare_avg_performance": summary.get("clare_avg_performance", 0.0),
                "clare_adapter_count": summary.get("clare_adapter_count", 0),
                "clare_total_routings": summary.get("clare_total_routings", 0),
                "clare_total_expansions": summary.get("clare_total_expansions", 0),
                "clare_ood_rate": summary.get("clare_ood_rate", 0.0),
            })
        
        # Get routing stats from router
        router = components.get("router")
        if router:
            clare_metrics["routing_stats"] = router.get_routing_stats()
        
        # Get expansion stats from expansion controller
        expansion_controller = components.get("expansion_controller")
        if expansion_controller:
            clare_metrics["expansion_stats"] = expansion_controller.get_expansion_stats()
        
        # Get adapter info
        adapter_manager = components.get("adapter_manager")
        if adapter_manager:
            clare_metrics["adapter_list"] = adapter_manager.list_adapters()
            
    except Exception as e:
        logger.debug(f"Could not collect CLARE metrics: {e}")
    
    return clare_metrics


def log_resilience_metrics(
    planner: Any,
    planner_infos: List[Dict[str, Any]],
    output_dir: str,
    episode_filename: str,
    env_interface: Any,
    info: Dict[str, Any],
) -> str:
    """
    Collect and log comprehensive resilience metrics from all components.
    
    This function aggregates metrics from:
    - Rebound: MTTR, MTBF, Recovery Ratio, B_epi
    - CycleVLA: Backtrack metrics, MBR usage, subtask completion
    - CoPAL: RSR, RTR, RSTC, Recovery Efficiency
    - Inner Monologue: Feedback success rate, thought generation, correction effectiveness
    - SayCan: Stability score, filter effectiveness, Affordance distribution
    
    Args:
        planner: LLMPlanner instance
        planner_infos: List of planner_info dictionaries from episode
        output_dir: Output directory for logs
        episode_filename: Episode filename identifier
        env_interface: Environment interface for episode ID
        info: Episode info dictionary (will be updated with summary metrics)
    
    Returns:
        Filepath of saved resilience metrics JSON file
    """
    # Get total planning steps
    total_planning_steps = info.get("total_step_count", 0)
    if not total_planning_steps and planner_infos:
        # Fallback: use replanning_count from last planner_info
        last_info = planner_infos[-1]
        replanning_count = last_info.get("replanning_count", {})
        if isinstance(replanning_count, dict):
            total_planning_steps = max(replanning_count.values()) if replanning_count else 0
        else:
            total_planning_steps = replanning_count
    
    # Collect metrics from each component
    rebound_metrics = collect_rebound_metrics(planner, total_planning_steps)
    cyclevla_metrics = collect_cyclevla_metrics(planner, total_planning_steps)
    copal_metrics = collect_copal_metrics(planner, total_planning_steps)
    inno_mono_metrics = collect_inner_monologue_metrics(planner, planner_infos)
    saycan_metrics = collect_saycan_metrics(planner, planner_infos)
    clare_metrics = collect_clare_metrics(planner, total_planning_steps)
    
    # Build comprehensive resilience metrics
    resilience_data = {
        "episode_id": getattr(env_interface.env.env.env._env.current_episode, "episode_id", "unknown"),
        "episode_filename": episode_filename,
        "total_planning_steps": total_planning_steps,
        "total_simulation_steps": info.get("total_step_count", 0),
        "rebound_metrics": rebound_metrics,
        "cyclevla_metrics": cyclevla_metrics,
        "copal_metrics": copal_metrics,
        "inner_monologue_metrics": inno_mono_metrics,
        "saycan_metrics": saycan_metrics,
        "clare_metrics": clare_metrics,
        "summary": {
            # Rebound summary
            "rebound_mttr": rebound_metrics["rebound_mttr"],
            "rebound_mtbf": rebound_metrics["rebound_mtbf"],
            "rebound_count": rebound_metrics["rebound_count"],
            "recovery_ratio": rebound_metrics["recovery_ratio"],
            "b_epi": rebound_metrics["b_epi"],
            # CycleVLA summary
            "vlm_backtrack_rate": cyclevla_metrics["vlm_backtrack_rate"],
            "mbr_sample_count": cyclevla_metrics["mbr_sample_count"],
            "subtask_completion_rate": cyclevla_metrics["subtask_completion_rate"],
            # CoPAL summary (aligned with Rebound)
            "copal_mttr": copal_metrics["copal_mttr"],
            "copal_mtbf": copal_metrics["copal_mtbf"],
            "copal_rsr": copal_metrics["copal_rsr"],
            "copal_rtr": copal_metrics["copal_rtr"],
            "copal_efficiency": copal_metrics["copal_efficiency"],
            "copal_failure_count": copal_metrics["copal_failure_count"],
            # Inner Monologue summary
            "feedback_success_rate": inno_mono_metrics["feedback_success_rate"],
            "thought_generation_rate": inno_mono_metrics["thought_generation_rate"],
            "correction_effectiveness": inno_mono_metrics["correction_effectiveness"],
            # SayCan summary
            "saycan_stability_score": saycan_metrics["saycan_stability_score"],
            "saycan_filter_effectiveness": saycan_metrics["saycan_filter_effectiveness"],
            "saycan_mean_total_score": saycan_metrics["mean_total_score"],
            "saycan_stability_violations": saycan_metrics["stability_violations"],
            # CLARE summary
            "clare_bwt": clare_metrics["clare_bwt"],
            "clare_fwt": clare_metrics["clare_fwt"],
            "clare_routing_accuracy": clare_metrics["clare_routing_accuracy"],
            "clare_adapter_count": clare_metrics["clare_adapter_count"],
            "clare_ood_rate": clare_metrics["clare_ood_rate"],
        },
        # Combined resilience score
        "combined_resilience": _calculate_combined_resilience(
            rebound_metrics, copal_metrics, cyclevla_metrics, inno_mono_metrics
        ),
    }
    
    # Save to file
    episode_id = getattr(env_interface.env.env.env._env.current_episode, "episode_id", "unknown")
    analyses_dir = os.path.join(output_dir, "analyses", "resilience")
    os.makedirs(analyses_dir, exist_ok=True)
    filename = f"resilience_metrics-episode_{episode_id}_{episode_filename}.json"
    target_filepath = os.path.join(analyses_dir, filename)
    
    _write_json(target_filepath, resilience_data)
    logger.info("Resilience metrics saved: %s", target_filepath)
    
    # Add summary metrics to info dictionary (for scalar extraction)
    # Rebound metrics
    info["resilience_rebound_mttr"] = rebound_metrics["rebound_mttr"]
    info["resilience_rebound_mtbf"] = rebound_metrics["rebound_mtbf"]
    info["resilience_rebound_count"] = rebound_metrics["rebound_count"]
    info["resilience_recovery_ratio"] = rebound_metrics["recovery_ratio"]
    info["resilience_b_epi"] = rebound_metrics["b_epi"]
    
    # CycleVLA metrics
    info["resilience_vlm_backtrack_rate"] = cyclevla_metrics["vlm_backtrack_rate"]
    info["resilience_mbr_sample_count"] = cyclevla_metrics["mbr_sample_count"]
    info["resilience_subtask_completion_rate"] = cyclevla_metrics["subtask_completion_rate"]
    
    # CoPAL metrics (aligned with Rebound naming for consistency)
    info["resilience_copal_mttr"] = copal_metrics["copal_mttr"]
    info["resilience_copal_mtbf"] = copal_metrics["copal_mtbf"]
    info["resilience_copal_rsr"] = copal_metrics["copal_rsr"]
    info["resilience_copal_rtr"] = copal_metrics["copal_rtr"]
    info["resilience_copal_efficiency"] = copal_metrics["copal_efficiency"]
    info["resilience_copal_failure_count"] = copal_metrics["copal_failure_count"]
    
    # Inner Monologue metrics
    info["resilience_feedback_success_rate"] = inno_mono_metrics["feedback_success_rate"]
    info["resilience_thought_generation_rate"] = inno_mono_metrics["thought_generation_rate"]
    info["resilience_correction_effectiveness"] = inno_mono_metrics["correction_effectiveness"]
    
    # SayCan metrics
    info["resilience_saycan_stability_score"] = saycan_metrics["saycan_stability_score"]
    info["resilience_saycan_filter_effectiveness"] = saycan_metrics["saycan_filter_effectiveness"]
    info["resilience_saycan_mean_total_score"] = saycan_metrics["mean_total_score"]
    info["resilience_saycan_stability_violations"] = saycan_metrics["stability_violations"]
    
    # CLARE metrics
    info["resilience_clare_bwt"] = clare_metrics["clare_bwt"]
    info["resilience_clare_fwt"] = clare_metrics["clare_fwt"]
    info["resilience_clare_routing_accuracy"] = clare_metrics["clare_routing_accuracy"]
    info["resilience_clare_adapter_count"] = clare_metrics["clare_adapter_count"]
    info["resilience_clare_ood_rate"] = clare_metrics["clare_ood_rate"]
    
    # Combined resilience score
    info["resilience_combined_score"] = resilience_data["combined_resilience"]["score"]
    info["resilience_availability"] = resilience_data["combined_resilience"]["availability"]
    
    return target_filepath


def _calculate_combined_resilience(
    rebound_metrics: Dict[str, Any],
    copal_metrics: Dict[str, Any],
    cyclevla_metrics: Dict[str, Any],
    inno_mono_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculate combined resilience score from all components.
    
    Formula:
        Availability = MTBF / (MTBF + MTTR)
        Resilience = α * Availability + β * RSR + γ * Correction_Effectiveness
        
    Where:
        - α = 0.4 (availability weight)
        - β = 0.4 (recovery success weight)
        - γ = 0.2 (correction effectiveness weight)
    """
    alpha = 0.4
    beta = 0.4
    gamma = 0.2
    
    # Use the best available MTTR/MTBF (prefer CoPAL if available, else Rebound)
    mttr = copal_metrics["copal_mttr"] if copal_metrics["copal_mttr"] > 0 else rebound_metrics["rebound_mttr"]
    mtbf = copal_metrics["copal_mtbf"] if copal_metrics["copal_mtbf"] > 0 else rebound_metrics["rebound_mtbf"]
    
    # Use the best available RSR/Recovery Ratio
    rsr = copal_metrics["copal_rsr"] if copal_metrics["copal_failure_count"] > 0 else rebound_metrics["recovery_ratio"]
    
    # Get correction effectiveness from Inner Monologue
    correction_eff = inno_mono_metrics["correction_effectiveness"]
    
    # Calculate availability
    availability = mtbf / (mtbf + mttr) if (mtbf + mttr) > 0 else 1.0
    
    # Calculate combined score
    score = alpha * availability + beta * rsr + gamma * correction_eff
    
    return {
        "score": score,
        "availability": availability,
        "recovery_success_rate": rsr,
        "correction_effectiveness": correction_eff,
        "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
        "formula": "α*Availability + β*RSR + γ*Correction_Effectiveness",
        "sources": {
            "mttr_source": "copal" if copal_metrics["copal_mttr"] > 0 else "rebound",
            "mtbf_source": "copal" if copal_metrics["copal_mtbf"] > 0 else "rebound",
            "rsr_source": "copal" if copal_metrics["copal_failure_count"] > 0 else "rebound",
        },
    }

