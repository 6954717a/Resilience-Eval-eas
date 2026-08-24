#!/usr/bin/env python3

"""
Evaluation logging helpers.

Keeps EvaluationRunner lean by moving file I/O and analysis logging here.
"""

import json
import logging
import os
import pickle
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
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


def _load_json(filepath: str) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return {}


def _write_json(filepath: str, data: Dict[str, Any]) -> None:
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


def log_planner_data(
    output_dir: str,
    episode_filename: str,
    agents: Dict[int, Any],
    planner_infos: List[Dict[str, Any]],
    current_instruction: str,
    env_interface: Any,
    log_detailed_traces: bool,
) -> None:
    """
    Logs planner data including prompts, traces, and other info to files.
    """
    latest_info = planner_infos[-1] if planner_infos else {}

    for agent in agents.values():
        if "prompts" in latest_info:
            file_path_prompts = os.path.join(
                output_dir,
                "prompts",
                str(agent.uid),
                f"prompt-{episode_filename}-{str(agent.uid)}.txt",
            )
            os.makedirs(os.path.dirname(file_path_prompts), exist_ok=True)
            with open(file_path_prompts, "w") as file:
                file.write(latest_info["prompts"][agent.uid])

        if "traces" in latest_info:
            file_path_traces = os.path.join(
                output_dir,
                "traces",
                str(agent.uid),
                f"trace-{episode_filename}-{str(agent.uid)}.txt",
            )
            os.makedirs(os.path.dirname(file_path_traces), exist_ok=True)
            with open(file_path_traces, "w") as file:
                file.write(latest_info["traces"][agent.uid])

    file_path_json = os.path.join(
        output_dir,
        "planner-log",
        f"planner-log-{episode_filename}.json",
    )

    if "actions_per_agent" in latest_info:
        actions_per_agent_path = os.path.join(
            output_dir,
            f"plan/{episode_filename}.txt",
        )
        os.makedirs(os.path.dirname(actions_per_agent_path), exist_ok=True)
        with open(actions_per_agent_path, "w") as file:
            file.write(str(latest_info["actions_per_agent"]))

    os.makedirs(os.path.dirname(file_path_json), exist_ok=True)

    planner_log: Dict[str, Any] = {
        "task": current_instruction,
        "steps": [],
    }

    keys_to_exclude = ["prompts", "traces", "print", "print_no_tags"]
    for i, planner_info in enumerate(planner_infos):
        step_info = {
            k: v
            for k, v in sorted(planner_info.items())
            if k not in keys_to_exclude
        }
        step_info["log_index"] = i
        planner_log["steps"].append(step_info)

    _write_json(file_path_json, planner_log)
    logger.info("Planner log saved: %s", file_path_json)

    if log_detailed_traces:
        save_detailed_traces(output_dir, episode_filename, current_instruction, env_interface)


def save_detailed_traces(
    output_dir: str,
    episode_filename: str,
    current_instruction: str,
    env_interface: Any,
) -> None:
    """
    Save detailed traces to a pickle file containing instruction, action history and state history.
    """
    for actions in env_interface.agent_action_history.values():
        for action in actions[:-1]:
            if action.response in [None, ""] and action.action[0] != "Done":
                action_history_string = "\n".join(
                    [f"{a.action}: {a.response}" for a in actions]
                )
                raise ValueError(
                    f"Agent {action.agent_uid} has a null response on {action.action}: "
                    f"Action history:\n{action_history_string}"
                )

    file_path_detailed_trace = os.path.join(
        output_dir,
        "detailed_traces",
        f"detailed_trace-{episode_filename}.pkl",
    )
    result = {
        "instruction": current_instruction,
        "action_history": env_interface.agent_action_history,
        "state_history": env_interface.agent_state_history,
    }

    os.makedirs(os.path.dirname(file_path_detailed_trace), exist_ok=True)
    with open(file_path_detailed_trace, "wb") as file:
        pickle.dump(result, file)
    logger.info("Detailed traces saved: %s", file_path_detailed_trace)


def log_rebound_analysis(
    info: Dict[str, Any],
    planner: Any,
    output_dir: str,
    episode_filename: str,
    env_interface: Any,
    critic: Optional[Any] = None,
) -> str:
    """
    Collect and save rebound analysis data.
    Also adds scalar summary metrics to info.
    """
    rebound_managers = {}
    if hasattr(planner, "rebound_managers"):
        rebound_managers = planner.rebound_managers
    elif isinstance(planner, dict):
        for _, sub_planner in planner.items():
            if hasattr(sub_planner, "rebound_managers"):
                rebound_managers.update(sub_planner.rebound_managers)

    rebound_data: Dict[str, Any] = {}
    total_steps = info.get("total_step_count", 0)

    if rebound_managers:
        for agent_uid, mgr in rebound_managers.items():
            metrics = mgr.get_metrics(total_steps)
            history = mgr.failure_history
            rebound_data[str(agent_uid)] = {
                "metrics": metrics,
                "failure_history": history,
                "active_failure_start": mgr.active_failure_start,
            }
    else:
        rebound_data["status"] = "No Rebound Managers Active"

    has_rebound_activity = False
    for agent_data in rebound_data.values():
        if isinstance(agent_data, dict) and agent_data.get("metrics", {}).get("rebound_count", 0) > 0:
            has_rebound_activity = True
            break

    if not has_rebound_activity and "status" not in rebound_data:
        rebound_data["status"] = "No Rebound Events Triggered"

    summary_mttr = 0.0
    summary_mtbf = 0.0
    summary_count = 0.0
    num_agents = 0

    if rebound_managers:
        for _, mgr in rebound_managers.items():
            metrics = mgr.get_metrics(total_steps)
            summary_mttr += float(metrics.get("rebound_mttr", 0.0))
            summary_mtbf += float(metrics.get("rebound_mtbf", 0.0))
            summary_count += float(metrics.get("rebound_count", 0.0))
            num_agents += 1

    if num_agents > 0:
        summary_mttr /= num_agents
        summary_mtbf /= num_agents

    info["rebound_mttr"] = summary_mttr
    info["rebound_mtbf"] = summary_mtbf
    info["rebound_count"] = summary_count

    episode_id = env_interface.env.env.env._env.current_episode.episode_id
    analyses_dir = os.path.join(output_dir, "analyses", "rebound")
    os.makedirs(analyses_dir, exist_ok=True)
    filename = f"rebound_analysis-episode_{episode_id}_{episode_filename}.json"
    target_filepath = os.path.join(analyses_dir, filename)

    save_data: Dict[str, Any] = {
        "episode_id": episode_id,
        "episode_filename": episode_filename,
        "rebound_summary": {
            "rebound_mttr": summary_mttr,
            "rebound_mtbf": summary_mtbf,
            "rebound_count": summary_count,
        },
        "rebound_analysis": rebound_data,
    }

    _write_json(target_filepath, save_data)
    logger.info("Rebound analysis saved: %s", target_filepath)
    return target_filepath


def log_adca_analysis(
    adca_result: Dict[str, Any],
    output_dir: str,
    episode_filename: str,
    env_interface: Any,
    current_instruction: str,
    critic: Optional[Any] = None,
) -> str:
    """
    Save ADCA analysis results to a JSON file.
    """
    adca_dir = os.path.join(output_dir, "analyses", "adca")
    os.makedirs(adca_dir, exist_ok=True)
    episode_id = env_interface.env.env.env._env.current_episode.episode_id
    filename = f"adca_analysis-episode_{episode_id}_{episode_filename}.json"
    filepath = os.path.join(adca_dir, filename)

    save_data = {
        "episode_id": episode_id,
        "episode_filename": episode_filename,
        "task_instruction": current_instruction,
        "adca_result": adca_result,
    }

    _write_json(filepath, save_data)
    logger.info("ADCA analysis saved: %s", filepath)
    return filepath


def write_critic_stats(
    critic_stats: Dict[str, Any],
    output_dir: str,
    episode_filename: str,
    env_interface: Any,
    critic: Optional[Any] = None,
) -> str:
    """
    Persist critic statistics to JSON. Returns the filepath written.
    """
    episode_id = env_interface.env.env.env._env.current_episode.episode_id
    analyses_dir = os.path.join(output_dir, "analyses", "critic")
    os.makedirs(analyses_dir, exist_ok=True)
    filename = f"critic_stats-episode_{episode_id}_{episode_filename}.json"
    target_filepath = os.path.join(analyses_dir, filename)

    save_data: Dict[str, Any] = {
        "episode_id": episode_id,
        "episode_filename": episode_filename,
        "critic_stats": critic_stats,
    }

    _write_json(target_filepath, save_data)
    logger.info("Critic stats saved: %s", target_filepath)
    return target_filepath

def log_saycan_analysis(
    saycan_data: Dict[str, Any],
    output_dir: str,
    episode_filename: str,
    env_interface: Any,
    planner_infos: Optional[List[Dict[str, Any]]] = None,
    current_instruction: Optional[str] = None,
    critic: Optional[Any] = None,
) -> str:
    """
    Save SayCan analysis results to a JSON file.
    Also adds scalar summary metrics to info (similar to log_rebound_analysis).
    """
    saycan_dir = os.path.join(output_dir, "analyses", "saycan")
    os.makedirs(saycan_dir, exist_ok=True)
    
    try:
        episode_id = env_interface.env.env.env._env.current_episode.episode_id
    except Exception:
        episode_id = "unknown"
    
    filename = f"saycan_analysis-episode_{episode_id}_{episode_filename}.json"
    filepath = os.path.join(saycan_dir, filename)

    # Extract summary for scalar metrics (similar to rebound_analysis)
    summary = saycan_data.get("summary", {})
    
    # Get total planning steps from planner_infos (not simulation steps)
    total_planning_steps = 0
    if planner_infos:
        last_info = planner_infos[-1] if planner_infos else {}
        replanning_count = last_info.get("replanning_count", {})
        if isinstance(replanning_count, dict):
            total_planning_steps = max(replanning_count.values()) if replanning_count else 0
        else:
            total_planning_steps = replanning_count if replanning_count else 0
    else:
        # Fallback: use total_steps from summary
        total_planning_steps = summary.get("total_steps", 0)
    
    # Extract metrics
    mean_say_score = summary.get("mean_say_score", 0.0)
    mean_can_score = summary.get("mean_can_score", 0.0)
    mean_total_score = summary.get("mean_total_score", 0.0)
    stability_violations = summary.get("stability_violations", 0)
    stability_violation_rate = summary.get("stability_violation_rate", 0.0)
    affordance_by_skill = saycan_data.get("affordance_analysis", {}).get("by_skill", {})
    
    # Calculate stability score (inverse of violation rate)
    stability_score = max(0.0, 1.0 - stability_violation_rate)
    
    # Add summary metrics to a top-level saycan_summary for consistency
    saycan_summary = {
        "total_planning_steps": total_planning_steps,
        "mean_say_score": mean_say_score,
        "mean_can_score": mean_can_score,
        "mean_total_score": mean_total_score,
        "stability_violations": stability_violations,
        "stability_violation_rate": stability_violation_rate,
        "stability_score": stability_score,
        "affordance_by_skill": affordance_by_skill,
    }

    save_data: Dict[str, Any] = {
        "episode_id": str(episode_id),
        "episode_filename": episode_filename,
        "task_instruction": current_instruction or "",
        "total_planning_steps": total_planning_steps,
        "saycan_summary": saycan_summary,
        "saycan_analysis": saycan_data,
    }

    _write_json(filepath, save_data)
    logger.info("SayCan analysis saved: %s", filepath)
    return filepath

def log_copal_analysis(
    info: Dict[str, Any],
    planner: Any,
    output_dir: str,
    episode_filename: str,
    env_interface: Any,
) -> str:
    """
    Collect and save CoPAL (Corrective Planning) analysis data.
    
    CoPAL metrics complement Rebound metrics:
    - RSR (Recovery Success Rate): Successful corrections / Total attempts
    - RTR (Recovery Token Ratio): Efficiency of correction
    - RSTC (Recovery Steps to Complete): Steps from failure to success
    
    These align with Rebound's MTTR/MTBF for comprehensive resilience measurement.
    """
    copal_data: Dict[str, Any] = {}
    
    # Get CoPAL components from planner
    use_copal = getattr(planner, "use_copal", False)
    copal_builder = getattr(planner, "copal_context_builder", None)
    recovery_metrics = getattr(planner, "copal_recovery_metrics", None)
    
    if use_copal and copal_builder:
        # Get statistics from CoPAL context builder
        stats = copal_builder.get_statistics()
        
        # Get history details
        history_items = []
        if hasattr(copal_builder, "history_manager") and hasattr(copal_builder.history_manager, "_history"):
            for step in list(copal_builder.history_manager._history)[-15:]:
                history_items.append({
                    "step_id": step.step_id,
                    "action": step.action,
                    "action_type": step.action_type,
                    "success": step.success,
                    "observation": step.observation[:100] if step.observation else "",
                })
        
        # Calculate CoPAL-specific resilience metrics
        total_steps = stats.get("total_steps", 0)
        failure_count = stats.get("failure_count", 0)
        
        # Default values (fallback)
        success_rate = stats.get("success_rate", 1.0)
        copal_mttr = 0.0
        copal_mtbf = 0.0
        copal_rtr = 0.0
        copal_rstc = 0.0
        copal_efficiency = 0.0
        
        # Use authoritative RecoveryMetrics if available
        if recovery_metrics:
            rm_stats = recovery_metrics.get_metrics()
            success_rate = rm_stats.get("copal_rsr", 1.0)
            copal_rtr = rm_stats.get("copal_rtr", 0.0)
            copal_rstc = rm_stats.get("copal_rstc", 0.0)
            copal_efficiency = rm_stats.get("copal_efficiency", 0.0)
            
            # Map RSTC to MTTR (Recovery Steps to Complete is semantically MTTR)
            copal_mttr = copal_rstc
            
            # Calculate MTBF: Total Steps / Failures
            # Use raw failure count from builder or recovery metrics
            if failure_count > 0:
                copal_mtbf = total_steps / failure_count
            else:
                copal_mtbf = float(total_steps) if total_steps > 0 else 0.0
                
        # Fallback to history analysis if no RecoveryMetrics (or strict verification)
        elif history_items:
            # Analyze failure-recovery patterns
            failure_starts = []
            recovery_steps = []
            last_failure_idx = -1
            
            for i, item in enumerate(history_items):
                if not item["success"]:
                    if last_failure_idx < 0:
                        last_failure_idx = i
                elif last_failure_idx >= 0:
                    # Recovery achieved
                    recovery_steps.append(i - last_failure_idx)
                    failure_starts.append(last_failure_idx)
                    last_failure_idx = -1
            
            if recovery_steps:
                copal_mttr = sum(recovery_steps) / len(recovery_steps)
            
            # MTBF: average steps between failure starts
            if len(failure_starts) > 1:
                intervals = [failure_starts[i+1] - failure_starts[i] for i in range(len(failure_starts)-1)]
                copal_mtbf = sum(intervals) / len(intervals)
            elif total_steps > 0:
                copal_mtbf = total_steps / max(1, failure_count)
        
        copal_data = {
            "enabled": True,
            "statistics": {
                "total_steps": total_steps,
                "failure_count": failure_count,
                "success_rate": success_rate,
                "consecutive_failures": stats.get("consecutive_failures", 0),
            },
            "resilience_metrics": {
                "copal_mttr": copal_mttr,
                "copal_mtbf": copal_mtbf,
                "copal_rsr": success_rate,
                "copal_rtr": copal_rtr,
                "copal_rstc": copal_rstc,
                "copal_efficiency": copal_efficiency,
                "copal_failure_ratio": failure_count / max(1, total_steps),
            },
            "action_history": history_items,
        }
        
        # Add summary metrics to info for CSV output
        info["copal_mttr"] = copal_mttr
        info["copal_mtbf"] = copal_mtbf
        info["copal_rsr"] = success_rate
        info["copal_rtr"] = copal_rtr
        info["copal_rstc"] = copal_rstc
        info["copal_efficiency"] = copal_efficiency
        info["copal_failure_count"] = failure_count
        info["copal_total_steps"] = total_steps
    else:
        copal_data = {
            "enabled": False,
            "status": "CoPAL not enabled for this run",
        }
        # Add zero metrics to info
        info["copal_mttr"] = 0.0
        info["copal_mtbf"] = 0.0
        info["copal_rsr"] = 1.0
        info["copal_rtr"] = 0.0
        info["copal_rstc"] = 0.0
        info["copal_efficiency"] = 0.0
        info["copal_failure_count"] = 0
        info["copal_total_steps"] = 0
    
    # Save to file
    episode_id = env_interface.env.env.env._env.current_episode.episode_id
    analyses_dir = os.path.join(output_dir, "analyses", "copal")
    os.makedirs(analyses_dir, exist_ok=True)
    filename = f"copal_analysis-episode_{episode_id}_{episode_filename}.json"
    target_filepath = os.path.join(analyses_dir, filename)
    
    save_data: Dict[str, Any] = {
        "episode_id": episode_id,
        "episode_filename": episode_filename,
        "task_percent_complete": info.get("task_percent_complete", 0.0),
        "task_state_success": info.get("task_state_success", False),
        "copal_analysis": copal_data,
    }
    
    _write_json(target_filepath, save_data)
    logger.info("CoPAL analysis saved: %s", target_filepath)
    return target_filepath


def log_resilience_summary(
    info: Dict[str, Any],
    planner: Any,
    output_dir: str,
    episode_filename: str,
    env_interface: Any,
    critic: Optional[Any] = None,
) -> str:
    """
    Generate a unified resilience summary combining Rebound and CoPAL metrics.
    
    Resilience Metrics Framework:
    
    1. Recovery Metrics (from Rebound):
       - MTTR (Mean Time To Recovery): Average steps to recover from failure
       - MTBF (Mean Time Between Failures): Average steps between failures
       - RR (Recovery Ratio): Errors fixed / Errors occurred
    
    2. Correction Metrics (from CoPAL):
       - RSR (Recovery Success Rate): Success rate of corrective actions
       - RTR (Recovery Token Ratio): Efficiency of recovery process
       - RSTC (Recovery Steps to Complete): Steps from failure to completion
    
    3. Combined Resilience Score:
       Resilience = α * (MTBF / (MTBF + MTTR)) + β * RSR
       where α = 0.6, β = 0.4 (configurable)
    """
    # Collect Rebound metrics
    rebound_mttr = info.get("rebound_mttr", 0.0)
    rebound_mtbf = info.get("rebound_mtbf", 0.0)
    rebound_count = info.get("rebound_count", 0.0)
    
    # Collect CoPAL metrics
    copal_mttr = info.get("copal_mttr", 0.0)
    copal_mtbf = info.get("copal_mtbf", 0.0)
    copal_rsr = info.get("copal_rsr", 1.0)
    copal_rtr = info.get("copal_rtr", 0.0)
    copal_rstc = info.get("copal_rstc", 0.0)
    copal_efficiency = info.get("copal_efficiency", 0.0)
    copal_failure_count = info.get("copal_failure_count", 0)
    
    # Calculate combined resilience score
    alpha = 0.6
    beta = 0.4
    
    # Use CoPAL metrics if available, otherwise Rebound
    # If CoPAL is enabled (mttr > 0 or failures > 0), prioritize it as it tracks closed-loop steps
    mttr = copal_mttr if copal_mttr > 0 else rebound_mttr
    mtbf = copal_mtbf if copal_mtbf > 0 else rebound_mtbf
    
    availability = mtbf / (mtbf + mttr) if (mtbf + mttr) > 0 else 1.0
    resilience_score = alpha * availability + beta * copal_rsr
    
    # Add to info for CSV output
    info["resilience_score"] = resilience_score
    info["resilience_availability"] = availability
    
    # Build summary
    summary_data: Dict[str, Any] = {
        "episode_id": env_interface.env.env.env._env.current_episode.episode_id,
        "episode_filename": episode_filename,
        "task_outcome": {
            "task_percent_complete": info.get("task_percent_complete", 0.0),
            "task_state_success": info.get("task_state_success", False),
        },
        "rebound_metrics": {
            "mttr": rebound_mttr,
            "mtbf": rebound_mtbf,
            "event_count": rebound_count,
        },
        "copal_metrics": {
            "mttr": copal_mttr,
            "mtbf": copal_mtbf,
            "rsr": copal_rsr,
            "rtr": copal_rtr,
            "rstc": copal_rstc,
            "efficiency": copal_efficiency,
            "failure_count": copal_failure_count,
        },
        "combined_resilience": {
            "resilience_score": resilience_score,
            "availability": availability,
            "formula": f"α*Availability + β*RSR (α={alpha}, β={beta})",
        },
    }
    
    # Save to file
    analyses_dir = os.path.join(output_dir, "analyses", "resilience")
    os.makedirs(analyses_dir, exist_ok=True)
    episode_id = env_interface.env.env.env._env.current_episode.episode_id
    filename = f"resilience_summary-episode_{episode_id}_{episode_filename}.json"
    target_filepath = os.path.join(analyses_dir, filename)
    
    _write_json(target_filepath, summary_data)
    logger.info("Resilience summary saved: %s", target_filepath)
    return target_filepath

