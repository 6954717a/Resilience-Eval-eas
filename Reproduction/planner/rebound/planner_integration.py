#!/usr/bin/env python3

"""
Rebound Planner Integration Module

This module contains Rebound logic to keep llm_planner minimal.
Rebound is limited to prompt/trace modifications only.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

# Import unified context compressor
try:
    from habitat_llm.context import ContextCompressor, LLMTags, format_context_update_as_turn
    _HAS_CONTEXT_COMPRESSOR = True
except ImportError:
    _HAS_CONTEXT_COMPRESSOR = False
    LLMTags = None
    format_context_update_as_turn = None

if TYPE_CHECKING:
    from habitat_llm.planner.llm_planner import LLMPlanner


def handle_rebound_side_effects(
    planner: "LLMPlanner", responses: Dict[int, str]
) -> Dict[int, str]:
    """
    Rebound is limited to prompt modifications only.
    Tool side effects are intentionally disabled.
    """
    return responses


def _assistant_turn_start(planner: "LLMPlanner") -> str:
    if planner.planner_config.planning_mode.lower() == "cot":
        return f"{planner.planner_config.llm.assistant_tag}Thought:"
    return f"{planner.planner_config.llm.assistant_tag}"


def _inject_guidance_as_turn(planner: "LLMPlanner", guidance: str, title: str = "Rebound Guidance") -> None:
    """
    Inject guidance as a tagged user turn to preserve ChatML/LLaMA format.
    Uses the unified context module for proper tag formatting.
    """
    if not guidance:
        return

    # Get tags from planner config
    user_tag = planner.planner_config.llm.user_tag
    eot_tag = planner.planner_config.llm.eot_tag
    assistant_turn = _assistant_turn_start(planner)

    prompt = planner.curr_prompt.rstrip()
    if prompt.endswith(assistant_turn):
        prompt = prompt[: -len(assistant_turn)].rstrip()

    # Use unified formatting if available
    if _HAS_CONTEXT_COMPRESSOR and format_context_update_as_turn is not None:
        tags = LLMTags(
            system_tag=planner.planner_config.llm.system_tag,
            user_tag=user_tag,
            assistant_tag=planner.planner_config.llm.assistant_tag,
            eot_tag=eot_tag,
        )
        guidance_block = format_context_update_as_turn(title, guidance, tags)
    else:
        # Fallback to manual formatting
        guidance_block = (
            f"\n{user_tag}[{title}]\n{guidance}\n{eot_tag}"
        )
    
    prompt += guidance_block

    if not prompt.endswith(assistant_turn):
        prompt += f"\n{assistant_turn}"

    planner.curr_prompt = prompt
    planner.trace += f"\n[{title}]\n{guidance}\n"


def _normalize_rebound_text(text: str, max_chars: int) -> str:
    if not text:
        return "none"
    normalized = " ".join(text.split())
    if len(normalized) > max_chars:
        return normalized[: max_chars - 3] + "..."
    return normalized


def _format_rebound_action(
    action: Tuple[str, str, Optional[str]], include_args: bool, max_arg_chars: int
) -> str:
    action_name, arg_1, arg_2 = action
    if not include_args:
        return action_name

    args = [arg for arg in (arg_1, arg_2) if arg not in (None, "")]
    if not args:
        return action_name

    args_str = _normalize_rebound_text(", ".join(str(arg) for arg in args), max_arg_chars)
    return f"{action_name}[{args_str}]"


def _build_rebound_context_block(
    planner: "LLMPlanner", detected_faults: Dict[int, List[str]]
) -> str:
    context_config = planner.rebound_config.get("context", {})
    max_response_chars = context_config.get("max_response_chars", 160)
    max_arg_chars = context_config.get("max_action_arg_chars", 60)
    include_args = context_config.get("include_action_args", True)

    lines = [f"[Rebound]: planning_step={planner.replanning_count}"]
    for agent_id, faults in detected_faults.items():
        last_action = planner.last_high_level_actions.get(agent_id, ("Unknown", "", ""))
        action_desc = _format_rebound_action(last_action, include_args, max_arg_chars)
        response = planner.latest_agent_response.get(agent_id, "")
        response_summary = _normalize_rebound_text(response, max_response_chars)
        fault_list = ", ".join(sorted(set(faults)))
        lines.append(
            f"[Rebound][Agent {agent_id}] faults={fault_list}; "
            f"last_action={action_desc}; response={response_summary}"
        )

    return "\n".join(lines)


def detect_rebound_faults(planner: "LLMPlanner") -> Dict[int, List[str]]:
    """
    Phase 1 of Rebound: Detection only.
    Uses planning-step semantics and only runs after a completed replan.
    """
    if not planner.rebound_enabled:
        return {}

    if planner.replanning_count == 0 or not planner.latest_agent_response:
        return {}

    detected_faults: Dict[int, List[str]] = {}
    detector_config = planner.rebound_config.get("detectors", {})

    for agent_id, response in planner.latest_agent_response.items():
        mgr = planner.rebound_managers.get(agent_id)
        if not mgr:
            continue

        last_action = planner.last_high_level_actions.get(agent_id, ("Unknown", "", ""))
        error_str = response if response and "fail" in response.lower() else None
        mgr.log_execution(last_action[0], last_action[1], response or "", 1.0, error_str)

        faults = mgr.detector.detect_faults(mgr.telemetry)

        # Immediate Error Detection with Trial-and-Error Tolerance (Issue 2)
        # Only trigger after N consecutive same-type failures to give LLM retry space
        if detector_config.get("immediate_error", True):
            if response and "fail" in response.lower():
                # Extract action type from last_action
                action_type = last_action[0] if last_action[0] else "Unknown"
                # Check if we should trigger intervention (N consecutive same-type failures)
                should_intervene = mgr.record_immediate_error(
                    planner.replanning_count, action_type, response
                )
                if should_intervene and "immediate_error" not in faults:
                    faults.append("immediate_error")
                    print(
                        f"[ReboundIntegration] Immediate error triggered for Agent {agent_id} "
                        f"after {mgr.immediate_error_window} consecutive {action_type} failures."
                    )
            else:
                # Success or no failure - clear the error buffer
                mgr.clear_immediate_error_buffer()

        faults = mgr.filter_faults(faults, planner.replanning_count)
        
        # Update metrics logic
        mgr.update_failure_metrics(faults, planner.replanning_count) # (since we are not using suggest_recovery in this integration mode)

        if faults:
            detected_faults[agent_id] = faults

    if len(planner.agents) > 1:
        stagnating_agents = [
            aid for aid, faults in detected_faults.items()
            if "progress_stagnation" in faults
        ]
        if stagnating_agents and len(stagnating_agents) < len(planner.agents):
            print(
                f"[ReboundIntegration] Suppressing stagnation. "
                f"Only {len(stagnating_agents)}/{len(planner.agents)} agents stuck."
            )
            for aid in stagnating_agents:
                detected_faults[aid].remove("progress_stagnation")
                if not detected_faults[aid]:
                    del detected_faults[aid]

    return detected_faults


def apply_rebound_context_modification(
    planner: "LLMPlanner", detected_faults: Dict[int, List[str]]
) -> None:
    """
    Phase 2 of Rebound: Context modification only.
    """
    if not detected_faults:
        return

    context_config = planner.rebound_config.get("context", {})

    all_faults = set()
    for faults in detected_faults.values():
        all_faults.update(faults)

    context_additions = []

    if "context_overflow" in all_faults:
        print("[ReboundIntegration] Applying context truncation for overflow.")
        keep_last_n = context_config.get("keep_last_n", 5)
        max_prompt_chars = context_config.get("max_prompt_chars", 8000)
        max_trace_chars = context_config.get("max_trace_chars", 6000)

        if len(planner.curr_prompt) > max_prompt_chars:
            if getattr(planner, "use_prompt_builder", False) and getattr(
                planner, "prompt_builder", None
            ):
                keep_last_turns = max(1, keep_last_n * 2)
                planner.truncate_prompt_history(keep_last_turns)
            else:
                marker = "\n...[Context Pruned]...\n"
                head_len = min(1200, max_prompt_chars // 4)
                tail_len = max_prompt_chars - head_len - len(marker)
                if tail_len > 0:
                    planner.curr_prompt = (
                        planner.curr_prompt[:head_len]
                        + marker
                        + planner.curr_prompt[-tail_len:]
                    )
                else:
                    planner.curr_prompt = planner.curr_prompt[:max_prompt_chars]

        trace_lines = planner.trace.split("\n")
        if len(trace_lines) > keep_last_n * 3:
            task_line = trace_lines[0] if trace_lines else ""
            recent_lines = trace_lines[-(keep_last_n * 3):]
            planner.trace = task_line + "\n...[Context Pruned]...\n" + "\n".join(recent_lines)

        if len(planner.trace) > max_trace_chars:
            planner.trace = planner.trace[-max_trace_chars:]
            planner.trace = "[Trace Pruned]...\n" + planner.trace

        context_additions.append(
            "Context was truncated to prevent overflow. "
            "Focus on recent observations and the original task."
        )

    context_additions.append(_build_rebound_context_block(planner, detected_faults))

    if "graph_inconsistency" in all_faults:
        context_additions.append(
            "World state may be inconsistent. "
            "Consider using perception or navigation to verify object locations before acting."
        )

    if "progress_stagnation" in all_faults:
        recent_actions = []
        for agent in planner.agents:
            mgr = planner.rebound_managers.get(agent.uid)
            if mgr:
                actions = mgr.telemetry.get_last_n_actions(3)
                for a in actions:
                    recent_actions.append(
                        f"{a.get('tool', 'Unknown')}: {a.get('result', '')[:50]}"
                    )
        action_summary = "; ".join(recent_actions[-5:]) if recent_actions else "No recent actions"
        context_additions.append(
            f"Progress has stalled. Recent actions: {action_summary}. "
            "Reflect on what might be blocking progress and consider alternative approaches."
        )

    if "tool_failure_burst" in all_faults:
        context_additions.append(
            "Multiple consecutive tool failures detected. "
            "Consider: (1) verifying preconditions, (2) trying an alternative tool, "
            "(3) navigating closer to the target, or (4) exploring to find the target."
        )

    if "immediate_error" in all_faults:
        # Use ContextCompressor for concise, action-oriented guidance
        for agent_id in detected_faults:
            if "immediate_error" in detected_faults[agent_id]:
                response = planner.latest_agent_response.get(agent_id, "")
                last_action = planner.last_high_level_actions.get(agent_id, ("", "", ""))
                action_str = f"{last_action[0]}[{last_action[1]}]" if last_action[0] else "Unknown"
                
                if _HAS_CONTEXT_COMPRESSOR:
                    # Use unified context compressor
                    compressor = ContextCompressor()
                    guidance = compressor.compress_rebound_guidance(
                        fault_type="immediate_error",
                        action=action_str,
                        response=response or "Action failed",
                    )
                    context_additions.append(guidance)
                else:
                    # Fallback to original format
                    if response:
                        context_additions.append(
                            f"Action failed. Error: {response[:80]}. "
                            "Navigate closer or try alternative approach."
                        )

    if context_additions:
        guidance = "\n".join(context_additions)
        if getattr(planner, "use_prompt_builder", False) and getattr(
            planner, "prompt_builder", None
        ):
            planner.enqueue_context_update("Rebound Guidance", guidance)
        else:
            _inject_guidance_as_turn(planner, guidance)
        print(f"[ReboundIntegration] Applied {len(context_additions)} context modifications.")


def try_execute_recovery(
    planner: "LLMPlanner",
    observations: Dict[str, Any],
    planner_info: Dict[str, Any],
    print_str: str
) -> Optional[Tuple[Dict[int, Any], Dict[str, Any], bool]]:
    """
    Rebound is limited to prompt modifications only.
    Runs after a completed replan step and never executes tools.
    """
    if not planner.rebound_enabled:
        return None

    detected_faults = detect_rebound_faults(planner)
    if not detected_faults:
        return None

    for agent_id in detected_faults:
        mgr = planner.rebound_managers.get(agent_id)
        if mgr and agent_id in planner.last_high_level_actions:
            mgr.pending_high_level_action = planner.last_high_level_actions[agent_id]

    apply_rebound_context_modification(planner, detected_faults)
    print("[ReboundIntegration] Context modified. Continuing to replan with guidance...")
    return None


def apply_rebound_termination_guard(planner: "LLMPlanner", llm_response: str) -> bool:
    """
    Prevent premature termination based on progress perception.
    Returns True if termination was overridden.
    """
    if not planner.rebound_enabled:
        return False

    if not planner.is_done or not planner.check_if_agent_done(llm_response):
        return False

    for agent in planner.agents:
        mgr = planner.rebound_managers.get(agent.uid)
        if not mgr:
            continue

        progress = mgr.current_progress if mgr.current_progress is not None else 0.0
        
        # Check if we should allow termination despite low progress to avoid infinite loops
        max_termination_blocks = planner.rebound_config.get("max_termination_blocks", 5)
        
        if progress == 0.0:
            # Check if progress has improved since last block
            if progress > mgr.last_progress_at_termination:
                # Progress made, reset counter
                mgr.termination_block_count = 0
                mgr.last_progress_at_termination = progress
            
            if mgr.termination_block_count >= max_termination_blocks:
                print(
                    f"[ReboundIntegration] Termination guard limit reached ({mgr.termination_block_count}). "
                    f"Allowing termination despite progress={progress:.2f}."
                )
                return False

            # Grace period check: Prevent termination but don't log as fault if very early
            grace_period = planner.rebound_config.get("termination_grace_period", 10)
            if planner.replanning_count < grace_period:
                print(
                    f"[ReboundIntegration] Early termination detected at step {planner.replanning_count}. "
                    "Forcing continuation (Grace Period)."
                )
                planner.is_done = False
                planner.replan_required = True
                return True

            mgr.termination_block_count += 1
            # Log the intervention for metric tracking
            mgr.log_termination_intervention(planner.replanning_count)
            print(
                "[ReboundIntegration] Premature termination detected "
                f"(progress={progress:.2f}). Forcing continuation."
                f"(Block {mgr.termination_block_count}/{max_termination_blocks})"
            )
            planner.is_done = False
            planner.replan_required = True
            return True

    return False
