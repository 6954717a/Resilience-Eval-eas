"""
Trace Extractor Module

Extracts key information from execution traces for LLM context generation.
Focuses on failure points and action sequences to enable precise suggestions.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def extract_trace_summary(
    trace_content: str,
    failure_points: Optional[List[int]] = None,
    max_lines: int = 8
) -> str:
    """
    Extract a concise summary of trace focusing on failures and key actions.
    
    Args:
        trace_content: Full trace file content as string
        failure_points: Optional list of failure step numbers from analysis
        max_lines: Maximum lines to include in summary
    
    Returns:
        Formatted trace summary string for LLM consumption
        
    Example output:
        Step 58: Pick[toy_bee_1] -> FAILED (not close enough)
        Step 68: Navigate[toy_bee_1] -> OK
        Step 81: Pick[toy_bee_1] -> OK
        Step 94: Rearrange[toy_bee_1] -> FAILED (already held)
    """
    lines = trace_content.splitlines()
    summary_entries: List[Tuple[int, str]] = []
    
    for i, line in enumerate(lines):
        # Detect failure lines
        if "Unexpected failure" in line or "Failed" in line:
            action = _extract_previous_action(lines, i)
            error = _extract_error_message(line)
            if action:
                summary_entries.append((i, f"{action} -> FAILED ({error})"))
        
        # Also capture successful actions that follow failures
        elif "Successful execution" in line and summary_entries:
            # Check if previous action exists
            action = _extract_previous_action(lines, i)
            if action:
                last_step = summary_entries[-1][0] if summary_entries else 0
                # Only include if close to a failure (within 20 lines)
                if i - last_step < 20:
                    summary_entries.append((i, f"{action} -> OK"))
    
    # Sort by step number and limit
    summary_entries.sort(key=lambda x: x[0])
    summary_entries = summary_entries[:max_lines]
    
    # Format output
    result_lines = []
    for step, entry in summary_entries:
        result_lines.append(f"Step {step}: {entry}")
    
    return "\n".join(result_lines) if result_lines else "No failures detected in trace"


def extract_failure_actions(trace_content: str) -> List[Dict[str, Any]]:
    """
    Extract detailed information about failed actions from trace.
    
    Args:
        trace_content: Full trace file content
        
    Returns:
        List of failure info dicts with keys:
        - step: int
        - action_type: str (e.g., "Pick", "Place")
        - action_full: str (full action string)
        - error: str
        - preceding_action: Optional[str]
    """
    lines = trace_content.splitlines()
    failures: List[Dict[str, Any]] = []
    
    for i, line in enumerate(lines):
        if "Unexpected failure" not in line and "Failed" not in line:
            continue
            
        # Extract action from previous lines
        action_full = _extract_previous_action(lines, i)
        if not action_full:
            continue
        
        # Parse action type
        action_type = action_full.split("[")[0] if "[" in action_full else action_full
        
        # Get error message
        error = _extract_error_message(line)
        
        # Get the action before this one (for context)
        preceding_action = None
        for j in range(i - 2, max(0, i - 10), -1):
            if "Action:" in lines[j]:
                preceding_action = _parse_action_line(lines[j])
                break
        
        failures.append({
            "step": i,
            "action_type": action_type,
            "action_full": action_full,
            "error": error,
            "preceding_action": preceding_action,
        })
    
    return failures


def extract_task_from_trace(trace_content: str) -> str:
    """Extract the task instruction from trace file."""
    lines = trace_content.splitlines()
    for line in lines[:5]:  # Task is usually in first few lines
        if line.startswith("Task:"):
            return line[5:].strip()
    return "Unknown task"


def load_trace_file(trace_path: Path) -> Optional[str]:
    """Load trace file content with error handling."""
    if not trace_path or not trace_path.exists():
        return None
    try:
        return trace_path.read_text(encoding="utf-8")
    except Exception:
        return None


# =============================================================================
# Internal Helper Functions
# =============================================================================

def _extract_previous_action(lines: List[str], current_idx: int) -> Optional[str]:
    """Find the most recent action before the current line."""
    for j in range(current_idx - 1, max(0, current_idx - 5), -1):
        action = _parse_action_line(lines[j])
        if action:
            return action
    return None


def _parse_action_line(line: str) -> Optional[str]:
    """Parse an action from a line like 'Agent_0_Action: Pick[object]'."""
    # Match patterns like "Agent_0_Action: Pick[...]" or "Agent_1_Action: Navigate[...]"
    match = re.search(r'Agent_\d+_Action:\s*(.+?)(?:\s*$|<\|)', line)
    if match:
        return match.group(1).strip()
    return None


def _extract_error_message(line: str) -> str:
    """Extract concise error message from failure line."""
    # Common error patterns
    patterns = [
        (r"Not close enough", "not close enough"),
        (r"occluded", "occluded"),
        (r"not found", "object not found"),
        (r"already held", "already held"),
        (r"not holding", "not holding object"),
        (r"Failed to pick", "pick failed"),
        (r"Failed to place", "place failed"),
        (r"Failed to navigate", "navigation failed"),
    ]
    
    line_lower = line.lower()
    for pattern, message in patterns:
        if pattern.lower() in line_lower:
            return message
    
    # Fallback: extract text after "Failed" or "-"
    if " - " in line:
        error_part = line.split(" - ", 1)[1]
        # Truncate to reasonable length
        return error_part[:40].strip()
    
    return "execution error"


def summarize_failures_for_prompt(failures: List[Dict[str, Any]], max_failures: int = 5) -> str:
    """
    Convert failure list to a compact string for LLM prompts.
    
    Example output:
        Pick[toy_bee_1] failed (not close enough) after Navigate[bedroom_2]
        Rearrange[toy_bee_1] failed (already held) after Pick[toy_bee_1]
    """
    if not failures:
        return "No failures recorded"
    
    lines = []
    for f in failures[:max_failures]:
        action = f.get("action_full", "Unknown")
        error = f.get("error", "error")
        preceding = f.get("preceding_action")
        
        if preceding:
            lines.append(f"{action} failed ({error}) after {preceding}")
        else:
            lines.append(f"{action} failed ({error})")
    
    return "\n".join(lines)
