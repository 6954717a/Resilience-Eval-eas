from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING, Tuple
import json
import numpy as np

if TYPE_CHECKING:
    from habitat_llm.planner.miqp_planner.params_module.scenario_params_task import ScenarioConfigTask

def extract_json_from_text(text: str, target_type: type = dict) -> Optional[str]:
    """Extract a JSON fragment from an LLM response with surrounding text."""
    if target_type == dict:
        start_char, end_char = '{', '}'
    elif target_type == list:
        start_char, end_char = '[', ']'
    else:
        return None

    try:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end > start:
            return text[start:end+1]
    except Exception:
        pass
    return None

def get_llm_config() -> Dict[str, Any]:
    """Return the role-tag configuration for the LLM planner."""
    return {
        'system_tag': '[SYSTEM]',
        'user_tag': '[USER]', 
        'assistant_tag': '[ASSISTANT]',
        'eot_tag': '[EOT]'
    }

def update_param_value(scenario_config: Union[Dict[str, Any], Any], key: str, value: Any) -> None:
    if isinstance(scenario_config, dict):
        scenario_config[key] = value
    else:
        try:
            if hasattr(scenario_config, 'scenario_params') and key in ['A', 'T', 'ws', 'Hs']:
                scenario_config.scenario_params[key] = value
                print(f"DEBUG: Updated scenario_params['{key}'] in ScenarioConfigTask")
            elif hasattr(scenario_config, 'update_global_task_var'):
                scenario_config.update_global_task_var(key, value)
        except AttributeError:
            print(f"Error: scenario_config is neither a dictionary nor an object with a supported update method")

def normalize_task_list(
    tasks: List[Dict[str, Any]],
    required_fields: List[str],
    defaults: Optional[Dict[str, Any]] = None,
    add_task_id: bool = False,
    clamp_priority: Optional[Tuple[int, int]] = None,
    clamp_duration: Optional[Tuple[float, float]] = None,
    include_can_parallel: bool = False,
    include_phase_group: bool = False,
    clean_prereqs: bool = False,
    log_invalid_prereqs: bool = False,
) -> List[Dict[str, Any]]:
    """
    Normalize a list of task dictionaries produced by LLMs.
    """
    if not tasks:
        return []

    defaults = defaults or {}
    normalized_tasks: List[Dict[str, Any]] = []

    for idx, task in enumerate(tasks):
        if not isinstance(task, dict) or not all(field in task for field in required_fields):
            continue

        normalized: Dict[str, Any] = {}
        if add_task_id:
            normalized["task_id"] = f"task_{idx}"

        for field in required_fields:
            normalized[field] = task[field]

        for key, default_value in defaults.items():
            normalized[key] = task.get(key, default_value)

        if "priority" in normalized and clamp_priority is not None:
            min_p, max_p = clamp_priority
            normalized["priority"] = max(min_p, min(max_p, normalized["priority"]))

        if "estimated_duration" in normalized and clamp_duration is not None:
            min_d, max_d = clamp_duration
            duration = normalized["estimated_duration"]
            normalized["estimated_duration"] = max(min_d, min(max_d, duration))

        if include_can_parallel:
            normalized["can_parallel"] = bool(
                task.get("can_parallel", defaults.get("can_parallel", False))
            )
        if include_phase_group:
            normalized["phase_group"] = task.get(
                "phase_group", defaults.get("phase_group", "execution")
            )

        prereqs = task.get("prerequisites", defaults.get("prerequisites", []))
        should_clean = clean_prereqs or add_task_id or log_invalid_prereqs
        if not isinstance(prereqs, list):
            prereqs = [] if should_clean else prereqs
        if should_clean:
            cleaned_prereqs = [
                p.strip() for p in prereqs if isinstance(p, str) and p.strip()
            ]
            if add_task_id and "task_id" in normalized:
                cleaned_prereqs = [
                    p for p in cleaned_prereqs if p != normalized["task_id"]
                ]
            normalized["prerequisites"] = cleaned_prereqs
        else:
            normalized["prerequisites"] = prereqs

        normalized_tasks.append(normalized)

    if add_task_id and normalized_tasks:
        valid_ids = {task["task_id"] for task in normalized_tasks}
        for task in normalized_tasks:
            prereqs = task.get("prerequisites", [])
            invalid = [p for p in prereqs if p not in valid_ids]
            if invalid and log_invalid_prereqs:
                print(
                    f"WARNING: Removed invalid prerequisites {invalid} from task {task.get('task_id')}"
                )
            task["prerequisites"] = [p for p in prereqs if p in valid_ids]

    return normalized_tasks
