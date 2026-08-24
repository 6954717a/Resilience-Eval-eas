# -*- coding: utf-8 -*-
"""
Common prompt definitions and LLM interaction utilities for evaluation.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Constants
THRESHOLD = 0

# ==============================================================================
# ADCA (Attribution-Driven Credit Assignment) Prompts
# ==============================================================================

ADCA_SYSTEM_PROMPT = f"""You are an expert *process reward evaluator*, specializing in **attributional analysis** of multi-step solution trajectories.

**INPUT STRUCTURE:** The single message you receive always contains three labelled sections:
  1.  **TASK DESCRIPTION**   – The user's original request.
  2.  **SOLUTION TRAJECTORY** – A strictly numbered list of assistant steps. Each step describes an `ACTION` taken (and optionally an `OBSERVATION`).
  3.  **OVERALL REWARD SCORE** – A scalar value representing the environment's final judgment on task completion. **>0** indicates the task was **successfully completed**. **≤0** indicates the task **failed or was incomplete**.

**YOUR TASK (STEP-LEVEL ATTRIBUTION):** Analyze how each step contributed to the final task outcome (success/failure).  
Judge **each step independently** based only on the provided ACTION/OBSERVATION and how later steps use (or fail to use) its results. The OVERALL score provides **context**, but must **not** determine labels by itself.

**EVALUATION RULES:**

*   **If OVERALL REWARD SCORE is POSITIVE (> {THRESHOLD:+.1f}) – SUCCESSFUL COMPLETION:**
    *   Mark a step as **GOOD** if it **directly advanced** the successful outcome or **enabled** later successful steps (e.g., establishing a needed state, retrieving information that is subsequently used, narrowing the solution path, or correctly finalizing).
    *   Mark a step as **BAD** if it was **irrelevant, redundant without new effect, later undone without net benefit, or counterproductive**.

*   **If OVERALL REWARD SCORE is NON-POSITIVE (≤ {THRESHOLD:+.1f}) – TASK FAILURE:**
    *   Mark a step as **GOOD** **only if** it **genuinely reduced the distance to a correct solution** — for example, by establishing a necessary precondition later relied upon, correcting an earlier direction with observable improvement, validating or narrowing in a way later steps actually use, or preventing deterioration that would otherwise occur.
    *   Mark a step as **BAD** if it **failed to advance** the solution (no new effect, outputs unused downstream, or repetition with no added value), **obscured or impeded** progress, or **finalized** an incorrect result.

**GUARDRAILS:**
* Do **not** label all steps the same unless every step independently meets the same criterion.
* **Unused or later-discarded outputs** generally indicate **BAD** unless the step’s value is still demonstrable elsewhere.
* **Finalization** that submits an incorrect or incomplete outcome must be **BAD**.
* Keep analyses **concise** and **text-bound** (no external assumptions).

**FOCUS:** Judge based on **objective contribution to task completion**, not effort or intent.

**OUTPUT FORMAT:** Reply IN THE REQUIRED OUTPUT FORMAT and output nothing else."""


def get_positive_mask(score: float, threshold: float = THRESHOLD) -> bool:
    """Determines whether score is positive based on the given threshold."""
    return score > threshold


def build_adca_prompt(
    query: str,
    trajectory: List[Dict[str, str]],
    outcome_score: float,
    max_step_chars: int = 2000
) -> List[Dict[str, str]]:
    """
    Constructs the evaluation prompt for ADCA.
    """
    is_pos = get_positive_mask(outcome_score)
    polarity = "positive" if is_pos else "negative"

    def _trim(s: str) -> str:
        if not s: return ""
        return s if len(s) <= max_step_chars else s[:max_step_chars] + "\n…"

    user_parts = [
        "### TASK DESCRIPTION",
        query,
        "",
        f"### SOLUTION TRAJECTORY  (total {len(trajectory)} steps)",
    ]

    for i, st in enumerate(trajectory):
        action_str = st.get("action", "")
        # Handle tuple/dict action representation if necessary
        if not isinstance(action_str, str):
            action_str = str(action_str)

        block = [
            f">>> EVAL-STEP {i} <<<",
            "<|ACTION|>",
            _trim(action_str),
            "<|END|>",
        ]
        obs = st.get("observation")
        if obs:
            if not isinstance(obs, str):
                obs = str(obs)
            block += ["<|OBSERVATION|>", _trim(obs), "<|END|>"]
        user_parts.append("\n".join(block))

    user_parts += [
        "",
        "---",
        f"**OVERALL REWARD SCORE {outcome_score:+.4f} ({polarity})**",
        "Evaluation reminder:",
        "• Positive SCORE → Did this step IMPROVE the answer?",
        "• Negative SCORE → DIAGNOSIS + FIX + EVIDENCE (quoted). If evidence missing → BAD.",
        "  (Continuing wrong plan / repeating same failure / finalising wrong result → BAD)",
        "",
        "REQUIRED OUTPUT FORMAT:",
        "Step 0 Analysis: <your reasoning>",
        "Step 0 Judgment: GOOD/BAD",
        "",
        "Step 1 Analysis: <your reasoning>",
        "Step 1 Judgment: GOOD/BAD",
        "",
        "[…continue for all steps…]",
    ]

    return [
        {"role": "system", "content": ADCA_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


# ==============================================================================
# Reward Shaper Prompts
# ==============================================================================

REWARD_SHAPING_SYSTEM_PROMPT = """You are an expert evaluator for robotic task execution.
You are assisting in training a robot by providing shaped reward signals.

Your Goal: Evaluate the quality of the current action in the context of the entire task history.

Consider the following criteria:
1.  **Goal Progress**: Did the action move closer to completing the task goal?
2.  **Action Rationality**: Was the action reasonable given the current state and task?
    *   **Exploration**: In early steps, exploration is valuable if the target is unknown.
    *   **Logical Consistency**: Are actions logically sequenced? (e.g., Navigate -> Pick -> Navigate -> Place).
    *   **Preconditions**: Does the action respect preconditions (e.g., holding an object before placing)?
3.  **Efficiency**: Was this an efficient step, or could a better action have been taken?
4.  **Perception & Completion**: Consider if the robot has successfully perceived the necessary objects and if the task is nearing completion.

Respond ONLY with a JSON object in this exact format:
{
    "goal_progress_score": <float between -1.0 and 1.0>,
    "rationality_score": <float between -1.0 and 1.0>,
    "efficiency_score": <float between -1.0 and 1.0>,
    "explanation": "<brief reasoning>"
}
Do NOT wrap the JSON in markdown or code fences."""

REWARD_SHAPING_USER_TEMPLATE = """Task Goal: {task_instruction}

Current Step: {step_count}

Context:
- Task Percent Complete: {task_percent_complete:.2f}
- Perception Complete: {perception_complete}

Previous Actions (Last 5):
{action_history}

Action Response:
{action_response}

Previous State: {state_description}

Action Taken: {action_description}

Next State: {next_state_description}

Base Environment Reward: {base_reward}"""


def build_reward_shaping_prompt(
    task_instruction: str,
    state_description: str,
    action_description: str,
    next_state_description: str,
    base_reward: float,
    step_count: int = 0,
    action_history: str = "None",
    task_percent_complete: float = 0.0,
    perception_complete: str = "Unknown",
    action_response: str = "None"
) -> List[Dict[str, str]]:
    """
    Constructs the evaluation prompt for Reward Shaper.
    Splits into System and User messages.
    """
    user_content = REWARD_SHAPING_USER_TEMPLATE.format(
        task_instruction=task_instruction,
        state_description=state_description,
        action_description=action_description,
        next_state_description=next_state_description,
        base_reward=base_reward,
        step_count=step_count,
        action_history=action_history,
        task_percent_complete=task_percent_complete,
        perception_complete=perception_complete,
        action_response=action_response
    )

    return [
        {"role": "system", "content": REWARD_SHAPING_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


# ==============================================================================
# Common LLM Interaction
# ==============================================================================

def call_llm_completion(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 1024,
    json_mode: bool = False
) -> Optional[str]:
    """
    Common wrapper for calling LLM chat completions.
    
    Args:
        client: The LLM client (e.g., OpenAI client).
        model: Model name.
        messages: List of message dicts.
        temperature: Sampling temperature.
        max_tokens: Max output tokens.
        json_mode: Whether to enforce JSON response (if supported).

    Returns:
        The response content string, or None if failed.
    """
    try:
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            # Note: response_format={"type": "json_object"} is supported by recent OpenAI models
            # We can conditionally add it if needed, or leave it to prompt instructions
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error(f"LLM API call failed: {e}")
        return None
