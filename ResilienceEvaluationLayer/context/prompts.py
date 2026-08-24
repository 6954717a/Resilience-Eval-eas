"""
LLM Prompt Templates for Context Refinement

All prompts are designed for Qwen3-8B-Instruct and similar models.
Output length is strictly controlled via prompt instructions and max_tokens.

Tag Format Support:
- Qwen/ChatML: <|im_start|>role\\n ... <|im_end|>\\n
- LLaMA: <|start_header_id|>role<|end_header_id|>\\n\\n ... <|eot_id|>
"""

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class LLMTags:
    """LLM format tags configuration."""
    system_tag: str = "<|im_start|>system\n"
    user_tag: str = "<|im_start|>user\n"
    assistant_tag: str = "<|im_start|>assistant\n"
    eot_tag: str = "<|im_end|>\n"
    
    @classmethod
    def qwen(cls) -> "LLMTags":
        """Qwen/ChatML format tags."""
        return cls(
            system_tag="<|im_start|>system\n",
            user_tag="<|im_start|>user\n",
            assistant_tag="<|im_start|>assistant\n",
            eot_tag="<|im_end|>\n",
        )
    
    @classmethod
    def llama(cls) -> "LLMTags":
        """LLaMA format tags."""
        return cls(
            system_tag="<|start_header_id|>system<|end_header_id|>\n\n",
            user_tag="<|start_header_id|>user<|end_header_id|>\n\n",
            assistant_tag="<|start_header_id|>assistant<|end_header_id|>\n\n",
            eot_tag="<|eot_id|>",
        )
    
    @classmethod
    def from_config(cls, config) -> "LLMTags":
        """Create tags from planner config."""
        if config is None:
            return cls.qwen()
        try:
            return cls(
                system_tag=getattr(config, "system_tag", cls.qwen().system_tag),
                user_tag=getattr(config, "user_tag", cls.qwen().user_tag),
                assistant_tag=getattr(config, "assistant_tag", cls.qwen().assistant_tag),
                eot_tag=getattr(config, "eot_tag", cls.qwen().eot_tag),
            )
        except Exception:
            return cls.qwen()


# =============================================================================
# Suggestion Refinement Prompt (Episode-level)
# =============================================================================
SUGGESTION_REFINEMENT_PROMPT = """You are an AI Agent Coach for a household robot. Generate exactly 2 action-oriented suggestions based on the execution analysis.

## Task
{task_instruction}

## Execution Summary
- Quality Score: {quality_score}
- Root Causes: {root_causes}

## Trace Excerpt (failures)
{trace_summary}

## Available Actions
Navigate[target], Pick[object], Place[object, on, furniture], Rearrange[object, on, furniture], Explore[room], Wait[], Done[]

## Rules
1. Each suggestion MUST reference a specific Action name (e.g., "Before Pick, always Navigate to the object first")
2. Each suggestion MUST be 25 words or less
3. Be specific, not generic (bad: "improve planning", good: "Navigate to furniture before Place")
4. Output ONLY a JSON list: ["suggestion1", "suggestion2"]

## Output:"""


# =============================================================================
# Batch Advice Prompt (Evolve batch-level)
# =============================================================================
BATCH_ADVICE_PROMPT = """You are an AI Agent Coach. Analyze the batch of episode summaries and generate 3 general advice items.

## Episode Summaries
{batch_summaries}

## Common Failure Patterns
{failure_patterns}

## Available Actions
Navigate[target], Pick[object], Place[object, on, furniture], Rearrange[object, on, furniture], Explore[room], Wait[], Done[]

## Rules
1. Each advice MUST reference a specific Action name
2. Each advice MUST be 20 words or less
3. Focus on preventing the most common failures
4. Output ONLY a JSON list: ["advice1", "advice2", "advice3"]

## Output:"""


# =============================================================================
# Rebound Guidance Prompt (Immediate fault recovery)
# =============================================================================
REBOUND_GUIDANCE_PROMPT = """Based on the execution failure below, generate a precise recovery instruction.

## Failed Action
{failed_action}

## Error Message
{error_message}

## Current State
{current_state}

## Rules
1. Output a single, actionable instruction (15 words or less)
2. Reference specific Action names
3. Do NOT output JSON, just the plain text instruction

## Recovery Instruction:"""


# =============================================================================
# Dialog-Preserving Context Compression Prompt
# =============================================================================
CONTEXT_HISTORY_COMPRESSION_PROMPT = """You compress long household-robot planning context while preserving dialog form.

## Task Instruction
{task_instruction}

## Protected Entity Names
These object, furniture, and room names must remain verbatim if they are mentioned:
{protected_keywords}

## Old Dialog History To Compress
{chat_history}

## Trajectory Summary
{trajectory_summary}

## World Description
{world_description}

## Agent Description
{agent_description}

## Rules
1. Return ONLY valid JSON.
2. Use this schema:
{{
  "summary_turns": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ],
  "compressed_world_description": "...",
  "compressed_agent_description": "..."
}}
3. Keep the output compatible with the existing chat log format. Do not invent a new section structure.
4. Preserve object/furniture names exactly. Do not rename, alias, or abbreviate them.
5. Summarize only older context. Do not rewrite the task, system prompt, action schema, or recent turns.
6. Keep only facts useful for the next action: verified locations, completed subgoals, failed attempts, blockers, and active commitments.
7. ``summary_turns`` must contain 2 to 6 turns total and should read like a compact continuation of the existing dialog.
8. Keep ``compressed_world_description`` and ``compressed_agent_description`` shorter than the originals while preserving task-relevant entities.

## Output JSON:"""


# =============================================================================
# Context Update Formatting (for injection into conversation)
# =============================================================================

def format_context_update_as_turn(
    title: str,
    content: str,
    tags: Optional[LLMTags] = None,
) -> str:
    """
    Format a context update as a proper user turn for injection.
    
    Args:
        title: Context update title (e.g., "Rebound Guidance")
        content: The context content
        tags: LLM format tags (defaults to Qwen/ChatML)
    
    Returns:
        Properly formatted user turn string
    
    Example (Qwen):
        <|im_start|>user
        [Rebound Guidance]
        Prior: Place failed (not close enough).
        Fix: Navigate to target first, then Place.
        <|im_end|>
    """
    if tags is None:
        tags = LLMTags.qwen()
    
    # Clean content
    content = content.strip()
    
    # Format as user turn with title
    formatted = (
        f"{tags.user_tag}"
        f"[{title}]\n"
        f"{content}\n"
        f"{tags.eot_tag}"
    )
    
    return formatted


def format_context_for_prompt_builder(
    title: str,
    content: str,
) -> str:
    """
    Format context for PromptBuilder's add_user_turn method.
    
    This format does NOT include tags (PromptBuilder adds them).
    
    Args:
        title: Context update title
        content: The context content
    
    Returns:
        Content string suitable for prompt_builder.add_user_turn()
    """
    content = content.strip()
    return f"[{title}]\n{content}"


def format_dialog_turns(turns: Iterable[dict], max_chars: Optional[int] = None) -> str:
    """Format dialog turns into a readable prompt block for summarization."""
    lines = []
    for turn in turns:
        role = str(turn.get("role", "user")).upper()
        content = str(turn.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{role}: {content}")
    text = "\n".join(lines) if lines else "No prior dialog turns."
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def format_protected_keywords(
    keywords: Iterable[str],
    max_items: int = 120,
    max_chars: int = 2000,
) -> str:
    """Format protected keywords for prompt injection."""
    cleaned = []
    seen = set()
    for keyword in keywords:
        text = str(keyword).strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    joined = ", ".join(cleaned) if cleaned else "None"
    if len(joined) > max_chars:
        return joined[: max_chars - 3] + "..."
    return joined


# =============================================================================
# Prompt Builder Utilities
# =============================================================================

def format_batch_summaries(summaries: list, max_episodes: int = 10) -> str:
    """Format episode summaries for batch advice prompt."""
    lines = []
    for ep in summaries[:max_episodes]:
        ep_id = ep.get("episode_id", "?")
        success = ep.get("task_state_success", 0)
        completion = ep.get("task_percent_complete", 0)
        
        analysis = ep.get("analysis", {}) or {}
        causes = analysis.get("root_causes", [])[:2]
        
        causes_str = "; ".join(causes) if causes else "none"
        lines.append(f"- Ep {ep_id}: success={success}, completion={completion:.0%}, causes=[{causes_str}]")
    
    return "\n".join(lines) if lines else "No episodes available"


def format_failure_patterns(summaries: list) -> str:
    """Extract common failure patterns from batch summaries."""
    from collections import Counter
    
    all_causes = []
    for ep in summaries:
        analysis = ep.get("analysis", {}) or {}
        all_causes.extend(analysis.get("root_causes", []))
    
    # Count and sort
    cause_counts = Counter(all_causes)
    top_causes = cause_counts.most_common(3)
    
    if not top_causes:
        return "No clear patterns detected"
    
    lines = []
    for cause, count in top_causes:
        # Truncate long causes
        cause_short = cause[:80] + "..." if len(cause) > 80 else cause
        lines.append(f"- ({count}x) {cause_short}")
    
    return "\n".join(lines)
