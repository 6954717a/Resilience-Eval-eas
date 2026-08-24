"""
Unified Context Management Module

Provides context compression, formatting, and LLM-based refinement for:
- Evolve: Episode history and batch-level advice
- Rebound: Fault detection and recovery guidance
- Phase: Phase-based planning guidance
"""

from habitat_llm.context.context_compressor import ContextCompressor, CompressionConfig
from habitat_llm.context.trace_extractor import (
    extract_trace_summary,
    extract_failure_actions,
)
from habitat_llm.context.prompts import (
    LLMTags,
    SUGGESTION_REFINEMENT_PROMPT,
    BATCH_ADVICE_PROMPT,
    REBOUND_GUIDANCE_PROMPT,
    CONTEXT_HISTORY_COMPRESSION_PROMPT,
    format_context_update_as_turn,
    format_context_for_prompt_builder,
    format_batch_summaries,
    format_dialog_turns,
    format_failure_patterns,
    format_protected_keywords,
)

__all__ = [
    "ContextCompressor",
    "CompressionConfig",
    "LLMTags",
    "extract_trace_summary",
    "extract_failure_actions",
    "SUGGESTION_REFINEMENT_PROMPT",
    "BATCH_ADVICE_PROMPT",
    "REBOUND_GUIDANCE_PROMPT",
    "CONTEXT_HISTORY_COMPRESSION_PROMPT",
    "format_context_update_as_turn",
    "format_context_for_prompt_builder",
    "format_batch_summaries",
    "format_dialog_turns",
    "format_failure_patterns",
    "format_protected_keywords",
]
