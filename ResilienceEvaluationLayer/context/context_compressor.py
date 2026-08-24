"""
Context Compressor Module

using Pattern Match method.

Unified context compression and LLM-based refinement for:
- Evolve: Episode history and batch-level advice
- Rebound: Fault detection and recovery guidance
- Phase: Phase-based planning guidance

This module ensures all context injected into the LLM is:
1. Action-oriented (references specific Action names)
2. Concise (strict character limits)
3. Non-redundant (doesn't repeat System Prompt rules)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from habitat_llm.context.prompts import (
    SUGGESTION_REFINEMENT_PROMPT,
    BATCH_ADVICE_PROMPT,
    REBOUND_GUIDANCE_PROMPT,
    CONTEXT_HISTORY_COMPRESSION_PROMPT,
    format_batch_summaries,
    format_dialog_turns,
    format_failure_patterns,
    format_protected_keywords,
)
from habitat_llm.context.trace_extractor import (
    extract_trace_summary,
    extract_failure_actions,
    summarize_failures_for_prompt,
    load_trace_file,
)

logger = logging.getLogger(__name__)


@dataclass
class CompressionConfig:
    """Configuration for context compression."""
    max_rebound_chars: int = 150
    max_evolve_chars: int = 300
    max_phase_chars: int = 200
    max_total_context_chars: int = 500
    default_max_tokens: int = 150
    default_temperature: float = 0.3
    enable_llm_refinement: bool = True
    model_name: str = "gpt-4"
    max_prompt_chars: int = 12000
    preserve_last_turns: int = 6
    protected_entity_sources: Tuple[str, ...] = (
        "task",
        "episode",
        "world_graph",
        "scene_graph",
    )
    llm_base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.1

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "CompressionConfig":
        if not payload:
            return cls()
        data = dict(payload)
        llm_cfg = data.get("llm", {}) or {}
        return cls(
            max_rebound_chars=int(data.get("max_rebound_chars", 150)),
            max_evolve_chars=int(data.get("max_evolve_chars", 300)),
            max_phase_chars=int(data.get("max_phase_chars", 200)),
            max_total_context_chars=int(data.get("max_total_context_chars", 500)),
            default_max_tokens=int(data.get("default_max_tokens", llm_cfg.get("max_tokens", 150))),
            default_temperature=float(data.get("default_temperature", llm_cfg.get("temperature", 0.3))),
            enable_llm_refinement=bool(data.get("enable_llm_refinement", True)),
            model_name=str(data.get("model_name", llm_cfg.get("model", "gpt-4"))),
            max_prompt_chars=int(data.get("max_prompt_chars", 12000)),
            preserve_last_turns=int(data.get("preserve_last_turns", 6)),
            protected_entity_sources=tuple(
                data.get(
                    "protected_entity_sources",
                    ("task", "episode", "world_graph", "scene_graph"),
                )
            ),
            llm_base_url=llm_cfg.get("base_url"),
            api_key_env=str(llm_cfg.get("api_key_env", "OPENAI_API_KEY")),
            llm_max_tokens=int(llm_cfg.get("max_tokens", 512)),
            llm_temperature=float(llm_cfg.get("temperature", 0.1)),
        )


class ContextCompressor:
    """
    Unified context compression and LLM-based refinement.
    
    Supports three context sources:
    - Rebound: Immediate fault recovery guidance
    - Evolve: Episode history and batch-level advice
    - Phase: Phase-based planning guidance
    
    Example usage:
        compressor = ContextCompressor(llm_client=my_client)
        
        # Rebound context
        guidance = compressor.compress_rebound_guidance(
            fault_type="immediate_error",
            action="Place[laptop_0, on, bench_10]",
            response="Failed to place! Not close enough."
        )
        
        # Evolve context with LLM refinement
        suggestions = compressor.refine_suggestions_with_llm(
            analysis=analysis_data,
            trace_path=trace_path
        )
    """
    
    # Character limits
    MAX_REBOUND_CHARS = 150
    MAX_EVOLVE_CHARS = 300
    MAX_PHASE_CHARS = 200
    MAX_TOTAL_CONTEXT_CHARS = 500
    
    # LLM settings
    DEFAULT_MAX_TOKENS = 150
    DEFAULT_TEMPERATURE = 0.3
    
    def __init__(
        self,
        llm_client: Any = None,
        model_name: str = "gpt-4",
        enable_llm_refinement: bool = True,
        config: Optional[CompressionConfig] = None,
    ):
        """
        Initialize the context compressor.
        
        Args:
            llm_client: Optional LLM client (OpenAI-compatible or local)
            model_name: Model name for API calls
            enable_llm_refinement: Whether to use LLM for suggestion refinement
            config: Optional configuration object
        """
        self.llm_client = llm_client
        
        # Use config if provided, otherwise use defaults/args
        if config:
            self.config = config
            self.model_name = config.model_name
            self.enable_llm_refinement = config.enable_llm_refinement and llm_client is not None
            
            # Apply limits from config
            self.MAX_REBOUND_CHARS = config.max_rebound_chars
            self.MAX_EVOLVE_CHARS = config.max_evolve_chars
            self.MAX_PHASE_CHARS = config.max_phase_chars
            self.MAX_TOTAL_CONTEXT_CHARS = config.max_total_context_chars
            self.DEFAULT_MAX_TOKENS = config.default_max_tokens
            self.DEFAULT_TEMPERATURE = config.default_temperature
        else:
            self.config = CompressionConfig() # Default config
            self.model_name = model_name
            self.enable_llm_refinement = enable_llm_refinement and llm_client is not None
        self._maybe_init_openai_client()

    def _maybe_init_openai_client(self) -> None:
        """Optionally build an OpenAI-compatible client from config."""
        if self.llm_client is not None:
            return
        if not self.config.llm_base_url:
            return
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            logger.warning(
                "Context compression LLM base_url is set but env %s is missing.",
                self.config.api_key_env,
            )
            return
        try:
            from openai import OpenAI

            self.llm_client = OpenAI(
                base_url=self.config.llm_base_url,
                api_key=api_key,
            )
            self.enable_llm_refinement = True
        except Exception as exc:
            logger.warning("Failed to initialize context compression client: %s", exc)
    
    # =========================================================================
    # Rebound Context Compression
    # =========================================================================
    
    def compress_rebound_guidance(
        self,
        fault_type: str,
        action: str,
        response: str,
        use_llm: bool = False,
        current_state: Optional[str] = None,
    ) -> str:
        """
        Compress Rebound fault information into action-oriented guidance.
        
        Args:
            fault_type: Type of fault (e.g., "immediate_error", "tool_failure_burst")
            action: The action that failed
            response: Error response from the action
            use_llm: Whether to use LLM for refinement (default: use rules)
            current_state: Optional current world state description
        
        Returns:
            Compressed guidance string (≤ MAX_REBOUND_CHARS)
        """
        if use_llm and self.enable_llm_refinement:
            return self._llm_rebound_guidance(action, response, current_state)
        
        return self._rule_based_rebound_guidance(action, response)
    
    def _rule_based_rebound_guidance(self, action: str, response: str) -> str:
        """Generate rebound guidance using predefined rules."""
        action_type = action.split("[")[0] if "[" in action else action
        error_keywords = self._extract_error_keywords(response)
        fix = self._generate_fix(action_type, error_keywords)
        
        result = (
            "[Context Update]\n"
            f"Prior: {action_type} failed ({error_keywords}).\n"
            f"Fix: {fix}"
        )
        
        return result[:self.MAX_REBOUND_CHARS]
    
    def _llm_rebound_guidance(
        self,
        action: str,
        response: str,
        current_state: Optional[str]
    ) -> str:
        """Use LLM to generate rebound guidance."""
        prompt = REBOUND_GUIDANCE_PROMPT.format(
            failed_action=action[:100],
            error_message=response[:150],
            current_state=current_state[:200] if current_state else "Unknown",
        )
        
        result = self._call_llm(prompt, max_tokens=50)
        if result:
            return f"[Context Update]\n{result[:self.MAX_REBOUND_CHARS - 20]}"
        
        # Fallback to rule-based
        return self._rule_based_rebound_guidance(action, response)
    
    # =========================================================================
    # Evolve Context Compression
    # =========================================================================
    
    def compress_evolve_guidance(
        self,
        summary: Dict[str, Any],
        trace_path: Optional[Path] = None,
    ) -> Optional[str]:
        """
        Compress Evolve episode summary into guidance.
        
        Args:
            summary: Episode summary dict from context_evolve
            trace_path: Optional path to trace file
        
        Returns:
            Compressed guidance string or None if not needed
        """
        completion = float(summary.get("task_percent_complete", 0.0))
        
        # Skip if task was successful
        if completion >= 1.0:
            return None
        
        # Get analysis info
        analysis = summary.get("analysis", {}) or {}
        root_causes = analysis.get("root_causes", [])[:2]
        
        # Get failure info from reward shaper
        reward = summary.get("reward_shaper", {}) or {}
        negative_actions = reward.get("top_negative_actions", [])
        
        if not root_causes and not negative_actions:
            return None
        
        # Build compact recap
        failed_action = negative_actions[0].split("(")[0] if negative_actions else "unknown"
        tip = self._generate_action_tip(failed_action)
        
        result = (
            "[Episode Recap]\n"
            f"Last: {int(completion * 100)}% complete, issue at {failed_action}.\n"
            f"Tip: {tip}"
        )
        
        return result[:self.MAX_EVOLVE_CHARS]
    
    def refine_suggestions_with_llm(
        self,
        analysis: Dict[str, Any],
        trace_path: Optional[Path] = None,
    ) -> List[str]:
        """
        Use LLM to refine analysis suggestions into action-oriented advice.
        
        Args:
            analysis: Analysis dict from analysis_*.json
            trace_path: Optional path to trace file
        
        Returns:
            List of 2 refined suggestions (each ≤ 100 chars)
        """
        if not self.enable_llm_refinement:
            return self._rule_based_suggestions(analysis)
        
        # Load and process trace
        trace_summary = "No trace available"
        if trace_path:
            trace_content = load_trace_file(trace_path)
            if trace_content:
                failures = extract_failure_actions(trace_content)
                trace_summary = summarize_failures_for_prompt(failures)
        
        # Build prompt
        prompt = SUGGESTION_REFINEMENT_PROMPT.format(
            task_instruction=analysis.get("task_instruction", "Unknown task")[:150],
            quality_score=f"{analysis.get('execution_quality_score', 0.0):.2f}",
            root_causes="; ".join(analysis.get("root_causes", [])[:2])[:200],
            trace_summary=trace_summary[:400],
        )
        
        # Call LLM
        response = self._call_llm(prompt, max_tokens=self.DEFAULT_MAX_TOKENS)
        
        # Parse JSON response
        suggestions = self._parse_json_list(response)
        if suggestions:
            # Enforce length limits
            return [s[:100] for s in suggestions[:2]]
        
        # Fallback
        return self._rule_based_suggestions(analysis)
    
    def generate_batch_advice(
        self,
        summaries: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Generate batch-level advice using LLM.
        
        Args:
            summaries: List of episode summary dicts
        
        Returns:
            List of 3 advice items (each ≤ 80 chars)
        """
        if not self.enable_llm_refinement or not summaries:
            return self._rule_based_batch_advice(summaries)
        
        # Build prompt
        prompt = BATCH_ADVICE_PROMPT.format(
            batch_summaries=format_batch_summaries(summaries)[:600],
            failure_patterns=format_failure_patterns(summaries)[:300],
        )
        
        # Call LLM
        response = self._call_llm(prompt, max_tokens=self.DEFAULT_MAX_TOKENS)
        
        # Parse JSON response
        advice = self._parse_json_list(response)
        if advice:
            return [a[:80] for a in advice[:3]]
        
        return self._rule_based_batch_advice(summaries)
    
    # =========================================================================
    # Phase Context Compression
    # =========================================================================
    
    def compress_phase_guidance(
        self,
        phase_name: str,
        phase_advice: str,
    ) -> str:
        """
        Compress phase-based guidance.
        
        Args:
            phase_name: Name of current phase (e.g., "exploration", "execution")
            phase_advice: Original phase advice
        
        Returns:
            Compressed phase guidance string
        """
        # Phase guidance is usually already concise, just enforce limit
        result = f"[Phase: {phase_name}]\n{phase_advice}"
        return result[:self.MAX_PHASE_CHARS]

    # =========================================================================
    # Dialog / Prompt Compression
    # =========================================================================

    def compress_prompt_state(
        self,
        *,
        task_instruction: str,
        prompt_history: List[Dict[str, str]],
        trajectory_summary: str,
        world_description: str,
        agent_description: str,
        protected_keywords: Optional[List[str]] = None,
        preserve_last_turns: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Compress older dialog turns plus world/agent descriptions into a
        dialog-preserving prompt state update.
        """
        if not prompt_history:
            return None

        keep_last_turns = preserve_last_turns or self.config.preserve_last_turns
        keep_last_turns = max(1, int(keep_last_turns))
        old_history = (
            prompt_history[:-keep_last_turns]
            if len(prompt_history) > keep_last_turns
            else prompt_history[:-1]
        )
        if not old_history and not world_description and not agent_description:
            return None

        if self.enable_llm_refinement:
            payload = self._llm_prompt_state_compression(
                task_instruction=task_instruction,
                old_history=old_history,
                trajectory_summary=trajectory_summary,
                world_description=world_description,
                agent_description=agent_description,
                protected_keywords=protected_keywords or [],
            )
            if payload:
                return payload

        return self._fallback_prompt_state_compression(
            old_history=old_history,
            world_description=world_description,
            agent_description=agent_description,
        )
    
    # =========================================================================
    # Context Combination
    # =========================================================================
    
    def combine_contexts(
        self,
        rebound_context: Optional[str] = None,
        evolve_context: Optional[str] = None,
        phase_context: Optional[str] = None,
    ) -> Optional[str]:
        """
        Combine multiple context sources with priority and length management.
        
        Priority: Rebound > Phase > Evolve
        
        Args:
            rebound_context: Immediate fault guidance
            evolve_context: Episode history guidance
            phase_context: Phase-based guidance
        
        Returns:
            Combined context string or None if all empty
        """
        parts = []
        remaining_chars = self.MAX_TOTAL_CONTEXT_CHARS
        
        # Priority 1: Rebound (immediate issues)
        if rebound_context and remaining_chars > 0:
            truncated = rebound_context[:min(len(rebound_context), remaining_chars)]
            parts.append(truncated)
            remaining_chars -= len(truncated) + 2  # +2 for separator
        
        # Priority 2: Phase (current phase guidance)
        if phase_context and remaining_chars > 50:
            truncated = phase_context[:min(len(phase_context), remaining_chars)]
            parts.append(truncated)
            remaining_chars -= len(truncated) + 2
        
        # Priority 3: Evolve (historical guidance)
        if evolve_context and remaining_chars > 50:
            truncated = evolve_context[:min(len(evolve_context), remaining_chars)]
            parts.append(truncated)
        
        if not parts:
            return None
        
        return "\n\n".join(parts)
    
    # =========================================================================
    # LLM Utilities
    # =========================================================================
    
    def _call_llm(self, prompt: str, max_tokens: int = 150) -> str:
        """Call LLM with prompt and return response text."""
        if not self.llm_client:
            return ""
        
        try:
            # OpenAI-compatible client
            if hasattr(self.llm_client, "chat") and hasattr(self.llm_client.chat, "completions"):
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.llm_temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            
            # Local LLM client with generate method
            elif hasattr(self.llm_client, "generate"):
                return self.llm_client.generate(
                    prompt,
                    stop=None,
                    max_length=max_tokens,
                ) or ""
            
            # Fallback for other interfaces
            elif callable(self.llm_client):
                return self.llm_client(prompt) or ""
                
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
        
        return ""

    def _parse_json_object(self, response: str) -> Dict[str, Any]:
        """Parse a JSON object from an LLM response."""
        if not response:
            return {}
        content = response.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        try:
            payload = json.loads(content.strip())
            return payload if isinstance(payload, dict) else {}
        except (json.JSONDecodeError, ValueError) as exc:
            logger.debug("JSON object parse failed: %s", exc)
            return {}
    
    def _parse_json_list(self, response: str) -> List[str]:
        """Parse JSON list from LLM response."""
        if not response:
            return []
        
        try:
            # Clean markdown if present
            content = response.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            result = json.loads(content.strip())
            if isinstance(result, list):
                return [str(item) for item in result]
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"JSON parse failed: {e}")
        
        return []

    def _llm_prompt_state_compression(
        self,
        *,
        task_instruction: str,
        old_history: List[Dict[str, str]],
        trajectory_summary: str,
        world_description: str,
        agent_description: str,
        protected_keywords: List[str],
    ) -> Optional[Dict[str, Any]]:
        prompt = CONTEXT_HISTORY_COMPRESSION_PROMPT.format(
            task_instruction=(task_instruction or "Unknown task")[:800],
            protected_keywords=format_protected_keywords(protected_keywords),
            chat_history=format_dialog_turns(old_history, max_chars=6000),
            trajectory_summary=(trajectory_summary or "No trajectory summary.")[:4000],
            world_description=(world_description or "No world description.")[:4000],
            agent_description=(agent_description or "No agent description.")[:2500],
        )
        payload = self._parse_json_object(
            self._call_llm(prompt, max_tokens=self.config.llm_max_tokens)
        )
        if not payload:
            return None

        raw_turns = payload.get("summary_turns") or []
        summary_turns: List[Dict[str, str]] = []
        if isinstance(raw_turns, list):
            for item in raw_turns[:6]:
                if not isinstance(item, Mapping):
                    continue
                role = str(item.get("role", "user")).strip().lower()
                content = str(item.get("content", "")).strip()
                if role not in {"user", "assistant"} or not content:
                    continue
                summary_turns.append({"role": role, "content": content})
        if not summary_turns:
            return None

        return {
            "summary_turns": summary_turns,
            "compressed_world_description": str(
                payload.get("compressed_world_description", world_description)
            ).strip(),
            "compressed_agent_description": str(
                payload.get("compressed_agent_description", agent_description)
            ).strip(),
        }

    def _fallback_prompt_state_compression(
        self,
        *,
        old_history: List[Dict[str, str]],
        world_description: str,
        agent_description: str,
    ) -> Dict[str, Any]:
        history_text = format_dialog_turns(old_history, max_chars=1600)
        return {
            "summary_turns": [
                {
                    "role": "user",
                    "content": (
                        "Compressed historical context:\n"
                        f"{history_text}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Acknowledged compressed history. I will use it as prior context "
                        "and rely on the recent turns for the next action."
                    ),
                },
            ],
            "compressed_world_description": world_description[:2000],
            "compressed_agent_description": agent_description[:1200],
        }
    
    # =========================================================================
    # Rule-Based Fallbacks
    # =========================================================================
    
    def _extract_error_keywords(self, response: str) -> str:
        """Extract key error type from response."""
        patterns = {
            "not close enough": "not close enough",
            "occluded": "occluded",
            "not found": "not found",
            "already held": "already held",
            "not holding": "not holding",
        }
        response_lower = response.lower()
        for pattern, keyword in patterns.items():
            if pattern in response_lower:
                return keyword
        return "error"
    
    def _generate_fix(self, action_type: str, error_keywords: str) -> str:
        """Generate fix suggestion based on action and error type."""
        fixes = {
            ("Place", "not close enough"): "Navigate to target first, then Place.",
            ("Place", "occluded"): "Navigate to different position, then Place.",
            ("Pick", "not close enough"): "Navigate to object first, then Pick.",
            ("Pick", "already held"): "Place current object first.",
            ("Navigate", "not found"): "Explore room to find target.",
            ("Rearrange", "already held"): "Use Navigate then Place instead.",
        }
        return fixes.get((action_type, error_keywords), f"Check {action_type} preconditions.")
    
    def _generate_action_tip(self, action_type: str) -> str:
        """Generate action-specific tip."""
        tips = {
            "Place": "Navigate closer before Place.",
            "Pick": "Navigate to object before Pick.",
            "Navigate": "Explore if target not found.",
            "Rearrange": "Use Navigate+Place if holding object.",
            "Explore": "Explore all rooms systematically.",
        }
        return tips.get(action_type, "Verify preconditions.")
    
    def _rule_based_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate suggestions using rules (no LLM)."""
        suggestions = []
        
        root_causes = analysis.get("root_causes", [])
        for cause in root_causes[:2]:
            cause_lower = cause.lower()
            if "close" in cause_lower or "proximity" in cause_lower:
                suggestions.append("Navigate to target before Pick or Place actions.")
            elif "held" in cause_lower or "holding" in cause_lower:
                suggestions.append("Place held object before picking another.")
            elif "unknown" in cause_lower or "track" in cause_lower:
                suggestions.append("Explore rooms to locate unknown objects.")
            elif "repeat" in cause_lower or "redundant" in cause_lower:
                suggestions.append("Change approach if action fails twice.")
        
        if not suggestions:
            suggestions = [
                "Navigate to target before manipulation.",
                "Verify action success before proceeding.",
            ]
        
        return suggestions[:2]
    
    def _rule_based_batch_advice(self, summaries: List[Dict[str, Any]]) -> List[str]:
        """Generate batch advice using rules (no LLM)."""
        return [
            "Navigate to target before Pick or Place.",
            "If action fails, try different approach.",
            "Explore rooms to find missing objects.",
        ]
