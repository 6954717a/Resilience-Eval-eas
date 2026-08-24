import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import unified context compressor and utilities
try:
    from habitat_llm.context import (
        ContextCompressor,
        BATCH_ADVICE_PROMPT,
        format_batch_summaries as ctx_format_batch_summaries,
        format_failure_patterns as ctx_format_failure_patterns,
    )
    _HAS_CONTEXT_COMPRESSOR = True
except ImportError:
    _HAS_CONTEXT_COMPRESSOR = False
    BATCH_ADVICE_PROMPT = None

from habitat_llm.planner.evolve.compressor_integration import ContextCompressorIntegration

def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_read_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def _iter_jsonl(path: Optional[Path]) -> List[Dict[str, Any]]:
    if not path or not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []
    return entries


def _find_latest_file(paths: List[Path]) -> Optional[Path]:
    latest_path = None
    latest_mtime = -1.0
    for path in paths:
        if not path or not path.exists():
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = path
    return latest_path


def _find_latest_match(roots: List[Path], pattern: str) -> Optional[Path]:
    candidates: List[Path] = []
    for root in roots:
        if not root or not root.exists():
            continue
        candidates.extend(root.glob(pattern))
    return _find_latest_file(candidates)


def _extract_action_types(action_desc: str) -> List[str]:
    if not action_desc:
        return []
    return re.findall(r"\b([A-Za-z_]+)\[", action_desc)


def _summarize_reward_shaper(log_path: Optional[Path]) -> Dict[str, Any]:
    entries = _iter_jsonl(log_path)
    if not entries:
        return {}

    llm_bonuses: List[float] = []
    shaped_rewards: List[float] = []
    action_scores: Dict[str, List[float]] = defaultdict(list)
    action_counts: Counter = Counter()
    negative_count = 0

    for entry in entries:
        llm_bonus = _as_float(entry.get("llm_bonus", 0.0))
        shaped_reward = _as_float(entry.get("shaped_reward", 0.0))
        llm_bonuses.append(llm_bonus)
        shaped_rewards.append(shaped_reward)
        if llm_bonus < 0:
            negative_count += 1

        action_desc = entry.get("action_desc", "")
        for action_type in _extract_action_types(action_desc):
            action_counts[action_type] += 1
            action_scores[action_type].append(llm_bonus)

    action_avg = {
        action: _mean(scores) for action, scores in action_scores.items() if scores
    }
    top_actions = [a for a, _ in action_counts.most_common(3)]
    top_negative = [
        f"{action}({_mean(action_scores[action]):.2f})"
        for action, score in sorted(action_avg.items(), key=lambda item: item[1])
        if score < 0
    ][:3]
    top_positive = [
        f"{action}({_mean(action_scores[action]):.2f})"
        for action, score in sorted(action_avg.items(), key=lambda item: -item[1])
        if score > 0
    ][:3]

    return {
        "entries": len(entries),
        "mean_llm_bonus": _mean(llm_bonuses),
        "mean_shaped_reward": _mean(shaped_rewards),
        "negative_bonus_ratio": negative_count / len(entries),
        "top_actions": top_actions,
        "top_negative_actions": top_negative,
        "top_positive_actions": top_positive,
    }


def _summarize_state_encoder(log_path: Optional[Path]) -> Dict[str, Any]:
    entries = _iter_jsonl(log_path)
    if not entries:
        return {}

    max_repeat = 0
    long_repeats: List[Tuple[int, int, int]] = []
    progress_events: List[str] = []
    last_progress: Optional[float] = None

    for entry in entries:
        record_type = entry.get("record_type")
        if record_type == "repeat":
            repeat_count = int(entry.get("repeat_count", 0))
            max_repeat = max(max_repeat, repeat_count)
            if repeat_count >= 50:
                start = int(entry.get("repeat_start_step", 0))
                end = int(entry.get("repeat_end_step", 0))
                long_repeats.append((repeat_count, start, end))
        elif record_type == "state":
            key_features = entry.get("key_features", {}) or {}
            progress = key_features.get("task_progress")
            if progress is None:
                continue
            try:
                progress_val = float(progress)
            except (TypeError, ValueError):
                continue
            if last_progress is None or progress_val != last_progress:
                step = entry.get("step_count", "unknown")
                progress_events.append(f"step {step} -> {progress_val:.2f}")
                last_progress = progress_val

    long_repeats.sort(key=lambda item: item[0], reverse=True)
    repeat_summaries = [
        f"{start}-{end}({count})" for count, start, end in long_repeats[:3]
    ]

    return {
        "max_repeat": max_repeat,
        "long_repeats": repeat_summaries,
        "progress_events": progress_events[-3:],
        "final_task_progress": last_progress,
    }


def _summarize_offline_analysis(path: Optional[Path]) -> Dict[str, Any]:
    data = _safe_read_json(path)
    if not data:
        return {}
    root_causes = data.get("root_causes", []) or []
    suggestions = data.get("improvement_suggestions", []) or []
    return {
        "execution_quality_score": data.get("execution_quality_score"),
        "root_causes": root_causes[:2],
        "improvement_suggestions": suggestions[:2],
        "overall_assessment": data.get("overall_assessment", ""),
        "failure_point_count": len(data.get("failure_points", []) or []),
    }


def _summarize_critic_stats(path: Optional[Path]) -> Dict[str, Any]:
    data = _safe_read_json(path)
    if not data:
        return {}
    return data.get("critic_stats", {}) or {}


def _summarize_adca(path: Optional[Path]) -> Dict[str, Any]:
    data = _safe_read_json(path)
    if not data:
        return {}
    adca_result = data.get("adca_result", {}) or {}
    step_flags = adca_result.get("step_flags", []) or []
    step_advantages = adca_result.get("step_advantages", []) or []
    good_ratio = _mean([1.0 if flag else 0.0 for flag in step_flags]) if step_flags else 0.0
    return {
        "good_step_ratio": good_ratio,
        "mean_advantage": _mean([_as_float(val, 0.0) for val in step_advantages]),
        "total_steps": len(step_flags),
    }


def _summarize_rebound(path: Optional[Path], info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    data = _safe_read_json(path)
    if data and "rebound_summary" in data:
        return data["rebound_summary"]
    if info:
        return {
            "rebound_mttr": info.get("rebound_mttr", 0.0),
            "rebound_mtbf": info.get("rebound_mtbf", 0.0),
            "rebound_count": info.get("rebound_count", 0.0),
        }
    return {}


def resolve_analysis_roots(config: Any, output_root: Path) -> List[Path]:
    roots: List[Path] = []
    roots.append(output_root / "analyses")
    try:
        results_dir = Path(config.paths.results_dir)
        roots.append(results_dir / "analyses")
    except Exception:
        pass

    critic_dir = None
    try:
        critic_cfg = config.evaluation.get("critic", None)
    except Exception:
        critic_cfg = getattr(config.evaluation, "critic", None)
    if critic_cfg:
        try:
            critic_dir = critic_cfg.get("analysis_save_dir", None)
        except Exception:
            critic_dir = getattr(critic_cfg, "analysis_save_dir", None)
    if critic_dir:
        critic_dir = Path(critic_dir)
        if not critic_dir.is_absolute():
            critic_dir = Path.cwd() / critic_dir
        roots.append(critic_dir)

    debug_dir = Path("Debug") / "analyses"
    if debug_dir.exists():
        roots.append(debug_dir)

    unique_roots: List[Path] = []
    seen: set = set()
    for root in roots:
        if not root:
            continue
        try:
            resolved = root.resolve()
        except Exception:
            resolved = root
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique_roots.append(resolved)
    return unique_roots


def _extract_replanning_count(info: Dict[str, Any]) -> float:
    value = info.get("replanning_count")
    if isinstance(value, dict):
        return float(sum(value.values()))
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def build_episode_summary(
    analysis_roots: List[Path],
    episode_id: str,
    episode_filename: str,
    instruction: str,
    info: Dict[str, Any],
    run_id: int,
) -> Dict[str, Any]:
    analysis_file = _find_latest_match(
        analysis_roots, f"analysis_ep{episode_id}_*.json"
    )
    reward_file = _find_latest_match(
        analysis_roots,
        f"reward_shaper/reward_shaper-episode_{episode_id}_{episode_filename}.jsonl",
    )
    state_file = _find_latest_match(
        analysis_roots,
        f"state_encoder/state_encoder-episode_{episode_id}_{episode_filename}.jsonl",
    )
    critic_file = _find_latest_match(
        analysis_roots,
        f"critic/critic_stats-episode_{episode_id}_{episode_filename}.json",
    )
    adca_file = _find_latest_match(
        analysis_roots,
        f"adca/adca_analysis-episode_{episode_id}_{episode_filename}.json",
    )
    rebound_file = _find_latest_match(
        analysis_roots,
        f"rebound/rebound_analysis-episode_{episode_id}_{episode_filename}.json",
    )

    summary = {
        "episode_id": str(episode_id),
        "episode_filename": episode_filename,
        "run_id": run_id,
        "instruction": instruction,
        "task_percent_complete": _as_float(info.get("task_percent_complete", 0.0)),
        "task_state_success": _as_float(info.get("task_state_success", 0.0)),
        "sim_step_count": _as_float(info.get("sim_step_count", 0.0)),
        "replanning_count": _extract_replanning_count(info),
        "runtime": _as_float(info.get("runtime", 0.0)),
        "analysis": _summarize_offline_analysis(analysis_file),
        "critic": _summarize_critic_stats(critic_file),
        "reward_shaper": _summarize_reward_shaper(reward_file),
        "state_encoder": _summarize_state_encoder(state_file),
        "adca": _summarize_adca(adca_file),
        "rebound": _summarize_rebound(rebound_file, info),
    }
    return summary


def _build_episode_advice(summary: Dict[str, Any]) -> List[str]:
    advice: List[str] = []
    analysis = summary.get("analysis", {}) or {}
    reward = summary.get("reward_shaper", {}) or {}
    state = summary.get("state_encoder", {}) or {}
    adca = summary.get("adca", {}) or {}
    rebound = summary.get("rebound", {}) or {}

    for cause in analysis.get("root_causes", []) or []:
        advice.append(f"Address: {cause}")

    for suggestion in analysis.get("improvement_suggestions", []) or []:
        advice.append(f"Improve: {suggestion}")

    if reward.get("top_negative_actions"):
        negative_actions = ", ".join(reward.get("top_negative_actions", []))
        advice.append(
            f"Low-value actions observed: {negative_actions}. Re-check preconditions before retrying."
        )

    if state.get("max_repeat", 0) >= 50:
        advice.append("Detected long loops; change target or re-sense when state stalls.")

    if rebound.get("rebound_count", 0) > 0:
        advice.append("After failures, re-check proximity/occlusion before repeating actions.")

    good_ratio = _as_float(adca.get("good_step_ratio", 0.0))
    if adca and good_ratio < 0.4:
        advice.append("Prioritize actions with positive advantage; re-check failed steps before retrying.")

    if _as_float(summary.get("task_percent_complete", 0.0)) < 1.0:
        advice.append("Verify each task object is picked, moved, and placed on the target receptacle.")

    if not advice:
        advice.append("Keep plans concise and confirm state changes after each action.")

    return advice[:4]


def _build_general_advice(batch_summaries: List[Dict[str, Any]]) -> List[str]:
    advice: List[str] = []
    if not batch_summaries:
        return [
            "Check preconditions (distance, visibility, holding state) before Pick or Place.",
            "After a failed action, reposition and re-sense before retrying.",
            "Resolve ambiguous targets by explicitly testing each candidate receptacle.",
        ]

    root_causes = Counter()
    has_repeats = False
    has_place_failures = False
    has_ambiguity = False
    for summary in batch_summaries:
        analysis = summary.get("analysis", {}) or {}
        for cause in analysis.get("root_causes", []) or []:
            root_causes[cause] += 1
            if "other" in cause.lower() or "ambigu" in cause.lower():
                has_ambiguity = True

        state = summary.get("state_encoder", {}) or {}
        if state.get("max_repeat", 0) >= 50:
            has_repeats = True

        reward = summary.get("reward_shaper", {}) or {}
        if any("Place" in action for action in reward.get("top_negative_actions", [])):
            has_place_failures = True

    if has_place_failures:
        advice.append("Check distance/visibility before Place; reposition if occluded or out of range.")
    if has_repeats:
        advice.append("If state does not change after repeated actions, switch target or re-sense.")
    if has_ambiguity:
        advice.append("Disambiguate references by checking each candidate receptacle once.")

    advice.append("Use action-response feedback to confirm success before moving to the next subgoal.")
    advice.append("Split subgoals across agents when possible to reduce idle time.")

    return advice[:3]


def _format_general_context(general_advice: List[str], episode_count: int) -> str:
    if not general_advice:
        return ""
    lines = [f"Batch Guidance (episodes={episode_count})"]
    for idx, item in enumerate(general_advice, 1):
        lines.append(f"{idx}) {item}")
    return "\n".join(lines)


class EvolutionContextManager:
    def __init__(
        self,
        output_root: Path,
        enabled: bool = True,
        batch_size: int = 0,
        max_history: int = 3,
        max_context_chars: int = 2000,
        llm_client: Any = None,
    ) -> None:
        self.enabled = enabled
        self.output_root = output_root
        self.batch_size = max(0, int(batch_size))
        self.max_history = max(1, int(max_history))
        self.max_context_chars = max(200, int(max_context_chars))
        self.llm_client = llm_client
        self.episode_history: Dict[str, List[Dict[str, Any]]] = {}
        self.context_by_episode: Dict[str, str] = {}
        self.general_context: Optional[str] = None
        self.batch_counter = 0

        self.evolve_dir = self.output_root / "analyses" / "evolve"
        self.evolve_dir.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.evolve_dir / f"evolution_context_proc{os.getpid()}.json"
        
        # Initialize Compressor Integration if client is available
        self.compressor_integration = None
        if self.llm_client:
            try:
                self.compressor_integration = ContextCompressorIntegration(self.llm_client)
            except Exception as e:
                print(f"[EvolutionContextManager] Failed to init CompressorIntegration: {e}")

        self._load()

    def _load(self) -> None:
        data = _safe_read_json(self.memory_path)
        if not data:
            return
        history = data.get("episode_history", {})
        contexts = data.get("context_by_episode", {})
        if isinstance(history, dict):
            self.episode_history = history
        if isinstance(contexts, dict):
            self.context_by_episode = contexts
        general_context = data.get("general_context")
        if isinstance(general_context, str):
            self.general_context = general_context
        self.batch_counter = int(data.get("batch_counter", 0))

    def _save(self) -> None:
        payload = {
            "episode_history": self.episode_history,
            "context_by_episode": self.context_by_episode,
            "general_context": self.general_context,
            "batch_counter": self.batch_counter,
        }
        try:
            with self.memory_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception:
            pass

    def _write_episode_summary(self, summary: Dict[str, Any]) -> None:
        episode_id = summary.get("episode_id", "unknown")
        episode_filename = summary.get("episode_filename", "episode")
        filepath = self.evolve_dir / f"episode_{episode_id}_{episode_filename}_summary.json"
        try:
            with filepath.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
        except Exception:
            pass

    def _write_batch_summary(self, episode_ids: List[str], general_advice: List[str]) -> None:
        filepath = self.evolve_dir / f"batch_{self.batch_counter:03d}_advice.json"
        payload = {
            "episode_ids": episode_ids,
            "general_advice": general_advice,
            "general_context": self.general_context,
        }
        try:
            with filepath.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception:
            pass

    def record_episode_summary(self, summary: Dict[str, Any]) -> None:
        if not self.enabled or not summary:
            return
        episode_id = str(summary.get("episode_id", ""))
        if not episode_id:
            return
        history = list(self.episode_history.get(episode_id, []))
        history.append(summary)
        if len(history) > self.max_history:
            history = history[-self.max_history :]
        self.episode_history[episode_id] = history
        self._write_episode_summary(summary)

    def evolve_batch(self, episode_ids: List[str]) -> Dict[str, str]:
        if not self.enabled:
            return {}
        batch_histories = [
            self.episode_history.get(str(episode_id), [])[-1]
            for episode_id in episode_ids
            if self.episode_history.get(str(episode_id))
        ]
        
        general_advice = []
        
        # Try ContextCompressor first (unified, action-oriented advice)
        if _HAS_CONTEXT_COMPRESSOR and self.llm_client:
            try:
                compressor = ContextCompressor(
                    llm_client=self.llm_client,
                    enable_llm_refinement=True,
                )
                general_advice = compressor.generate_batch_advice(batch_histories)
            except Exception as e:
                print(f"[EvolutionContextManager] ContextCompressor failed: {e}")
                general_advice = []
        
        # Fallback to legacy LLM advice
        if not general_advice and self.llm_client:
            try:
                general_advice = self._generate_llm_advice(batch_histories)
            except Exception as e:
                print(f"[EvolutionContextManager] LLM generation failed: {e}")
                general_advice = []
        
        # Final fallback to rule-based
        if not general_advice:
            general_advice = _build_general_advice(batch_histories)
            
        if batch_histories:
            self.general_context = _format_general_context(
                general_advice, len(batch_histories)
            )
        if episode_ids:
            self.batch_counter += 1
            self._write_batch_summary(episode_ids, general_advice)

        updated: Dict[str, str] = {}
        for episode_id in episode_ids:
            history = self.episode_history.get(str(episode_id), [])
            if not history:
                continue
            context = self._build_context(str(episode_id), history)
            if context:
                self.context_by_episode[str(episode_id)] = context
                updated[str(episode_id)] = context
        if updated:
            self._save()
        return updated

    def _generate_llm_advice(self, histories: List[Dict[str, Any]]) -> List[str]:
        """
        Generate general advice using LLM based on batch histories.
        Uses unified BATCH_ADVICE_PROMPT from context module for consistent formatting.
        """
        if not histories:
            return []

        # Use unified prompt template if available
        if _HAS_CONTEXT_COMPRESSOR and BATCH_ADVICE_PROMPT is not None:
            # Use context module formatting
            batch_summaries = ctx_format_batch_summaries(histories)
            failure_patterns = ctx_format_failure_patterns(histories)
            
            prompt = BATCH_ADVICE_PROMPT.format(
                batch_summaries=batch_summaries[:600],
                failure_patterns=failure_patterns[:300],
            )
        else:
            # Fallback to legacy prompt
            summaries_text = ""
            for h in histories:
                analysis = h.get("analysis", {}) or {}
                root_causes = "; ".join(analysis.get("root_causes", []) or [])
                suggestions = "; ".join(analysis.get("improvement_suggestions", []) or [])
                summaries_text += (
                    f"- Ep {h.get('episode_id')}: Success={h.get('task_state_success')}; "
                    f"RootCauses=[{root_causes}]; Suggestions=[{suggestions}]\n"
                )

            prompt = (
                "You are an expert AI Agent Coach. Analyze the following batch of execution summaries "
                "from a household service robot. \n"
                "Identify common failure patterns and provide exactly 3 actionable, generalizable pieces of advice "
                "that the agent should follow in future episodes to avoid these mistakes.\n"
                "Format your response as a JSON list of 3 strings (e.g., [\"advice 1\", \"advice 2\", \"advice 3\"]).\n"
                "Do not output markdown code blocks, just the raw JSON string.\n\n"
                f"Batch Data:\n{summaries_text}\n\n"
                "Advice:"
            )

        # Call LLM with stricter temperature for consistent output
        response_content = ""
        if hasattr(self.llm_client, "chat") and hasattr(self.llm_client.chat, "completions"):
            # OpenAI Client
            response = self.llm_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=150,   # Limit output length
            )
            response_content = response.choices[0].message.content
        elif hasattr(self.llm_client, "generate"):
            # Local/Habitat LLM Client
            response_content = self.llm_client.generate(prompt, max_new_tokens=150)
        else:
            print("[EvolutionContextManager] Unknown LLM client interface.")
            return []

        # Clean and parse JSON
        try:
            import json
            # Strip markdown if present
            content = response_content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            advice = json.loads(content.strip())
            if isinstance(advice, list):
                # Enforce length limit per advice
                return [str(a)[:80] for a in advice[:3]]
        except Exception as e:
            print(f"[EvolutionContextManager] Failed to parse LLM response: {e}; Content: {response_content}")
            
        return []

    def get_context(self, episode_id: str) -> Optional[str]:
        if not self.enabled:
            return None
        episode_context = self.context_by_episode.get(str(episode_id))
        return self._combine_context(episode_context)

    def _combine_context(self, episode_context: Optional[str]) -> Optional[str]:
        parts: List[str] = []
        if self.general_context:
            parts.append(self.general_context)
        if episode_context:
            parts.append(episode_context)
        if not parts:
            return None
        combined = "\n\n".join(parts)
        if len(combined) > self.max_context_chars:
            combined = combined[: self.max_context_chars - 3].rstrip() + "..."
        return combined

    def _build_context(
        self,
        episode_id: str,
        history: List[Dict[str, Any]],
    ) -> str:
        last = history[-1]
        success_flags = [
            1.0 if _as_float(h.get("task_state_success", 0.0)) >= 0.5 else 0.0
            for h in history
        ]
        success_rate = _mean(success_flags)
        quality_scores = [
            _as_float(h.get("analysis", {}).get("execution_quality_score", 0.0))
            for h in history
            if h.get("analysis", {}).get("execution_quality_score") is not None
        ]
        quality_avg = _mean(quality_scores) if quality_scores else None

        critic = last.get("critic", {}) or {}
        reward = last.get("reward_shaper", {}) or {}
        state = last.get("state_encoder", {}) or {}
        adca = last.get("adca", {}) or {}
        analysis = last.get("analysis", {}) or {}

        episode_advice = _build_episode_advice(last)

        # Advanced Retry Hint using Compressor
        retry_hint = ""
        # Only generate hint if failure and compressor is available
        if (
            _as_float(last.get("task_state_success", 0.0)) < 1.0 
            and self.compressor_integration
            and last.get("rebound", {}).get("rebound_count", 0) > 0 # Use rebound as proxy for difficulty/trace availability
        ):
             # We need the trace. Currently build_episode_summary doesn't load full trace content.
             # We might need to load it on demand or store path in summary.
             # For now, let's assume we can rely on standard advice or need to extend this.
             # Ideally we'd call self.compressor_integration.generate_retry_hint(trace, last.get('instruction'))
             pass

        lines = [
            f"Episode Guidance (episode_id={episode_id}, runs={len(history)})",
            f"Task: {last.get('instruction', '')}",
            (
                "Signals: "
                f"success_rate={success_rate:.2f}; "
                f"completion={_as_float(last.get('task_percent_complete', 0.0)):.2f}; "
                f"steps={int(_as_float(last.get('sim_step_count', 0.0)))}; "
                f"replans={int(_as_float(last.get('replanning_count', 0.0)))}"
            ),
        ]

        extra_signals: List[str] = []
        if quality_avg is not None:
            extra_signals.append(
                f"quality_avg={quality_avg:.2f}"
            )
        if critic:
            extra_signals.append(
                f"critic_return={_as_float(critic.get('mean_return', 0.0)):.2f}"
            )
        if adca:
            extra_signals.append(
                f"adca_good={_as_float(adca.get('good_step_ratio', 0.0)):.2f}"
            )
        if reward:
            extra_signals.append(
                f"reward_bonus={_as_float(reward.get('mean_llm_bonus', 0.0)):.2f}"
            )
        if state:
            repeats = ", ".join(state.get("long_repeats", []))
            if repeats:
                extra_signals.append(f"stagnation={repeats}")
        if extra_signals:
            lines.append("Signals+: " + "; ".join(extra_signals))

        lines.append("Episode Advice:")
        for idx, item in enumerate(episode_advice, 1):
            lines.append(f"{idx}) {item}")

        context = "\n".join(line for line in lines if line)
        return context


def apply_evolution_context(
    eval_runner: Any,
    context: Optional[str],
    episode_id: str,
    run_id: int,
) -> None:
    title = f"Evolution Context (episode {episode_id}, run {run_id})"
    planners = eval_runner.planner
    if isinstance(planners, dict):
        for planner in planners.values():
            if hasattr(planner, "set_evolution_context"):
                planner.set_evolution_context(title, context or "")
    else:
        if hasattr(planners, "set_evolution_context"):
            planners.set_evolution_context(title, context or "")
