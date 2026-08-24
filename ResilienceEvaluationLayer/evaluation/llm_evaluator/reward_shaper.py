"""
LLM Reward Shaper

Uses LLM API to provide semantic-level reward signals that augment
environment-based rewards. This helps with sparse reward problems.
"""

import json
import logging
import os
from typing import Dict, Any, Optional, Tuple
import hashlib

from .prompt_evaluation import (
    call_llm_completion,
    build_reward_shaping_prompt
)

logger = logging.getLogger(__name__)


class LLMRewardShaper:
    """
    Uses LLM to augment environment rewards with semantic understanding.

    The shaped reward is:
        shaped_reward = base_reward + shaping_weight * llm_bonus

    Where llm_bonus is computed from LLM evaluation of:
    - Goal progress (weight 0.4)
    - Action rationality (weight 0.4)
    - Efficiency (weight 0.2)
    """

    def __init__(self, llm_client: Any, config: Dict[str, Any]):
        """
        Initialize LLM Reward Shaper.

        Args:
            llm_client: LLM client (e.g., OpenAI client)
            config: Configuration dict with keys:
                - shaping_weight: Weight for LLM bonus (default 0.3)
                - llm_model: Model to use (default 'gpt-3.5-turbo')
                - llm_call_frequency: Call LLM every N steps (default 1) [LEGACY]
                - call_on_planning_steps: Align LLM calls with planning steps (default True)
                - planning_step_window: Dict with 'post_steps' for post-replan evaluation
                - evaluate_on_rebound: Trigger evaluation on Rebound events (default True)
                - cache_size: Max cache entries (default 1000)
        """
        self.llm_client = llm_client
        self.shaping_weight = config.get('shaping_weight', 0.3)
        self.llm_model = config.get('llm_model', 'gpt-3.5-turbo')

        # Planning-step alignment (NEW)
        self.call_on_planning_steps = config.get('call_on_planning_steps', True)
        self.planning_step_window = config.get('planning_step_window', {'post_steps': 2})
        self.evaluate_on_rebound = config.get('evaluate_on_rebound', True)
        self.last_planning_step = -1

        # Legacy frequency mode (backward compatible)
        self.call_frequency = config.get('llm_call_frequency', 1)

        self.cache_size = config.get('cache_size', 1000)
        self.json_mode = bool(config.get('reward_shaper_json_mode', False))
        self.debug = bool(config.get('debug', config.get('DEBUG', False)))
        self.save_shaping_logs = config.get('save_shaping_logs', self.debug)
        self.log_frequency = max(1, int(config.get('reward_shaper_log_frequency', 1)))
        self.log_max_entries = max(0, int(config.get('reward_shaper_log_max_entries', 10000)))
        self.log_cache_hits = bool(config.get('reward_shaper_log_cache_hits', False))
        self.log_dir = config.get('reward_shaper_log_dir')
        if not self.log_dir:
            analysis_dir = config.get('analysis_save_dir')
            if analysis_dir:
                self.log_dir = os.path.join(analysis_dir, "reward_shaper")
        if self.save_shaping_logs and self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
        self._log_entries = 0

        # Cache for similar states (to reduce API costs)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.step_count = 0

        mode_str = "planning-step" if self.call_on_planning_steps else f"frequency-{self.call_frequency}"
        logger.info(f"Initialized LLMRewardShaper: weight={self.shaping_weight}, "
                   f"model={self.llm_model}, mode={mode_str}")

    def shape_reward(
        self,
        state: Dict[str, Any],
        action: Any,
        next_state: Dict[str, Any],
        base_reward: float,
        task_instruction: str,
        info: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute shaped reward using LLM evaluation.

        Args:
            state: Current world_state_dict
            action: Action taken
            next_state: Next world_state_dict
            base_reward: Base environment reward
            task_instruction: Task goal description
            info: Additional info dictionary containing context like step_count, history, metrics

        Returns:
            shaped_reward: base_reward + shaping_weight * llm_bonus
        """
        self.step_count += 1

        # Extract planning step context from info
        planning_step = info.get('planning_step_count', 0) if info else 0
        sim_step = info.get('sim_step_count', self.step_count) if info else self.step_count

        # Decide whether to call LLM based on mode
        should_call = self._should_call_llm(planning_step, sim_step, info or {})

        if not should_call:
            return base_reward

        # Convert states and actions to text descriptions
        state_desc = self._state_to_text(state)
        next_state_desc = self._state_to_text(next_state)
        action_desc = self._action_to_text(action)

        # Extract context from info
        step_cnt = 0
        action_history_str = "None"
        action_response_str = "None"
        task_percent_complete = 0.0
        perception_complete = "Unknown"
        proposition_tracker_summary = None

        if info:
            step_cnt = info.get('step_count', self.step_count)
            task_percent_complete = info.get('task_percent_complete', 0.0)
            
            # Format action history
            # Expecting info['action_history'] to be a list of action strings or objects
            if 'action_history' in info and isinstance(info['action_history'], list):
                # Take last 5 actions
                history = info['action_history'][-5:] if len(info['action_history']) > 5 else info['action_history']
                # Convert to strings
                hist_strs = []
                for idx, item in enumerate(history):
                    # Handle ActionHistoryElement objects if passed directly
                    if isinstance(item, dict):
                        action_text = self._format_action_value(item.get("action"))
                        response_text = item.get("response", "")
                    elif hasattr(item, 'to_string'):
                        action_text = item.to_string()
                        response_text = getattr(item, "response", "")
                    elif hasattr(item, 'action'):
                        action_text = self._format_action_value(item.action)
                        response_text = getattr(item, "response", "")
                    else:
                        action_text = str(item)
                    if response_text:
                        hist_strs.append(f"{idx+1}. {action_text} -> {response_text}")
                    else:
                        hist_strs.append(f"{idx+1}. {action_text}")
                action_history_str = "\n".join(hist_strs) if hist_strs else "None"

            raw_action_response = info.get('action_response')
            if raw_action_response is None:
                raw_action_response = info.get('action_responses')
            action_response_str = self._extract_action_response(raw_action_response)

            # Check for perception complete
            # If not explicitly provided, we might infer from other metrics or just pass what we have
            if 'perception_complete' in info:
                perception_complete = str(info['perception_complete'])
            elif 'auto_eval_proposition_tracker' in info:
                proposition_tracker_summary = self._summarize_proposition_tracker(
                    info.get('auto_eval_proposition_tracker')
                )
                perception_complete = proposition_tracker_summary or perception_complete

        # Check cache (include step context in key if we want context-sensitive caching, 
        # but usually caching is state-action pair specific. 
        # Adding context breaks cache hit rate significantly. 
        # For now, let's keep cache key based on state/action/next_state to be safe, 
        # or maybe we accept that context matters and skip cache?
        # Include task instruction and progress to avoid cross-task collisions.
        cache_context = f"{task_instruction}|{task_percent_complete:.2f}|{action_response_str}"
        cache_key = self._compute_cache_key(state_desc, action_desc, cache_context)
        
        cache_hit = False
        llm_result: Dict[str, Any] = {}
        llm_response_text: Optional[str] = None

        if cache_key in self.cache:
            cached = self.cache[cache_key]
            llm_bonus = float(cached.get("llm_bonus", 0.0))
            llm_result = cached.get("llm_result", {}) or {}
            llm_response_text = cached.get("llm_response")
            cache_hit = True
            logger.debug(f"Cache hit for reward shaping: bonus={llm_bonus:.3f}")
        else:
            # Call LLM API
            llm_bonus, llm_result, llm_response_text = self._query_llm(
                state_desc, action_desc, next_state_desc,
                base_reward, task_instruction,
                step_count=step_cnt,
                action_history=action_history_str,
                task_percent_complete=task_percent_complete,
                perception_complete=perception_complete,
                action_response=action_response_str
            )

            # Update cache (with size limit)
            if len(self.cache) >= self.cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]

            self.cache[cache_key] = {
                "llm_bonus": llm_bonus,
                "llm_result": llm_result,
                "llm_response": llm_response_text,
            }

        # Compute shaped reward
        shaped_reward = base_reward + self.shaping_weight * llm_bonus

        logger.debug(f"Reward shaping: base={base_reward:.3f}, bonus={llm_bonus:.3f}, "
                    f"shaped={shaped_reward:.3f}")

        if self.save_shaping_logs:
            self._maybe_log_shaping(
                info=info,
                cache_key=cache_key,
                cache_hit=cache_hit,
                task_instruction=task_instruction,
                state_desc=state_desc,
                action_desc=action_desc,
                action_repr=str(action),
                next_state_desc=next_state_desc,
                base_reward=base_reward,
                shaped_reward=shaped_reward,
                llm_bonus=llm_bonus,
                llm_result=llm_result,
                llm_response_text=llm_response_text,
                step_count=step_cnt,
                task_percent_complete=task_percent_complete,
                perception_complete=perception_complete,
                action_response=action_response_str,
                action_history=action_history_str,
                proposition_tracker_summary=proposition_tracker_summary,
            )

        return shaped_reward

    def _query_llm(
        self,
        state_desc: str,
        action_desc: str,
        next_state_desc: str,
        base_reward: float,
        task_instruction: str,
        step_count: int = 0,
        action_history: str = "None",
        task_percent_complete: float = 0.0,
        perception_complete: str = "Unknown",
        action_response: str = "None"
    ) -> Tuple[float, Dict[str, Any], Optional[str]]:
        """
        Query LLM for reward bonus.

        Args:
            state_desc: Text description of current state
            action_desc: Text description of action
            next_state_desc: Text description of next state
            base_reward: Base environment reward
            task_instruction: Task instruction
            step_count: Current step count
            action_history: String representation of recent actions
            task_percent_complete: Percentage of task completion
            perception_complete: Indicator of perception status

        Returns:
            (llm_bonus, llm_result, llm_response_text)
        """
        # Format prompt
        messages = build_reward_shaping_prompt(
            task_instruction=task_instruction,
            state_description=state_desc,
            action_description=action_desc,
            next_state_description=next_state_desc,
            base_reward=base_reward,
            step_count=step_count,
            action_history=action_history,
            task_percent_complete=task_percent_complete,
            perception_complete=perception_complete,
            action_response=action_response
        )

        try:
            # Call LLM API using common utility
            response_text = call_llm_completion(
                self.llm_client,
                self.llm_model,
                messages,
                temperature=0.1,
                max_tokens=200,
                json_mode=self.json_mode
            )

            if not response_text:
                return 0.0, {}, None

            # Parse JSON response
            result = json.loads(response_text)

            # Compute weighted bonus
            # Adjusted weights: Goal Progress 0.4, Rationality 0.4, Efficiency 0.2
            llm_bonus = (
                result.get('goal_progress_score', 0.0) * 0.4 +
                result.get('rationality_score', 0.0) * 0.4 +
                result.get('efficiency_score', 0.0) * 0.2
            )

            # Clamp to [-1, 1]
            llm_bonus = max(-1.0, min(1.0, llm_bonus))

            logger.info(f"LLM evaluation: bonus={llm_bonus:.3f}, "
                       f"explanation={result.get('explanation', 'N/A')}")

            return llm_bonus, result, response_text

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"Response text: {response_text}")
            return 0.0, {}, response_text

        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return 0.0, {}, None

    def _state_to_text(self, state: Dict[str, Any]) -> str:
        """
        Convert world_state_dict to text description.

        Args:
            state: world_state_dict

        Returns:
            description: Text description
        """
        parts = []

        # Agent information
        agent_poses = state.get('agent_poses', {})
        if agent_poses and 0 in agent_poses:
            pos = agent_poses[0].get('position', [0, 0, 0])
            parts.append(f"Agent at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

        # Object information (task-relevant first, limit to 3)
        object_positions = state.get('object_positions', {})
        task_context = state.get('task_context', {}) or {}
        task_objects = task_context.get("objects", []) or []
        task_entities = task_context.get("entities", []) or []
        candidate_names = task_objects if task_objects else task_entities
        if candidate_names:
            candidate_names = [n for n in candidate_names if n in object_positions]
        else:
            candidate_names = list(object_positions.keys())

        for obj_name in candidate_names[:3]:
            obj_data = object_positions.get(obj_name, {})
            parent = obj_data.get('parent', 'unknown')
            parts.append(f"{obj_name} on {parent}")

        if not parts:
            return "State unknown"
        return "; ".join(parts)

    def _action_to_text(self, action: Any) -> str:
        """
        Convert action to text description.

        Args:
            action: Action (can be dict, tuple, or string)

        Returns:
            description: Text description
        """
        if hasattr(action, 'to_string'):
            return action.to_string()

        if isinstance(action, dict):
            if any(k in action for k in ("action_type", "action", "target", "args")):
                return self._format_action_value(action)
            return self._format_action_mapping(action)

        if hasattr(action, 'action'):
            return self._format_action_value(action.action)

        if isinstance(action, (tuple, list)):
            return self._format_action_value(action)

        return str(action)

    @staticmethod
    def _format_action_value(action: Any) -> str:
        if isinstance(action, dict):
            action_type = action.get("action_type") or action.get("action") or action.get("type")
            target = action.get("target") or action.get("object") or action.get("obj")
            args = action.get("args")
            if action_type and target:
                return f"{action_type}[{target}]"
            if action_type and args is not None:
                return f"{action_type}[{args}]"
            if action_type:
                return str(action_type)
            return str(action)

        if isinstance(action, (tuple, list)) and len(action) > 0:
            action_type = action[0]
            action_arg = action[1] if len(action) > 1 else None
            if action_arg is None or action_arg == "":
                return str(action_type)
            return f"{action_type}[{action_arg}]"

        return str(action)

    def _format_action_mapping(self, action_map: Dict[Any, Any]) -> str:
        if not action_map:
            return "no_action"
        parts = []
        for agent_id, action_val in sorted(action_map.items(), key=lambda item: str(item[0])):
            formatted = self._format_action_value(action_val)
            if formatted:
                parts.append(f"Agent {agent_id}: {formatted}")
        return "; ".join(parts) if parts else "no_action"

    @staticmethod
    def _extract_action_response(raw_response: Any) -> str:
        if raw_response is None:
            return "None"
        if isinstance(raw_response, dict):
            parts = []
            for agent_id, response in sorted(raw_response.items(), key=lambda item: str(item[0])):
                if response:
                    parts.append(f"Agent {agent_id}: {response}")
            return "\n".join(parts) if parts else "None"
        if isinstance(raw_response, (tuple, list)):
            parts = [str(resp) for resp in raw_response if resp]
            return "\n".join(parts) if parts else "None"
        response_text = str(raw_response)
        return response_text if response_text else "None"

    @staticmethod
    def _summarize_proposition_tracker(tracker: Any) -> Optional[str]:
        if not tracker:
            return None
        if isinstance(tracker, dict):
            total_props = tracker.get("propositions")
            total_count = len(total_props) if isinstance(total_props, list) else None
            satisfied_at = tracker.get("proposition_satisfied_at", [])
            satisfied_count = None
            if isinstance(satisfied_at, list):
                satisfied_count = sum(
                    1 for val in satisfied_at if isinstance(val, (int, float)) and val >= 0
                )
            state_seq = tracker.get("state_sequence", [])
            steps = len(state_seq) if isinstance(state_seq, list) else None
            parts = []
            if total_count is not None:
                parts.append(f"props_total={total_count}")
            if satisfied_count is not None:
                parts.append(f"props_satisfied={satisfied_count}")
            if steps is not None:
                parts.append(f"steps={steps}")
            return ", ".join(parts) if parts else None
        return str(tracker)

    def _compute_cache_key(self, state_desc: str, action_desc: str, context: str = "") -> str:
        """
        Compute cache key from state and action descriptions.

        Args:
            state_desc: State description
            action_desc: Action description
            context: Additional context (e.g. percent complete)

        Returns:
            cache_key: Hash-based cache key
        """
        combined = f"{state_desc}|{action_desc}|{context}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _should_call_llm(
        self,
        planning_step: int,
        sim_step: int,
        info: Dict[str, Any]
    ) -> bool:
        """
        Decide whether to call LLM based on planning-step alignment or legacy frequency.

        Args:
            planning_step: Current planning step count (replanning_count)
            sim_step: Current simulation step count
            info: Info dict with context (rebound_recovery_active, last_replan_sim_step, etc.)

        Returns:
            should_call: True if LLM should be called
        """
        # Mode 1: Planning-step alignment (recommended)
        if self.call_on_planning_steps:
            # Priority 1: Rebound events (if enabled)
            if self.evaluate_on_rebound:
                if info.get('rebound_recovery_active', False):
                    logger.debug(f"LLM call triggered by Rebound recovery event at sim_step={sim_step}")
                    return True
                if info.get('rebound_tags'):
                    logger.debug(f"LLM call triggered by Rebound tags: {info.get('rebound_tags')}")
                    return True

            # Priority 2: Planning step changes (Stability)
            planning_changed = (planning_step != self.last_planning_step)
            if planning_changed:
                self.last_planning_step = planning_step
                logger.debug(f"LLM call triggered by planning step change: {planning_step}")
                return True

            # Priority 3: Post-replan window
            # Evaluate N steps after each replan to capture action execution outcomes
            last_replan_sim_step = info.get('last_replan_sim_step', 0)
            steps_since_replan = sim_step - last_replan_sim_step
            post_steps = self.planning_step_window.get('post_steps', 2)
            if 0 < steps_since_replan <= post_steps:
                logger.debug(f"LLM call triggered by post-replan window: "
                           f"{steps_since_replan}/{post_steps} steps after replan")
                return True

            return False

        # Mode 2: Legacy frequency mode (backward compatible)
        return (sim_step % self.call_frequency == 0)

    def _maybe_log_shaping(
        self,
        info: Optional[Dict[str, Any]],
        cache_key: str,
        cache_hit: bool,
        task_instruction: str,
        state_desc: str,
        action_desc: str,
        action_repr: str,
        next_state_desc: str,
        base_reward: float,
        shaped_reward: float,
        llm_bonus: float,
        llm_result: Dict[str, Any],
        llm_response_text: Optional[str],
        step_count: int,
        task_percent_complete: float,
        perception_complete: str,
        action_response: str,
        action_history: str,
        proposition_tracker_summary: Optional[str],
    ) -> None:
        if not self.save_shaping_logs or not self.log_dir:
            return
        if self.log_max_entries == 0:
            return
        if self._log_entries >= self.log_max_entries:
            return
        if step_count % self.log_frequency != 0:
            return
        if cache_hit and not self.log_cache_hits:
            return

        filepath = self._get_log_filepath(info)
        if not filepath:
            return

        episode_id = None
        episode_filename = None
        if info:
            episode_id = info.get("episode_id")
            episode_filename = info.get("episode_filename")

        payload = {
            "episode_id": episode_id,
            "episode_filename": episode_filename,
            "step_count": step_count,
            "cache_key": cache_key,
            "cache_hit": cache_hit,
            "llm_model": self.llm_model,
            "base_reward": base_reward,
            "llm_bonus": llm_bonus,
            "shaped_reward": shaped_reward,
            "task_percent_complete": task_percent_complete,
            "perception_complete": perception_complete,
            "action_response": action_response,
            "task_instruction": task_instruction,
            "state_desc": state_desc,
            "action_desc": action_desc,
            "action_repr": action_repr,
            "next_state_desc": next_state_desc,
            "action_history": action_history,
            "llm_result": llm_result,
            "llm_response_text": llm_response_text,
            "proposition_tracker_summary": proposition_tracker_summary,
            "llm_called": not cache_hit,
        }

        self._write_log_line(filepath, payload)
        self._log_entries += 1

    def _get_log_filepath(self, info: Optional[Dict[str, Any]]) -> Optional[str]:
        if not self.log_dir:
            return None
        episode_id = None
        episode_filename = None
        if info:
            episode_id = info.get("episode_id")
            episode_filename = info.get("episode_filename")
        if episode_id is None:
            filename = "reward_shaper-log.jsonl"
        else:
            safe_name = str(episode_filename or "episode")
            safe_name = safe_name.replace("\\\\", "_").replace("/", "_")
            filename = f"reward_shaper-episode_{episode_id}_{safe_name}.jsonl"
        return os.path.join(self.log_dir, filename)

    @staticmethod
    def _write_log_line(filepath: str, payload: Dict[str, Any]) -> None:
        try:
            with open(filepath, "a", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, default=str)
                handle.write("\n")
        except Exception as exc:
            logger.debug("RewardShaper log failed: %s", exc)

    def clear_cache(self):
        """Clear the reward shaping cache."""
        self.cache.clear()
        logger.info("Cleared LLM reward shaping cache")
