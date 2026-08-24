"""
LLM Offline Trajectory Analyzer

Analyzes complete episode trajectories after execution to provide:
- Failure point identification
- Root cause analysis
- Improvement suggestions
- Pattern recognition

This is intended for debugging and offline learning, not online decision-making.
"""

import json
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)


# Prompt template for trajectory analysis
TRAJECTORY_ANALYSIS_PROMPT = """You are an expert robot task execution analyst.

Task: {task_instruction}

Execution Trajectory:
{formatted_trajectory}

Final Outcome:
- Success: {success}
- Percent Complete: {percent_complete:.2f}
- Total Reward: {total_reward:.2f}
- Number of Simulation Steps: {num_steps}
- Number of Planning Steps: {planning_step_count}

Please analyze this trajectory and provide:
1. Key failure points (if any) - which steps went wrong?
2. Root causes - why did failures occur?
3. Improvement suggestions - what could be done better?
4. Overall assessment - summary of execution quality

Respond ONLY with a JSON object in this exact format:
{{
    "failure_points": [<list of step indices where failures occurred>],
    "root_causes": ["<cause 1>", "<cause 2>", ...],
    "improvement_suggestions": ["<suggestion 1>", "<suggestion 2>", ...],
    "overall_assessment": "<1-2 sentence summary>",
    "execution_quality_score": <float between 0.0 and 1.0>
}}
"""


class LLMOfflineAnalyzer:
    """
    Analyzes episode trajectories offline using LLM.

    This provides interpretable feedback for debugging and improvement.
    """

    def __init__(self, llm_client: Any, config: Dict[str, Any]):
        """
        Initialize LLM Offline Analyzer.

        Args:
            llm_client: LLM client (e.g., OpenAI client)
            config: Configuration dict with keys:
                - llm_model: Model to use (default 'gpt-4')
                - save_analyses: Whether to save analysis reports (default True)
                - analysis_save_dir: Directory to save reports (default './analyses')
                - analyze_frequency: Analyze every N episodes (default 1)
        """
        self.llm_client = llm_client
        self.llm_model = config.get('llm_model', 'gpt-4')
        self.save_analyses = config.get('save_analyses', True)
        self.analysis_save_dir = config.get('analysis_save_dir', './analyses')
        self.analyze_frequency = config.get('analyze_frequency', 1)
        self.proc_id = config.get('proc_id')
        if self.proc_id is None:
            self.proc_id = (
                os.environ.get("LOCAL_RANK")
                or os.environ.get("RANK")
                or os.environ.get("SLURM_PROCID")
                or os.environ.get("PROC_ID")
                or os.environ.get("PROCESS_INDEX")
            )
        if self.proc_id is None:
            self.proc_id = str(os.getpid())

        self.episode_count = 0

        # Create save directory if needed
        if self.save_analyses:
            os.makedirs(self.analysis_save_dir, exist_ok=True)

        logger.info(f"Initialized LLMOfflineAnalyzer: model={self.llm_model}, "
                   f"save_dir={self.analysis_save_dir}")

    def analyze_episode(
        self,
        trajectory: List[Tuple[Any, Any, float, Any, bool]],
        task_instruction: str,
        success: bool,
        total_reward: float,
        percent_complete: float = 0.0,
        episode_id: Optional[Any] = None,
        episode_filename: Optional[str] = None,
        planning_step_count: int = 0
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Analyze a complete episode trajectory.

        Args:
            trajectory: List of (state, action, reward, next_state, done) tuples
            task_instruction: Task goal description
            success: Whether task succeeded
            total_reward: Total reward accumulated
            percent_complete: Percentage of task completed
            planning_step_count: Number of high-level planning steps

        Returns:
            tuple: (analysis_dict, saved_filepath)
        """
        self.episode_count += 1

        # Only analyze at specified frequency
        if self.episode_count % self.analyze_frequency != 0:
            logger.debug(f"Skipping analysis for episode {self.episode_count} "
                        f"(frequency={self.analyze_frequency})")
            return {}, None

        # Format trajectory for LLM
        formatted_traj = self._format_trajectory(trajectory)
        num_steps = len(trajectory)

        # Call LLM for analysis
        analysis = self._query_llm(
            task_instruction=task_instruction,
            formatted_trajectory=formatted_traj,
            success=success,
            total_reward=total_reward,
            num_steps=num_steps,
            percent_complete=percent_complete,
            planning_step_count=planning_step_count
        )

        if not analysis:
             return {}, None
        
        # Override execution_quality_score if necessary
        # Only force to 0 if BOTH success is False AND percent_complete is 0
        if not success and percent_complete == 0.0:
            analysis['execution_quality_score'] = 0.0
            logger.info("Forced execution_quality_score to 0.0 (success=False, percent_complete=0.0)")

        # Add metadata
        analysis['episode_number'] = self.episode_count
        analysis['episode_id'] = episode_id
        analysis['episode_filename'] = episode_filename
        analysis['proc_id'] = self.proc_id
        analysis['timestamp'] = datetime.now().isoformat()
        analysis['task_instruction'] = task_instruction

        # Save analysis if enabled
        filepath = None
        if self.save_analyses and analysis:
            filepath = self._save_analysis(analysis)

        return analysis, filepath

    def _query_llm(
        self,
        task_instruction: str,
        formatted_trajectory: str,
        success: bool,
        total_reward: float,
        num_steps: int,
        percent_complete: float = 0.0,
        planning_step_count: int = 0
    ) -> Dict[str, Any]:
        """
        Query LLM for trajectory analysis.

        Args:
            task_instruction: Task instruction
            formatted_trajectory: Formatted trajectory string
            success: Task success
            total_reward: Total reward
            num_steps: Number of steps
            percent_complete: Percentage of task completed
            planning_step_count: Number of planning steps

        Returns:
            analysis: Analysis results dictionary
        """
        # Format prompt
        prompt = TRAJECTORY_ANALYSIS_PROMPT.format(
            task_instruction=task_instruction,
            formatted_trajectory=formatted_trajectory,
            success=success,
            total_reward=total_reward,
            num_steps=num_steps,
            percent_complete=percent_complete,
            planning_step_count=planning_step_count
        )

        try:
            # Call LLM API
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Low temperature for consistent analysis
                max_tokens=800
            )

            response_text = response.choices[0].message.content.strip()

            # Clean markdown code blocks if present
            if response_text.startswith("```"):
                lines = response_text.splitlines()
                # Remove the first line (starts with ```)
                lines = lines[1:]
                # Remove the last line if it ends with ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                response_text = "\n".join(lines).strip()

            # Parse JSON response
            analysis = json.loads(response_text)

            logger.info(f"LLM Analysis for episode {self.episode_count}: "
                       f"{analysis.get('overall_assessment', 'N/A')}")

            return analysis

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM analysis as JSON: {e}")
            logger.error(f"Response text: {response_text}")
            return {}

        except Exception as e:
            logger.error(f"LLM API call failed for trajectory analysis: {e}")
            return {}

    def _format_trajectory(
        self,
        trajectory: List[Tuple[Any, Any, float, Any, bool]]
    ) -> str:
        """
        Format trajectory for LLM consumption.

        Args:
            trajectory: List of (state, action, reward, next_state, done) tuples

        Returns:
            formatted: Human-readable trajectory string
        """
        lines = []

        for step_idx, step_data in enumerate(trajectory):
             # Unpack first 5 elements, keep optional metadata when present.
            if len(step_data) >= 5:
                state, action, reward, next_state, done = step_data[:5]
            else:
                 # Fallback if somehow data is malformed
                continue
            planning_step = step_data[5] if len(step_data) > 5 else None
            sim_step = step_data[6] if len(step_data) > 6 else None
            event_tags = step_data[7] if len(step_data) > 7 else ""

            # Format state (simplified)
            state_str = self._state_to_short_text(state)

            # Format action
            action_str = self._action_to_short_text(action)

            # Format reward
            reward_str = f"{reward:.2f}"

            # Combine
            prefix_parts = [f"TraceStep {step_idx}"]
            if planning_step is not None:
                prefix_parts.append(f"P{planning_step}")
            if sim_step is not None:
                prefix_parts.append(f"S{sim_step}")
            if event_tags:
                prefix_parts.append(f"[{event_tags}]")
            prefix = " ".join(prefix_parts)
            line = f"{prefix}: {state_str} -> {action_str} -> reward={reward_str}"

            if done:
                line += " [DONE]"

            lines.append(line)

        # Limit to last 20 steps to avoid token limits
        if len(lines) > 20:
            lines = lines[-20:]
            lines.insert(0, f"... (showing last 20 of {len(trajectory)} steps) ...")

        return "\n".join(lines)

    def _state_to_short_text(self, state: Any) -> str:
        """Convert state to short text description."""
        if isinstance(state, str):
            return state
        if isinstance(state, dict):
            agent_poses = state.get('agent_poses', {})
            if agent_poses and 0 in agent_poses:
                pos = agent_poses[0].get('position', [0, 0, 0])
                return f"Agent@({pos[0]:.1f},{pos[1]:.1f})"
        return "State"

    def _action_to_short_text(self, action: Any) -> str:
        """Convert action to short text description."""
        if isinstance(action, dict):
            action_type = action.get('action_type', 'unknown')
            target = action.get('target', '')
            planning_step = action.get('planning_step_count')
            sim_step = action.get('sim_step_count')
            recent_response = action.get('recent_response', '')
            event_tags = action.get('event_tags', [])
            base = f"{action_type}({target})" if target else str(action_type)
            suffix_parts = []
            if planning_step is not None:
                suffix_parts.append(f"P{planning_step}")
            if sim_step is not None:
                suffix_parts.append(f"S{sim_step}")
            if event_tags:
                suffix_parts.append("/".join(str(tag) for tag in event_tags))
            if recent_response:
                suffix_parts.append(f"resp={recent_response}")
            if suffix_parts:
                return f"{base} [{' | '.join(suffix_parts)}]"
            return base
        elif isinstance(action, (tuple, list)) and len(action) > 0:
            return str(action[0])
        return str(action)

    def _save_analysis(self, analysis: Dict[str, Any]) -> Optional[str]:
        """
        Save analysis to file.

        Args:
            analysis: Analysis results dictionary
            
        Returns:
            filepath: Path to the saved file, or None if failed
        """
        episode_id = analysis.get('episode_id')
        episode_tag = episode_id if episode_id is not None else analysis.get('episode_number', 0)
        proc_tag = str(analysis.get('proc_id', 'unknown')).replace("\\", "_").replace("/", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_ep{episode_tag}_proc{proc_tag}_{timestamp}.json"
        filepath = os.path.join(self.analysis_save_dir, filename)

        try:
            with open(filepath, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Saved trajectory analysis to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
            return None

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics from all saved analyses.

        Returns:
            stats: Dictionary with summary statistics
        """
        if not self.save_analyses:
            return {}

        # Load all saved analyses
        analyses = []
        for filename in os.listdir(self.analysis_save_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.analysis_save_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        analysis = json.load(f)
                        analyses.append(analysis)
                except Exception as e:
                    logger.error(f"Failed to load {filepath}: {e}")

        if not analyses:
            return {}

        # Compute statistics
        quality_scores = [a.get('execution_quality_score', 0.0) for a in analyses]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        # Aggregate root causes
        all_causes = []
        for a in analyses:
            all_causes.extend(a.get('root_causes', []))

        # Count common causes
        cause_counts = {}
        for cause in all_causes:
            cause_counts[cause] = cause_counts.get(cause, 0) + 1

        stats = {
            'total_analyses': len(analyses),
            'average_quality_score': avg_quality,
            'common_root_causes': sorted(cause_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }

        return stats

