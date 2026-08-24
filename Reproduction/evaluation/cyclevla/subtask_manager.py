# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree

"""
Subtask Manager for CycleVLA.

This module implements:
- Subtask: Data structure representing a single subtask
- SubtaskManager: Manages task decomposition, progress estimation, and boundary detection

Reference: CycleVLA Paper (arxiv.org/abs/2601.02295)
- Section IV-A: Learning Stop and Progress Signals for Subtask Execution
- The paper uses LLM to decompose demonstrations into subtasks with precise
  start/end timestamps and language instructions.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Subtask:
    """
    Represents a single subtask in the CycleVLA framework.
    
    Corresponds to CycleVLA's subtask decomposition where each subtask
    has a description, preconditions, and postconditions.
    
    Attributes:
        idx: Subtask index (0-based)
        instruction: Brief action description (e.g., "grasp apple")
        preconditions: What must be true before starting
        postconditions: What will be true after completion
        expected_tools: Which tools will be used
        status: pending / active / completed / failed
        start_step: Step count when subtask started
        end_step: Step count when subtask ended (if completed)
    """
    idx: int
    instruction: str
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    expected_tools: List[str] = field(default_factory=list)
    status: str = "pending"
    start_step: Optional[int] = None
    end_step: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "idx": self.idx,
            "instruction": self.instruction,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "expected_tools": self.expected_tools,
            "status": self.status,
            "start_step": self.start_step,
            "end_step": self.end_step,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Subtask":
        """Create from dictionary."""
        return cls(
            idx=data.get("idx", 0),
            instruction=data.get("instruction", ""),
            preconditions=data.get("preconditions", []),
            postconditions=data.get("postconditions", []),
            expected_tools=data.get("expected_tools", []),
            status=data.get("status", "pending"),
            start_step=data.get("start_step"),
            end_step=data.get("end_step"),
        )


class SubtaskManager:
    """
    Manages subtask decomposition, progress tracking, and boundary detection.
    
    This class implements the core logic from CycleVLA's subtask-aware execution:
    1. Decompose high-level instruction into atomic subtasks using LLM
    2. Track progress within each subtask
    3. Detect subtask boundaries (when progress >= threshold)
    
    Reference: CycleVLA Section IV-A
    - "We introduce a pipeline that uses LLMs to decompose demonstrations 
       into subtasks with precise start/end timestamps and language instructions"
    """
    
    # Prompt template for subtask decomposition
    DECOMPOSITION_PROMPT = '''You are a task decomposition expert for robotic manipulation.

Task Instruction: {instruction}

Current Environment State:
{world_state_summary}

Available Tools: {available_tools}

Decompose this task into a sequence of minimal atomic subtasks. Each subtask should be a single primitive action.

For each subtask, provide:
- instruction: Brief action description (use tool names when possible)
- preconditions: List of conditions that must be true before starting
- postconditions: List of conditions that will be true after completion
- expected_tools: List of tool names that will be used

Output as a JSON array:
[
  {{
    "instruction": "navigate to the table",
    "preconditions": ["agent is mobile"],
    "postconditions": ["agent is near table"],
    "expected_tools": ["Navigate"]
  }},
  ...
]

Important:
- Use atomic actions (navigate, pick, place, open, close)
- Each subtask should use 1-2 tools maximum
- Preconditions of subtask N should match postconditions of subtask N-1
- Be specific about object names from the environment

Output only the JSON array, no additional text.'''
    
    def __init__(self, llm_client: Any, config: Dict[str, Any]):
        """
        Initialize SubtaskManager.
        
        Args:
            llm_client: LLM client for decomposition (can be planner.llm)
            config: Configuration dictionary with optional keys:
                - progress_threshold: Threshold for boundary detection (default 0.8)
                - max_subtasks: Maximum number of subtasks (default 10)
                - decomposition_model: Model to use for decomposition
        """
        self.llm_client = llm_client
        self.config = config
        self.progress_threshold = config.get("progress_threshold", 0.8)
        self.max_subtasks = config.get("max_subtasks", 10)
        
        # State
        self.subtasks: List[Subtask] = []
        self.current_idx: int = 0
        self.step_count: int = 0
        # Track recent progress for stagnation detection
        self._recent_progress: List[float] = []
        self._stagnation_window = 6  # steps
        self._stagnation_delta = 0.05  # minimal improvement to be considered progress
        
        # Tool name mapping for progress estimation
        self.tool_categories = {
            "navigation": ["Navigate", "Explore", "FindAgentActionTool", "FindObjectTool", "FindFurnitureTool", "FindRoomTool"],
            "manipulation": ["Pick", "Place", "Open", "Close", "Rearrange"],
            "perception": ["Wait", "Done"],
        }
    
    def reset(self):
        """Reset manager state for new episode."""
        self.subtasks = []
        self.current_idx = 0
        self.step_count = 0
    
    def decompose_task(
        self, 
        instruction: str, 
        world_state: Dict[str, Any],
        available_tools: Optional[List[str]] = None
    ) -> List[Subtask]:
        """
        Use LLM to decompose task into subtask sequence.
        
        This implements CycleVLA's LLM-based subtask decomposition.
        
        Args:
            instruction: High-level task instruction
            world_state: Current environment state dict
            available_tools: List of available tool names
        
        Returns:
            List of Subtask objects
        """
        if available_tools is None:
            available_tools = ["Navigate", "Pick", "Place", "Open", "Close", "Wait", "Done"]
        
        # Build world state summary
        world_summary = self._summarize_world_state(world_state)
        
        # Build prompt
        prompt = self.DECOMPOSITION_PROMPT.format(
            instruction=instruction,
            world_state_summary=world_summary,
            available_tools=", ".join(available_tools)
        )
        
        try:
            # Call LLM for decomposition
            if hasattr(self.llm_client, 'generate'):
                # Use planner's LLM wrapper
                # Note: VLLM wrapper uses 'stop' parameter, not 'stop_str'
                response = self.llm_client.generate(prompt, stop=None)
            elif hasattr(self.llm_client, 'chat'):
                # Use OpenAI-style client
                response = self.llm_client.chat.completions.create(
                    model=self.config.get("decomposition_model", "gpt-4-turbo"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                ).choices[0].message.content
            else:
                logger.warning("LLM client type not recognized, using fallback decomposition")
                return self._fallback_decomposition(instruction)
            
            # Parse JSON response
            subtasks = self._parse_decomposition_response(response)
            
            if not subtasks:
                logger.warning("Empty subtask list from LLM, using fallback")
                return self._fallback_decomposition(instruction)
            
            # Limit number of subtasks
            if len(subtasks) > self.max_subtasks:
                logger.warning(f"Truncating subtasks from {len(subtasks)} to {self.max_subtasks}")
                subtasks = subtasks[:self.max_subtasks]
            
            self.subtasks = subtasks
            self.current_idx = 0
            
            # Mark first subtask as active
            if self.subtasks:
                self.subtasks[0].status = "active"
                self.subtasks[0].start_step = self.step_count
            
            logger.info(f"Decomposed task into {len(self.subtasks)} subtasks")
            return self.subtasks
            
        except Exception as e:
            logger.error(f"Error in task decomposition: {e}")
            return self._fallback_decomposition(instruction)
    
    def _summarize_world_state(self, world_state: Dict[str, Any]) -> str:
        """Summarize world state for prompt."""
        summary_parts = []
        
        if "objects" in world_state:
            obj_list = []
            for obj_name, obj_info in world_state.get("objects", {}).items():
                location = obj_info.get("location", "unknown")
                obj_list.append(f"- {obj_name} (at {location})")
            if obj_list:
                summary_parts.append("Objects:\n" + "\n".join(obj_list[:10]))
        
        if "furniture" in world_state:
            furn_list = []
            for furn_name, furn_info in world_state.get("furniture", {}).items():
                furn_list.append(f"- {furn_name}")
            if furn_list:
                summary_parts.append("Furniture:\n" + "\n".join(furn_list[:10]))
        
        if "agent_holding" in world_state:
            held = world_state.get("agent_holding")
            if held:
                summary_parts.append(f"Agent is holding: {held}")
            else:
                summary_parts.append("Agent is not holding anything")
        
        if "rooms" in world_state:
            rooms = list(world_state.get("rooms", {}).keys())
            if rooms:
                summary_parts.append(f"Rooms: {', '.join(rooms[:5])}")
        
        return "\n".join(summary_parts) if summary_parts else "No detailed state available"
    
    def _parse_decomposition_response(self, response: str) -> List[Subtask]:
        """Parse LLM response into Subtask objects."""
        subtasks = []
        
        # Try to extract JSON from response
        try:
            # Find JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                for idx, item in enumerate(data):
                    subtask = Subtask(
                        idx=idx,
                        instruction=item.get("instruction", f"Step {idx}"),
                        preconditions=item.get("preconditions", []),
                        postconditions=item.get("postconditions", []),
                        expected_tools=item.get("expected_tools", []),
                        status="pending"
                    )
                    subtasks.append(subtask)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
        
        return subtasks
    
    def _fallback_decomposition(self, instruction: str) -> List[Subtask]:
        """
        Fallback decomposition when LLM fails.
        Creates a single subtask from the instruction.
        """
        subtask = Subtask(
            idx=0,
            instruction=instruction,
            preconditions=[],
            postconditions=["task completed"],
            expected_tools=["Navigate", "Pick", "Place"],
            status="active",
            start_step=self.step_count
        )
        self.subtasks = [subtask]
        self.current_idx = 0
        return self.subtasks
    
    def estimate_progress(
        self,
        subtask: Subtask,
        action_history: List[Any],
        world_state: Dict[str, Any]
    ) -> float:
        """
        Estimate subtask progress ∈ [0, 1].
        
        This implements progress estimation similar to CycleVLA's progress signal,
        but using heuristics instead of a trained model.
        
        Methods:
        1. Action count: steps since subtask start / expected steps
        2. Tool match: how many expected tools have been used
        3. Postcondition check: rough check of postconditions
        
        Args:
            subtask: Current subtask
            action_history: List of executed actions
            world_state: Current world state
        
        Returns:
            Progress value between 0 and 1
        """
        if subtask.status == "completed":
            return 1.0
        
        progress_components = []
        
        # 1. Action count progress
        if subtask.start_step is not None:
            steps_in_subtask = self.step_count - subtask.start_step
            # Assume each subtask takes ~3-5 steps on average
            expected_steps = max(3, len(subtask.expected_tools) * 2)
            action_progress = min(1.0, steps_in_subtask / expected_steps)
            progress_components.append(("action_count", action_progress, 0.3))
        
        # 2. Tool usage progress
        if subtask.expected_tools and action_history:
            used_tools = set()
            # Check recent actions for tool usage
            recent_actions = action_history[-10:] if len(action_history) > 10 else action_history
            for action in recent_actions:
                if hasattr(action, 'action') and len(action.action) > 0:
                    raw_tool_name = action.action[0]
                    tool_name = self._normalize_tool_name(raw_tool_name)
                    used_tools.add(tool_name)
            
            expected_set = set(subtask.expected_tools)
            if expected_set:
                tool_progress = len(used_tools & expected_set) / len(expected_set)
                progress_components.append(("tool_usage", tool_progress, 0.4))
        
        # 3. Simple postcondition heuristics
        postcond_progress = self._check_postconditions(subtask, world_state)
        progress_components.append(("postconditions", postcond_progress, 0.3))
        
        # Weighted average
        if progress_components:
            total_weight = sum(w for _, _, w in progress_components)
            weighted_sum = sum(p * w for _, p, w in progress_components)
            progress = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            progress = 0.0
        
        progress = min(1.0, max(0.0, progress))
        self._record_progress(progress)
        return progress
    
    def _check_postconditions(self, subtask: Subtask, world_state: Dict[str, Any]) -> float:
        """
        Check postcondition satisfaction (heuristic).
        
        This is a simplified check - in practice, you'd want more 
        sophisticated condition parsing.
        """
        if not subtask.postconditions:
            return 0.5  # No postconditions = assume halfway
        
        satisfied = 0
        for cond in subtask.postconditions:
            cond_lower = cond.lower()
            
            # Check for holding conditions
            if "holding" in cond_lower or "grasp" in cond_lower:
                agent_holding = world_state.get("agent_holding")
                if agent_holding:
                    # Check if the held object matches
                    for word in cond_lower.split():
                        if word in str(agent_holding).lower():
                            satisfied += 1
                            break
            
            # Check for location conditions
            elif "near" in cond_lower or "at" in cond_lower:
                # Simplified: assume some progress if we have agent position
                if world_state.get("agent_poses"):
                    satisfied += 0.5
            
            # Check for placement conditions
            elif "on" in cond_lower or "in" in cond_lower:
                # Check object positions
                objects = world_state.get("objects", {})
                for obj_name, obj_info in objects.items():
                    if obj_name.lower() in cond_lower:
                        satisfied += 0.5
                        break
        
        return satisfied / len(subtask.postconditions) if subtask.postconditions else 0.5
    
    def is_boundary(self, progress: float, threshold: Optional[float] = None) -> bool:
        """
        Check if at subtask boundary (progress >= threshold).
        
        This corresponds to CycleVLA's boundary detection where the VLM
        is invoked when progress reaches τp.
        
        Args:
            progress: Current progress value
            threshold: Optional threshold override
        
        Returns:
            True if at boundary
        """
        if threshold is None:
            threshold = self.progress_threshold
        
        # Standard threshold check
        if progress >= threshold:
            return True
            
        # If progress is close or stagnating, allow an early boundary to avoid being stuck.
        # We lowered the threshold to catch stagnation earlier.
        if self._is_stagnating() and progress >= max(0.3, 0.5 * threshold):
            return True
            
        return False

    def _normalize_tool_name(self, tool_name: str) -> str:
        """
        Map semantically equivalent tools to canonical names.
        Example: Explore is treated as Navigate for progress estimation.
        """
        if tool_name.lower().startswith("explore"):
            return "Navigate"
        return tool_name

    def _record_progress(self, progress: float):
        """Keep a rolling window of recent progress to detect stagnation."""
        self._recent_progress.append(progress)
        if len(self._recent_progress) > self._stagnation_window:
            self._recent_progress.pop(0)

    def _is_stagnating(self) -> bool:
        """Detect if progress has plateaued within the recent window."""
        if len(self._recent_progress) < self._stagnation_window:
            return False
        delta = max(self._recent_progress) - min(self._recent_progress)
        return delta < self._stagnation_delta
    
    def mark_complete(self, subtask_idx: int):
        """
        Mark subtask as completed and advance to next.
        
        Args:
            subtask_idx: Index of subtask to mark complete
        """
        if 0 <= subtask_idx < len(self.subtasks):
            self.subtasks[subtask_idx].status = "completed"
            self.subtasks[subtask_idx].end_step = self.step_count
            
            # Activate next subtask if available
            next_idx = subtask_idx + 1
            if next_idx < len(self.subtasks):
                self.subtasks[next_idx].status = "active"
                self.subtasks[next_idx].start_step = self.step_count
                self.current_idx = next_idx
            
            logger.info(f"Subtask {subtask_idx} completed, advanced to {self.current_idx}")
    
    def mark_failed(self, subtask_idx: int):
        """Mark subtask as failed."""
        if 0 <= subtask_idx < len(self.subtasks):
            self.subtasks[subtask_idx].status = "failed"
            self.subtasks[subtask_idx].end_step = self.step_count
    
    def get_current_subtask(self) -> Optional[Subtask]:
        """Return current active subtask or None if all done."""
        if 0 <= self.current_idx < len(self.subtasks):
            return self.subtasks[self.current_idx]
        return None
    
    def advance_step(self):
        """Increment step counter."""
        self.step_count += 1
    
    def rollback_to(self, subtask_idx: int):
        """
        Rollback to specified subtask.
        
        Resets status of subtasks after the target index.
        """
        if not (0 <= subtask_idx < len(self.subtasks)):
            return
        
        # Reset current and later subtasks
        for i in range(subtask_idx, len(self.subtasks)):
            self.subtasks[i].status = "pending" if i > subtask_idx else "active"
            self.subtasks[i].end_step = None
            if i == subtask_idx:
                self.subtasks[i].start_step = self.step_count
        
        self.current_idx = subtask_idx
        logger.info(f"Rolled back to subtask {subtask_idx}")
    
    def get_subtask_summary(self) -> str:
        """Get a text summary of all subtasks and their status."""
        if not self.subtasks:
            return "No subtasks defined"
        
        lines = []
        for st in self.subtasks:
            status_marker = {
                "pending": "○",
                "active": "●",
                "completed": "✓",
                "failed": "✗"
            }.get(st.status, "?")
            lines.append(f"[{status_marker}] {st.idx}: {st.instruction}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize manager state."""
        return {
            "subtasks": [st.to_dict() for st in self.subtasks],
            "current_idx": self.current_idx,
            "step_count": self.step_count,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], llm_client: Any, config: Dict) -> "SubtaskManager":
        """Deserialize manager state."""
        mgr = cls(llm_client, config)
        mgr.subtasks = [Subtask.from_dict(st) for st in data.get("subtasks", [])]
        mgr.current_idx = data.get("current_idx", 0)
        mgr.step_count = data.get("step_count", 0)
        return mgr
