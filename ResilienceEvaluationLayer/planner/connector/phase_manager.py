from typing import Dict, List, Any, Optional, Tuple
import json

from habitat_llm.planner.prompting import (
    get_miqp_prompt,
    get_sequencing_system_prompt,
)
from habitat_llm.planner.connector.planner_utils import (
    extract_json_from_text,
    get_llm_config,
    normalize_task_list,
)

class PhaseManager:
    """
    负责管理任务执行的阶段（Phases）。
    此类将所有与阶段创建、状态管理和推进相关的逻辑与PerceptionConnector解耦。
    """
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self.task_execution_phases: List[Dict[str, Any]] = []
        self.current_phase_index: int = 0
        self.completed_tasks: List[str] = []
        self.phase_step_count: Dict[int, int] = {}  # Track steps per phase

    def set_execution_phases(self, phases: List[Dict[str, Any]]):
        """设置任务执行阶段"""
        self.task_execution_phases = phases
        self.current_phase_index = 0
        self.completed_tasks = []
        self.phase_step_count = {i: 0 for i in range(len(phases))}  # Initialize step counts
        print(f"DEBUG: PhaseManager initialized with {len(phases)} phases.")
        
    def decompose_and_initialize_phases(
        self,
        instruction: str,
        world_description: str,
        agent_info_string: str,
        llm_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        使用LLM分解指令，并返回结构化的子任务列表。
        """
        if not self.llm_client:
            raise ValueError("LLM client not initialized in PhaseManager. Cannot decompose task.")

        # 使用LLM进行初步分解
        structured_subtasks = self._decompose_with_sequencing_prompt(
            instruction, world_description, agent_info_string, llm_config
        )
        
        return structured_subtasks

    def _decompose_with_sequencing_prompt(
        self,
        instruction: str,
        world_description: str,
        agent_info_string: str,
        llm_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """使用专门的序列化分解prompt"""
        prompt_template = get_miqp_prompt("sequencing_decomposition", get_llm_config())
        sequencing_prompt = prompt_template(
            instruction, world_description, agent_info_string
        )

        try:
            messages = [
                {"role": "system", "content": get_sequencing_system_prompt()},
                {"role": "user", "content": sequencing_prompt},
            ]
            api_params = {
                "model": llm_config.get("gpt_version", "gpt-4o-2024-11-20"),
                "messages": messages,
                "max_tokens": llm_config.get("max_tokens", 1500),
                "temperature": 0.1
            }

            response = self.llm_client.chat.completions.create(**api_params)
            decomposed_text = response.choices[0].message.content.strip()
            
            structured_tasks = self._parse_sequenced_decomposition_response(decomposed_text)
            
            if structured_tasks:
                print(f"DEBUG: Successfully parsed {len(structured_tasks)} sequenced subtasks by PhaseManager")
                return structured_tasks
            else:
                print("Warning: PhaseManager failed to parse sequenced subtasks, returning empty list.")
                return []
                
        except Exception as e:
            print(f"Error in PhaseManager during LLM call for task decomposition: {e}")
            raise

    def _parse_sequenced_decomposition_response(self, response_text: str) -> List[Dict[str, Any]]:
        """解析带序列信息的任务分解响应"""
        try:
            json_text = extract_json_from_text(response_text, list)
            if not json_text:
                return []

            structured_tasks = json.loads(json_text)
            validated_tasks = normalize_task_list(
                structured_tasks,
                required_fields=['task_type', 'target', 'description'],
                defaults={
                    'priority': 3,
                    'estimated_duration': 5.0,
                    'preferred_agent': None,
                    'prerequisites': [],
                    'can_parallel': False,
                    'phase_group': 'execution',
                },
                add_task_id=True,
                clamp_priority=(1, 5),
                clamp_duration=(1.0, 60.0),
                include_can_parallel=True,
                include_phase_group=True,
                log_invalid_prereqs=True,
            )

            print(f"DEBUG: Validated {len(validated_tasks)} tasks from LLM response in PhaseManager")
            return validated_tasks
            
        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: JSON parsing for sequenced decomposition failed in PhaseManager: {e}")
            return []

    def get_current_phase_tasks(self) -> Optional[Dict[str, Any]]:
        """ 获取当前阶段的任务"""
        if (self.current_phase_index < len(self.task_execution_phases)):
            return self.task_execution_phases[self.current_phase_index]
        return None

    def advance_to_next_phase(self) -> bool:
        """ 推进到下一个执行阶段"""
        if self.current_phase_index < len(self.task_execution_phases) - 1:
            current_phase = self.task_execution_phases[self.current_phase_index]
            for task in current_phase['tasks']:
                if task['task_id'] not in self.completed_tasks:
                    self.completed_tasks.append(task['task_id'])
            
            # Reset step count for previous phase
            old_idx = self.current_phase_index
            self.current_phase_index += 1
            if old_idx in self.phase_step_count:
                print(f"DEBUG: Phase {old_idx + 1} completed after {self.phase_step_count[old_idx]} steps")
            self.phase_step_count[self.current_phase_index] = 0  # Reset for new phase
            print(f"DEBUG: Advanced to phase {self.current_phase_index + 1}/{len(self.task_execution_phases)}")
            return True
        
        # Mark tasks from the last phase as complete as well
        if self.current_phase_index == len(self.task_execution_phases) -1:
            current_phase = self.task_execution_phases[self.current_phase_index]
            for task in current_phase['tasks']:
                if task['task_id'] not in self.completed_tasks:
                    self.completed_tasks.append(task['task_id'])
        
        return False
    
    def increment_phase_step(self):
        """当前阶段的步数计数"""
        if self.current_phase_index in self.phase_step_count:
            self.phase_step_count[self.current_phase_index] += 1
    
    def get_current_phase_step_count(self) -> int:
        """获取当前阶段的步数"""
        return self.phase_step_count.get(self.current_phase_index, 0)

    def is_current_phase_complete(self, agent_statuses: Dict[int, str]) -> bool:
        """ 检查当前阶段是否完成"""
        current_phase = self.get_current_phase_tasks()
        if not current_phase:
            return True
        
        try:
            if not agent_statuses:
                return False
            
            completion_keywords = [
                'completed', 'done', 'finished', 'success', 'successful', 'achievement',
                'reached', 'arrived', 'found', 'picked', 'placed', 'opened', 'closed',
                'execution!', 'goal', 'target', 'complete', 'accomplish'
            ]
            
            failure_keywords = [
                'failed', 'error', 'stuck', 'impossible', 'cannot', 'unable', 'fail',
                'blocked', 'obstacle', 'collision', 'timeout', 'unreachable'
            ]
            
            completed_agents = 0
            failed_agents = 0
            total_relevant_agents = 0
            
            for agent_id, status in agent_statuses.items():
                if status and isinstance(status, str) and status.strip():
                    total_relevant_agents += 1
                    status_lower = status.lower()
                    
                    has_completion = any(keyword in status_lower for keyword in completion_keywords)
                    has_failure = any(keyword in status_lower for keyword in failure_keywords)
                    
                    if has_completion:
                        completed_agents += 1
                    elif has_failure:
                        failed_agents += 1
            
            current_phase_tasks = current_phase.get('tasks', [])
            required_agents = current_phase.get('required_agents', 1)
            
            is_single_task_phase = len(current_phase_tasks) == 1
            
            if is_single_task_phase:
                phase_complete = completed_agents >= 1 and failed_agents == 0
            else:
                min_required_completions = min(required_agents, len(current_phase_tasks))
                phase_complete = (completed_agents >= min_required_completions and 
                                failed_agents < total_relevant_agents // 2)
            
            if phase_complete:
                print(f"[SUCCESS] Phase {current_phase['phase_id']} marked as COMPLETE by PhaseManager!")
            
            return phase_complete
            
        except Exception as e:
            print(f"[ERROR] Phase completion check failed in PhaseManager: {e}")
            import traceback
            traceback.print_exc()
            return False 

