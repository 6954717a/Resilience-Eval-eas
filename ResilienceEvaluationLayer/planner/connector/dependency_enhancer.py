from typing import List, Dict, Any, Tuple
from collections import deque, defaultdict

class TaskDependencyEnhancer:
    """
    基于任务语义和拓扑结构，将原始子任务列表转换为带依赖关系的、分阶段的执行计划。
    这个类将所有相关的依赖增强和任务组织逻辑从PerceptionConnector中解耦出来。
    """

    def structure_and_phase(
        self,
        structured_subtasks: List[Dict[str, Any]],
        max_agents: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, List[str]]]:
        """
        主入口方法，接收原始子任务列表，返回增强和分阶段后的任务计划。

        Args:
            structured_subtasks: 从LLM获取的原始子任务列表。
            max_agents: 最大智能体数量。

        Returns:
            A tuple containing:
            - enhanced_subtasks: 经过语义增强和依赖清理后的完整任务列表。
            - execution_phases: 按阶段组织的可并行执行的任务计划。
            - task_dependency_graph: 优化后的任务依赖图。
        """
        # 1. 首先进行语义增强，添加隐含的依赖关系 (例如 Pick -> Navigate -> Place)
        semantically_enhanced_tasks = self._enhance_semantic_dependencies(structured_subtasks)

        # 2. 构建并优化任务依赖图
        task_dependency_graph = self._build_dependency_graph(semantically_enhanced_tasks)

        # 基于优化后的图更新任务的先决条件
        task_dict = {task['task_id']: task for task in semantically_enhanced_tasks}
        for task_id, prereqs in task_dependency_graph.items():
            if task_id in task_dict:
                task_dict[task_id]['prerequisites'] = prereqs
        
        updated_tasks = list(task_dict.values())
        
        # 3. 构建依赖图并组织为执行阶段
        execution_phases = self._organize_tasks_into_phases(updated_tasks, max_agents)

        # 4. 从最终阶段中提取完整的、有序的任务列表
        final_task_list = [task for phase in execution_phases for task in phase['tasks']]

        return final_task_list, execution_phases, task_dependency_graph

    def _enhance_semantic_dependencies(self, structured_subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于任务语义增强依赖关系，特别处理复合动作（如Pick-Navigate-Place）。"""
        enhanced_tasks = [task.copy() for task in structured_subtasks]
        
        pick_tasks: List[Tuple[int, Dict[str, Any]]] = []
        navigate_tasks: List[Tuple[int, Dict[str, Any]]] = []
        place_tasks: List[Tuple[int, Dict[str, Any]]] = []
        
        for i, task in enumerate(enhanced_tasks):
            task_type = task.get('task_type', '')
            if task_type == 'Pick': pick_tasks.append((i, task))
            elif task_type == 'Navigate': navigate_tasks.append((i, task))
            elif task_type == 'Place': place_tasks.append((i, task))
        
        for nav_idx, nav_task in navigate_tasks:
            if self._is_carrying_navigation_task(nav_task, pick_tasks, place_tasks):
                required_picks = self._find_required_pick_tasks(nav_task, pick_tasks)
                current_prereqs = set(nav_task.get('prerequisites', []))
                for _, pick_task in required_picks:
                    current_prereqs.add(pick_task['task_id'])
                enhanced_tasks[nav_idx]['prerequisites'] = list(current_prereqs)
                enhanced_tasks[nav_idx]['is_carrying_navigation'] = True
        
        for place_idx, place_task in place_tasks:
            required_picks = self._find_pick_tasks_for_place(place_task, pick_tasks)
            required_navs = self._find_carrying_navigation_for_place(place_task, navigate_tasks)
            current_prereqs = set(place_task.get('prerequisites', []))
            for _, pick_task in required_picks:
                current_prereqs.add(pick_task['task_id'])
            for _, nav_task in required_navs:
                current_prereqs.add(nav_task['task_id'])
            enhanced_tasks[place_idx]['prerequisites'] = list(current_prereqs)
        
        self._validate_enhanced_dependencies(enhanced_tasks)
        return enhanced_tasks

    def _organize_tasks_into_phases(
        self,
        structured_subtasks: List[Dict[str, Any]],
        max_agents: int
    ) -> List[Dict[str, Any]]:
        """将任务组织为可并行执行的阶段"""
        sorted_tasks = self._topological_sort_tasks(structured_subtasks)
        
        phases = []
        remaining_tasks = sorted_tasks.copy()
        completed_task_ids = set()
        safety_counter = 0
        max_iterations = len(sorted_tasks) * 2

        while remaining_tasks and safety_counter < max_iterations:
            safety_counter += 1
            current_phase_tasks = []
            
            # 查找所有在当前可以执行的任务
            runnable_tasks = [
                task for task in remaining_tasks
                if all(prereq in completed_task_ids for prereq in task.get('prerequisites', []))
            ]

            # 按并行能力和优先级排序
            runnable_tasks.sort(key=lambda t: (not t.get('can_parallel', False), -t.get('priority', 3)))

            # 填充当前阶段
            for task in runnable_tasks:
                if len(current_phase_tasks) >= max_agents:
                    break
                current_phase_tasks.append(task)
                if not task.get('can_parallel', False):
                    break
            
            if not current_phase_tasks and remaining_tasks:
                current_phase_tasks.append(remaining_tasks[0])

            # 更新状态
            for task in current_phase_tasks:
                remaining_tasks.remove(task)
                completed_task_ids.add(task['task_id'])
            
            if current_phase_tasks:
                phases.append({
                    'phase_id': len(phases),
                    'tasks': current_phase_tasks,
                    'max_parallel_tasks': len(current_phase_tasks),
                    'estimated_duration': max((task.get('estimated_duration', 5.0) for task in current_phase_tasks), default=5.0),
                    'required_agents': min(len(current_phase_tasks), max_agents)
                })

        if remaining_tasks:
            if phases:
                phases[-1]['tasks'].extend(remaining_tasks)
            else:
                phases.append({
                    'phase_id': 0, 'tasks': remaining_tasks, 'max_parallel_tasks': len(remaining_tasks),
                    'estimated_duration': max((task.get('estimated_duration', 5.0) for task in remaining_tasks), default=5.0),
                    'required_agents': min(len(remaining_tasks), max_agents)
                })
        return phases

    def _topological_sort_tasks(self, structured_subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """对任务进行拓扑排序，自动处理循环依赖"""
        cleaned_tasks = self._clean_task_dependencies(structured_subtasks)
        task_dict = {task['task_id']: task for task in cleaned_tasks}
        
        in_degree = {task_id: 0 for task_id in task_dict}
        graph = {task_id: [] for task_id in task_dict}
        
        for task_id, task in task_dict.items():
            for prereq in task.get('prerequisites', []):
                if prereq in graph:
                    graph[prereq].append(task_id)
                    in_degree[task_id] += 1
        
        queue = deque([task_id for task_id, degree in in_degree.items() if degree == 0])
        sorted_task_ids = []
        
        while queue:
            current = queue.popleft()
            sorted_task_ids.append(current)
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(sorted_task_ids) != len(task_dict):
            # 处理循环依赖
            remaining_ids = set(task_dict.keys()) - set(sorted_task_ids)
            remaining_tasks = [task_dict[task_id] for task_id in remaining_ids]
            remaining_tasks.sort(key=lambda x: x.get('priority', 3), reverse=True)
            sorted_task_ids.extend([task['task_id'] for task in remaining_tasks])
            
        return [task_dict[task_id] for task_id in sorted_task_ids]

    def _build_dependency_graph(self, structured_subtasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """ 构建任务依赖关系图并检测和修复循环依赖"""
        dependency_graph = {}
        task_ids = {task['task_id'] for task in structured_subtasks}
        
        for task in structured_subtasks:
            task_id = task['task_id']
            prerequisites = task.get('prerequisites', [])
            
            cleaned_prerequisites = [prereq for prereq in prerequisites if prereq != task_id and prereq in task_ids]
            dependency_graph[task_id] = cleaned_prerequisites
        
        cycles = self._detect_dependency_cycles(dependency_graph)
        if cycles:
            print(f"WARNING: Detected {len(cycles)} dependency cycles:")
            for i, cycle in enumerate(cycles):
                print(f"  Cycle {i+1}: {' -> '.join(cycle + [cycle[0]])}")
            
            dependency_graph = self._break_dependency_cycles(dependency_graph, cycles)
            
        return dependency_graph

    def _detect_dependency_cycles(self, dependency_graph: Dict[str, List[str]]) -> List[List[str]]:
        """检测依赖图中的循环"""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                try:
                    cycle_start_index = path.index(node)
                    cycles.append(path[cycle_start_index:])
                except ValueError:
                    pass
                return
            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in dependency_graph.get(node, []):
                if neighbor in dependency_graph:
                    dfs(neighbor, path + [neighbor])
            
            rec_stack.remove(node)

        for task_id in dependency_graph:
            if task_id not in visited:
                dfs(task_id, [task_id])
        
        # 去重
        unique_cycles = []
        seen_cycles = set()
        for cycle in cycles:
            sorted_cycle = tuple(sorted(cycle))
            if sorted_cycle not in seen_cycles:
                unique_cycles.append(cycle)
                seen_cycles.add(sorted_cycle)
        
        return unique_cycles

    def _break_dependency_cycles(self, dependency_graph: Dict[str, List[str]], cycles: List[List[str]]) -> Dict[str, List[str]]:
        """通过移除循环中的一条边来打破依赖循环"""
        cleaned_graph = {k: list(v) for k, v in dependency_graph.items()}
        for cycle in cycles:
            if len(cycle) < 2:
                continue

            # 尝试移除从最后一个节点到第一个节点的依赖
            u, v = cycle[-1], cycle[0]
            if v in cleaned_graph.get(u, []):
                cleaned_graph[u].remove(v)
                print(f"INFO: Broke cycle by removing dependency: {u} -> {v}")
        return cleaned_graph

    def _inject_navigate_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        作为安全保障，检查每个Pick/Open任务，如果缺少前置的Navigate，则强制注入一个。
        """
        newly_added_tasks = []
        task_dict = {task['task_id']: task for task in tasks}
        
        # 构建一个包含所有父子关系的依赖图，便于查找
        full_dependency_graph = {tid: set(p.get('prerequisites', [])) for tid, p in task_dict.items()}
        for _ in range(len(task_dict)): # 迭代以传播依赖关系
            for tid, prereqs in full_dependency_graph.items():
                for p in list(prereqs):
                    if p in full_dependency_graph:
                        full_dependency_graph[tid].update(full_dependency_graph[p])

        interaction_tasks = [t for t in tasks if t.get('task_type') in ['Pick', 'Open']]

        for i, task in enumerate(interaction_tasks):
            has_navigate_prereq = False
            # 检查所有直接和间接的先决条件
            prereq_ids = full_dependency_graph.get(task['task_id'], set())
            
            for prereq_id in prereq_ids:
                if prereq_id in task_dict and task_dict[prereq_id].get('task_type') == 'Navigate':
                    # 检查导航目标是否与交互目标相关
                    nav_target = task_dict[prereq_id].get('target', '')
                    interaction_target = task.get('target', '')
                    if self._are_targets_related(nav_target, interaction_target):
                        has_navigate_prereq = True
                        break
            
            if not has_navigate_prereq:
                # 需要注入一个新的Navigate任务
                # 假设target是 'object_name'，我们需要导航到它的父物体
                # 这个信息通常在world_state里，这里我们只能做一个合理的猜测
                # 我们假设Place任务的目标是容器，而Pick任务的目标是物体
                # 导航目标应该是这个物体的父容器名，但我们在这里无法直接获取
                # LLM应该在分解时提供这个信息。这里我们只能创建一个通用的导航
                # 例如，导航到物体本身的名字，让执行器去解析
                
                nav_target = task.get('target', '')
                nav_task_id = f"injected_nav_{i}_{task['task_id']}"
                
                new_nav_task = {
                    'task_id': nav_task_id,
                    'task_type': 'Navigate',
                    'target': nav_target,
                    'description': f"Auto-injected: Navigate to {nav_target} before {task['task_type']}.",
                    'priority': task.get('priority', 3) + 1, # 导航应有更高优先级
                    'estimated_duration': 10.0,
                    'prerequisites': [],
                    'can_parallel': False, # 通常导航不能并行
                    'phase_group': task.get('phase_group', 'execution')
                }
                
                print(f"INFO: Injecting missing Navigate task for {task['task_type']}->{task['target']}")
                
                # 将新导航任务设置为原任务的先决条件
                current_prereqs = task.get('prerequisites', [])
                current_prereqs.append(nav_task_id)
                task['prerequisites'] = current_prereqs
                
                newly_added_tasks.append(new_nav_task)

        return tasks + newly_added_tasks
        
    def _clean_task_dependencies(self, structured_subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """清理任务依赖关系，移除自循环和无效引用"""
        task_ids = {task['task_id'] for task in structured_subtasks}
        cleaned_tasks = []
        for task in structured_subtasks:
            cleaned_task = task.copy()
            prereqs = task.get('prerequisites', [])
            cleaned_task['prerequisites'] = [p for p in prereqs if p in task_ids and p != task['task_id']]
            cleaned_tasks.append(cleaned_task)
        return cleaned_tasks

    def _is_carrying_navigation_task(self, nav_task: Dict[str, Any], pick_tasks: List, place_tasks: List) -> bool:
        """判断是否为携带物品的导航任务"""
        nav_target = nav_task.get('target', '').lower()
        nav_description = nav_task.get('description', '').lower()
        place_targets = {p_task.get('target', '').lower() for _, p_task in place_tasks}
        if nav_target in place_targets: return True
        
        carrying_keywords = ['carry', 'carrying', 'bring', 'take', 'transport', 'move', 'deliver']
        if any(keyword in nav_description for keyword in carrying_keywords): return True
        
        try:
            nav_order = int(nav_task.get('task_id', 'task_0').split('_')[1])
            has_pick_before = any(int(p_task.get('task_id', 'task_99').split('_')[1]) < nav_order for _, p_task in pick_tasks)
            has_place_after = any(int(p_task.get('task_id', 'task_0').split('_')[1]) > nav_order for _, p_task in place_tasks)
            return has_pick_before and has_place_after
        except (ValueError, IndexError):
            return False

    def _find_required_pick_tasks(self, nav_task: Dict[str, Any], pick_tasks: List) -> List[Tuple[int, Dict[str, Any]]]:
        """找到导航任务所需的所有前置Pick任务"""
        try:
            nav_order = int(nav_task.get('task_id', 'task_0').split('_')[1])
            return [(idx, p_task) for idx, p_task in pick_tasks if int(p_task.get('task_id', 'task_99').split('_')[1]) < nav_order]
        except (ValueError, IndexError):
            return []

    def _find_pick_tasks_for_place(self, place_task: Dict[str, Any], pick_tasks: List) -> List[Tuple[int, Dict[str, Any]]]:
        """找到Place任务对应的Pick任务"""
        desc = place_task.get('description', '').lower()
        return [(idx, p_task) for idx, p_task in pick_tasks if p_task.get('target', '').lower() in desc]

    def _find_carrying_navigation_for_place(self, place_task: Dict[str, Any], navigate_tasks: List) -> List[Tuple[int, Dict[str, Any]]]:
        """找到Place任务对应的携带导航任务"""
        place_target = place_task.get('target', '').lower()
        related_navs = []
        for nav_idx, nav_task in navigate_tasks:
            if nav_task.get('is_carrying_navigation') and self._are_targets_related(nav_task.get('target', '').lower(), place_target):
                related_navs.append((nav_idx, nav_task))
        return related_navs

    def _are_targets_related(self, target1: str, target2: str) -> bool:
        """判断两个目标是否相关"""
        if not target1 or not target2: return False
        if target1 == target2: return True
        words1 = set(target1.replace('_', ' ').split())
        words2 = set(target2.replace('_', ' ').split())
        return len(words1.intersection(words2)) > 0

    def _validate_enhanced_dependencies(self, enhanced_tasks: List[Dict[str, Any]]) -> None:
        """验证增强后的依赖关系，防止循环依赖"""
        task_ids = {task['task_id'] for task in enhanced_tasks}
        for task in enhanced_tasks:
            prereqs = task.get('prerequisites', [])
            task['prerequisites'] = [p for p in prereqs if p in task_ids and p != task['task_id']]
