# SayCan 组件结构说明

## 一、目录结构

```
temp_modify/habitat_llm/
├── evaluation/saycan/              # 评估与分析组件
│   ├── __init__.py
│   ├── affordance_model.py         # Affordance 模型（增强版）
│   ├── saycan_analyzer.py          # SayCan 分析器
│   ├── state_assessor.py           # 状态评估器
│   ├── critic_integration.py       # Critic 集成
│   └── README.md
│
└── planner/saycan/                  # Planner 集成组件
    ├── __init__.py
    ├── saycan_planner.py            # SayCan Planner（主类）
    ├── candidate_scorer.py          # 候选评分器
    ├── stability_monitor.py         # 稳定性监控器
    ├── planner_integration.py       # Planner 集成函数
    ├── prompts.py                   # SayCan prompts
    ├── README.md
    ├── IMPLEMENTATION_SUMMARY.md
    └── COMPONENT_STRUCTURE.md       # 本文件
```

## 二、组件详细说明

### 2.1 evaluation/saycan/ - 评估与分析层

#### AffordanceModel (`affordance_model.py`)

**职责**: 计算动作的物理可行性 P(Can)

**核心方法**:
- `get_affordance(skill, target, agent_id) -> float`: 返回可行性分数
- `get_affordance_with_details(skill, target, agent_id) -> Tuple[float, Dict]`: 返回分数和详细信息

**支持的技能**:
- `Navigate`: 基于距离和 NavMesh 路径可达性
- `Pick`: 基于距离、可见性（Raycast）、手部状态
- `Place`: 基于距离、手部持有状态
- `Open/Close`: 类似 Pick
- `Wait/Done`: 总是可行（1.0）

**详细信息字段**:
- `distance`: 到目标的距离
- `visibility`: 是否可见（使用 Raycast）
- `reachability`: 是否可达（NavMesh 路径）
- `failure_reason`: 低分原因（target_too_far, target_occluded, target_unreachable 等）

**配置参数**:
- `max_nav_distance`: 最大导航距离（默认 20.0）
- `pick_distance_thresh`: Pick 距离阈值（默认 2.0）
- `visibility_check_enabled`: 是否启用可见性检查（默认 True）

#### SayCanAnalyzer (`saycan_analyzer.py`)

**职责**: 收集和分析 SayCan 数据

**核心方法**:
- `record_step(step, candidates, selected, stability_triggered, fallback_action)`: 记录 planning step 数据
- `analyze_episode() -> Dict`: 计算 Episode 级统计

**数据结构**:
```python
@dataclass
class SayCanStepData:
    step: int
    candidates: List[ScoredCandidate]
    selected_action: Optional[str]
    selected_rank: int
    stability_triggered: bool
    say_score: float
    can_score: float
    total_score: float
```

**分析内容**:
- **分数分布**: Say/Can/Total 的均值、方差、分位数
- **候选分析**: LLM 推荐 vs 最终选择的排名变化
- **稳定性指标**: 阈值违反次数、分数方差
- **Affordance 分布**: 按技能类型的分数分布

#### StateAssessor (`state_assessor.py`)

**职责**: 评估世界状态，为 Affordance 计算提供上下文

**核心方法**:
- `assess_world_state(agent_id) -> Dict`: 提取 agent 位置、对象位置、可见性等信息
- `check_visibility(target_name, agent_id) -> Tuple[bool, Dict]`: 使用 Raycast 检查可见性

#### CriticIntegration (`critic_integration.py`)

**职责**: 集成 A2C Critic 与 SayCan，对比 Affordance 与价值函数

**核心方法**:
- `compare_affordance_vs_critic(affordance_score, critic_value, action, state) -> Dict`: 比较并记录
- `compute_correlation(window) -> float`: 计算相关性

### 2.2 planner/saycan/ - Planner 集成层

#### SayCanPlanner (`saycan_planner.py`)

**职责**: 主 Planner 类，实现完整的 SayCan 流程

**继承关系**: `SayCanPlanner(LLMPlanner)`

**核心流程**:
1. `generate_action_response()`:
   - 调用 `_get_say_candidates()` 获取 LLM 候选
   - 使用 `CandidateScorer` 评分
   - 使用 `StabilityMonitor` 检查稳定性
   - 调用 `SayCanAnalyzer.record_step()` 记录数据
   - 返回格式化的动作响应

**初始化组件**:
- `AffordanceModel`: 计算 Can 分数
- `CandidateScorer`: 融合 Say × Can
- `StabilityMonitor`: 稳定性检查
- `SayCanAnalyzer`: 数据收集和分析

#### CandidateScorer (`candidate_scorer.py`)

**职责**: 实现 Say × Can 融合逻辑

**核心方法**:
- `score_candidates(candidates, agent_id, env_interface) -> List[ScoredCandidate]`:
  - 输入: LLM 候选列表 `[{action, confidence}, ...]`
  - 输出: 评分后的候选列表，按 total_score 排序

**融合公式**: `total_score = say_score × can_score`

#### StabilityMonitor (`stability_monitor.py`)

**职责**: 监控稳定性，过滤低分动作

**核心方法**:
- `check_stability(best_candidate, threshold) -> Tuple[bool, str, Optional[str]]`:
  - 返回: `(is_stable, action_to_take, reason)`
  - 如果 `total_score < threshold`: 返回 `(False, fallback_action, reason)`
  - 否则: 返回 `(True, best_candidate.action, None)`

**稳定性指标**:
- `threshold_violations`: 阈值违反次数
- `score_variance`: 分数方差
- `recent_variance`: 最近 N 步的方差

#### Planner Integration (`planner_integration.py`)

**职责**: 提供 SayCan 与 LLMPlanner 的集成函数

**核心函数**:
- `apply_saycan_selection(planner, candidates) -> Optional[str]`: 应用 SayCan 选择（未来扩展）
- `inject_saycan_context(planner, guidance) -> None`: 注入 SayCan 指导到 prompt

## 三、数据流详解

### 3.1 Planning Step 执行流程

```
1. LLMPlanner.get_next_action()
   ↓
2. SayCanPlanner.generate_action_response() [如果使用 SayCanPlanner]
   ↓
3. _get_say_candidates()
   - 调用 LLM 生成 Top-K 候选动作
   - 返回: [{"action": "Navigate[kitchen]", "confidence": 0.9}, ...]
   ↓
4. CandidateScorer.score_candidates()
   - 对每个候选:
     a. 解析 skill 和 target
     b. 调用 AffordanceModel.get_affordance_with_details()
     c. 计算 total = say × can
   - 按 total_score 排序
   - 返回: List[ScoredCandidate]
   ↓
5. StabilityMonitor.check_stability()
   - 检查 best_candidate.total_score >= threshold
   - 如果否: 返回 fallback_action ("Explore" 或 "Wait")
   - 如果是: 返回 best_candidate.action
   ↓
6. SayCanAnalyzer.record_step()
   - 记录: step, candidates, selected, stability_triggered
   - 存储到内存 (self.step_data)
   ↓
7. 返回格式化的动作响应
   - 包含 Thought trace 和动作字符串
```

### 3.2 Episode 收尾分析流程

```
1. EvaluationRunner.run_instruction() [Episode 结束]
   ↓
2. 检查 planner.saycan_analyzer 是否存在
   ↓
3. SayCanAnalyzer.analyze_episode()
   - 计算分数分布统计
   - 分析候选动作排名变化
   - 分析 Affordance 分数分布
   - 返回: Dict[str, Any]
   ↓
4. log_saycan_analysis()
   - 保存到: output_dir/analyses/saycan/saycan_analysis-episode_{id}_{filename}.json
   - 包含: saycan_summary, saycan_analysis
   ↓
5. collect_saycan_metrics() [在 log_resilience_metrics 中]
   - 提取稳定性指标
   - 计算 saycan_stability_score, saycan_filter_effectiveness
   ↓
6. log_resilience_metrics()
   - 集成到综合韧性指标
   - 保存到: output_dir/analyses/resilience/resilience_metrics-episode_{id}_{filename}.json
```

## 四、关键数据结构

### ScoredCandidate

```python
@dataclass
class ScoredCandidate:
    action: str                      # 动作字符串，如 "Navigate[kitchen]"
    say_score: float                # LLM 置信度 (0.0 - 1.0)
    can_score: float                # Affordance 可行性 (0.0 - 1.0)
    total_score: float              # 融合分数 = say × can
    affordance_details: Dict        # Affordance 计算详情
    rank: int                        # 排名 (0 = 最佳)
```

### SayCanStepData

```python
@dataclass
class SayCanStepData:
    step: int                        # Planning step 编号
    candidates: List[ScoredCandidate] # 所有候选动作
    selected_action: Optional[str]   # 最终选择的动作
    selected_rank: int               # 选中动作的排名
    stability_triggered: bool        # 是否触发稳定性检查
    fallback_action: Optional[str]   # 回退动作（如果触发）
    say_score: float                # 选中动作的 Say 分数
    can_score: float                # 选中动作的 Can 分数
    total_score: float               # 选中动作的 Total 分数
```

## 五、配置参数说明

### SayCan 配置

```yaml
evaluation:
  planner:
    plan_config:
      saycan:
        enabled: true                # 是否启用 SayCan
        num_candidates: 5            # LLM 候选数量
        stability_threshold: 0.2     # 稳定性阈值
        fallback_action: "Explore"   # 低分时的回退动作
        affordance:                  # Affordance 配置
          max_nav_distance: 20.0
          pick_distance_thresh: 2.0
          visibility_check_enabled: true
        analyzer:                    # 分析器配置
          enabled: true
```

### 参数说明

- **num_candidates**: LLM 生成的候选动作数量（默认 5）
- **stability_threshold**: 最低可接受的总分（默认 0.2）。如果最佳候选的总分低于此值，触发回退动作
- **fallback_action**: 稳定性违反时的回退动作（"Explore" 或 "Wait"）
- **max_nav_distance**: Navigate 动作的最大距离（超过此距离，分数接近 0）
- **pick_distance_thresh**: Pick 动作的距离阈值（超过此距离，需要先 Navigate）
- **visibility_check_enabled**: 是否启用可见性检查（使用 Raycast）

## 六、输出文件格式

### SayCan 分析 JSON

```json
{
  "episode_id": "123",
  "episode_filename": "episode_123_0",
  "task_instruction": "Bring me the apple",
  "saycan_summary": {
    "mean_say_score": 0.75,
    "mean_can_score": 0.82,
    "mean_total_score": 0.62,
    "stability_violations": 2,
    "stability_violation_rate": 0.1,
    "affordance_by_skill": {
      "navigate": 0.85,
      "pick": 0.78,
      "place": 0.90
    }
  },
  "saycan_analysis": {
    "step_data": [...],
    "summary": {...},
    "candidate_analysis": {...},
    "affordance_analysis": {...}
  }
}
```

## 七、使用示例

### 示例 1: 使用 SayCanPlanner

```python
# 在配置中指定
evaluation:
  planner:
    _target_: habitat_llm.planner.saycan.saycan_planner.SayCanPlanner
    plan_config:
      saycan:
        enabled: true
        num_candidates: 5
        stability_threshold: 0.2
```

### 示例 2: 在 LLMPlanner 中启用分析器

```python
# 仅用于数据收集，不改变动作选择
evaluation:
  planner:
    plan_config:
      saycan:
        enabled: true  # 启用分析器
        analyzer:
          enabled: true
```

## 八、与现有系统的集成点

### 1. EvaluationRunner 集成

**位置**: `evaluation/evaluation_runner.py`

**修改**:
- 在 `run_instruction()` 收尾部分添加 SayCan 分析调用
- 新增 `_log_saycan_analysis()` 方法

### 2. Resilience Metrics 集成

**位置**: `evaluation/resilience_metrics_logging.py`

**修改**:
- 新增 `collect_saycan_metrics()` 函数
- 在 `log_resilience_metrics()` 中集成 SayCan 指标

### 3. LLMPlanner 集成

**位置**: `planner/llm_planner.py`

**修改**:
- 在 `__init__()` 中初始化 SayCan 分析器（可选）
- 在 `reset()` 中重置 SayCan 分析器
- 在 `get_next_action()` 中记录 SayCan 数据到 planner_info

## 九、测试建议

1. **单元测试**:
   - CandidateScorer.score_candidates()
   - StabilityMonitor.check_stability()
   - AffordanceModel.get_affordance_with_details()

2. **集成测试**:
   - SayCanPlanner 完整流程
   - SayCanAnalyzer 数据收集和统计

3. **稳定性测试**:
   - 遮挡场景（物体被部分遮挡）
   - 位置偏移（物体位置微小变化）
   - 感知噪声（WorldGraph 不完整）

## 十、扩展方向

1. **增强 Affordance 模型**:
   - 使用 Critic 价值函数增强 Affordance 预测
   - 学习型 Affordance 模型（替代启发式）

2. **稳定性测试框架**:
   - 自动化稳定性测试场景
   - 稳定性指标的可视化

3. **跨 Episode 学习**:
   - 从历史 Episode 中学习 Affordance 模式
   - 动态调整稳定性阈值
