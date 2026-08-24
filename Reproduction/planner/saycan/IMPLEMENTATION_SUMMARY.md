# SayCan 实现总结

## 实现完成情况

所有 12 个任务已完成：

✅ **Phase 1: 核心组件**
- Task 1: 增强 AffordanceModel
- Task 2: 创建 CandidateScorer
- Task 3: 创建 StabilityMonitor
- Task 4: 创建 SayCanPlanner

✅ **Phase 2: 集成与分析**
- Task 5: 创建 Planner Integration
- Task 6: 创建 SayCanAnalyzer
- Task 7: 创建 StateAssessor
- Task 9: 扩展日志记录

✅ **Phase 3: 高级功能**
- Task 8: 创建 CriticIntegration
- Task 10: 集成到 EvaluationRunner
- Task 11: 集成到 Resilience Metrics
- Task 12: 集成到 LLMPlanner

## 文件结构

### evaluation/saycan/ (状态评估、Critic 集成、分析)

```
evaluation/saycan/
├── __init__.py                    # 模块导出
├── affordance_model.py            # 增强的 Affordance 模型
├── saycan_analyzer.py             # SayCan 分析器
├── state_assessor.py              # 状态评估器
├── critic_integration.py          # Critic 集成
└── README.md                      # 模块文档
```

### planner/saycan/ (Planner 集成、候选评分、稳定性监控)

```
planner/saycan/
├── __init__.py                    # 模块导出
├── saycan_planner.py              # SayCan Planner（主类）
├── candidate_scorer.py           # 候选动作评分器
├── stability_monitor.py           # 稳定性监控器
├── planner_integration.py         # Planner 集成函数
├── prompts.py                     # SayCan 专用 prompts
├── README.md                      # 模块文档
└── IMPLEMENTATION_SUMMARY.md      # 实现总结（本文件）
```

## 核心组件说明

### 1. AffordanceModel

**位置**: `evaluation/saycan/affordance_model.py`

**功能**:
- 计算 P(success | state, skill)
- 返回详细计算信息（距离、可见性、可达性、失败原因）
- 支持 Navigate, Pick, Place, Open/Close, Wait/Done

**关键方法**:
- `get_affordance()`: 向后兼容，仅返回分数
- `get_affordance_with_details()`: 返回 (score, details) 元组

**增强特性**:
- 可见性检查（使用 Raycast）
- 路径距离计算（使用 NavMesh）
- 详细失败原因记录

### 2. CandidateScorer

**位置**: `planner/saycan/candidate_scorer.py`

**功能**:
- 实现 Say × Can 融合逻辑
- 对候选动作进行评分和排序

**核心方法**:
- `score_candidates()`: 输入 LLM 候选，输出评分后的候选列表

**数据结构**:
- `ScoredCandidate`: 包含 action, say_score, can_score, total_score, affordance_details, rank

### 3. StabilityMonitor

**位置**: `planner/saycan/stability_monitor.py`

**功能**:
- 监控 SayCan 稳定性
- 触发低分动作过滤
- 跟踪分数方差

**核心方法**:
- `check_stability()`: 检查最佳候选是否满足阈值
- `track_score_variance()`: 计算分数方差

### 4. SayCanPlanner

**位置**: `planner/saycan/saycan_planner.py`

**功能**:
- 继承 LLMPlanner
- 实现完整的 SayCan 流程
- 集成所有组件（CandidateScorer, StabilityMonitor, SayCanAnalyzer）

**核心流程**:
1. 获取 LLM 候选（Say）
2. 使用 CandidateScorer 评分（Can）
3. 使用 StabilityMonitor 检查稳定性
4. 记录数据到 SayCanAnalyzer
5. 返回格式化的动作响应

### 5. SayCanAnalyzer

**位置**: `evaluation/saycan/saycan_analyzer.py`

**功能**:
- 收集每个 planning step 的 SayCan 数据
- 计算 Episode 级统计

**分析内容**:
- 分数分布（Say/Can/Total 的均值、方差）
- 候选动作排名变化
- 稳定性指标
- Affordance 分数分布（按技能类型）

### 6. StateAssessor

**位置**: `evaluation/saycan/state_assessor.py`

**功能**:
- 评估世界状态
- 提取 agent 位置、对象位置、可见性信息

### 7. CriticIntegration

**位置**: `evaluation/saycan/critic_integration.py`

**功能**:
- 对比 Affordance 与 Critic 价值函数
- 计算相关性
- 记录一致性指标

## 数据流

### Planning Step 数据流

```
SayCanPlanner.generate_action_response()
  ↓
_get_say_candidates() → List[Dict]  # LLM 候选
  ↓
CandidateScorer.score_candidates()
  ├─ AffordanceModel.get_affordance_with_details() → (score, details)
  └─ 计算 total = say × can
  ↓
StabilityMonitor.check_stability()
  └─ 如果 total < threshold → 触发 fallback
  ↓
SayCanAnalyzer.record_step() → 存储到内存
  ↓
返回格式化的动作响应
```

### Episode 收尾数据流

```
EvaluationRunner.run_instruction() [Episode 结束]
  ↓
SayCanAnalyzer.analyze_episode() → Dict[str, Any]
  ├─ 计算分数分布统计
  ├─ 候选动作排名变化分析
  └─ Affordance 分数分布分析
  ↓
log_saycan_analysis() → 保存 JSON
  ↓
collect_saycan_metrics() → 提取稳定性指标
  ↓
log_resilience_metrics() → 集成到综合韧性指标
```

## 配置示例

### 完整启用 SayCan

```yaml
evaluation:
  planner:
    _target_: habitat_llm.planner.saycan.saycan_planner.SayCanPlanner
    plan_config:
      saycan:
        enabled: true
        num_candidates: 5
        stability_threshold: 0.2
        fallback_action: "Explore"
        affordance:
          max_nav_distance: 20.0
          pick_distance_thresh: 2.0
          visibility_check_enabled: true
        analyzer:
          enabled: true
```

### 仅在 LLMPlanner 中启用分析器

```yaml
evaluation:
  planner:
    plan_config:
      saycan:
        enabled: true  # 仅启用分析器，不改变动作选择
        analyzer:
          enabled: true
```

## 输出文件

### SayCan 分析结果

```
output_dir/analyses/saycan/saycan_analysis-episode_{id}_{filename}.json
```

**内容**:
- `saycan_summary`: 摘要指标
- `saycan_analysis`: 详细分析数据

### 韧性指标

```
output_dir/analyses/resilience/resilience_metrics-episode_{id}_{filename}.json
```

**包含 SayCan 指标**:
- `saycan_stability_score`
- `saycan_filter_effectiveness`
- `saycan_mean_total_score`
- `saycan_stability_violations`

## 与现有系统的协同

### SayCan + Rebound

- **SayCan**: 动作执行前过滤不可行动作（Stability，预防失败）
- **Rebound**: 动作失败后恢复（Resilience，恢复失败）

### SayCan + Critic

- **Affordance**: 启发式可行性评估
- **Critic**: 学习型价值函数

可以通过 `CriticIntegration` 对比一致性。

### SayCan + ADCA

- **SayCan**: 步骤级动作选择
- **ADCA**: 步骤级动作质量评估

可以分析 SayCan 选择的动作在 ADCA 评估中的表现。

## 使用建议

1. **完整 SayCan 模式**: 使用 `SayCanPlanner`，获得完整的 Say × Can 融合和稳定性检查
2. **分析模式**: 在 `LLMPlanner` 中启用 `saycan.analyzer.enabled`，仅收集数据用于分析
3. **稳定性测试**: 通过调整 `stability_threshold` 和观察 `stability_violations` 来测试系统稳定性

## 下一步

1. 运行测试验证功能
2. 调整配置参数优化性能
3. 分析 SayCan 数据以改进 Affordance 模型
4. 探索使用 Critic 增强 Affordance 的可能性
