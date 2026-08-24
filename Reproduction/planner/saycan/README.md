# SayCan Planner Module

## 概述

SayCan Planner 实现了 "Do As I Can, Not As I Say" 的核心机制，通过融合 LLM 置信度（Say）和 Affordance 可行性（Can）来选择动作。

## 核心组件

### 1. SayCanPlanner (`saycan_planner.py`)

主 Planner 类，继承自 `LLMPlanner`，实现完整的 SayCan 流程。

**核心流程：**
1. **Say**: 从 LLM 获取 Top-K 候选动作（带置信度）
2. **Can**: 使用 AffordanceModel 计算每个候选的可行性
3. **Fuse**: 融合分数 = Say × Can
4. **Select**: 选择总分最高的动作
5. **Stability Check**: 如果最高分 < threshold，触发回退动作

**配置示例：**
```yaml
evaluation:
  planner:
    plan_config:
      saycan:
        enabled: true
        num_candidates: 5
        stability_threshold: 0.2
        fallback_action: "Explore"
        affordance:
          max_nav_distance: 20.0
          pick_distance_thresh: 2.0
```

### 2. CandidateScorer (`candidate_scorer.py`)

实现 Say × Can 融合逻辑。

**核心方法：**
- `score_candidates()`: 对候选动作进行评分和排序

**数据结构：**
```python
@dataclass
class ScoredCandidate:
    action: str
    say_score: float
    can_score: float
    total_score: float
    affordance_details: Dict[str, Any]
    rank: int
```

### 3. StabilityMonitor (`stability_monitor.py`)

监控 SayCan 稳定性，过滤低分动作。

**核心方法：**
- `check_stability()`: 检查最佳候选是否满足稳定性阈值
- `track_score_variance()`: 跟踪分数方差（稳定性指标）

### 4. Planner Integration (`planner_integration.py`)

提供 SayCan 与 LLMPlanner 的集成函数。

**核心函数：**
- `apply_saycan_selection()`: 应用 SayCan 选择逻辑
- `inject_saycan_context()`: 注入 SayCan 指导到 prompt

## 使用方式

### 方式 1: 使用 SayCanPlanner（推荐）

在配置中指定使用 SayCanPlanner：

```yaml
evaluation:
  planner:
    _target_: habitat_llm.planner.saycan.saycan_planner.SayCanPlanner
    plan_config:
      saycan:
        enabled: true
        ...
```

### 方式 2: 在 LLMPlanner 中启用 SayCan 分析器

仅用于数据收集和分析，不改变动作选择逻辑：

```yaml
evaluation:
  planner:
    plan_config:
      saycan:
        enabled: true  # 仅启用分析器
        analyzer:
          enabled: true
```

## 数据流

```
LLM 生成候选 → CandidateScorer 评分 → StabilityMonitor 检查 → 选择动作
  ↓
SayCanAnalyzer.record_step() → 存储到内存
  ↓
Episode 收尾:
  SayCanAnalyzer.analyze_episode() → 计算统计
  ↓
  log_saycan_analysis() → 保存 JSON
```

## 输出文件

SayCan 分析结果保存在：
```
output_dir/analyses/saycan/saycan_analysis-episode_{id}_{filename}.json
```

包含：
- `saycan_summary`: 摘要指标（均值、方差、稳定性违反次数）
- `saycan_analysis`: 详细分析（步骤数据、候选分析、Affordance 分布）
