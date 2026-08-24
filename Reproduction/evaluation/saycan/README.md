# SayCan Evaluation Components

## 概述

SayCan 评估组件提供状态评估、Affordance 计算、Critic 集成和分析功能。

## 核心组件

### 1. AffordanceModel (`affordance_model.py`)

计算 P(success | state, skill)，即动作在当前物理状态下的可行性。

**核心方法：**
- `get_affordance()`: 返回分数（向后兼容）
- `get_affordance_with_details()`: 返回分数和详细计算信息

**支持的技能类型：**
- Navigate: 基于距离和路径可达性
- Pick: 基于距离、可见性和手部状态
- Place: 基于距离、手部持有状态
- Open/Close: 类似 Pick
- Wait/Done: 总是可行（返回 1.0）

**详细记录：**
- `distance`: 到目标的距离
- `visibility`: 是否可见（使用 Raycast）
- `reachability`: 是否可达（NavMesh 路径）
- `failure_reason`: 低分原因（如果适用）

### 2. SayCanAnalyzer (`saycan_analyzer.py`)

收集和分析 SayCan 数据。

**核心方法：**
- `record_step()`: 记录每个 planning step 的数据
- `analyze_episode()`: 计算 Episode 级统计

**分析内容：**
- 分数分布（Say/Can/Total 的均值、方差、分位数）
- 候选动作排名变化（LLM 推荐 vs 最终选择）
- 稳定性指标（阈值违反次数、分数方差）
- Affordance 分数分布（按技能类型）

### 3. StateAssessor (`state_assessor.py`)

评估世界状态，为 Affordance 计算提供上下文。

**核心方法：**
- `assess_world_state()`: 提取 agent 位置、对象位置等信息
- `check_visibility()`: 使用 Raycast 检查可见性

### 4. CriticIntegration (`critic_integration.py`)

集成 A2C Critic 与 SayCan，对比 Affordance 与价值函数。

**核心方法：**
- `compare_affordance_vs_critic()`: 比较 Affordance 分数与 Critic 价值
- `compute_correlation()`: 计算相关性

## 与现有系统的集成

### 与 Rebound 的协同

- **SayCan (Pre-act)**: 过滤不可行动作，预防失败（Stability）
- **Rebound (Post-act)**: 失败后恢复（Resilience）

### 与 Critic 的协同

- **Affordance**: 启发式可行性评估（快速但可能不准确）
- **Critic**: 学习型价值函数（准确但需要训练）

可以通过 `CriticIntegration` 对比两者的一致性，并探索使用 Critic 增强 Affordance 的可能性。

### 与 ADCA 的协同

- **SayCan**: 在步骤级选择动作
- **ADCA**: 在步骤级评估动作质量

可以分析 SayCan 选择的动作在 ADCA 评估中的表现。

## 配置示例

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
          visibility_check_enabled: true
        analyzer:
          enabled: true
```
