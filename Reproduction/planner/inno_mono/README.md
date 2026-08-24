# Inner Monologue Implementation

## 概述

Inner Monologue 机制实现了从 Open-Loop 到 Closed-Loop 的跃迁，通过将环境反馈转化为自然语言并注入到 LLM Prompt 中，强制 LLM 在每次反馈后生成推理（Thought/Monologue），从而提升系统的 Rebound 和 Stability 能力。

## 参考

- **论文**: [Inner Monologue: Embodied Reasoning through Planning with Language Models](https://arxiv.org/abs/2207.05608)
- **核心思想**: 将环境反馈（成功/失败、场景描述、自身状态）转化为自然语言，注入到 LLM Prompt 中，强制生成推理过程

## 架构

### 目录结构

```
habitat_llm/
├── evaluation/
│   └── inno_mono/                    # 状态评估与 Critic 集成
│       ├── __init__.py
│       ├── critic_feedback_extractor.py  # 从 Critic 提取反馈（可选）
│       └── state_assessor.py         # 状态评估器（可选）
│
└── planner/
    └── inno_mono/                    # 反馈生成与上下文管理
        ├── __init__.py
        ├── feedback_generator.py      # 核心反馈生成器
        ├── success_detector.py        # 成功/失败检测
        ├── scene_describer.py         # 场景描述生成
        ├── self_state_reporter.py     # 自身状态报告
        └── planner_integration.py     # 与 LLMPlanner 集成
```

### 核心组件

#### 1. FeedbackGenerator (`feedback_generator.py`)

**职责**: 整合多源反馈，生成结构化的 Inner Monologue 反馈

**功能**:
- 整合 Success Detection、Scene Description、Self State 反馈
- 可选整合 Critic 和 Rebound 反馈
- 格式化反馈为自然语言文本

**使用示例**:
```python
feedback_generator = FeedbackGenerator(
    config=inner_mono_config,
    env_interface=env_interface
)

feedback = feedback_generator.generate_feedback(
    agent_responses=responses,
    last_actions=last_actions,
    world_state=world_state
)

feedback_text = feedback_generator.format_feedback_as_text(feedback)
```

#### 2. SuccessDetector (`success_detector.py`)

**职责**: 检测动作执行是否成功

**功能**:
- 解析 agent response 判断成功/失败
- 生成自然语言反馈消息
- 可选验证状态变化

**检测逻辑**:
- 成功模式: "successful execution", "success", "completed" 等
- 失败模式: "fail", "error", "cannot", "unable" 等
- 进行中: "still in progress"

#### 3. SceneDescriber (`scene_describer.py`)

**职责**: 生成场景描述反馈

**功能**:
- 描述对象位置和状态
- 描述 Agent 位置和手持物
- 可选描述家具/房间信息

**复用组件**:
- `PromptContextBuilder.build_world_description()`
- `get_world_descr()` from `habitat_llm.llm.instruct.utils`

#### 4. SelfStateReporter (`self_state_reporter.py`)

**职责**: 报告 Agent 自身状态

**功能**:
- 报告 Agent 位置
- 报告 Agent 手持物
- 可选报告旋转/朝向

#### 5. Planner Integration (`planner_integration.py`)

**职责**: 提供与 LLMPlanner 的集成函数

**关键函数**:
- `generate_and_inject_feedback()`: 生成反馈并注入到 prompt
- `ensure_thought_generation()`: 确保 Thought 生成

## 配置

### 最小配置

```yaml
evaluation:
  planner:
    plan_config:
      inner_monologue:
        enabled: true
        feedback_sources:
          - success_detection
          - scene_description
          - self_state
        force_thought_generation: true
```

### 完整配置

```yaml
evaluation:
  planner:
    plan_config:
      inner_monologue:
        enabled: true
        feedback_sources:
          - success_detection
          - scene_description
          - self_state
        use_critic_feedback: false
        use_rebound_feedback: false
        force_thought_generation: true
        success_detector:
          check_state_changes: false
          verbose_feedback: true
        scene_describer:
          max_length: 200
          include_rooms: true
          include_furniture: true
        self_state_reporter:
          include_position: true
          include_rotation: false
          include_holdings: true
```

## 集成点

### 1. LLMPlanner 初始化

在 `llm_planner.py::__init__()` 中：
- 读取 `inner_monologue.enabled` 配置
- 初始化 `FeedbackGenerator` 及其子组件

### 2. 反馈生成与注入

在 `llm_planner.py::_add_responses_to_prompt()` 中：
- 获取世界状态
- 生成 Inner Monologue 反馈
- 注入到 `PromptBuilder` 或 `curr_prompt`
- 强制生成 Thought

### 3. Prompt 模板

在 `qwen_few_shot_centralized_motoronly.yaml` 中：
- 添加 Inner Monologue 相关规则
- 要求 LLM 在收到 Feedback 后生成 Thought

## 数据流

```
Agent Response
    ↓
_add_responses_to_prompt()
    ↓
FeedbackGenerator.generate_feedback()
    ├─> SuccessDetector.detect_success()
    ├─> SceneDescriber.describe_scene()
    ├─> SelfStateReporter.report_state()
    └─> (可选) CriticFeedbackExtractor.extract_feedback()
    ↓
format_feedback_as_text()
    ↓
注入到 PromptBuilder / curr_prompt
    ↓
强制添加 "Thought:" 提示
    ↓
LLM 生成推理和下一步动作
```

## 反馈格式示例

### 成功场景

```
Agent 0: Success - Agent 0 successfully executed Pick[apple_0]. Successful execution!
Scene: Objects: apple_0 is on table_0; Agents: Agent 0 is at position [1.2, 0.0, 3.5] and holding apple_0
State: Agent 0 is at position [1.2, 0.0, 3.5] and holding apple_0.
```

### 失败场景

```
Agent 0: Failure - Agent 0 failed to execute Pick[apple_0]. Target object or location not found.
Scene: Objects: apple_0 is on table_0; Agents: Agent 0 is at position [1.2, 0.0, 3.5] and hands free
State: Agent 0 is at position [1.2, 0.0, 3.5] and hands free.
```

## 与现有机制的协调

### 与 Context Update

- Inner Monologue Feedback 使用相同的 `PromptBuilder.add_user_turn()` 机制
- 标题使用 "Feedback"，与 "Context Update" 区分
- LLM 被指示将这些都视为"只读上下文"

### 与 Rebound Guidance

- Rebound Guidance 在故障检测后注入，优先级更高
- Inner Monologue Feedback 在每个动作后注入，提供常规反馈
- 两者可以共存

### 与 CoT (Chain of Thought)

- CoT 模式已经要求生成 Thought
- Inner Monologue 强化了这一要求
- 当 `inner_monologue_enabled=True` 时，即使 CoT 未启用，也强制生成 Thought

## 复用组件

- `PerceptionConnector.extract_world_state()` - 世界状态提取
- `PromptContextBuilder` - 场景描述生成
- `PromptBuilder.add_user_turn()` - 反馈注入
- `A2CCritic` - 状态评估（可选）
- `ReboundManager` - 故障检测（可选）

## 测试

### 单元测试

测试各个组件的功能：
- `SuccessDetector.detect_success()` - 成功/失败检测
- `SceneDescriber.describe_scene()` - 场景描述
- `SelfStateReporter.report_state()` - 状态报告
- `FeedbackGenerator.generate_feedback()` - 反馈生成

### 集成测试

测试与 LLMPlanner 的集成：
- 反馈生成和注入流程
- Thought 强制生成
- Prompt 格式正确性

### 端到端测试

测试完整流程：
- 反馈闭环机制
- LLM 基于反馈调整计划
- 任务完成率提升

## 注意事项

1. **World State 提取**: 如果 `PerceptionConnector.extract_world_state()` 返回空字典，会使用 `_build_minimal_world_state()` 作为回退
2. **错误处理**: 所有反馈生成步骤都有异常处理，不会中断主规划流程
3. **性能**: 反馈生成是同步的，但不会显著影响性能（主要是字符串操作）
4. **Prompt 长度**: 注意控制反馈文本长度，避免 prompt 过长

## 未来改进

1. **Critic 深度集成**: 从 Critic 提取更详细的评估信息
2. **Rebound 深度集成**: 将 Rebound 故障信息整合到反馈中
3. **反馈历史管理**: 压缩和总结历史反馈
4. **性能优化**: 异步反馈生成（如果需要）
