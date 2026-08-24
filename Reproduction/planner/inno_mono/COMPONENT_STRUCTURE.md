# Inner Monologue 组件结构说明

## 一、组件层次结构

```
Inner Monologue System
│
├── Core Feedback Generation (planner/inno_mono/)
│   │
│   ├── FeedbackGenerator (核心协调器)
│   │   ├── SuccessDetector (成功/失败检测)
│   │   ├── SceneDescriber (场景描述)
│   │   ├── SelfStateReporter (自身状态)
│   │   ├── CriticFeedbackExtractor (可选，Critic 反馈)
│   │   └── ReboundFeedback (可选，Rebound 反馈)
│   │
│   └── PlannerIntegration (集成接口)
│       ├── generate_and_inject_feedback()
│       └── ensure_thought_generation()
│
└── State Assessment (evaluation/inno_mono/, 可选)
    ├── CriticFeedbackExtractor (Critic 反馈提取)
    └── StateAssessor (状态评估)
```

## 二、各组件详细说明

### 1. SuccessDetector (`success_detector.py`)

**职责**: 检测动作执行是否成功

**输入**:
- `agent_id`: Agent 标识符
- `action`: (action_name, arg1, arg2) 元组
- `response`: Agent 响应字符串
- `world_state`: 世界状态字典（可选，用于状态变化验证）

**输出**:
```python
{
    "success": bool,
    "message": str,  # 自然语言反馈
    "reason": str,   # 简要原因
    "action": str,
    "response": str
}
```

**核心逻辑**:
1. 解析 `response` 字符串，识别成功/失败关键词
2. 成功模式: "successful execution", "success", "completed" 等
3. 失败模式: "fail", "error", "cannot", "unable" 等
4. 生成自然语言反馈消息

**复用逻辑**:
- 参考 `rebound/planner_integration.py::detect_rebound_faults()` 的错误检测
- 复用 `agent.process_high_level_action()` 返回的 response 解析

### 2. SceneDescriber (`scene_describer.py`)

**职责**: 生成场景描述反馈

**输入**:
- `world_state`: 世界状态字典
- `env_interface`: 环境接口

**输出**: 场景描述字符串

**核心逻辑**:
1. **对象描述** (`_describe_objects()`):
   - 从 `world_state["object_positions"]` 提取对象位置
   - 格式: "apple_0 is on table_0"
   - 限制: 最多 5 个对象

2. **Agent 描述** (`_describe_agents()`):
   - 从 `world_state["agent_poses"]` 和 `agent_holdings` 提取
   - 格式: "Agent 0 is at position [1.2, 0.0, 3.5] and holding apple_0"

3. **家具描述** (`_describe_furniture()`, 可选):
   - 从 `world_state["furniture_positions"]` 提取
   - 或使用 `PromptContextBuilder.build_world_description()`

**复用组件**:
- `PromptContextBuilder.build_world_description()`
- `get_world_descr()` from `habitat_llm.llm.instruct.utils`

### 3. SelfStateReporter (`self_state_reporter.py`)

**职责**: 报告 Agent 自身状态

**输入**:
- `agent_id`: Agent 标识符
- `world_state`: 世界状态字典

**输出**: 状态描述字符串

**核心逻辑**:
1. 从 `world_state["agent_poses"][agent_id]` 提取位置
2. 从 `world_state["agent_holdings"][agent_id]` 提取手持物
3. 可选提取旋转/朝向
4. 格式化为自然语言: "Agent 0 is at position [1.2, 0.0, 3.5] and holding apple_0."

**数据源**:
- `world_state_dict` 中的 `agent_poses` 和 `agent_holdings`
- 参考 `PromptContextBuilder.build_agent_status_prompt()`

### 4. FeedbackGenerator (`feedback_generator.py`)

**职责**: 整合多源反馈，生成结构化 Inner Monologue 反馈

**输入**:
- `agent_responses`: Dict[int, str] - Agent 响应
- `last_actions`: Dict[int, Tuple[str, str, str]] - 上次执行的动作
- `world_state`: Dict[str, Any] - 世界状态
- `critic_feedback`: Optional[Dict] - Critic 反馈（可选）
- `rebound_feedback`: Optional[Dict] - Rebound 反馈（可选）

**输出**: 结构化反馈字典

**核心流程**:
```python
feedback = {
    "success_detection": {
        agent_id: SuccessDetector.detect_success(...)
    },
    "scene_description": SceneDescriber.describe_scene(...),
    "self_state": {
        agent_id: SelfStateReporter.report_state(...)
    },
    "critic_feedback": critic_feedback,  # 可选
    "rebound_feedback": rebound_feedback  # 可选
}
```

**格式化方法** (`format_feedback_as_text()`):
- 将结构化反馈转换为自然语言文本
- 格式: "Agent X: Success/Failure - ...\nScene: ...\nState: ..."

### 5. PlannerIntegration (`planner_integration.py`)

**职责**: 提供与 LLMPlanner 的集成接口

**关键函数**:

1. **`generate_and_inject_feedback()`**:
   - 生成反馈
   - 格式化反馈
   - 注入到 `PromptBuilder` 或 `curr_prompt`

2. **`ensure_thought_generation()`**:
   - 确保在反馈后添加 "Thought:" 提示
   - 强制 LLM 生成推理

**参考**:
- `rebound/planner_integration.py::apply_rebound_context_modification()` 的注入方式
- `llm_planner.py::_add_responses_to_prompt()` 的 prompt 管理方式

### 6. CriticFeedbackExtractor (`evaluation/inno_mono/critic_feedback_extractor.py`)

**职责**: 从 A2CCritic 提取反馈信息（可选）

**输入**:
- `critic`: A2CCritic 实例
- `state`: 当前世界状态
- `action`: 执行的动作
- `reward`: 环境奖励

**输出**: Critic 反馈字典

**核心逻辑**:
1. 获取状态价值估计: `critic.evaluate(state, action)`
2. 获取 reward shaping 结果（如果启用）
3. 转换为自然语言反馈

**复用组件**:
- `A2CCritic.evaluate()` - 获取状态价值
- `LLMRewardShaper.shape_reward()` - 获取 shaping 结果

### 7. StateAssessor (`evaluation/inno_mono/state_assessor.py`)

**职责**: 状态评估（可选，封装 StateEncoder）

**功能**:
- 状态编码: `state_encoder.encode(world_state)`
- 状态描述生成: `state_encoder._generate_state_description(world_state)`
- 状态比较: `compare_states(state1, state2)`

**复用组件**:
- `StateEncoder.encode()` - 状态编码
- `StateEncoder._generate_state_description()` - 状态描述生成

## 三、数据流详解

### 3.1 反馈生成流程

```
[Agent 执行动作]
    ↓
[process_high_level_action() 返回 response]
    ↓
[_add_responses_to_prompt() 被调用]
    ↓
[收集所有 Agent 的 responses]
    ↓
[Inner Monologue 反馈生成]
    ├─> 获取 world_state (PerceptionConnector.extract_world_state())
    ├─> SuccessDetector.detect_success() 对每个 Agent
    ├─> SceneDescriber.describe_scene()
    ├─> SelfStateReporter.report_state() 对每个 Agent
    └─> (可选) CriticFeedbackExtractor.extract_feedback()
    ↓
[FeedbackGenerator.generate_feedback() 整合]
    ↓
[FeedbackGenerator.format_feedback_as_text() 格式化]
    ↓
[注入到 PromptBuilder / curr_prompt]
    ↓
[强制添加 "Thought:" 提示]
    ↓
[LLM 生成推理和下一步动作]
```

### 3.2 反馈格式示例

**成功场景**:
```
Agent 0: Success - Agent 0 successfully executed Pick[apple_0]. Successful execution!
Scene: Objects: apple_0 is on table_0; Agents: Agent 0 is at position [1.2, 0.0, 3.5] and holding apple_0
State: Agent 0 is at position [1.2, 0.0, 3.5] and holding apple_0.
```

**失败场景**:
```
Agent 0: Failure - Agent 0 failed to execute Pick[apple_0]. Target object or location not found.
Scene: Objects: apple_0 is on table_0; Agents: Agent 0 is at position [1.2, 0.0, 3.5] and hands free
State: Agent 0 is at position [1.2, 0.0, 3.5] and hands free.
```

**多 Agent 场景**:
```
Agent 0: Success - Agent 0 successfully executed Pick[apple_0]. Successful execution!
Agent 1: Failure - Agent 1 failed to execute Navigate[table_0]. Path is blocked by obstacle.
Scene: Objects: apple_0 is on table_0; Agents: Agent 0 is at position [1.2, 0.0, 3.5] and holding apple_0; Agent 1 is at position [2.0, 0.0, 4.0] and hands free
State: Agent 0 is at position [1.2, 0.0, 3.5] and holding apple_0.; Agent 1 is at position [2.0, 0.0, 4.0] and hands free.
```

## 四、配置参数说明

### SuccessDetector 配置

```yaml
success_detector:
  check_state_changes: false  # 是否验证状态变化
  verbose_feedback: true      # 是否生成详细反馈
```

### SceneDescriber 配置

```yaml
scene_describer:
  max_length: 200            # 最大描述长度
  include_rooms: true        # 是否包含房间信息
  include_furniture: true    # 是否包含家具信息
  focus_on_changes: false    # 是否关注变化
```

### SelfStateReporter 配置

```yaml
self_state_reporter:
  include_position: true     # 是否包含位置
  include_rotation: false   # 是否包含旋转/朝向
  include_holdings: true    # 是否包含手持物
  position_precision: 1     # 位置精度（小数位数）
```

### CriticFeedbackExtractor 配置

```yaml
critic_feedback_extractor:
  include_value_estimate: true    # 是否包含价值估计
  include_reward_shaping: false   # 是否包含 reward shaping
  value_threshold_low: 0.3        # 低价值阈值
  value_threshold_high: 0.7       # 高价值阈值
```

## 五、错误处理

所有组件都实现了完善的错误处理：

1. **FeedbackGenerator**:
   - 如果某个反馈源失败，其他源仍能正常工作
   - 返回部分反馈而不是完全失败

2. **SuccessDetector**:
   - 如果 response 解析失败，默认返回失败状态
   - 状态变化验证失败不影响主流程

3. **SceneDescriber**:
   - 如果 world_state 不完整，返回部分描述
   - 如果 env_interface 访问失败，跳过相关描述

4. **SelfStateReporter**:
   - 如果 world_state 缺少某些字段，返回可用信息
   - 位置提取失败时返回 "unknown"

5. **LLMPlanner 集成**:
   - 如果 Inner Monologue 初始化失败，自动禁用
   - 如果反馈生成失败，不影响主规划流程
   - 所有异常都被捕获并记录，不会中断执行

## 六、性能考虑

1. **同步执行**: 反馈生成是同步的，但主要是字符串操作，开销很小
2. **World State 提取**: 使用 `PerceptionConnector.extract_world_state()`，如果失败使用轻量级回退
3. **Prompt 长度**: 通过 `max_length` 配置控制反馈文本长度
4. **可选组件**: Critic 和 Rebound 反馈是可选的，默认禁用

## 七、扩展点

### 1. 添加新的反馈源

在 `FeedbackGenerator.generate_feedback()` 中添加：
```python
# 新反馈源
if "new_feedback_source" in feedback_sources:
    new_feedback = new_feedback_generator.generate(...)
    feedback["new_feedback"] = new_feedback
```

### 2. 自定义反馈格式

重写 `FeedbackGenerator.format_feedback_as_text()`:
```python
def format_feedback_as_text(self, feedback: Dict[str, Any]) -> str:
    # 自定义格式化逻辑
    ...
```

### 3. 集成新的评估组件

在 `evaluation/inno_mono/` 中添加新组件，然后在 `FeedbackGenerator` 中集成。

## 八、测试建议

### 单元测试

1. **SuccessDetector**:
   - 测试成功/失败检测逻辑
   - 测试各种 response 格式

2. **SceneDescriber**:
   - 测试场景描述生成
   - 测试空/不完整 world_state 处理

3. **SelfStateReporter**:
   - 测试状态报告生成
   - 测试多 Agent 场景

4. **FeedbackGenerator**:
   - 测试反馈整合
   - 测试格式化

### 集成测试

1. **与 LLMPlanner 集成**:
   - 测试反馈生成和注入
   - 测试 Thought 强制生成

2. **与 PromptBuilder 集成**:
   - 测试反馈正确注入
   - 测试格式正确性

### 端到端测试

1. **完整流程**:
   - 运行一个完整 episode
   - 验证反馈生成和注入
   - 验证 LLM 基于反馈调整计划

2. **错误场景**:
   - 测试 world_state 为空的情况
   - 测试组件初始化失败的情况
