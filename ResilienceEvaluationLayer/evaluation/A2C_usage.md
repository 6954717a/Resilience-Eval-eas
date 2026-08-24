 A2C Critic 功能使用说明手册

  一、环境准备

  1.1 依赖检查

  A2C Critic 功能需要以下依赖：

  # 必需依赖
  - PyTorch >= 2.0.0 (通过conda安装)
  - sentence-transformers >= 2.2.0 (已在requirements.txt中)
  - scipy >= 1.10.0 (已在requirements.txt中)
  - transformers (已在requirements.txt中)

  # 可选依赖 (用于LLM功能)
  - OpenAI API密钥 (用于LLM奖励塑形和离线分析)

  1.2 安装依赖

  cd /mnt/mydata/Proj/partnr-planner

  # 安装Python依赖
  pip install -r requirements.txt

  # 验证PyTorch安装
  python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"

  # 验证sentence-transformers安装
  python -c "import sentence_transformers; print('sentence-transformers已安装')"

  二、基本使用方法

  2.1 启用A2C Critic（基础模式，无LLM）

  cd /mnt/mydata/Proj/partnr-planner

  python -m habitat_llm.examples.planner_demo \
      --config-name evaluation/centralized_evaluation_runner_with_critic.yaml \
      habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
      evaluation.critic.enabled=true \
      evaluation.critic.use_llm_shaping=false \
      evaluation.critic.use_llm_offline=false \
      evaluation.planner.plan_config.llm.inference_mode=hf

  说明：
  - 使用专门的配置文件 centralized_evaluation_runner_with_critic.yaml
  - critic.enabled=true：启用A2C Critic
  - use_llm_shaping=false：禁用LLM奖励塑形（节省API成本）
  - use_llm_offline=false：禁用LLM离线分析

  2.2 启用A2C Critic（完整模式，含LLM功能）

  # 首先设置OpenAI API密钥
  export OPENAI_API_KEY="your-openai-api-key-here"

  python -m habitat_llm.examples.planner_demo \
      --config-name evaluation/centralized_evaluation_runner_with_critic.yaml \
      habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
      evaluation.critic.enabled=true \
      evaluation.critic.use_llm_shaping=true \
      evaluation.critic.use_llm_offline=true \
      evaluation.critic.llm_model="gpt-3.5-turbo" \
      evaluation.planner.plan_config.llm.inference_mode=hf

  说明：
  - use_llm_shaping=true：启用LLM奖励塑形
  - use_llm_offline=true：启用离线轨迹分析
  - llm_model="gpt-3.5-turbo"：使用GPT-3.5（也可用"gpt-4"）

  2.3 禁用A2C Critic（使用原有基线）

  python -m habitat_llm.examples.planner_demo \
      --config-name baselines/centralized_zero_shot_react_summary.yaml \
      habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
      evaluation.planner.plan_config.llm.inference_mode=hf

  说明：
  - 使用原有配置文件，不包含critic配置
  - 系统会自动跳过A2C功能，保持向后兼容

  三、配置参数详解

  3.1 核心RL参数

  critic:
    enabled: true                    # 启用/禁用开关
    gamma: 0.80                      # 折扣因子 (0-1之间)
    gae_lambda: 0.65                 # GAE lambda参数 (平衡偏差-方差)
    n_step: 5                        # n-step回报 (当前未使用GAE)

  参数说明：
  - gamma：未来奖励的折扣率，越接近1越重视长期奖励
  - gae_lambda：GAE参数，控制优势估计的偏差-方差权衡
    - 0.0 = 低方差高偏差
    - 1.0 = 高方差低偏差
    - 推荐值：0.95

  3.2 神经网络架构

  critic:
    text_dim: 384                    # SentenceTransformer输出维度
    numerical_dim: 32                # 数值特征维度
    encoder_hidden_dim: 256          # StateEncoder MLP隐藏层维度
    state_dim: 128                   # 最终编码状态维度
    value_hidden_dims: [256, 128, 64]  # ValueNetwork隐藏层维度列表
    dropout_rate: 0.1                # Dropout正则化率

  架构流程：
  世界状态 → SentenceTransformer(384维) →
          → 数值特征(32维) →
          → StateEncoder(256维隐藏层) → 状态向量(128维) →
          → ValueNetwork([256,128,64]隐藏层) → V(s)值(1维)

  3.3 优化参数

  critic:
    value_lr: 0.001                  # Adam优化器学习率

  3.4 LLM集成参数

  critic:
    use_llm_shaping: true            # 启用LLM奖励塑形
    use_llm_offline: true            # 启用LLM离线轨迹分析
    shaping_weight: 0.3              # LLM奖励加成权重
    llm_call_frequency: 5            # 每N步调用一次LLM (成本控制)
    llm_model: "gpt-3.5-turbo"       # LLM模型选择
    cache_size: 1000                 # LLM结果缓存大小

  成本控制建议：
  - llm_call_frequency: 5：每5步调用一次LLM（降低API成本）
  - llm_model: "gpt-3.5-turbo"：使用GPT-3.5而非GPT-4（更便宜）
  - cache_size: 1000：缓存LLM结果避免重复调用

  3.5 离线分析参数

  critic:
    save_analyses: true              # 保存轨迹分析到JSON
    analysis_save_dir: "./analyses"  # 分析报告保存目录
    analyze_frequency: 1             # 每N个episode分析一次

  3.6 检查点保存

  critic:
    save_checkpoints: true           # 启用检查点保存
    checkpoint_frequency: 10         # 每N个episode保存一次
    checkpoint_dir: "./checkpoints"  # 检查点保存目录

  3.7 设备配置

  critic:
    device: "cpu"                    # "cpu" 或 "cuda"
                                     # 如果cuda不可用会自动降级到cpu

  3.8 日志配置

  critic:
    log_statistics: true             # 在episode结束时记录训练统计

  四、使用示例

  4.1 示例1：快速测试（无LLM，单个episode）

  python -m habitat_llm.examples.planner_demo \
      --config-name evaluation/centralized_evaluation_runner_with_critic.yaml \
      habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
      evaluation.critic.enabled=true \
      evaluation.critic.use_llm_shaping=false \
      evaluation.critic.use_llm_offline=false \
      evaluation.critic.device="cpu" \
      evaluation.planner.plan_config.llm.inference_mode=hf

  预期输出：
  INFO: A2C Critic initialized successfully
  INFO:   - Gamma: 0.99
  INFO:   - GAE Lambda: 0.95
  INFO:   - LLM Shaping: False
  INFO:   - LLM Offline: False

  ... [运行过程] ...

  ============================================================
  A2C CRITIC STATISTICS
  ============================================================
  Value Loss:          0.0234
  Mean Predicted Value: 0.4567
  Mean Return:         0.4321
  Episodes Trained:    1
  ============================================================

  4.2 示例2：完整训练（含LLM，多个episodes）

  export OPENAI_API_KEY="sk-xxx..."

  python -m habitat_llm.examples.planner_demo \
      --config-name evaluation/centralized_evaluation_runner_with_critic.yaml \
      habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
      evaluation.critic.enabled=true \
      evaluation.critic.use_llm_shaping=true \
      evaluation.critic.use_llm_offline=true \
      evaluation.critic.llm_call_frequency=10 \
      evaluation.critic.save_checkpoints=true \
      evaluation.critic.checkpoint_frequency=5 \
      evaluation.planner.plan_config.llm.inference_mode=hf

  生成的文件：
  ./checkpoints/critic/
    ├── checkpoint_episode_5.pt
    ├── checkpoint_episode_10.pt
    └── ...

  ./analyses/
    ├── episode_0_analysis.json
    ├── episode_1_analysis.json
    └── ...

  4.3 示例3：GPU加速训练

  python -m habitat_llm.examples.planner_demo \
      --config-name evaluation/centralized_evaluation_runner_with_critic.yaml \
      habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
      evaluation.critic.enabled=true \
      evaluation.critic.device="cuda" \
      evaluation.planner.plan_config.llm.inference_mode=hf

  4.4 示例4：自定义网络架构

  python -m habitat_llm.examples.planner_demo \
      --config-name evaluation/centralized_evaluation_runner_with_critic.yaml \
      habitat.dataset.data_path="data/datasets/partnr_episodes/v0_0/val_mini.json.gz" \
      evaluation.critic.enabled=true \
      evaluation.critic.state_dim=256 \
      evaluation.critic.value_hidden_dims="[512,256,128]" \
      evaluation.critic.dropout_rate=0.2 \
      evaluation.critic.value_lr=0.0005 \
      evaluation.planner.plan_config.llm.inference_mode=hf

  五、输出文件说明

  5.1 检查点文件 (Checkpoints)

  位置： ./checkpoints/critic/checkpoint_episode_N.pt

  内容：
  {
      'state_encoder_state_dict': {...},  # StateEncoder权重
      'value_network_state_dict': {...},  # ValueNetwork权重
      'optimizer_state_dict': {...},      # 优化器状态
      'episode': N,                       # Episode编号
      'statistics': {...}                 # 训练统计
  }

  加载检查点：
  from habitat_llm.evaluation.critic import A2CCritic
  import torch

  # 加载检查点
  checkpoint = torch.load('./checkpoints/critic/checkpoint_episode_10.pt')
  critic.state_encoder.load_state_dict(checkpoint['state_encoder_state_dict'])
  critic.value_network.load_state_dict(checkpoint['value_network_state_dict'])

  5.2 LLM分析报告

  位置： ./analyses/episode_N_analysis.json

  内容示例：
  {
      "episode_id": 0,
      "task_instruction": "Put the apple on the table",
      "total_steps": 45,
      "task_success": true,
      "trajectory_analysis": {
          "efficiency_score": 0.85,
          "critical_decisions": [...],
          "improvement_suggestions": [...]
      },
      "reward_shaping_stats": {
          "total_llm_calls": 9,
          "cache_hits": 3,
          "avg_bonus_reward": 0.15
      }
  }

  5.3 训练日志

  在标准输出中查看：
  ============================================================
  A2C CRITIC STATISTICS
  ============================================================
  Value Loss:          0.0234
  Mean Predicted Value: 0.4567
  Mean Return:         0.4321
  Episodes Trained:    10
  ============================================================

  六、常见问题 (FAQ)

  Q1: A2C Critic 和原有评估系统的关系？

  答： A2C Critic 是可选的附加功能，不会影响原有评估系统：
  - 不启用时：系统完全按原来方式运行
  - 启用时：在原有评估基础上增加价值函数学习和LLM分析
  - 向后兼容：所有原有配置文件仍然可用

  Q2: 为什么只支持CentralizedEvaluationRunner？

  答： 设计选择：
  - 中心化评估有全局视角，更适合价值函数学习
  - 去中心化评估需要为每个agent独立训练critic（未来可扩展）

  Q3: LLM功能会产生多少API成本？

  答： 可通过参数控制：
  llm_call_frequency: 10  # 每10步调用1次 → 降低10倍成本
  llm_model: "gpt-3.5-turbo"  # 使用GPT-3.5 → 比GPT-4便宜15倍
  cache_size: 1000  # 缓存结果 → 避免重复调用

  估算： 对于50步的episode：
  - 无缓存，每步调用：50次 × $0.002 = $0.10
  - frequency=10，有缓存：5次 × 0.5 (缓存命中) × $0.002 = $0.005

  Q4: 如何检查A2C Critic是否正常工作？

  答： 查看启动日志：
  ✅ 正常启动：
  INFO: A2C Critic initialized successfully
  INFO:   - Gamma: 0.99
  INFO:   - GAE Lambda: 0.95

  ❌ 启动失败：
  ERROR: Failed to import A2CCritic: No module named 'torch'
  WARNING: A2C Critic unavailable. Install: torch, transformers, sentence-transformers

  Q5: 训练的价值函数可以用于其他任务吗？

  答： 可以，通过加载检查点：
  # 在新任务中加载已训练的critic
  critic = A2CCritic(config=new_config, env_interface=new_env)
  checkpoint = torch.load('checkpoint_episode_100.pt')
  critic.state_encoder.load_state_dict(checkpoint['state_encoder_state_dict'])
  critic.value_network.load_state_dict(checkpoint['value_network_state_dict'])

  Q6: 如何调整学习率和网络大小？

  答： 通过命令行覆盖：
  evaluation.critic.value_lr=0.0001 \
  evaluation.critic.state_dim=256 \
  evaluation.critic.value_hidden_dims="[512,256,128]"

  Q7: GPU内存不足怎么办？

  答： 降低网络大小或使用CPU：
  # 方案1：使用CPU
  evaluation.critic.device="cpu"

  # 方案2：减小网络
  evaluation.critic.state_dim=64 \
  evaluation.critic.value_hidden_dims="[128,64]"

  七、故障排除

  问题1: ModuleNotFoundError: No module named 'torch'

  原因： PyTorch未安装

  解决：
  # 通过conda安装
  conda install pytorch torchvision torchaudio -c pytorch

  # 或通过pip安装
  pip install torch>=2.0.0

  问题2: ImportError: sentence-transformers

  原因： sentence-transformers未安装

  解决：
  pip install sentence-transformers>=2.2.0

  问题3: CUDA out of memory

  原因： GPU内存不足

  解决：
  # 切换到CPU
  evaluation.critic.device="cpu"

  # 或减小批次大小/网络大小
  evaluation.critic.state_dim=64

  问题4: OpenAI API密钥错误

  症状： openai.error.AuthenticationError

  解决：
  # 检查API密钥是否设置
  echo $OPENAI_API_KEY

  # 重新设置
  export OPENAI_API_KEY="sk-your-key-here"

  # 或禁用LLM功能
  evaluation.critic.use_llm_shaping=false \
  evaluation.critic.use_llm_offline=false

  问题5: 检查点加载失败

  症状： RuntimeError: Error loading state dict

  原因： 网络架构不匹配

  解决：
  # 确保配置与保存时一致
  # 检查checkpoint中的配置
  checkpoint = torch.load('checkpoint.pt')
  print(checkpoint.keys())  # 查看保存了哪些信息

  问题6: 训练统计全是0

  原因： critic未被正确调用

  检查：
  # 确认critic_enabled为True
  print(evaluation_runner.critic_enabled)  # 应该是True

  # 确认critic不为None
  print(evaluation_runner.critic)  # 应该是A2CCritic对象

  八、性能优化建议

  8.1 训练速度优化

  # 1. 使用GPU
  evaluation.critic.device="cuda"

  # 2. 减小LLM调用频率
  evaluation.critic.llm_call_frequency=20

  # 3. 减小检查点保存频率
  evaluation.critic.checkpoint_frequency=50

  8.2 内存优化

  # 1. 减小网络大小
  evaluation.critic.state_dim=64 \
  evaluation.critic.value_hidden_dims="[128,64]"

  # 2. 减小缓存大小
  evaluation.critic.cache_size=500

  # 3. 禁用不需要的功能
  evaluation.critic.save_analyses=false

  8.3 成本优化（LLM）

  # 1. 使用更便宜的模型
  evaluation.critic.llm_model="gpt-3.5-turbo"

  # 2. 增加调用间隔
  evaluation.critic.llm_call_frequency=50

  # 3. 增加缓存
  evaluation.critic.cache_size=5000

  # 4. 仅在需要时分析
  evaluation.critic.analyze_frequency=10

  九、高级用法

  9.1 自定义奖励塑形函数

  编辑配置文件添加自定义权重：
  critic:
    use_llm_shaping: true
    shaping_weight: 0.5  # 增加LLM奖励影响
    llm_call_frequency: 3  # 更频繁调用

  9.2 多GPU训练

  # 使用特定GPU
  CUDA_VISIBLE_DEVICES=0 python -m habitat_llm.examples.planner_demo \
      --config-name evaluation/centralized_evaluation_runner_with_critic.yaml \
      evaluation.critic.device="cuda"

  9.3 继续训练

  # 在配置中指定检查点路径（需自行实现）
  evaluation.critic.resume_from="./checkpoints/critic/checkpoint_episode_50.pt"

  ---
  联系与支持

  如遇到问题，请检查：
  1. 依赖是否完整安装
  2. 配置文件语法是否正确
  3. 日志输出中的错误信息

  祝使用愉快！