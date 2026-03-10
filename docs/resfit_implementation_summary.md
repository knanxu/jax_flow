# ResFiT Implementation Summary

## 完成状态

已按照 `docs/resfit_plan.md` 的详细计划，完成了 ResFiT (Residual Fine-Tuning) 算法的完整实现。

## 实现的组件

### Phase 1: 基础组件（已完成）

#### Task A: Replay Buffer ✓
- **文件**: `jax_flow/data/replay_buffer.py`
- **功能**:
  - 纯 NumPy 环形 buffer
  - 支持 dict 观测（image + lowdim）
  - N-step returns 计算
  - `OfflineReplayBuffer.from_dataset()` 用于从 BC 推理填充离线数据

#### Task B: RandomShiftsAug ✓
- **文件**: `jax_flow/networks/encoders/random_shifts.py`
- **功能**:
  - DrQ 风格随机平移数据增强
  - Pad → 随机裁剪回原始尺寸
  - 训练时增强，推理时不变

#### Task C: ResidualActor 修改 ✓
- **文件**: `jax_flow/networks/residual_actor.py`
- **新增**: `add_exploration_noise()` 函数
- **功能**: TD3 风格 clipped Gaussian 探索噪声

#### Task D: Value 验证 ✓
- **文件**: `jax_flow/networks/value.py`
- **确认**: 支持 `num_ensembles=10` (RED-Q)
- **工具函数**: 复用 `spatial_emb_critic.py` 中的 `redq_q_value` 和 `ensemble_mean_q`

### Phase 2: 环境与 Checkpoint（已完成）

#### Task E: Residual Environment Wrapper ✓
- **文件**: `jax_flow/envs/residual_wrapper.py`
- **功能**:
  - 内嵌冻结 BC 策略
  - 内部管理 BC action queue（将 chunk 拆分为单步）
  - RL agent 只输出单步残差
  - 观测 dict 自动添加 `base_action` 字段
  - 不需要外层 ActionChunkingWrapper

#### Task F: Checkpoint 扩展 ✓
- **文件**: `jax_flow/core/checkpoint.py`
- **新增函数**:
  - `save_resfit_checkpoint()`: 保存多 TrainState checkpoint
  - `load_resfit_checkpoint()`: 加载 ResFiT checkpoint

### Phase 3: 核心 Agent（已完成）

#### Task G: ResFiT Agent ✓
- **文件**: `jax_flow/agents/IL_RLFiT/resfit_agent.py`
- **架构**:
  - 3 个独立 TrainState (encoder, actor, critic)
  - 2 个 target 网络 (Polyak 更新)
  - RED-Q ensemble (10 个 Q-heads)
  - Encoder 只从 critic loss 更新（actor loss 中 stop_gradient）
- **方法**:
  - `create()`: 初始化 agent，可选从 BC checkpoint 加载 encoder
  - `update_critic()`: 更新 encoder + critic
  - `update_actor()`: 更新 actor（encoder stop_gradient）
  - `sample_actions()`: 采样动作（带探索噪声）
  - `eval_actions()`: 确定性动作（评估用）

### Phase 4: 训练与验证（已完成）

#### Task H: Evaluation 适配 ✓
- **文件**: `jax_flow/core/evaluation.py`
- **修改**:
  - 确保 obs_batch 转为 numpy array
  - 添加注释说明 ResFiT 和 BC agent 的 action 格式差异
  - 核心逻辑无需改动（ResidualEnvWrapper 处理一切）

#### Task I: 训练脚本 + 配置 ✓
- **训练脚本**: `scripts/train_resfit.py`
- **配置文件**:
  - `configs/resfit/default.yaml`: 算法默认配置
  - `configs/resfit/task/square_lowdim.yaml`: Square lowdim 任务
  - `configs/resfit/task/square_image.yaml`: Square image 任务

**训练流程（3 阶段）**:
1. **Phase 1: Random Exploration** (10,000 env steps)
   - 随机残差动作探索
   - 填充 online replay buffer
2. **Phase 2: Critic Warmup** (10,000 gradient steps)
   - 只更新 encoder + critic
   - 不更新 actor
3. **Phase 3: Full Training** (300,000 env steps)
   - UTD ratio = 4（每个 env step 做 4 次 critic 更新）
   - 每 4 次 critic 更新做 1 次 actor 更新
   - 50/50 online/offline 数据混合
   - 定期评估和 checkpoint

## 关键设计决策

### 1. Action Chunking 处理
- **ResFiT 原版**: BC 策略内部维护 action queue，RL 只看到单步
- **我们的实现**: ResidualEnvWrapper 内部管理 BC action queue
  - BC 一次输出 `(horizon, action_dim)` chunk
  - Wrapper 拆分为单步，逐步执行
  - RL actor 每步输出单步残差 `(action_dim,)`
  - 不需要外层 ActionChunkingWrapper

### 2. 多 TrainState 架构
- Encoder, Actor, Critic 各自独立 optimizer
- Encoder 学习率 = Critic 学习率 (1e-4)
- Actor 学习率极小 (1e-6)
- Encoder 只从 critic loss 获得梯度（actor loss 中 stop_gradient）

### 3. Image vs Lowdim 模式
- **Image 模式**:
  - Encoder: MultiImageEncoder (ViT backbone, return_patches=True)
  - Critic: SpatialEmbCritic (共享 trunk + vmap ensemble)
  - Actor: 接收 spatial embedding + prop features
- **Lowdim 模式**:
  - Encoder: MLPEncoder
  - Critic: Value (concat obs + action → MLP ensemble)
  - Actor: 接收 encoded obs features

### 4. RED-Q Ensemble
- 10 个 Q-heads (num_q=10)
- Target Q: 随机选 2 个取 min (min_q_heads=2)
- Actor loss: 全部 10 个 head 的 mean

## 文件清单

### 新建文件
```
jax_flow/
├── agents/IL_RLFiT/
│   ├── __init__.py
│   └── resfit_agent.py
├── data/
│   └── replay_buffer.py
├── envs/
│   └── residual_wrapper.py
└── networks/encoders/
    └── random_shifts.py

configs/resfit/
├── default.yaml
└── task/
    ├── square_lowdim.yaml
    └── square_image.yaml

scripts/
└── train_resfit.py

docs/
└── resfit_implementation_summary.md  # 本文件
```

### 修改文件
```
jax_flow/
├── agents/__init__.py              # 导出 ResFiTAgent
├── data/__init__.py                # 导出 ReplayBuffer
├── envs/__init__.py                # 导出 ResidualEnvWrapper
├── networks/__init__.py            # 导出 ResidualActor, Value, 工具函数
├── networks/encoders/__init__.py   # 导出 RandomShiftsAug
├── networks/residual_actor.py      # 添加 add_exploration_noise()
├── core/checkpoint.py              # 添加 save/load_resfit_checkpoint()
└── core/evaluation.py              # 小改：确保 obs_batch 转 numpy
```

## 使用方法

### 1. 训练 BC 策略（前置步骤）
```bash
python scripts/train_bc.py task=square_lowdim
# 或
python scripts/train_bc.py task=square_image
```

### 2. ResFiT 微调
```bash
# Lowdim 模式
python scripts/train_resfit.py \
    bc_checkpoint=checkpoints/square_lowdim_meanflow_mlp/best_model.pkl \
    task=square_lowdim

# Image 模式
python scripts/train_resfit.py \
    bc_checkpoint=checkpoints/square_image_meanflow_vit/best_model.pkl \
    task=square_image
```

### 3. 配置覆盖
```bash
python scripts/train_resfit.py \
    bc_checkpoint=path/to/bc.pkl \
    task=square_lowdim \
    algorithm.total_timesteps=500000 \
    algorithm.actor_lr=5e-7 \
    algorithm.eval_interval=5000
```

## 超参数对照（ResFiT 论文 vs 本实现）

| 参数 | 论文值 | 本实现 | 说明 |
|------|--------|--------|------|
| actor_lr | 1e-6 | 1e-6 | 残差 actor 学习率极小 |
| critic_lr | 1e-4 | 1e-4 | critic + encoder 共用 |
| tau | 0.005 | 0.005 | Polyak 系数 |
| action_scale | 0.1 | 0.1 | 残差范围 ±10% |
| num_q | 10 | 10 | RED-Q ensemble |
| min_q_heads | 2 | 2 | target Q 子采样 |
| UTD ratio | 4 | 4 | critic 更新频率 |
| batch_size | 256 | 256 | 128 online + 128 offline |
| n_step | 3 | 3 | N-step returns |
| gamma | 0.99 | 0.99 | 折扣因子 |
| stddev | 0.05 | 0.05 | 探索噪声 |
| learning_starts | 10,000 | 10,000 | 随机探索步数 |
| critic_warmup | 10,000 | 10,000 | critic-only 预热 |
| total_timesteps | 300,000 | 300,000 | 总训练步数 |

## 验证状态

- ✅ 所有模块语法正确
- ✅ 所有模块可正常 import
- ✅ 配置文件格式正确
- ⏳ 端到端训练验证（需要 BC checkpoint）

## 下一步

### Task J: 端到端验证
1. 训练一个 BC checkpoint（square_lowdim）
2. 运行 ResFiT 训练脚本
3. 验证训练流程正常
4. 检查评估结果
5. 调试任何运行时问题

### 可选扩展
- Prioritized Experience Replay (PER)
- HL-Gauss / C51 distributional critic
- Actor spatial embedding（ResFiT 中 actor 也可用 SpatialEmb）
- 多环境并行采集（VecEnv）
- 其他任务配置（lift, can, stack 等）

## 注意事项

1. **BC checkpoint 必须先训练好**，ResFiT 依赖冻结的 BC 策略
2. **数据集路径**: 训练脚本会从 BC checkpoint 中读取 dataset_path
3. **观测模式一致性**: ResFiT 的 obs_type 必须与 BC checkpoint 一致
4. **Action normalizer**: 从 BC checkpoint 的 normalizers 中加载
5. **W&B logging**: 可选，需要安装 wandb 并配置

## 参考

- 论文: ResFiT (arXiv:2509.19301)
- 代码: https://github.com/amazon-far/residual-offpolicy-rl
- 计划文档: `docs/resfit_plan.md`
