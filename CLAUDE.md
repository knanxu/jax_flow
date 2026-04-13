# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Documentation Rule

**IMPORTANT**: When making improvements to the codebase, add a brief 1-2 sentence summary to the relevant section in this file. Do NOT create separate detailed documentation files unless explicitly requested. Keep this file concise and focused on key information.

## Project Overview

**jax_flow** is a JAX-based imitation learning and reinforcement learning framework for robotic manipulation tasks, focusing on flow matching policies and offline-to-online RL fine-tuning.

### Motivation

Build a high-quality, reusable JAX codebase for robotics research with:
1. **Standard JAX code**: Pure functional programming, immutable state, JIT-friendly
2. **Rich flow policies**: Flow matching, MeanFlow, MIP from much-ado-about-noising
3. **Multiple networks**: MLP (priority), UNet, Transformer (DiT)
4. **Full environment support**: Robomimic MimicGen DexMimicGen Push-T Kitchen (lowdim + image)
5. **Clean config management**: Hydra + OmegaConf (YAML configs)
6. **Offline RL**: DDPG + BC constraint (DDPGBCAgent)
7. **Extensibility**: Standard agent design (jaxrl_m, qc patterns)

### Commands

### Setup
```bash
pip install -e .
pip install -e .[dev]
pip install -e .[robomimic]
pip install -e .[pusht]
pip install -e .[kitchen]
```

### Data Download
```bash
# Lowdim 数据集（支持自动下载）
python scripts/download_data.py --task square --obs_type lowdim

# Image 数据集方案 1：从 demo 文件生成（robomimic 官方方式）
python scripts/generate_image_dataset.py --task square --dataset_type ph

# Image 数据集方案 2：从 MimicGen 下载（推荐，如果任务可用）
python scripts/download_data.py --task stack --obs_type image --source mimicgen

# 批量下载所有数据集
python scripts/download_all_datasets.py

# 训练时自动下载缺失的 lowdim 数据集
python scripts/train_bc.py task=square_lowdim  # 会自动检查并下载
```

### Training
```bash
# Behavior cloning (Hydra config)
python scripts/train_bc.py task=lift_lowdim

# Override parameters
python scripts/train_bc.py task=lift_lowdim optimization.lr=1e-3 optimization.batch_size=128

# Disable W&B logging
python scripts/train_bc.py task=lift_lowdim wandb.enabled=false

# Adjust evaluation frequency
python scripts/train_bc.py task=lift_lowdim eval.eval_interval=20000

# Offline RL: DDPG + BC on flow policy
python scripts/train_offline_rl.py task=square_lowdim algorithm.bc_checkpoint=path/to/bc/best_model.pkl

# Push-T: state/keypoint/image observations
python scripts/train_bc.py task=pusht_state
python scripts/train_bc.py task=pusht_keypoint
python scripts/train_bc.py task=pusht_image

# Kitchen: state observations with p1-p7 metrics
python scripts/train_bc.py task=kitchen_state

# SpeedTuning: Rainbow DQN speed adjustment on frozen BC policy
python scripts/train_speed_tuning.py task=square_lowdim \
    algorithm.bc_checkpoint=path/to/bc/best_model.pkl
python scripts/train_speed_tuning.py task=square_lowdim \
    algorithm.bc_checkpoint=path/to/bc/best_model.pkl \
    algorithm.max_speed=3.0 algorithm.speed_granularity=0.2
```

### Evaluation
```bash
# Evaluate a trained checkpoint
python scripts/eval_bc.py --checkpoint checkpoints/square_lowdim_meanflow_mlp/best_model.pkl

# Evaluate with custom settings
python scripts/eval_bc.py --checkpoint path/to/checkpoint.pkl --num_episodes 100 --save_video

# Evaluate with rendering
python scripts/eval_bc.py --checkpoint path/to/checkpoint.pkl --render
```

### Data Download
```bash
# Robomimic lowdim 数据集（支持自动下载）
python scripts/download_data.py --task square --obs_type lowdim

# Robomimic image 数据集（从 demo 文件生成）
python scripts/generate_image_dataset.py --task square --dataset_type ph

# MimicGen 数据集（一键下载+转换全部 9 个任务）
python scripts/download_mimicgen.py

# MimicGen 单任务手动下载+转换
cd ~/.robomimic/mimicgen/core
wget -c "https://huggingface.co/datasets/amandlek/mimicgen_datasets/resolve/main/core/stack_d0.hdf5"
python scripts/convert_mimicgen.py --input ~/.robomimic/mimicgen/core/stack_d0.hdf5 --task stack

# 批量转换所有已下载的 MimicGen 原始文件
python scripts/convert_mimicgen.py --all

# DexMimicGen 数据集（一键下载+转换全部 3 个 Panda 双臂任务）
python scripts/download_dexmimicgen.py

# DexMimicGen 单任务下载+转换
python scripts/download_dexmimicgen.py --tasks two_arm_threading

# 批量转换所有已下载的 DexMimicGen 原始文件
python scripts/convert_dexmimicgen.py --all
```

**数据源对比**：
- **Robomimic**: lift, can, square, transport, tool_hang（lowdim 可直接下载，image 需从 demo 生成）
- **MimicGen**: stack, stack_three, threading, coffee, kitchen, hammer_cleanup, mug_cleanup, pick_place, nut_assembly（HuggingFace 下载 `{task}_d0.hdf5` 后用 `convert_mimicgen.py` 转换）
- **DexMimicGen**: two_arm_threading, two_arm_three_piece_assembly, two_arm_transport（Panda 双臂任务，HuggingFace `MimicGen/dexmimicgen_datasets` 下载后用 `convert_dexmimicgen.py` 转换，存储于 `~/.dexmimicgen/datasets/`）
- **Push-T**: pusht（zarr 格式，HuggingFace `ChaoyiPan/mip-dataset` 自动下载，支持 state/keypoint/image）
- **Kitchen**: kitchen（MJL 二进制日志，HuggingFace `ChaoyiPan/mip-dataset` 自动下载，60D obs + 9D action）

数据集路径优先级：绝对路径 > 项目相对路径 > ~/.dexmimicgen/datasets/ > ~/.robomimic/mimicgen/core/ > ~/.robomimic/ > 自动下载
DatasetManager 会自动根据 task 名判断数据源（MIMICGEN_TASKS / DEXMIMICGEN_TASKS / PUSHT_TASKS / KITCHEN_TASKS 集合中的任务自动走对应路径）

**任务分发机制**: task YAML 中 `task_source` 字段（robomimic/pusht/kitchen）控制数据集和环境的创建路由。`make_dataset()` 和 `make_env()` 分发函数根据 `task_source` 路由到对应后端。

### Testing
```bash
pytest
ruff format .
ruff check --fix .
pyright jax_flow/
```


## Architecture

### Key Design Decisions

- **No FlowMap**: 网络直接输出速度场 `velocity`，不输出最终动作。推理时通过 Euler/Heun ODE 求解器积分得到 action。
- **网络签名**: `(at, s, t, obs) -> velocity`，其中 `at: (batch, horizon, action_dim)`，网络内部 flatten horizon 维度处理。
- **Action Chunking**: 网络预测 `horizon` 步动作序列，环境只执行前 `act_exec_steps` 步（Diffusion Policy 风格）。
- **Encoder 与核心网络分离**: Encoder 独立编码观测为 `cond` 向量，核心网络（MLP/UNet/Transformer）接收 `cond` 作为条件。
- **6D Rotation Representation**: Robomimic/MimicGen 任务默认使用连续 rotation_6d 表示（`abs_action: true`），将 axis_angle(3D) 转为 rot6d(6D)，单臂 action 从 7D→10D，双臂从 14D→20D。避免 axis_angle 在 ±π 处的拓扑不连续性，提升旋转密集任务的成功率。

### Core Components

1. **TrainState + ModuleDict** (`jax_flow/agents/train_state.py`)
   - Custom `flax.struct.PyTreeNode` with `apply_loss_fn` (qc pattern)
   - ModuleDict: 单个 TrainState 管理多个网络（encoder, flow, critic 等）
   - `select(name)` helper for module access
   - `extra_variables`: 存储 batch_stats 等非 params 变量，lowdim 任务为 None

2. **Agents** (`jax_flow/agents/`)
   - `BCAgent`: Flow matching BC, immutable PyTreeNode。图像任务使用 GroupNorm ResNet18（从头训练，统一 LR，对齐 Diffusion Policy）
   - `DDPGBCAgent`: Offline RL (DDPG + BC constraint)，单 ModuleDict TrainState + multi_transform optimizer，冻结 encoder，联合优化 flow policy (Q loss + BC loss) + critic
   - `CriticState`: Critic 训练状态管理，用于 BC warmup 阶段
   - `RainbowDQNAgent`: SpeedTuning 速度策略（Rainbow DQN: Double Q + C51 分布式 Q + Dueling 架构），冻结 BC encoder，离散速度动作空间，PER 外部管理
   - Pattern: `create()` → `update(batch)` → `sample_actions(obs)`

3. **SpeedTuning** (`jax_flow/agents/speed_tuning/`)
   - `rainbow_agent.py`: RainbowDQNAgent，复用 BC encoder（冻结），DuelingC51Network 输出速度离散动作
   - `networks.py`: DuelingC51Network（Dueling + C51 分布式 Q-learning，num_atoms=51）
   - `speed_tuning_env.py`: SpeedTuningEnvWrapper，封装冻结 BC 策略 + 线性时间插值 + k_skip 步执行
   - `per_buffer.py`: SumTree + PrioritizedReplayBuffer（优先经验回放）
   - `interpolation.py`: temporal_interpolate（线性时间插值）+ make_speed_options（速度范围生成，min=1.0）

3. **Flow Matching** (`jax_flow/flow/`)
   - `interpolant.py`: Linear / trigonometric interpolation
   - `losses.py`: flow_loss, mip_loss, mf_loss (mf_loss 待完整实现 JVP)
   - `samplers.py`: Euler, Heun, MIP samplers
   - 网络直接处理 horizon 维度，无需 vmap

4. **Networks** (`jax_flow/networks/`)
   - `mlp.py`: MLP，`@nn.compact`，输入 `(at, s, t, obs)` 输出 `velocity`
   - `unet.py`: 1D UNet (对齐 ChiUNet from much-ado-about-noising)，dim_mult 控制通道倍增，GELU 激活，FiLM conditioning
   - `transformer.py`: DiT-style transformer
   - `value.py`: Q-function ensemble (nn.vmap), 支持 RED-Q (num_ensembles=10)
   - `embeddings.py`: Fourier / Learned timestep embeddings
   - `encoders/`: ResNet18Encoder (GroupNorm, 完整 layer1-4), SpatialSoftmax, MultiImageEncoder, MinViTEncoder, RandomShiftsAug

5. **Data** (`jax_flow/data/`)
   - `robomimic_dataset.py`: Lowdim HDF5 dataset, per-episode 存储 + 全局索引。支持 `sample_sequence()` 用于离线 RL（加载 rewards/dones/next_obs，per-episode 采样不跨 episode，可配置 reward_offset）
   - `robomimic_image_dataset.py`: Image HDF5 dataset, 支持多相机 + lowdim 混合。同样支持 `sample_sequence()` 用于离线 RL
   - `rotation_utils.py`: axis_angle ↔ rotation_6d 转换，`transform_action_to_6d` / `undo_transform_action` 支持单臂(7↔10D)和双臂(14↔20D)
   - `pusht_dataset.py`: Push-T zarr 数据集，支持 state/keypoint/image 三种 obs type，HuggingFace 自动下载
   - `kitchen_dataset.py`: Kitchen MJL 二进制日志数据集，60D obs + 9D action，HuggingFace 自动下载
   - `normalizer.py`: MinMax, Image, Identity normalizers
   - `dataset_manager.py`: 自动下载和路径解析，支持 robomimic/mimicgen/dexmimicgen/pusht/kitchen，智能处理 image 数据集
   - `replay_buffer.py`: NumPy 环形 replay buffer, 支持 dict obs + N-step returns + offline 填充

6. **Environments** (`jax_flow/envs/`)
   - `robomimic_env.py`: RobomimicWrapper, RobomimicImageWrapper, FrameStackWrapper, ActionChunkingWrapper, DEXMIMICGEN_ENVS 注册表（自动推断双臂 obs/image keys）
   - `pusht/`: Push-T 环境（pymunk 物理引擎），支持 state(5D)/keypoint(20D)/image obs，PushTWrapper 归一化 + success 判定
   - `kitchen/`: Kitchen 环境（MuJoCo Franka），thirdparty/ 包含完整 adept_envs，KitchenWrapper 桥接 old gym → Gymnasium + p1-p7 指标

7. **Configuration** (`configs/`)
   - Hydra + OmegaConf, YAML 格式
   - 层级: `task/`, `network/`, `flow/`, `optimization/`

8. **Checkpoint & Evaluation** (`jax_flow/core/`)
   - `checkpoint.py`: save/load checkpoint (pickle format), 包含 params + opt_state + config + normalizers
   - `evaluation.py`: rollout_episode, evaluate_policy (success rate + videos + Kitchen p1-p7 指标)
   - 训练时自动保存 best model + 定期 checkpoint，支持 W&B 视频上传
   - Kitchen p1-p7: 完成 ≥k 个子任务的 episode 比例（k=1..7），用于评估多任务完成度

### Training Pipeline for BC

```
Dataset → sample_batch → {observations, actions}
                              ↓
Encoder(observations) → cond: (batch, cond_dim)
                              ↓
t ~ Uniform(0,1), x0 ~ N(0,I), x_t = interp(t, x0, actions)
                              ↓
Network(x_t, t, t, cond) → predicted_velocity
                              ↓
Loss = MSE(predicted_velocity, target_velocity)
```

### Evaluation Pipeline

```
Environment.reset() → obs
                       ↓
FrameStack(obs) → obs_history: (obs_steps, obs_dim)
                       ↓
Agent.eval_actions(obs_history) → actions: (horizon, action_dim)
                       ↓
ActionChunking → execute first act_exec_steps
                       ↓
Environment.step() → next_obs, reward, done, info
                       ↓
Record success rate, episode length, return
Save videos (if enabled) → upload to W&B
```

## Configuration

```
configs/
├── config.yaml              # Main config (defaults + logging)
├── task/lift_lowdim.yaml    # horizon=16, obs_steps=2, act_steps=8
├── network/mlp.yaml         # hidden_dims=[512,512,512], emb_dim=256
├── flow/meanflow.yaml       # interp=linear, sampler=euler
└── optimization/default.yaml # lr=1e-4, batch_size=256, steps=300k
```

## References

- **much-ado-about-noising**: Flow matching methods (MIP, MeanFlow)
- **Diffusion Policy**: Action chunking, UNet, obs_steps/horizon/act_steps 设计
- **jaxrl_m**: JAX RL patterns, ensemblize
- **robomimic / mimicgen**: Environments and datasets
