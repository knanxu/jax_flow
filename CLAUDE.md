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
4. **Full environment support**: Robomimic MimicGen DexMimicGen (lowdim + image)
5. **Clean config management**: Hydra + OmegaConf (YAML configs)
6. **Offline-to-online RL**: ACFQL algorithm from qc project
7. **Extensibility**: Standard agent design (jaxrl_m, qc patterns)

### Commands

### Setup
```bash
pip install -e .
pip install -e .[dev]
pip install -e .[robomimic]
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

**数据源对比**：
- **Robomimic**: lift, can, square, transport, tool_hang（lowdim 可直接下载，image 需从 demo 生成）
- **MimicGen**: stack, stack_three, threading, coffee, kitchen 等（lowdim + image 都可直接下载，推荐用于 image）
- **注意**: MimicGen 没有 square 任务，square 的 image 数据集需要从 robomimic demo 生成

数据集路径优先级：绝对路径 > 项目相对路径 > ~/.robomimic/ > 自动下载 

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

### Core Components

1. **TrainState + ModuleDict** (`jax_flow/agents/train_state.py`)
   - Custom `flax.struct.PyTreeNode` with `apply_loss_fn` (qc pattern)
   - ModuleDict: 单个 TrainState 管理多个网络（encoder, flow, critic 等）
   - `select(name)` helper for module access

2. **Agents** (`jax_flow/agents/`)
   - `BCAgent`: Flow matching BC, immutable PyTreeNode
   - `ACFQLAgent`: Offline-to-online RL (待完善)
   - Pattern: `create()` → `update(batch)` → `sample_actions(obs)`

3. **Flow Matching** (`jax_flow/flow/`)
   - `interpolant.py`: Linear / trigonometric interpolation
   - `losses.py`: flow_loss, mip_loss, mf_loss (mf_loss 待完整实现 JVP)
   - `samplers.py`: Euler, Heun, MIP samplers
   - 网络直接处理 horizon 维度，无需 vmap

4. **Networks** (`jax_flow/networks/`)
   - `mlp.py`: MLP，`@nn.compact`，输入 `(at, s, t, obs)` 输出 `velocity`
   - `unet.py`: 1D UNet for sequence modeling
   - `transformer.py`: DiT-style transformer
   - `value.py`: Q-function ensemble (nn.vmap)
   - `embeddings.py`: Fourier / Learned timestep embeddings
   - `encoders/`: ResNet18Encoder, SpatialSoftmax, MultiImageEncoder

5. **Data** (`jax_flow/data/`)
   - `robomimic_dataset.py`: Lowdim HDF5 dataset, per-episode 存储 + 全局索引
   - `robomimic_image_dataset.py`: Image HDF5 dataset, 支持多相机 + lowdim 混合
   - `normalizer.py`: MinMax, Image, Identity normalizers
   - `dataset_manager.py`: 自动下载和路径解析，支持 robomimic/mimicgen，智能处理 image 数据集

6. **Environments** (`jax_flow/envs/robomimic_env.py`)
   - `RobomimicWrapper`: Lowdim obs, normalized actions
   - `RobomimicImageWrapper`: Dict obs (images + lowdim)
   - `FrameStackWrapper`: 观测历史堆叠（支持 array 和 dict obs）
   - `ActionChunkingWrapper`: 执行 action 序列的前 N 步

7. **Configuration** (`configs/`)
   - Hydra + OmegaConf, YAML 格式
   - 层级: `task/`, `network/`, `flow/`, `optimization/`

8. **Checkpoint & Evaluation** (`jax_flow/core/`)
   - `checkpoint.py`: save/load checkpoint (pickle format), 包含 params + opt_state + config + normalizers
   - `evaluation.py`: rollout_episode, evaluate_policy (success rate + videos)
   - 训练时自动保存 best model + 定期 checkpoint，支持 W&B 视频上传

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

## Implementation Roadmap

**Phase 1: Foundation** (✅ Completed)
- [x] TrainState, ModuleDict, config system
- [x] Data loading (lowdim + image), normalizers
- [x] Environment wrappers (lowdim + image + action chunking)
- [x] Frame stacking

**Phase 2: Flow Matching + Networks** (✅ Completed)
- [x] Interpolant (linear, trig)
- [x] Losses (flow_loss, mip_loss, mf_loss placeholder)
- [x] Samplers (Euler, Heun, MIP)
- [x] MLP network with horizon support
- [x] Timestep embeddings (Fourier, Learned)
- [x] Image encoders (ResNet18 + SpatialSoftmax + MultiImage)

**Phase 3: BC Agent + Training** (✅ Completed)
- [x] BCAgent (create, update, sample_actions)
- [x] Training script (scripts/train_bc.py)
- [x] Data download script

**Phase 4: 验证与调试** (✅ Completed)
- [x] 数据集管理系统（自动下载 + 路径解析）
- [x] Image 数据集生成脚本
- [x] Square lowdim 端到端训练验证
- [x] Square image 端到端训练验证
- [x] Evaluation script (rollout + success rate)
- [x] Checkpoint save/load (pickle format)
- [x] W&B logging 集成（训练 + eval + 视频上传）

**Phase 5: ACFQL + Extensions**
- [ ] 完善 ACFQLAgent
- [ ] MeanFlow 完整实现 (JVP)
- [ ] UNet / Transformer networks
- [ ] 图像数据增强 (random crop, color jitter)

## References

- **qc (ACFQL)**: Agent design, TrainState, ACFQL algorithm
- **much-ado-about-noising**: Flow matching methods (MIP, MeanFlow)
- **Diffusion Policy**: Action chunking, UNet, obs_steps/horizon/act_steps 设计
- **jaxrl_m**: JAX RL patterns, ensemblize
- **robomimic / mimicgen**: Environments and datasets
