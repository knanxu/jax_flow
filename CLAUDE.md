# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**jax_flow** is a JAX-based imitation learning and reinforcement learning framework for robotic manipulation tasks, focusing on flow matching policies and offline-to-online RL fine-tuning.

### Motivation

Build a high-quality, reusable JAX codebase for robotics research with:
1. **Standard JAX code**: Pure functional programming, immutable state, JIT-friendly
2. **Rich flow policies**: Flow matching, MeanFlow, MIP from much-ado-about-noising
3. **Multiple networks**: MLP (priority), UNet, Transformer (DiT)
4. **Full environment support**: Robomimic and MimicGen (lowdim + image)
5. **Clean config management**: Hydra + OmegaConf (YAML configs)
6. **Offline-to-online RL**: ACFQL algorithm from qc project
7. **Extensibility**: Standard agent design (jaxrl_m, qc patterns)

## Commands

### Setup
```bash
pip install -e .
pip install -e .[dev]
pip install -e .[robomimic]
```

### Training
```bash
# Behavior cloning (Hydra config)
python scripts/train_bc.py task=lift_lowdim

# Override parameters
python scripts/train_bc.py task=lift_lowdim optimization.lr=1e-3 optimization.batch_size=128
```

### Data Download
```bash
python scripts/download_data.py --task lift --obs_type lowdim
python scripts/download_data.py --task lift --obs_type image
```

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
   - `value.py`: Q-function ensemble (nn.vmap)
   - `embeddings.py`: Fourier / Learned timestep embeddings
   - `encoders.py`: IdentityEncoder, MLPEncoder + factory
   - `encoders/`: ResNet18Encoder, SpatialSoftmax, MultiImageEncoder

5. **Data** (`jax_flow/data/`)
   - `robomimic_dataset.py`: Lowdim HDF5 dataset, per-episode 存储 + 全局索引
   - `robomimic_image_dataset.py`: Image HDF5 dataset, 支持多相机 + lowdim 混合
   - `normalizer.py`: MinMax, Image, Identity normalizers

6. **Environments** (`jax_flow/envs/robomimic_env.py`)
   - `RobomimicWrapper`: Lowdim obs, normalized actions
   - `RobomimicImageWrapper`: Dict obs (images + lowdim)
   - `FrameStackWrapper`: 观测历史堆叠（支持 array 和 dict obs）
   - `ActionChunkingWrapper`: 执行 action 序列的前 N 步

7. **Configuration** (`configs/`)
   - Hydra + OmegaConf, YAML 格式
   - 层级: `task/`, `network/`, `flow/`, `optimization/`

### Training Pipeline

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

**Phase 4: 验证与调试** (Next)
- [ ] 下载数据集，端到端训练验证
- [ ] Evaluation script (rollout + success rate)
- [ ] Checkpoint save/load
- [ ] W&B logging 集成

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
