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
5. **Clean config management**: ml_collections + absl.flags (JAX community standard)
6. **Offline-to-online RL**: ACFQL algorithm from qc project
7. **Extensibility**: Standard agent design (jaxrl, rlax, qc patterns)

## Commands

### Setup
```bash
pip install -e .
pip install -e .[dev]
pip install -e .[robomimic]
```

### Training
```bash
# Behavior cloning
python examples/train_bc.py --env_name=lift --config=configs/bc_config.py

# ACFQL offline-to-online
python examples/train_acfql.py --env_name=lift --config=configs/acfql_config.py

# Override config
python examples/train_bc.py --config.lr=1e-4 --config.batch_size=256
```

### Evaluation
```bash
python examples/eval_policy.py --checkpoint_path=checkpoints/latest.pkl --env_name=lift --num_episodes=50
```

### Testing
```bash
pytest
pytest tests/test_flow_matching.py -v
pytest --cov=jax_flow --cov-report=html
```

### Code Quality
```bash
ruff format .
ruff check --fix .
pyright jax_flow/
pre-commit run --all-files
```
**Naming Convention:**
```
{task}-{obs_type}-{dataset_type}

Examples:
- "lift-low_dim-ph"    → Lift task, state obs, proficient-human data
- "square-image-ph"    → Square task, image obs, proficient-human data
- "stack-image-mg"     → Stack task, image obs, machine-generated data
```

**Supported Tasks:**
- Robomimic: lift, can, square, transport, tool_hang
- Mimicgen: stack, stack_three, threading, coffee, kitchen, pick_place, etc.

## Architecture

### Core Components

1. **TrainState** (`jax_flow/agents/train_state.py`)
   - Custom `flax.struct.PyTreeNode` with `apply_loss_fn` method (qc pattern)
   - Manages params, opt_state, step in immutable state
   - `select(name)` helper for ModuleDict

2. **Agent System** (`jax_flow/agents/`)
   - `BCAgent`: Flow matching behavior cloning
   - `ACFQLAgent`: Offline-to-online RL with flow matching
   - Immutable PyTreeNode agents (qc pattern)
   - Single TrainState with ModuleDict for multi-network management

3. **Flow Matching** (`jax_flow/flow/`)
   - `interpolant.py`: Linear and trigonometric interpolation
   - `flow_map.py`: Flax Module learning velocity field
   - `losses.py`: flow_loss, mip_loss, mf_loss (MeanFlow)
   - `samplers.py`: Euler, Heun samplers

4. **Networks** (`jax_flow/networks/`)
   - `mlp.py`: MLP (priority implementation)
   - `unet.py`: UNet (Diffusion Policy style, later)
   - `transformer.py`: DiT (later)
   - `encoders.py`: Identity, MLP, Image encoders
   - Unified interface: `(x, s, t, cond) -> velocity`

5. **Data** (`jax_flow/data/`)
   - `robomimic_dataset.py`: HDF5 loading for robomimic/mimicgen
   - `normalizer.py`: Observation/action normalization
   - Supports all robomimic tasks (lift, can, square, transport, tool_hang)
   - Supports all mimicgen tasks (stack, threading, coffee, etc.)

6. **Environments** (`jax_flow/envs/`)
   - `robomimic_env.py`: Environment wrapper
   - `wrappers.py`: Lowdim/Image wrappers
   - Unified Gymnasium interface

7. **Configuration** (`configs/`)
   - Uses ml_collections + absl.flags
   - Hierarchical structure: `task/`, `network/`, `optimization/`
   - Python config files (not YAML)

### Training Pipeline

BC training flow (`examples/train_bc.py`):
1. Load ml_collections config
2. Create environment and dataset
3. Initialize BCAgent with `BCAgent.create()`
4. Training loop:
   - Sample batch from dataloader
   - `agent, info = agent.update(batch)` (immutable update)
   - Periodic evaluation and logging
5. Save checkpoints

ACFQL training flow (`examples/train_acfql.py`):
1. Load BC checkpoint or train from scratch
2. Create ACFQLAgent with BC flow + critic
3. Online interaction loop:
   - Collect trajectories with `agent.sample_actions()`
   - Store in replay buffer
   - Update with mixed offline/online data
   - BC flow loss + distillation loss + Q loss

### Key Design Patterns

- **Immutable State**: All state in PyTreeNode, updates return new objects
- **Pure Functions**: Core logic (loss, update, sample) are pure functions
- **ModuleDict**: Single TrainState manages multiple networks
- **Factory Pattern**: `@classmethod create()` for agent initialization
- **Functional Pipeline**: Side effects only in training scripts

## Configuration Structure

```
configs/
├── bc_config.py              # BC main config
├── acfql_config.py           # ACFQL main config
├── task/
│   ├── lift_lowdim.py
│   ├── lift_image.py
│   └── ...
├── network/
│   ├── mlp.py
│   └── ...
└── optimization/
    └── default.py
```

Key parameters:
- `task.env_name`: Environment name (lift, can, square, etc.)
- `task.obs_type`: "lowdim" or "image"
- `task.horizon`: Action prediction horizon (10)
- `task.obs_steps`: Observation history (2)
- `task.act_steps`: Action execution steps (8)
- `network.network_type`: "mlp", "unet", "transformer"
- `network.hidden_dims`: Network hidden dimensions
- `optimization.lr`: Learning rate
- `optimization.batch_size`: Batch size
- `optimization.gradient_steps`: Total training steps
- `flow.flow_type`: "flow_matching", "meanflow", "mip"
- `flow.flow_steps`: Number of ODE steps

## Implementation Roadmap

**Phase 1: Foundation** (✅ Completed)
- [x] TrainState with apply_loss_fn
- [x] ModuleDict helper
- [x] Basic config system (ml_collections)
- [x] Data loading (robomimic HDF5)
- [x] Environment wrappers
- [x] Normalizers (MinMax, Image)
- [x] Frame stacking wrapper

**Phase 2: Flow Matching** (Next)
- [ ] Interpolant (linear, trig)
- [ ] FlowMap module
- [ ] Losses (flow_loss, mip_loss, mf_loss)
- [ ] Samplers (Euler, Heun)
- [ ] MLP network
- [ ] Timestep embeddings

**Phase 3: BC Agent**
- [ ] BCAgent implementation
- [ ] Training script
- [ ] Evaluation script

**Phase 4: ACFQL**
- [ ] Critic network
- [ ] ACFQLAgent implementation
- [ ] Replay buffer
- [ ] Online training script

**Phase 5: Extensions**
- [ ] UNet and Transformer networks
- [ ] Image encoders (ResNet-18 + spatial softmax)
- [ ] Image dataset with augmentation
- [ ] More RL algorithms

## References

- **qc (ACFQL)**: Agent design, TrainState, ACFQL algorithm
- **much-ado-about-noising**: Flow matching methods
- **jaxrl / jaxrl_m**: JAX RL patterns
- **rlax**: RL algorithm components
- **robomimic / mimicgen**: Environments and datasets
