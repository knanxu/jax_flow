# JAX Flow

JAX-based flow matching framework for robotic manipulation — imitation learning and offline-to-online RL.

## Installation

Requires Python >= 3.10.

```bash
# 1. Install JAX (choose one)
pip install jax[cpu]       # CPU only
pip install jax[cuda12]    # CUDA 12 (recommended for training)

# 2. Install project core
pip install -e .
```

根据你要使用的任务环境，安装对应的可选依赖：

```bash
# Robomimic 环境（lift, can, square, transport, tool_hang）
pip install -e ".[robomimic]"

# Push-T 环境
pip install -e ".[pusht]"

# Kitchen 环境
pip install -e ".[kitchen]"

# 全部安装
pip install -e ".[all]"
```

MimicGen 特有环境（coffee, threading, hammer_cleanup 等）需额外安装 mimicgen 包：

```bash
pip install "mimicgen @ git+https://github.com/NVlabs/mimicgen.git"
```

DexMimicGen 双臂任务需额外安装 dexmimicgen 包：

```bash
pip install -e /path/to/dexmimicgen
```

## Datasets

支持五类数据源：

| 数据源 | 任务 | 存储位置 | obs 类型 |
|--------|------|----------|----------|
| Robomimic | lift, can, square, transport, tool_hang | `~/.robomimic/{task}/ph/` | lowdim, image |
| MimicGen | stack, stack_three, threading, coffee, kitchen, hammer_cleanup, mug_cleanup, pick_place, nut_assembly | `~/.robomimic/mimicgen/core/{task}/ph/` | lowdim, image |
| DexMimicGen | two_arm_threading, two_arm_three_piece_assembly, two_arm_transport | `~/.dexmimicgen/datasets/{task}/ph/` | lowdim, image |
| Push-T | pusht | HuggingFace cache (自动下载) | state, keypoint, image |
| Kitchen | kitchen | HuggingFace cache (自动下载) | state |

### Robomimic

Robomimic 的 lowdim 和 image 数据集获取方式不同，这是最容易踩坑的地方。

**Lowdim 数据集** — 可直接下载，训练时缺失也会自动下载：

```bash
python scripts/download_data.py --task square --obs_type lowdim
```

**Image 数据集** — 不能直接下载，必须先下载 demo（raw）文件，再渲染生成 image 数据：

```bash
# Step 1: 下载 demo 文件（包含 states 轨迹，用于渲染图像）
# 必须指定 --download_dir ~/.robomimic，否则 robomimic 默认下载到 site-packages/datasets/ 下
python -m robomimic.scripts.download_datasets \
    --tasks square \
    --dataset_types ph \
    --hdf5_types raw \
    --download_dir ~/.robomimic

# Step 2: 从 demo 文件渲染生成 image 数据集
python scripts/generate_image_dataset.py --task square --dataset_type ph
```

每个任务的相机配置已内置（lift/can/square: agentview + robot0_eye_in_hand 84x84, transport: 4 cameras 84x84, tool_hang: sideview + robot0_eye_in_hand 240x240）。

生成后的 image 数据集较大（每个任务约 2-3 GB），确保磁盘空间充足。

**批量下载所有 Robomimic lowdim 数据集：**

```bash
python scripts/download_all_datasets.py
```

> 注意：`download_all_datasets.py` 会尝试下载 lowdim 并生成 image，但 image 生成依赖 demo 文件存在。如果 demo 文件未提前下载，image 部分会跳过。

### MimicGen

MimicGen 数据集从 HuggingFace 下载原始 HDF5 文件（同时包含 lowdim 和 image），然后转换为分离格式。

```bash
# 一键下载全部 9 个任务并自动转换（推荐）
python scripts/download_mimicgen.py

# 或手动单任务下载 + 转换
python scripts/download_data.py --task stack --obs_type lowdim --source mimicgen

# 或完全手动：下载原始文件 → 转换
wget -c "https://huggingface.co/datasets/amandlek/mimicgen_datasets/resolve/main/core/stack_d0.hdf5" \
     -P ~/.robomimic/mimicgen/core/
python scripts/convert_mimicgen.py --input ~/.robomimic/mimicgen/core/stack_d0.hdf5 --task stack

# 批量转换所有已下载的原始文件
python scripts/convert_mimicgen.py --all
```

转换后生成：
- `~/.robomimic/mimicgen/core/{task}/ph/low_dim_v141.hdf5`（提取 lowdim keys）
- `~/.robomimic/mimicgen/core/{task}/ph/image_v141.hdf5`（symlink 到原始文件）

### DexMimicGen

DexMimicGen 是 Panda 双臂任务（act_dim=14），从 HuggingFace `MimicGen/dexmimicgen_datasets` 下载。

```bash
# 一键下载全部 3 个双臂任务并转换（推荐）
python scripts/download_dexmimicgen.py

# 下载指定任务
python scripts/download_dexmimicgen.py --tasks two_arm_threading

# 批量转换所有已下载的原始文件
python scripts/convert_dexmimicgen.py --all
```

转换后生成：
- `~/.dexmimicgen/datasets/{task}/ph/low_dim_v141.hdf5`
- `~/.dexmimicgen/datasets/{task}/ph/image_v141.hdf5`

### Push-T

Push-T 数据集从 HuggingFace `ChaoyiPan/mip-dataset` 自动下载（zarr 格式），无需手动操作。

训练时会自动下载缺失的数据集。需要安装 Push-T 依赖：

```bash
pip install -e ".[pusht]"
```

### Kitchen

Kitchen 数据集从 HuggingFace `ChaoyiPan/mip-dataset` 自动下载（MJL 二进制日志），无需手动操作。

训练时会自动下载缺失的数据集。需要安装 Kitchen 依赖：

```bash
pip install -e ".[kitchen]"
```

## Training

使用 Hydra 配置系统，通过命令行覆盖参数。

### Behavior Cloning (BC)

```bash
# 默认配置（lift lowdim + MLP + MeanFlow）
python scripts/train_bc.py

# Robomimic 任务
python scripts/train_bc.py task=lift_lowdim
python scripts/train_bc.py task=square_lowdim
python scripts/train_bc.py task=square_image
python scripts/train_bc.py task=can_lowdim
python scripts/train_bc.py task=transport_lowdim
python scripts/train_bc.py task=tool_hang_lowdim

# MimicGen 任务
python scripts/train_bc.py task=stack_lowdim
python scripts/train_bc.py task=stack_image
python scripts/train_bc.py task=coffee_lowdim
python scripts/train_bc.py task=threading_lowdim

# DexMimicGen 双臂任务
python scripts/train_bc.py task=dex_threading_lowdim
python scripts/train_bc.py task=dex_threading_image
python scripts/train_bc.py task=dex_three_piece_assembly_lowdim
python scripts/train_bc.py task=dex_transport_lowdim

# Push-T 任务（state / keypoint / image 三种 obs）
python scripts/train_bc.py task=pusht_state
python scripts/train_bc.py task=pusht_keypoint
python scripts/train_bc.py task=pusht_image

# Kitchen 任务
python scripts/train_bc.py task=kitchen_state

# 自定义参数
python scripts/train_bc.py task=square_lowdim optimization.lr=3e-4 optimization.batch_size=256

# 切换网络 / flow 方法
python scripts/train_bc.py task=square_lowdim network=unet flow=mip
python scripts/train_bc.py task=square_lowdim flow=flow_matching
python scripts/train_bc.py task=square_lowdim flow=meanflow_stable

# 禁用 W&B
python scripts/train_bc.py task=square_lowdim wandb.enabled=false

# 调整 eval 频率
python scripts/train_bc.py task=square_lowdim eval.eval_interval=20000
```

### ResFiT (Residual RL Fine-Tuning)

需要先训练好 BC checkpoint，再用 TD3 + RED-Q 做残差 RL 微调：

```bash
# Step 1: 先训练 BC
python scripts/train_bc.py task=square_lowdim

# Step 2: 用 BC checkpoint 做 ResFiT
python scripts/train_resfit.py task=square_lowdim bc_checkpoint=checkpoints/square_lowdim_meanflow_mlp/best_model.pkl

# Image 任务
python scripts/train_resfit.py task=square_image bc_checkpoint=checkpoints/square_image/best_model.pkl
```

### ACFQL (Offline-to-Online RL)

```bash
python scripts/train_acfql.py --config-name acfql_default task=square_lowdim

# 切换 actor 类型
python scripts/train_acfql.py --config-name acfql_default task=square_lowdim algorithm.actor_type=distill-ddpg
```

### DQC (Decoupled Q-Chunking)

```bash
python scripts/train_dqc.py --config-name dqc_default task=square_lowdim

# 自定义 backup horizon 和 policy chunk size
python scripts/train_dqc.py --config-name dqc_default task=square_lowdim algorithm.backup_horizon=25 algorithm.policy_chunk_size=5
```

### Offline RL (DDPG + BC)

在离线数据集上联合训练 flow policy（actor）和 critic。算法核心：`actor_loss = q_weight * q_loss + alpha * bc_loss`，其中 BC loss 作用于完整 action chunk（horizon 步），Q loss 作用于实际执行的前 `act_exec_steps` 步。

**前置条件**：需要一个 BC checkpoint 来读取网络架构配置（encoder type、network type、hidden dims 等）。默认不加载预训练权重，所有参数随机初始化。

```bash
# 先训练一个 BC（仅用于获取网络架构配置）
python scripts/train_bc.py task=square_lowdim flow=mip
# 产出: checkpoints/square_lowdim_mip_mlp/best_model.pkl
```

#### 模式 1：BC warmup → Offline RL

先用纯 BC loss 训练 encoder + flow + critic（`q_weight=0`），warmup 结束后加入 Q loss（`q_weight=1.0`）。warmup 阶段 critic 也在同步训练（通过 critic_loss），这样切换到 RL 阶段时 critic 已经有了合理的 Q 值估计。

```bash
# Lowdim: 前 100k 步纯 BC，之后加入 Q loss
python scripts/train_offline_rl.py task=square_lowdim \
    algorithm.bc_checkpoint=checkpoints/square_lowdim_mip_mlp/best_model.pkl \
    algorithm.bc_warmup_steps=100000 \
    algorithm.freeze_encoder=false

# Image: Push-T
python scripts/train_offline_rl.py task=pusht_image \
    algorithm.bc_checkpoint=checkpoints/pusht_image_mip_unet/best_model.pkl \
    algorithm.bc_warmup_steps=50000 \
    algorithm.freeze_encoder=false
```

#### 模式 2：直接 Offline RL（无 warmup）

从第 0 步开始 BC loss 和 Q loss 同时作用。

```bash
python scripts/train_offline_rl.py task=square_lowdim \
    algorithm.bc_checkpoint=checkpoints/square_lowdim_mip_mlp/best_model.pkl \
    algorithm.bc_warmup_steps=0
```

#### 关键配置项

配置文件：`configs/offline_rl/default.yaml`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bc_checkpoint` | null | BC checkpoint 路径（必填，用于读取网络架构） |
| `load_bc_weights` | false | 是否加载 BC 预训练权重初始化 encoder/flow |
| `freeze_encoder` | true | 是否冻结 encoder（false 时 encoder 参与 BC loss 更新） |
| `bc_warmup_steps` | 0 | BC warmup 步数（0=直接 RL，>0=先 BC 再 RL） |
| `q_weight` | 1.0 | Q loss 权重（warmup 阶段自动设为 0） |
| `alpha` | 1.0 | BC loss 权重 |
| `actor_lr` | 1e-4 | Flow policy 学习率（和 BC 一致） |
| `critic_lr` | 3e-4 | Critic 学习率 |
| `critic_hidden_dims` | [2048,2048,2048] | Critic 隐藏层（DSRL 风格） |
| `reward_offset` | -1.0 | 奖励偏移（DSRL 风格：{0,1} → {-1,0}） |
| `normalize_q_loss` | true | Q loss 归一化（MeanFlowQL 风格） |
| `gradient_steps` | 500000 | 总训练步数 |

#### 训练流程

```
dataset.sample_sequence() → {obs(B,obs_steps,D), actions(B,horizon,A), rewards, next_obs, masks}
                                         ↓
                   ┌──────────── total_loss ────────────┐
                   │                                     │
             critic_loss                            actor_loss
         Q(s,a) vs target_Q               q_weight * q_loss + alpha * bc_loss
                   │                                     │
          更新 critic 参数                         更新 flow 参数
                   │                             (encoder 可选冻结)
          target_critic EMA                     flow EMA (用于 eval)
```

#### 更多示例

```bash
# 加载 BC 预训练权重（而非随机初始化）
python scripts/train_offline_rl.py task=square_lowdim \
    algorithm.bc_checkpoint=path/to/bc.pkl \
    algorithm.load_bc_weights=true \
    algorithm.freeze_encoder=true

# 调整 loss 权重
python scripts/train_offline_rl.py task=square_lowdim \
    algorithm.bc_checkpoint=path/to/bc.pkl \
    algorithm.alpha=0.5 \
    algorithm.q_weight=2.0

# 禁用 W&B + 调整 eval 频率
python scripts/train_offline_rl.py task=square_lowdim \
    algorithm.bc_checkpoint=path/to/bc.pkl \
    algorithm.bc_warmup_steps=100000 \
    wandb.enabled=false \
    eval.eval_interval=50000
```

## Evaluation

```bash
# 评估 BC checkpoint
python scripts/eval_bc.py --checkpoint checkpoints/square_lowdim_meanflow_mlp/best_model.pkl

# 自定义评估参数
python scripts/eval_bc.py --checkpoint path/to/best_model.pkl --num_episodes 100

# 保存视频 + 渲染
python scripts/eval_bc.py --checkpoint path/to/best_model.pkl --save_video --render
```

## License

MIT
