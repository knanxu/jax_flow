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
python -m robomimic.scripts.download_datasets \
    --tasks square \
    --dataset_types ph \
    --hdf5_types raw

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
