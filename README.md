# JAX Flow

JAX-based flow matching framework for robotic manipulation — imitation learning and offline-to-online RL.

## Installation

Requires Python >= 3.10.

```bash
# Install JAX (choose one)
pip install jax[cpu]       # CPU
pip install jax[cuda12]    # CUDA 12

# Install project
pip install -e .

# Install robomimic environment support (required for training/eval)
pip install -e .[robomimic]
```

MimicGen 特有环境（coffee, kitchen, threading 等）需额外安装：

```bash
pip install "mimicgen @ git+https://github.com/NVlabs/mimicgen.git"
```

DexMimicGen 双臂任务需额外安装：

```bash
pip install -e /path/to/dexmimicgen
```

## Datasets

支持三类数据源，所有数据统一存放在 `~/.robomimic/` 或 `~/.dexmimicgen/` 下。

| 数据源 | 任务 | 说明 |
|--------|------|------|
| Robomimic | lift, can, square, transport, tool_hang | lowdim 可直接下载，image 需从 demo 生成 |
| MimicGen | stack, stack_three, threading, coffee, kitchen, hammer_cleanup, mug_cleanup, pick_place, nut_assembly | HuggingFace 下载后转换 |
| DexMimicGen | two_arm_threading, two_arm_three_piece_assembly, two_arm_transport | Panda 双臂，act_dim=14 |

### Robomimic

```bash
# 手动下载 lowdim
python scripts/download_data.py --task square --obs_type lowdim

# 生成 image 数据集（从 demo 渲染）
python scripts/generate_image_dataset.py --task square --dataset_type ph
```

训练时如果 lowdim 数据缺失会自动下载。

### MimicGen

```bash
# 一键下载全部 9 个任务并转换
python scripts/download_mimicgen.py

# 或手动单任务
wget -c "https://huggingface.co/datasets/amandlek/mimicgen_datasets/resolve/main/core/stack_d0.hdf5" \
     -P ~/.robomimic/mimicgen/core/
python scripts/convert_mimicgen.py --input ~/.robomimic/mimicgen/core/stack_d0.hdf5 --task stack
```

### DexMimicGen

```bash
# 一键下载全部 3 个双臂任务并转换
python scripts/download_dexmimicgen.py

# 下载指定任务
python scripts/download_dexmimicgen.py --tasks two_arm_threading
```

## Training

使用 Hydra 配置系统，通过命令行覆盖参数。

### Behavior Cloning

```bash
# 默认配置（lift lowdim + MLP + MeanFlow）
python scripts/train_bc.py

# 指定任务
python scripts/train_bc.py task=square_lowdim
python scripts/train_bc.py task=square_image
python scripts/train_bc.py task=dex_threading_lowdim

# 自定义参数
python scripts/train_bc.py task=square_lowdim optimization.lr=3e-4 optimization.batch_size=256

# 切换网络 / flow 方法
python scripts/train_bc.py network=unet flow=mip

# 禁用 W&B
python scripts/train_bc.py task=square_lowdim wandb.enabled=false
```

### ResFiT (Residual RL Fine-Tuning)

需要先训练好 BC checkpoint，再用 TD3 + RED-Q 做残差 RL 微调。

```bash
python scripts/train_resfit.py task=square_lowdim bc_checkpoint=path/to/best_model.pkl
```

### ACFQL / DQC (Offline-to-Online RL)

```bash
python scripts/train_acfql.py --config-name acfql_default task=square_lowdim
python scripts/train_dqc.py --config-name dqc_default task=square_lowdim
```

## Evaluation

```bash
python scripts/eval_bc.py --checkpoint path/to/best_model.pkl --num_episodes 50
python scripts/eval_bc.py --checkpoint path/to/best_model.pkl --save_video --render
```

## License

MIT
