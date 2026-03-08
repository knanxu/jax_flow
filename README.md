# JAX Flow

基于 JAX 的机器人操作任务模仿学习框架，专注于 Flow Matching 策略和离线到在线强化学习。

## 特性

- 🚀 **高性能**: 基于 JAX，支持 JIT 编译和自动向量化
- 🎯 **Flow Matching**: 实现多种 flow matching 方法（Flow Matching, MIP, MeanFlow）
- 🏗️ **模块化设计**: 清晰的架构，易于扩展和定制
- 🤖 **机器人支持**: 完整集成 Robomimic 和 MimicGen 数据集和环境
- 🎨 **多种网络**: MLP、UNet、Transformer (DiT)
- 📊 **完整训练流程**: 训练、评估、checkpoint 管理、W&B 日志
- ⚙️ **配置驱动**: 使用 Hydra 进行灵活的配置管理

## 安装

### 基础安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/jax_flow.git
cd jax_flow

# 创建虚拟环境（推荐 Python 3.10+）
conda create -n jax_flow python=3.10
conda activate jax_flow

# 安装 JAX（根据你的硬件选择）
# CPU only
pip install jax[cpu]

# CUDA 12
pip install jax[cuda12]

# 安装项目
pip install -e .
```

### 开发安装

```bash
# 安装开发依赖（代码检查、测试等）
pip install -e .[dev]

# 安装 Robomimic 支持（必需）
pip install -e .[robomimic]
```

## 数据准备

项目支持两类数据源：**Robomimic**（5 个任务）和 **MimicGen**（9 个任务）。所有数据统一存放在 `~/.robomimic/` 目录下。

### 数据源对比

| 数据源 | 任务 | Lowdim | Image | 说明 |
|--------|------|--------|-------|------|
| Robomimic | lift, can, square, transport, tool_hang | 直接下载 | 需从 demo 生成 | robomimic 官方数据 |
| MimicGen | stack, stack_three, threading, coffee, kitchen, hammer_cleanup, mug_cleanup, pick_place, nut_assembly | 下载+转换 | 下载+转换 | HuggingFace 上的 MimicGen 数据 |

### 数据目录结构

```
~/.robomimic/
├── lift/ph/low_dim_v15.hdf5          # Robomimic lowdim
├── square/ph/low_dim_v15.hdf5
├── square/ph/image_v141.hdf5         # Robomimic image (从 demo 生成)
├── ...
└── mimicgen/core/
    ├── stack_d0.hdf5                 # MimicGen 原始文件 (image+lowdim)
    ├── stack/ph/
    │   ├── low_dim_v141.hdf5         # 转换后的 lowdim
    │   └── image_v141.hdf5           # symlink -> stack_d0.hdf5
    ├── coffee_d0.hdf5
    ├── coffee/ph/
    │   ├── low_dim_v141.hdf5
    │   └── image_v141.hdf5
    └── ...
```

### Robomimic 数据集

```bash
# 方法 1: 训练时自动下载 lowdim 数据集
python scripts/train_bc.py task=square_lowdim

# 方法 2: 手动下载
python scripts/download_data.py --task square --obs_type lowdim

# 生成 image 数据集（Robomimic 的 image 数据不能直接下载，需从 demo 生成）
python scripts/generate_image_dataset.py --task square --dataset_type ph
```

### MimicGen 数据集

MimicGen 数据在 HuggingFace 上以 `core/{task}_d0.hdf5` 格式存储，每个文件同时包含 image 和 lowdim 数据。需要先下载原始文件，再转换为项目使用的 lowdim/image 分离格式。

**完整流程（推荐）：**

```bash
# 一键下载全部 9 个 MimicGen 任务并转换
python scripts/download_mimicgen.py
```

**逐步操作：**

```bash
# 步骤 1: 下载原始文件（以 stack 为例，约 1.1GB）
cd ~/.robomimic/mimicgen/core
wget -c "https://huggingface.co/datasets/amandlek/mimicgen_datasets/resolve/main/core/stack_d0.hdf5"

# 步骤 2: 转换为 lowdim/image 格式
python scripts/convert_mimicgen.py --input ~/.robomimic/mimicgen/core/stack_d0.hdf5 --task stack

# 或批量转换所有已下载的文件
python scripts/convert_mimicgen.py --all

# 重新转换（如脚本更新后需要刷新）
python scripts/convert_mimicgen.py --all --force
```

**代理设置：** 如果使用代理下载，注意 `ALL_PROXY=socks://...` 会导致 huggingface_hub 报错，建议用 wget 下载或临时取消 `ALL_PROXY`：

```bash
ALL_PROXY= all_proxy= wget -c "https://huggingface.co/datasets/amandlek/mimicgen_datasets/resolve/main/core/stack_d0.hdf5"
```

**各任务原始文件大小参考：**

| 任务 | 大小 | 任务 | 大小 |
|------|------|------|------|
| stack | 1.1 GB | coffee | 1.9 GB |
| threading | 1.7 GB | stack_three | 2.4 GB |
| hammer_cleanup | 3.1 GB | mug_cleanup | 3.2 GB |
| nut_assembly | 3.4 GB | kitchen | 7.5 GB |
| pick_place | 9.0 GB | **总计** | **~33 GB** |

### MimicGen 环境依赖

MimicGen 特有的环境（Coffee, Kitchen, Threading 等）需要安装 mimicgen 包：

```bash
pip install "mimicgen @ git+https://github.com/NVlabs/mimicgen.git"
```

注意：部分 MimicGen 环境可能与当前 robosuite 版本（1.5.x）存在兼容性问题。基础环境（Stack, NutAssembly, PickPlace）可直接使用，框架会自动处理 MimicGen 数据中的环境名称映射（如 `Stack_D0` → `Stack`）和 controller 配置格式转换。

**环境可用性：**

| 环境评估可用 | 仅训练可用（环境评估需 mimicgen 兼容 robosuite 1.5） |
|-------------|----------------------------------------------|
| stack, nut_assembly, pick_place | stack_three, threading, coffee, kitchen, hammer_cleanup, mug_cleanup |

### 数据集路径解析优先级

`DatasetManager` 按以下顺序查找数据集：
1. 绝对路径（如果存在）
2. 项目根目录的相对路径
3. `~/.robomimic/mimicgen/core/{task}/ph/`（MimicGen 任务）
4. `~/.robomimic/{task}/ph/`（Robomimic 任务，v15 然后 v141）
5. 返回配置中的路径（可能不存在，触发自动下载）

## 快速开始

### 训练 BC 模型

```bash
# 使用默认配置（lift lowdim + MLP + MeanFlow）
python scripts/train_bc.py

# 训练 square 任务（lowdim）
python scripts/train_bc.py task=square_lowdim

# 训练 image 观测任务
python scripts/train_bc.py task=square_image

# 自定义超参数
python scripts/train_bc.py \
    task=square_lowdim \
    optimization.lr=3e-4 \
    optimization.batch_size=256 \
    optimization.gradient_steps=500000

# 禁用 W&B 日志
python scripts/train_bc.py task=square_lowdim wandb.enabled=false

# 调整评估频率
python scripts/train_bc.py task=square_lowdim eval.eval_interval=20000
```

**训练过程**:
- 自动下载缺失的数据集
- 每 10000 步进行一次环境评估（可配置）
- 每 10000 步保存 checkpoint
- 保存 best model（基于 validation loss）
- 上传评估视频到 W&B（如果启用）

### 评估模型

```bash
# 评估 best model
python scripts/eval_bc.py \
    --checkpoint checkpoints/square_lowdim_meanflow_mlp/best_model.pkl \
    --num_episodes 100

# 保存评估视频
python scripts/eval_bc.py \
    --checkpoint checkpoints/square_lowdim_meanflow_mlp/best_model.pkl \
    --save_video \
    --num_videos 5 \
    --output_dir eval_results/square

# 实时渲染
python scripts/eval_bc.py \
    --checkpoint checkpoints/square_lowdim_meanflow_mlp/best_model.pkl \
    --render \
    --num_episodes 10
```

## 项目结构

```
jax_flow/
├── jax_flow/                   # 核心代码
│   ├── agents/                 # Agent 实现
│   │   ├── bc_agent.py         # Behavior Cloning agent
│   │   ├── acfql_agent.py      # ACFQL agent (待完善)
│   │   └── train_state.py      # 自定义 TrainState
│   ├── core/                   # 核心工具
│   │   ├── checkpoint.py       # Checkpoint 保存/加载
│   │   ├── evaluation.py       # 策略评估
│   │   ├── types.py            # 类型定义
│   │   └── utils.py            # 工具函数
│   ├── data/                   # 数据加载
│   │   ├── robomimic_dataset.py        # Lowdim 数据集
│   │   ├── robomimic_image_dataset.py  # Image 数据集
│   │   ├── normalizer.py               # 数据归一化
│   │   └── dataset_manager.py          # 数据集管理和下载
│   ├── envs/                   # 环境包装器
│   │   └── robomimic_env.py    # Robomimic 环境 wrapper
│   ├── flow/                   # Flow matching 实现
│   │   ├── interpolant.py      # 插值方法
│   │   ├── losses.py           # Flow matching 损失
│   │   └── samplers.py         # ODE 求解器
│   └── networks/               # 神经网络
│       ├── mlp.py              # MLP 网络
│       ├── unet.py             # UNet 网络
│       ├── transformer.py      # Transformer 网络
│       ├── value.py            # Q-function ensemble
│       └── encoders/           # 观测编码器
│           ├── base.py         # 基础编码器
│           ├── resnet.py       # ResNet18 图像编码器
│           ├── spatial_softmax.py  # Spatial Softmax
│           └── multi_image.py  # 多相机图像编码器
├── configs/                    # Hydra 配置
│   ├── config.yaml             # 主配置
│   ├── task/                   # 任务配置
│   ├── network/                # 网络配置
│   ├── flow/                   # Flow 配置
│   └── optimization/           # 优化器配置
├── scripts/                    # 训练和工具脚本
│   ├── train_bc.py             # BC 训练脚本
│   ├── eval_bc.py              # 评估脚本
│   ├── download_data.py        # 数据下载
│   ├── download_all_datasets.py    # 批量下载
│   ├── generate_image_dataset.py   # 生成 image 数据集
│   └── create_all_configs.py   # 生成所有任务配置
└── tests/                      # 测试
```

## 配置系统

使用 Hydra 进行配置管理，支持命令行覆盖：

```bash
# 覆盖任务
python scripts/train_bc.py task=can_lowdim

# 覆盖网络
python scripts/train_bc.py network=unet

# 覆盖 flow 方法
python scripts/train_bc.py flow=mip

# 覆盖多个参数
python scripts/train_bc.py \
    task=lift_lowdim \
    network=mlp \
    flow=meanflow \
    optimization.lr=1e-4 \
    optimization.batch_size=256 \
    seed=42
```

### 主要配置文件

**config.yaml** - 主配置
```yaml
# 评估配置
eval:
  num_episodes: 50          # 评估 episode 数量
  eval_interval: 10000      # 评估间隔（训练步数）
  save_video: true          # 是否保存视频
  num_videos: 3             # 保存视频数量
  max_episode_steps: 500    # 最大 episode 长度

# Checkpoint 配置
checkpoint:
  save_freq: 10000          # 保存间隔
  keep_last_n: 3            # 保留最近 N 个 checkpoint
  save_best: true           # 保存 best model
  save_normalizers: true    # 保存 normalizer

# W&B 配置
wandb:
  enabled: true             # 启用 W&B
  log_videos: true          # 上传视频
  log_video_interval: 1     # 视频上传间隔
```

**task/*.yaml** - 任务配置
```yaml
name: square_lowdim
env_name: Square
obs_type: lowdim

dataset:
  path: ~/.robomimic/datasets/square/ph/low_dim.hdf5
  horizon: 16               # 动作预测长度
  obs_steps: 2              # 观测历史长度
  act_steps: 8              # 动作执行步数
```

## Flow Matching 方法

项目实现了多种 flow matching 方法：

### 1. Flow Matching (标准)
学习速度场 `v_t(x_t, obs)`，通过 ODE 求解生成动作。

```bash
python scripts/train_bc.py flow=flow_matching
```

### 2. MIP (Moment Interpolation Policy)
使用高阶矩插值，提升生成质量。

```bash
python scripts/train_bc.py flow=mip
```

### 3. MeanFlow (默认)
预测条件均值，支持一步生成（待完整实现 JVP）。

```bash
python scripts/train_bc.py flow=meanflow
```

## 网络架构

### MLP (默认)
简单高效的多层感知机，适合 lowdim 观测。

```bash
python scripts/train_bc.py network=mlp
```

### UNet
1D UNet，适合序列建模，参考 Diffusion Policy。

```bash
python scripts/train_bc.py network=unet
```

### Transformer
DiT 风格的 Transformer，适合长序列。

```bash
python scripts/train_bc.py network=transformer
```

## Checkpoint 管理

训练过程会自动保存 checkpoint：

```
checkpoints/square_lowdim_meanflow_mlp/
├── best_model.pkl              # 最佳模型（基于 val loss）
├── checkpoint_10000.pkl        # 定期保存
├── checkpoint_20000.pkl
└── checkpoint_30000.pkl
```

每个 checkpoint 包含：
- `params`: 网络参数
- `opt_state`: 优化器状态
- `config`: 训练配置（包含任务、网络、数据集路径等）
- `normalizers`: 数据归一化器
- `step`: 训练步数
- `best_val_loss`: 最佳验证损失

## W&B 日志

如果启用 W&B，会自动记录：

**训练指标**:
- `train/loss`: 训练损失
- `train/grad_norm`: 梯度范数
- `val/loss`: 验证损失

**评估指标**:
- `eval/success_rate`: 成功率
- `eval/avg_length`: 平均 episode 长度
- `eval/avg_return`: 平均回报
- `eval/video_*`: 评估视频

## 开发指南

### 代码风格

项目遵循 JAX 最佳实践：

```python
# 纯函数式编程
@jax.jit
def update(agent: BCAgent, batch: Batch) -> Tuple[BCAgent, InfoDict]:
    # 返回新的 agent，不修改输入
    new_agent, info = agent.update(batch)
    return new_agent, info

# 使用 PyTreeNode 管理状态
class BCAgent(flax.struct.PyTreeNode):
    rng: Any
    network: TrainState
    config: Any = nonpytree_field()
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_checkpoint_eval.py

# 带覆盖率
pytest --cov=jax_flow --cov-report=html
```

### 代码检查

```bash
# 格式化代码
ruff format .

# 检查代码质量
ruff check .

# 自动修复
ruff check --fix .

# 类型检查
pyright jax_flow/
```

## 常见问题

### Q: 如何调整评估频率？
A: 修改 `eval.eval_interval` 参数：
```bash
python scripts/train_bc.py task=square_lowdim eval.eval_interval=20000
```

### Q: 如何禁用视频保存（加速训练）？
A: 设置 `eval.save_video=false`：
```bash
python scripts/train_bc.py task=square_lowdim eval.save_video=false
```

### Q: 数据集下载失败怎么办？
A: 检查网络连接，或手动下载数据集到 `~/.robomimic/datasets/` 目录。

### Q: Checkpoint 文件太大怎么办？
A: 调整 `checkpoint.keep_last_n` 减少保存的 checkpoint 数量：
```bash
python scripts/train_bc.py task=square_lowdim checkpoint.keep_last_n=2
```

### Q: 如何使用 GPU？
A: 安装 CUDA 版本的 JAX：
```bash
pip install jax[cuda12]
```

## 性能优化建议

1. **减少评估频率**: 增大 `eval.eval_interval`（如 20000 或 50000）
2. **减少视频数量**: 设置 `eval.num_videos=1` 或 `eval.save_video=false`
3. **减少评估 episodes**: 设置 `eval.num_episodes=20`
4. **禁用 W&B**: 设置 `wandb.enabled=false`
5. **增大 batch size**: 充分利用 GPU 内存

## 路线图

**Phase 4: 验证与调试** (✅ 已完成)
- [x] 数据集管理系统（自动下载 + 路径解析）
- [x] Image 数据集生成脚本
- [x] Lowdim 端到端训练验证
- [x] Image 端到端训练验证
- [x] Evaluation script (rollout + success rate)
- [x] Checkpoint save/load
- [x] W&B logging 集成

**Phase 5: ACFQL + Extensions** (进行中)
- [ ] 完善 ACFQLAgent
- [ ] MeanFlow 完整实现 (JVP)
- [ ] 图像数据增强 (random crop, color jitter)
- [ ] MimicGen 数据集下载支持
- [ ] 多 GPU 训练支持

## 引用

如果你在研究中使用了本项目，请引用：

```bibtex
@software{jax_flow,
  title = {JAX Flow: Flow Matching for Robotic Manipulation},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/jax_flow}
}
```

## 许可证

MIT License

## 致谢

本项目参考了以下优秀工作：
- [much-ado-about-noising](https://github.com/...) - Flow matching 方法参考
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) - UNet 架构和 action chunking
- [qc](https://github.com/...) - ACFQL 算法和 agent 设计
- [jaxrl_m](https://github.com/dibyaghosh/jaxrl_m) - JAX RL 实现参考
- [Robomimic](https://robomimic.github.io/) - 数据集和环境
- [MimicGen](https://mimicgen.github.io/) - 数据生成和扩展
