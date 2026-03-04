# JAX Flow

基于 JAX 的机器人操作任务模仿学习框架，专注于 Flow Matching 方法（特别是 MeanFlow）。

## 特性

- 🚀 **高性能**: 基于 JAX，支持 JIT 编译和自动向量化
- 🎯 **MeanFlow**: 实现最新的 one-step flow matching 方法
- 🏗️ **模块化**: 清晰的架构设计，易于扩展
- 🤖 **机器人支持**: 集成 Robomimic 和 MimicGen 环境
- 🎨 **多种网络**: MLP、UNet、Transformer
- ⚙️ **配置驱动**: 使用 Hydra 进行灵活的配置管理

## 安装

### 基础安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/jax_flow.git
cd jax_flow

# 创建虚拟环境
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
# 安装开发依赖
pip install -e .[dev]

# 安装 Robomimic 支持
pip install -e .[robomimic]
```

## 快速开始

### 训练 MeanFlow 模型

```bash
# 使用默认配置（Lift 任务 + MLP + MeanFlow）
python scripts/train_bc.py

# 使用图像观察 + UNet
python scripts/train_bc.py task=lift_image network=unet

# 使用 Transformer + Rectified Flow
python scripts/train_bc.py network=transformer flow=rectified_flow

# 自定义超参数
python scripts/train_bc.py \
    task=lift_lowdim \
    flow=meanflow \
    network=mlp \
    optimization.lr=1e-4 \
    optimization.batch_size=256
```

### 评估模型

```bash
# 评估训练好的模型
python scripts/eval_policy.py --checkpoint_path=checkpoints/latest.pkl

# 可视化评估
python scripts/eval_policy.py \
    --checkpoint_path=checkpoints/latest.pkl \
    --render=True \
    --num_episodes=10
```

## 项目结构

```
jax_flow/
├── jax_flow/              # 核心代码
│   ├── core/              # 核心工具和类型
│   ├── flow/              # Flow matching 实现
│   │   ├── interpolants.py
│   │   ├── samplers.py
│   │   ├── losses.py
│   │   └── policies/      # Flow policies
│   ├── networks/          # 神经网络架构
│   │   ├── mlp.py
│   │   ├── unet.py
│   │   ├── transformer.py
│   │   └── encoders/
│   ├── agents/            # BC Agent
│   ├── datasets/          # 数据加载
│   ├── envs/              # 环境包装器
│   └── training/          # 训练循环
├── configs/               # Hydra 配置
├── scripts/               # 训练和评估脚本
└── tests/                 # 测试
```

## MeanFlow 简介

MeanFlow 是一种新型的 flow matching 方法，通过预测条件均值而非瞬时速度场，实现一步生成：

- **标准 Flow Matching**: 学习 `v_t(x_t, obs)`，需要多步 ODE 求解
- **MeanFlow**: 学习 `v̄_{s,t}(x_s, obs)`，可以一步到达目标

关键优势：
- ⚡ 推理速度快（一步生成）
- 🎯 训练稳定（自蒸馏约束）
- 📈 性能优秀（在多个任务上超越标准方法）

详见论文: "One-step flow map"

## 配置系统

使用 Hydra 进行配置管理，支持命令行覆盖：

```bash
# 基础配置
python scripts/train_bc.py

# 覆盖任务
python scripts/train_bc.py task=can_image

# 覆盖多个参数
python scripts/train_bc.py \
    task=lift_lowdim \
    network=mlp \
    flow=meanflow \
    optimization.lr=1e-4 \
    optimization.batch_size=256 \
    seed=42
```

配置文件位于 `configs/` 目录：
- `task/`: 任务和环境配置
- `network/`: 网络架构配置
- `flow/`: Flow policy 配置
- `optimization/`: 训练超参数

## 开发指南

### 代码风格

项目遵循 JAX 最佳实践：

```python
# 纯函数式编程
@jax.jit
def update(agent: BCAgent, batch: Batch) -> Tuple[BCAgent, InfoDict]:
    # 返回新的 agent，不修改输入
    ...
    return new_agent, info

# 使用 jaxtyping 进行类型标注
def forward(
    x: Float[Array, "batch horizon act_dim"],
    t: Float[Array, "batch"]
) -> Float[Array, "batch horizon act_dim"]:
    ...
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_flow_policies.py

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

## 性能优化

- ✅ JIT 编译关键函数
- ✅ 使用 `jax.vmap` 进行向量化
- ✅ 避免 Python 循环
- ✅ 合理的 batch size
- 🔄 多 GPU 支持（计划中）

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

以及 MeanFlow 论文：

```bibtex
@article{meanflow2024,
  title={One-step flow map},
  author={...},
  journal={...},
  year={2024}
}
```

## 许可证

MIT License

## 致谢

本项目参考了以下优秀工作：
- [much-ado-about-noising](https://github.com/...) - PyTorch 实现参考
- [Diffusion Policy](https://github.com/...) - UNet 架构参考
- [JAXRL](https://github.com/ikostrikov/jaxrl) - JAX 实现参考
- [RLax](https://github.com/deepmind/rlax) - JAX RL 工具
