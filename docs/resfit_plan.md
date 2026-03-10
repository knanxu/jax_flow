# ResFiT 实现计划

## 1. 项目概述

### 1.1 目标

在 jax_flow 框架中实现 ResFiT（Residual Fine-Tuning）算法，支持"先 BC 训练 → 再 RL 微调"的两阶段 pipeline。RL 阶段冻结 BC 策略，训练一个小的残差策略输出修正量，最终执行动作 = BC 动作 + 残差动作。

### 1.2 参考

- 论文：ResFiT (arXiv:2509.19301)
- 代码：https://github.com/amazon-far/residual-offpolicy-rl
- 核心思想：TD3 + RED-Q ensemble + 残差动作 + 50% online/offline 混合训练

### 1.3 设计决策（已确认）

| 决策项 | 选择 |
|--------|------|
| TrainState 管理 | 多 TrainState（encoder_state, actor_state, critic_state 各自独立 optimizer） |
| RL encoder 策略 | 可选：随机初始化新 encoder 或从 BC checkpoint 加载 encoder 权重 |
| 观测模式 | 同时支持 lowdim 和 image |
| Replay Buffer | 纯 NumPy 实现 |
| BC 策略集成 | 环境内嵌 BC（ResidualEnvWrapper） |
| Action Chunking | 残差也是 action chunk（与 BC 一致） |
| 数据增强 | 同时支持 RandomShiftsAug 和 CropRandomizer |
| 项目结构 | Agent 在 `jax_flow/agents/IL_RLFiT/`，其他组件按功能归类 |
| 离线数据 | BC 推理生成 base_action 填充 offline buffer |
| 复用组件 | ResidualActor, SpatialEmbCritic, Value, 所有 Encoder |

---

## 2. 整体架构

### 2.1 目录结构（新增/修改文件）

```
jax_flow/
├── agents/
│   └── IL_RLFiT/
│       ├── __init__.py
│       └── resfit_agent.py          # [新建] ResFiT Agent 核心
├── data/
│   └── replay_buffer.py             # [新建] NumPy replay buffer
├── envs/
│   └── residual_wrapper.py          # [新建] Residual env wrapper
├── networks/
│   ├── residual_actor.py            # [已有，需修改] 添加探索噪声支持
│   ├── spatial_emb_critic.py        # [已有，可直接复用]
│   ├── value.py                     # [已有，需小改] 支持 RED-Q num_q=10
│   └── encoders/
│       └── random_shifts.py         # [新建] RandomShiftsAug
├── core/
│   ├── checkpoint.py                # [需修改] 支持 ResFiT checkpoint
│   └── evaluation.py                # [需修改] 支持残差策略 eval
configs/
├── resfit/
│   ├── default.yaml                 # [新建] ResFiT 算法配置
│   └── task/                        # [新建] 各任务的 RL 微调配置
scripts/
└── train_resfit.py                  # [新建] ResFiT 训练脚本
```

### 2.2 数据流概览

```
Phase 1: BC 训练（已有）
  Dataset → BCAgent.update() → 保存 checkpoint

Phase 2: RL 微调（本次实现）
  加载 BC checkpoint → 冻结 BC 策略
                          ↓
  ResidualEnvWrapper(env, frozen_bc)
    env.reset() → obs → BC(obs) → base_action
    env.step(residual) → combined = base + residual → env 执行
                          ↓
  ResFiTAgent:
    encoder(obs) → features
    critic(features, combined_action) → Q values     [encoder + critic 更新]
    actor(stop_grad(features), base_action) → residual [actor 更新]
                          ↓
  Online buffer + Offline buffer → 50/50 混合采样
```

### 2.3 梯度流

```
Critic 更新（每步 4 次）:
  obs_images → encoder (可训练) → features
  next_obs_images → encoder (no grad) → next_features
  critic(features, state, action) → Q values
  critic_target(next_features, next_state, next_action) → target Q (no grad)
  Loss = MSE(Q, reward + gamma * target_Q)
  → encoder_opt.step() + critic_opt.step()
  → Polyak: critic → critic_target (tau=0.005)

Actor 更新（每 4 次 critic 更新做 1 次）:
  features = stop_gradient(features)  # 关键：encoder 不从 actor loss 获得梯度
  actor(features, base_action) → residual
  combined = clamp(base_action + residual, -1, 1)
  Q = mean(critic_ensemble(features, state, combined))  # 10 个 head 取 mean
  Loss = -Q.mean()
  → actor_opt.step() only
  → Polyak: actor → actor_target (tau=0.005)
```

---

## 3. 子任务拆分与实现顺序

按依赖关系排列，前面的任务是后面的前置条件：

```
Task A: Replay Buffer           ← 无依赖
Task B: RandomShiftsAug         ← 无依赖
Task C: ResidualActor 修改      ← 无依赖
Task D: Value critic 修改       ← 无依赖
Task E: Residual Env Wrapper    ← 依赖 BC checkpoint 加载能力
Task F: Checkpoint 扩展         ← 无依赖
Task G: ResFiT Agent            ← 依赖 A, B, C, D, E, F
Task H: Evaluation 扩展         ← 依赖 G
Task I: 训练脚本 + 配置         ← 依赖 G, H
Task J: 端到端验证              ← 依赖 I
```

---

## 4. 各组件详细设计

### 4.1 Task A: Replay Buffer (`jax_flow/data/replay_buffer.py`)

纯 NumPy 环形 buffer，支持 dict obs（image + lowdim）和 N-step returns。

#### 类设计

```python
class ReplayBuffer:
    """NumPy 环形 replay buffer，支持 dict 观测和 N-step returns。"""

    def __init__(
        self,
        capacity: int = 200_000,
        obs_shape: dict[str, tuple] | tuple = ...,  # {"image": (H,W,C), "lowdim": (d,)} 或 (obs_dim,)
        action_dim: int = ...,
        n_step: int = 3,
        gamma: float = 0.99,
    )

    def add(self, obs, action, reward, next_obs, done):
        """添加单个 transition。内部处理 N-step return 计算。"""

    def sample(self, batch_size: int, rng=None) -> dict:
        """均匀随机采样，返回 JAX-ready 的 dict。"""
        # 返回: {obs, action, reward, next_obs, done, discount}

    def __len__(self) -> int: ...
```

#### N-step Returns 实现

```
维护一个长度为 n_step 的临时 buffer：
  当 buffer 满时，计算：
    reward_n = r_0 + gamma*r_1 + gamma^2*r_2
    next_obs_n = obs at step n (or terminal obs)
    discount_n = gamma^n * (1 - done_any)
  写入主 buffer
  如果中途 done，立即 flush 剩余 transitions
```

#### Offline Buffer 填充

```python
class OfflineReplayBuffer(ReplayBuffer):
    """从 demo 数据集 + BC 推理填充的 offline buffer。"""

    @classmethod
    def from_dataset(cls, dataset, bc_agent, action_normalizer):
        """遍历 dataset 每个 episode：
        1. 对每个 obs 调用 bc_agent.eval_actions() 得到 base_action
        2. 存储 (obs, demo_action, reward=0, next_obs, done)
           其中 action = demo_action（ground truth combined action）
           base_action 存入 obs dict 供 actor 使用
        3. reward 设为 0（demo 数据没有 reward，或设为 sparse reward=1 at success）
        """
```

#### 关键细节

- 存储格式：obs 和 next_obs 支持 dict（image 存 uint8 节省内存，采样时转 float32）
- 采样输出：NumPy array，调用方负责转 JAX（避免 buffer 内部依赖 JAX）
- image obs 存储为 uint8 (H,W,C)，采样时归一化到 [0,1] float32
- base_action 作为 obs dict 的一个字段存储（与 ResFiT env wrapper 一致）

---

### 4.2 Task B: RandomShiftsAug (`jax_flow/networks/encoders/random_shifts.py`)

DrQ 风格的随机平移数据增强，用于 RL 训练中的 image 观测。

```python
class RandomShiftsAug:
    """JAX 实现的随机平移增强（DrQ 风格）。

    训练时：pad 图像 → 随机裁剪回原始大小（等效于随机平移 ±pad 像素）
    推理时：不做任何变换
    """

    def __init__(self, pad: int = 4): ...

    def __call__(self, images, rng, training=True):
        """
        Args:
            images: (batch, H, W, C) float32 [0, 1]
            rng: JAX PRNG key
            training: bool

        Returns:
            augmented images: (batch, H, W, C)
        """
        # if not training: return images
        # 1. jnp.pad(images, ((0,0),(pad,pad),(pad,pad),(0,0)), mode='constant')
        # 2. 对每个 batch 样本随机选择 crop 起点
        # 3. jax.lax.dynamic_slice 裁剪回 (H, W)
```

与现有 CropRandomizer 的区别：
- CropRandomizer：裁剪到更小尺寸（如 84→76），改变输出分辨率
- RandomShiftsAug：pad 后裁剪回原始尺寸，输出分辨率不变，只是平移

两者通过配置选择：`aug_type: "random_shifts" | "crop" | "none"`

---

### 4.3 Task C: ResidualActor 修改 (`jax_flow/networks/residual_actor.py`)

当前 ResidualActor 已基本完成，需要添加以下功能：

#### 需要添加的内容

1. 探索噪声支持（TD3 风格 TruncatedNormal）：

```python
class ResidualActor(nn.Module):
    # ... 现有属性不变 ...
    action_dim: int
    hidden_dim: int = 1024
    num_layers: int = 2
    action_scale: float = 0.1
    layer_norm: bool = True

    @nn.compact
    def __call__(self, features, base_action):
        # ... 现有实现不变，返回确定性残差动作 ...

# 在模块外添加探索噪声函数：
def add_exploration_noise(action, rng, stddev, clip=0.3):
    """TD3 风格 clipped Gaussian 探索噪声。

    Args:
        action: 确定性动作 (batch, action_dim)
        rng: JAX PRNG key
        stddev: 噪声标准差（ResFiT 默认 0.05）
        clip: 噪声裁剪范围（默认 0.3）

    Returns:
        noisy_action clipped to [-1, 1]
    """
    noise = jax.random.normal(rng, action.shape) * stddev
    noise = jnp.clip(noise, -clip, clip)
    return jnp.clip(action + noise, -1.0, 1.0)
```

2. 可选的 prop（proprioceptive state）输入：

```python
# image 模式下 actor 输入 = [spatial_features, prop, base_action]
# lowdim 模式下 actor 输入 = [encoded_obs, base_action]
# 当前实现已经通过 features 参数统一处理，不需要改接口
# 但需要确保 features 可以是 spatial embedding 后的结果
```

#### 不需要改动的部分

- 零初始化最后一层 ✓
- Tanh * action_scale ✓
- Dense → LayerNorm → ReLU 架构 ✓

---

### 4.4 Task D: Value Critic 修改 (`jax_flow/networks/value.py`)

当前 Value 类已支持 vmap ensemble，但默认 num_ensembles=2。需要：

1. 确保 num_ensembles=10 时正常工作（RED-Q 风格）
2. 添加 `redq_q_value` 和 `ensemble_mean_q` 工具函数（与 SpatialEmbCritic 一致）

```python
# 在 value.py 末尾添加：
def redq_q_value_from_value(q_values, rng, min_q_heads=2):
    """RED-Q for Value critic. 与 spatial_emb_critic.redq_q_value 相同逻辑。"""
    num_q = q_values.shape[0]
    indices = jax.random.choice(rng, num_q, shape=(min_q_heads,), replace=False)
    return jnp.min(q_values[indices], axis=0)

def ensemble_mean_q_from_value(q_values):
    """Mean Q for Value critic."""
    return jnp.mean(q_values, axis=0)
```

实际上 `spatial_emb_critic.py` 中的 `redq_q_value` 和 `ensemble_mean_q` 已经是通用的，
可以直接用于 Value 的输出（都是 `(num_q, batch)` 形状）。所以 Value 本身不需要改动，
只需在 agent 中统一调用 `redq_q_value` 即可。

**结论：Value 类无需修改，只需确认 num_ensembles=10 可正常初始化。**

---

### 4.5 Task E: Residual Environment Wrapper (`jax_flow/envs/residual_wrapper.py`)

将冻结的 BC 策略嵌入环境，RL agent 只看到残差动作空间。

#### 类设计

```python
class ResidualEnvWrapper(gym.Wrapper):
    """残差 RL 环境 wrapper。

    内嵌冻结 BC 策略。env.step() 接收残差动作，
    内部计算 combined = base_action + residual 后执行。

    观测 dict 中自动添加 'base_action' 字段供 actor 使用。
    """

    def __init__(
        self,
        env,                    # 已包装好的 robomimic env（含 FrameStack + ActionChunking）
        bc_agent,               # 冻结的 BCAgent 实例
        action_normalizer=None, # action normalizer（与 BC 训练一致）
    ):
        super().__init__(env)
        self.bc_agent = bc_agent
        self.action_normalizer = action_normalizer
        self._last_base_action = None  # (horizon, action_dim) 或 (action_dim,)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # 用 BC 策略计算初始 base_action
        self._update_base_action(obs)
        obs = self._augment_obs(obs)
        return obs, info

    def step(self, residual_action):
        """
        Args:
            residual_action: 残差动作。
                如果 ActionChunking 模式：(horizon, action_dim) 或 None（使用 buffer）
                如果单步模式：(action_dim,)

        处理逻辑：
            1. combined = base_action + residual_action（在 normalized [-1,1] 空间）
            2. combined = clip(combined, -1, 1)
            3. 传给底层 env 执行
            4. 用 BC 策略对 next_obs 计算新的 base_action
        """
        if residual_action is not None:
            combined = np.clip(self._last_base_action + residual_action, -1, 1)
        else:
            combined = None  # ActionChunking buffer 模式

        obs, reward, terminated, truncated, info = self.env.step(combined)

        # 记录 combined action 供 replay buffer 使用
        info["combined_action"] = combined
        info["base_action"] = self._last_base_action

        # 更新 base_action
        if self.env.needs_replan() if hasattr(self.env, 'needs_replan') else True:
            self._update_base_action(obs)

        obs = self._augment_obs(obs)
        return obs, reward, terminated, truncated, info

    def _update_base_action(self, obs):
        """调用冻结 BC 策略计算 base_action。"""
        # obs → batch 化 → bc_agent.eval_actions() → unbatch
        # 结果存入 self._last_base_action
        ...

    def _augment_obs(self, obs):
        """在 obs 中添加 base_action 字段。"""
        if isinstance(obs, dict):
            obs["base_action"] = self._last_base_action
        else:
            # lowdim: 返回 dict 包装
            obs = {"obs": obs, "base_action": self._last_base_action}
        return obs

    def needs_replan(self):
        """透传给底层 ActionChunkingWrapper。"""
        if hasattr(self.env, 'needs_replan'):
            return self.env.needs_replan()
        return True
```

#### Action Chunking 交互细节

```
BC 策略输出: base_actions = (horizon, action_dim)  # 如 (16, 7)
RL 残差输出: residual_actions = (horizon, action_dim)

combined_actions = clip(base_actions + residual_actions, -1, 1)

ActionChunkingWrapper 接收 combined_actions:
  - 存入 buffer
  - 每步取 buffer[idx] 执行
  - 执行 act_exec_steps 步后 needs_replan() = True
  - 此时 ResidualEnvWrapper 重新调用 BC 策略 + RL actor

时序：
  t=0: BC(obs_0) → base_0, Actor(obs_0, base_0) → res_0
       combined_0 = base_0 + res_0, 存入 ActionChunking buffer
  t=1..7: 从 buffer 取 combined[1..7] 执行（不调用 BC 和 Actor）
  t=8: needs_replan → BC(obs_8) → base_8, Actor(obs_8, base_8) → res_8
       ...
```

#### Lowdim 模式适配

lowdim 模式下 obs 原本是 ndarray `(obs_steps, obs_dim)`，wrapper 需要将其转为 dict：
```python
{"obs": obs_array, "base_action": base_action_array}
```
这样 agent 可以统一通过 `obs["base_action"]` 获取 base action。

---

### 4.6 Task F: Checkpoint 扩展 (`jax_flow/core/checkpoint.py`)

需要支持 ResFiT 的多 TrainState checkpoint。

#### 新增函数

```python
def save_resfit_checkpoint(
    checkpoint_path,
    agent,          # ResFiTAgent 实例
    step,
    normalizers=None,
    metadata=None,
):
    """保存 ResFiT checkpoint。

    保存内容：
    - encoder_params, encoder_opt_state
    - actor_params, actor_opt_state
    - critic_params, critic_opt_state
    - target_actor_params, target_critic_params
    - config
    - normalizers
    - bc_checkpoint_path（用于恢复时重新加载 BC 策略）
    """

def load_resfit_checkpoint(checkpoint_path):
    """加载 ResFiT checkpoint。"""

def restore_resfit_agent(checkpoint_path, bc_checkpoint_path, ex_obs, ex_actions):
    """从 checkpoint 恢复 ResFiT agent。
    1. 加载 BC checkpoint 重建冻结 BC 策略
    2. 加载 RL checkpoint 恢复 encoder/actor/critic 状态
    """
```

---

### 4.7 Task G: ResFiT Agent (`jax_flow/agents/IL_RLFiT/resfit_agent.py`)

核心 RL agent，管理 3 个独立 TrainState + 冻结 BC 策略。

#### 类设计

```python
class ResFiTAgent(flax.struct.PyTreeNode):
    """ResFiT: Residual Fine-Tuning Agent.

    3 个独立 TrainState（各自 optimizer）:
    - encoder_state: ViT/MLP encoder（只从 critic loss 更新）
    - critic_state: SpatialEmbCritic 或 Value（RED-Q ensemble）
    - actor_state: ResidualActor

    2 个 target 网络（无 optimizer，Polyak 更新）:
    - target_critic_params
    - target_actor_params
    """

    rng: Any
    encoder_state: TrainState
    critic_state: TrainState
    actor_state: TrainState
    target_critic_params: Any
    target_actor_params: Any
    config: Any = nonpytree_field()
```

#### create() 方法

```python
@classmethod
def create(
    cls,
    seed: int,
    ex_obs: dict,           # 示例观测（含 base_action）
    ex_actions,             # 示例动作 (batch, action_dim)
    config: dict,
    bc_checkpoint_path: str = None,  # 可选：从 BC encoder 初始化
):
    """创建 ResFiT agent。

    步骤：
    1. 创建 encoder（ViT 或 MLP，可选从 BC checkpoint 加载权重）
    2. 创建 critic（SpatialEmbCritic 或 Value，取决于 obs_type）
    3. 创建 actor（ResidualActor）
    4. 为每个组件创建独立 TrainState + optimizer
    5. 初始化 target 网络（深拷贝 critic 和 actor params）
    """

    # Encoder
    if config["obs_type"] == "image":
        encoder_def = MultiImageEncoder(
            image_backbone="vit",
            return_patches=True,  # image 模式返回 patch features
            ...
        )
    else:
        encoder_def = create_encoder(encoder_type="mlp", ...)

    # 可选：从 BC checkpoint 加载 encoder 权重
    if bc_checkpoint_path and config.get("init_encoder_from_bc", False):
        bc_ckpt = load_checkpoint(bc_checkpoint_path)
        # 提取 BC encoder params，用于初始化 RL encoder
        ...

    encoder_state = TrainState.create(
        encoder_def, encoder_params,
        tx=optax.adamw(lr=config["critic_lr"])  # encoder 用 critic_lr
    )

    # Critic
    if config["obs_type"] == "image":
        critic_def = SpatialEmbCritic(num_q=10, ...)
    else:
        critic_def = Value(num_ensembles=10, ...)

    critic_state = TrainState.create(
        critic_def, critic_params,
        tx=optax.adamw(lr=config["critic_lr"])
    )

    # Actor
    actor_def = ResidualActor(
        action_dim=action_dim,
        action_scale=config.get("action_scale", 0.1),
    )
    actor_state = TrainState.create(
        actor_def, actor_params,
        tx=optax.adamw(lr=config["actor_lr"])  # actor_lr 很小，默认 1e-6
    )

    # Target networks
    target_critic_params = critic_params  # 深拷贝
    target_actor_params = actor_params

    return cls(...)
```

#### update_critic() 方法

```python
@jax.jit
def update_critic(self, batch):
    """更新 encoder + critic。

    Args:
        batch: {obs, action, reward, next_obs, done, discount, base_action, next_base_action}

    Returns:
        (new_agent, info_dict)

    梯度流：
        encoder(obs) → features [可训练]
        encoder(next_obs) → next_features [stop_gradient]
        target_actor(next_features, next_base_action) → next_residual [no grad]
        next_combined = clip(next_base_action + next_residual, -1, 1)
        target_critic(next_features, next_combined) → target_Q [no grad]
        target_Q = reward + discount * redq_min(target_Q)  # min of random 2
        critic(features, combined_action) → Q_all  # 10 heads
        loss = mean((Q_all - target_Q)^2)
        → grad → encoder_state.apply_gradients + critic_state.apply_gradients
    """
    rng, noise_rng, redq_rng = jax.random.split(self.rng, 3)

    def critic_loss_fn(encoder_params, critic_params):
        # 1. Encode observations
        features = self.encoder_state(batch["obs"], params=encoder_params, training=True)
        next_features = jax.lax.stop_gradient(
            self.encoder_state(batch["next_obs"], params=encoder_params, training=True)
        )

        # 2. Target actor → next residual (no grad)
        next_residual = self.actor_state(
            next_features_flat, batch["next_base_action"],
            params=self.target_actor_params
        )
        next_combined = jnp.clip(batch["next_base_action"] + next_residual, -1, 1)

        # 3. Target Q
        target_q_all = self.critic_state(
            next_features, next_combined,
            params=self.target_critic_params
        )  # (num_q, batch)
        target_q = redq_q_value(target_q_all, redq_rng, min_q_heads=2)
        target_q = batch["reward"] + batch["discount"] * target_q
        target_q = jax.lax.stop_gradient(target_q)

        # 4. Current Q
        q_all = self.critic_state(features, batch["action"], params=critic_params)
        critic_loss = jnp.mean((q_all - target_q[None, :]) ** 2)

        return critic_loss, {"critic_loss": critic_loss, "q_mean": jnp.mean(q_all)}

    # 计算梯度（对 encoder 和 critic 参数）
    (loss, info), (enc_grads, crit_grads) = jax.value_and_grad(
        critic_loss_fn, argnums=(0, 1), has_aux=True
    )(self.encoder_state.params, self.critic_state.params)

    # 梯度裁剪
    enc_grads = optax.clip_by_global_norm(1.0).update(enc_grads, ...)[0]  # 简化示意
    crit_grads = optax.clip_by_global_norm(1.0).update(crit_grads, ...)[0]

    new_encoder = self.encoder_state.apply_gradients(enc_grads)
    new_critic = self.critic_state.apply_gradients(crit_grads)

    # Polyak update target critic
    new_target_critic = jax.tree.map(
        lambda p, tp: config["tau"] * p + (1 - config["tau"]) * tp,
        new_critic.params, self.target_critic_params
    )

    return self.replace(
        encoder_state=new_encoder,
        critic_state=new_critic,
        target_critic_params=new_target_critic,
        rng=rng,
    ), info
```

#### update_actor() 方法

```python
@jax.jit
def update_actor(self, batch):
    """更新 actor（encoder 不更新）。

    关键：features 经过 stop_gradient，encoder 不从 actor loss 获得梯度。

    Loss = -mean(ensemble_mean_Q(critic(stop_grad(features), combined)))
    """
    rng = ...

    def actor_loss_fn(actor_params):
        # features 已经 stop_gradient
        features = jax.lax.stop_gradient(
            self.encoder_state(batch["obs"], training=False)
        )

        residual = self.actor_state(
            features_flat, batch["base_action"],
            params=actor_params
        )
        combined = jnp.clip(batch["base_action"] + residual, -1, 1)

        q_all = self.critic_state(features, combined)  # (num_q, batch)
        q = ensemble_mean_q(q_all)  # mean over 10 heads

        actor_loss = -jnp.mean(q)
        return actor_loss, {"actor_loss": actor_loss}

    (loss, info), grads = jax.value_and_grad(
        actor_loss_fn, has_aux=True
    )(self.actor_state.params)

    new_actor = self.actor_state.apply_gradients(grads)

    # Polyak update target actor
    new_target_actor = jax.tree.map(
        lambda p, tp: config["tau"] * p + (1 - config["tau"]) * tp,
        new_actor.params, self.target_actor_params
    )

    return self.replace(
        actor_state=new_actor,
        target_actor_params=new_target_actor,
        rng=rng,
    ), info
```

#### sample_actions() 方法

```python
@jax.jit
def sample_actions(self, obs, base_action, rng, stddev=0.0):
    """采样动作（用于环境交互）。

    Args:
        obs: 观测（不含 base_action）
        base_action: BC 策略输出的 base action
        rng: PRNG key
        stddev: 探索噪声标准差（0 = 确定性）

    Returns:
        residual_action: 残差动作（传给 ResidualEnvWrapper）
    """
    features = self.encoder_state(obs, training=False)
    residual = self.actor_state(features_flat, base_action)

    if stddev > 0:
        residual = add_exploration_noise(residual, rng, stddev)

    return residual

@jax.jit
def eval_actions(self, obs, base_action):
    """确定性动作（用于评估）。"""
    return self.sample_actions(obs, base_action, rng=jax.random.PRNGKey(0), stddev=0.0)
```

#### Image vs Lowdim 模式差异

```
Image 模式:
  encoder = MultiImageEncoder(image_backbone="vit", return_patches=True)
  encoder(obs_dict) → (patch_features, prop_features)
  critic = SpatialEmbCritic(patch_features, action, prop_features)
  actor 输入 features = concat(spatial_emb(patch_features, prop), prop, base_action)
    → 实际上 actor 接收的 features 需要先经过 spatial embedding 或 flatten

Lowdim 模式:
  encoder = MLPEncoder 或 IdentityEncoder
  encoder(obs_array) → features (batch, feat_dim)
  critic = Value(features, action)  # concat(features, action) → MLP → Q
  actor 输入 features = encoded_obs
```

agent 内部通过 `config["obs_type"]` 分支处理这两种模式。

---

### 4.8 Task H: Evaluation 扩展 (`jax_flow/core/evaluation.py`)

需要支持 ResFiT agent 的评估（残差策略 + BC 策略组合）。

#### 修改方案

当前 `rollout_episode()` 调用 `agent.eval_actions(obs_batch)` 获取动作。
对于 ResFiT，环境已经是 ResidualEnvWrapper，所以：

```python
# 方案 1（推荐）：不改 evaluation.py，让 ResidualEnvWrapper 处理一切
# agent.eval_actions(obs_batch) 返回残差动作
# ResidualEnvWrapper.step(residual) 内部加上 base_action 执行
# → evaluation.py 完全不需要改动！

# 唯一需要确认的是 obs_batch 的格式：
# ResidualEnvWrapper 返回的 obs 包含 base_action 字段
# agent.eval_actions() 需要能处理这种 obs dict
```

**结论：evaluation.py 可能不需要修改。** 只要 ResFiTAgent.eval_actions() 接口与 BCAgent 一致
（接收 obs，返回 action sequence），现有 rollout_episode() 就能直接使用。

需要确认的边界情况：
- `needs_replan()` 透传是否正确
- obs dict 中 base_action 的 batch 化处理
- 视频录制是否正常工作

但 `eval_actions()` 的接口需要适配：当前 BCAgent 的签名是 `eval_actions(obs)`，
而 ResFiTAgent 需要从 obs dict 中提取 base_action。两种方案：

```python
# 方案 A：ResFiTAgent.eval_actions(obs_dict) 内部提取 base_action
def eval_actions(self, obs):
    base_action = obs.pop("base_action")  # 或 obs["base_action"]
    return self.sample_actions(obs_without_base, base_action, ...)

# 方案 B：让 evaluation.py 感知 agent 类型
# 不推荐，破坏统一接口
```

选择方案 A：ResFiTAgent.eval_actions() 接收完整 obs dict（含 base_action），
内部自行拆分。这样 evaluation.py 完全不需要改动。

---

### 4.9 Task I: 训练脚本 (`scripts/train_resfit.py`)

#### 训练流程（3 阶段）

```python
# ============================================================
# Phase 0: 初始化
# ============================================================
# 1. 加载 Hydra 配置
# 2. 加载 BC checkpoint → 创建冻结 BC agent
# 3. 创建环境 → 包装为 ResidualEnvWrapper
# 4. 加载 demo 数据集 → 用 BC 推理填充 offline buffer
# 5. 创建 ResFiT agent
# 6. 创建 online replay buffer（空）
# 7. 初始化 W&B

# ============================================================
# Phase 1: 随机探索 Warmup（填充 online buffer）
# ============================================================
# 持续 learning_starts 步（默认 10,000 env steps）
obs, info = env.reset()
while len(online_buffer) < config.learning_starts:
    # 残差 = 小随机噪声（base policy + noise 探索）
    residual = np.random.uniform(-1, 1, action_dim) * config.random_action_noise_scale  # 0.2
    next_obs, reward, done, truncated, info = env.step(residual)

    # 存储 combined action（env wrapper 内部计算的 base + residual）
    online_buffer.add(
        obs=obs,
        action=info["combined_action"],  # combined action in [-1, 1]
        reward=reward,
        next_obs=next_obs,
        done=done or truncated,
    )
    obs = next_obs if not (done or truncated) else env.reset()[0]

# ============================================================
# Phase 2: Critic-Only Warmup
# ============================================================
# 持续 critic_warmup_steps 步（默认 10,000 gradient steps）
# 只更新 encoder + critic，不更新 actor
for step in range(config.critic_warmup_steps):
    online_batch = online_buffer.sample(config.online_batch_size)   # 128
    offline_batch = offline_buffer.sample(config.offline_batch_size) # 128
    batch = merge_batches(online_batch, offline_batch)
    agent, info = agent.update_critic(batch)

    if step % log_interval == 0:
        log_metrics(info, step)

# ============================================================
# Phase 3: 正常训练
# ============================================================
# 持续 total_timesteps 步（默认 300,000 env steps）
global_step = 0
obs, info = env.reset()

while global_step < config.total_timesteps:
    # --- 1. 环境交互 ---
    stddev = linear_schedule(config.stddev_max, config.stddev_min, config.stddev_steps, global_step)
    residual = agent.sample_actions(obs_without_base, obs["base_action"], rng, stddev)

    # 可选：progressive clipping
    if config.progressive_clipping_steps > 0:
        scale = min(1.0, global_step / config.progressive_clipping_steps)
        residual = residual * scale

    next_obs, reward, done, truncated, info = env.step(residual)
    online_buffer.add(obs, info["combined_action"], reward, next_obs, done or truncated)
    obs = next_obs if not (done or truncated) else env.reset()[0]
    global_step += 1

    # --- 2. 梯度更新（UTD=4）---
    for i in range(config.utd_ratio):  # 默认 4
        online_batch = online_buffer.sample(config.online_batch_size)
        offline_batch = offline_buffer.sample(config.offline_batch_size)
        batch = merge_batches(online_batch, offline_batch)

        # Critic 更新（每次都做）
        agent, critic_info = agent.update_critic(batch)

        # Actor 更新（每 utd_ratio 次 critic 更新做 1 次）
        if (i + 1) % config.utd_ratio == 0:
            agent, actor_info = agent.update_actor(batch)

    # --- 3. 日志 & 评估 ---
    if global_step % config.eval_interval == 0:
        eval_results = evaluate_policy(agent, eval_env, ...)
        log_eval(eval_results)
        save_checkpoint_if_best(agent, eval_results)

    if global_step % config.log_interval == 0:
        log_metrics({**critic_info, **actor_info}, global_step)
```

#### Offline Buffer 填充细节

```python
def fill_offline_buffer(dataset, bc_agent, offline_buffer):
    """用 BC 推理填充 offline replay buffer。

    遍历 demo 数据集的每个 episode：
    1. 对每个 timestep 的 obs 调用 bc_agent.eval_actions() 得到 base_action
    2. 存储 transition:
       - obs: 原始观测 + base_action 字段
       - action: demo 的 ground truth action（作为 combined action）
       - reward: 0（demo 数据无 reward）或 sparse reward
       - next_obs: 下一步观测 + 下一步 base_action
       - done: episode 结束标志
    """
    for episode in dataset.episodes:
        obs_seq = episode["observations"]   # (T, obs_dim) 或 dict
        act_seq = episode["actions"]        # (T, action_dim)

        for t in range(len(obs_seq) - 1):
            # BC 推理得到 base_action
            obs_batch = obs_seq[t:t+1]  # batch dim
            base_actions = bc_agent.eval_actions(obs_batch)  # (1, horizon, action_dim)
            base_action = base_actions[0, 0]  # 取第一步

            obs_with_base = {**obs_seq[t], "base_action": base_action}
            next_base_actions = bc_agent.eval_actions(obs_seq[t+1:t+2])
            next_obs_with_base = {**obs_seq[t+1], "base_action": next_base_actions[0, 0]}

            offline_buffer.add(
                obs=obs_with_base,
                action=act_seq[t],          # GT combined action
                reward=0.0,                 # 或 sparse reward
                next_obs=next_obs_with_base,
                done=(t == len(obs_seq) - 2),
            )
```

---

### 4.10 Task I (续): 配置文件

#### `configs/resfit/default.yaml`

```yaml
# ResFiT 算法配置
algorithm:
  name: resfit

  # === 训练阶段 ===
  total_timesteps: 300000
  learning_starts: 10000        # Phase 1: 随机探索步数
  critic_warmup_steps: 10000    # Phase 2: critic-only 预热步数

  # === Optimizer ===
  actor_lr: 1.0e-6              # actor 学习率（很小！）
  critic_lr: 1.0e-4             # critic + encoder 学习率
  weight_decay: 0.0

  # === 残差动作 ===
  action_scale: 0.1             # 残差动作范围 [-0.1, 0.1]
  progressive_clipping_steps: 0 # 0 = 禁用渐进裁剪

  # === 探索噪声 ===
  stddev_max: 0.05              # 探索噪声标准差（恒定）
  stddev_min: 0.05
  stddev_steps: 300000
  stddev_clip: 0.3              # 噪声裁剪范围
  random_action_noise_scale: 0.2 # Phase 1 随机探索噪声

  # === Replay Buffer ===
  buffer_size: 200000
  batch_size: 256
  online_batch_size: 128        # 50% online
  offline_batch_size: 128       # 50% offline
  n_step: 3                     # N-step returns
  gamma: 0.99

  # === 更新频率 ===
  utd_ratio: 4                  # 每个 env step 做 4 次 critic 更新
  actor_update_cadence: 4       # 每 4 次 critic 更新做 1 次 actor 更新

  # === Target 网络 ===
  tau: 0.005                    # Polyak averaging 系数

  # === Critic ===
  num_q: 10                     # RED-Q ensemble 大小
  min_q_heads: 2                # target Q 随机选 2 个取 min
  critic_hidden_dim: 1024
  critic_num_layers: 2
  critic_layer_norm: true
  spatial_emb_dim: 1024         # SpatialEmbCritic trunk 维度
  grad_clip_norm: 1.0

  # === Actor ===
  actor_hidden_dim: 1024
  actor_num_layers: 2
  actor_layer_norm: true

  # === Encoder ===
  init_encoder_from_bc: false   # 是否从 BC checkpoint 初始化 encoder
  aug_type: random_shifts       # 数据增强: random_shifts / crop / none
  aug_pad: 4                    # RandomShiftsAug pad 像素

  # === 评估 ===
  eval_interval: 10000
  eval_episodes: 50
  eval_max_steps: 500

  # === Checkpoint ===
  checkpoint_interval: 50000
  keep_last_n: 3

# BC checkpoint 路径（必须指定）
bc_checkpoint: ???
```

#### `configs/resfit/task/square_image.yaml`（示例任务配置）

```yaml
# Square Image 任务的 ResFiT 微调配置
defaults:
  - /resfit/default

task:
  env_name: Square
  obs_type: image
  dataset_path: null  # 自动解析
  image_keys: [agentview_image, robot0_eye_in_hand_image]
  lowdim_keys: [robot0_eef_pos, robot0_gripper_qpos]
  horizon: 16
  obs_steps: 2
  act_steps: 8
  max_episode_steps: 400
  crop_shape: [76, 76]

network:
  image_backbone: vit
  vit_embed_dim: 128
  vit_num_heads: 4
  vit_depth: 1
```

---

## 5. 关键超参数对照表（ResFiT 论文 vs 本实现）

| 参数 | ResFiT 论文/代码 | 本实现默认值 | 说明 |
|------|-----------------|-------------|------|
| actor_lr | 1e-6 | 1e-6 | 残差 actor 学习率极小 |
| critic_lr | 1e-4 | 1e-4 | critic + encoder 共用 |
| tau | 0.005 | 0.005 | Polyak 系数 |
| action_scale | 0.1 | 0.1 | 残差范围 ±10% |
| last_layer_init | 0.0 | 0.0 (zeros) | actor 初始输出 ≈ 0 |
| num_q | 10 | 10 | RED-Q ensemble |
| min_q_heads | 2 | 2 | target Q 子采样 |
| policy_gradient_type | ensemble_mean | ensemble_mean | actor 用全部 head 均值 |
| UTD ratio | 4 | 4 | critic 更新频率 |
| actor_update_cadence | 4 | 4 | 每 4 次 critic 更新 1 次 actor |
| batch_size | 256 | 256 | 128 online + 128 offline |
| n_step | 3 | 3 | N-step returns |
| gamma | 0.99 | 0.99 | 折扣因子 |
| stddev | 0.05 (constant) | 0.05 | 探索噪声 |
| learning_starts | 10,000 | 10,000 | 随机探索步数 |
| critic_warmup | 10,000 | 10,000 | critic-only 预热 |
| total_timesteps | 300,000 | 300,000 | 总训练步数 |
| buffer_size | 200,000 | 200,000 | replay buffer 容量 |
| grad_clip_norm | 1.0 | 1.0 | 梯度裁剪 |
| aug | RandomShiftsAug(4) | RandomShiftsAug(4) | 图像增强 |
| encoder | MinViT(d=128,h=4,L=1) | MinViT(d=128,h=4,L=1) | 轻量 ViT |
| critic arch | SpatialEmbQEnsemble | SpatialEmbCritic | 共享 trunk + vmap heads |

---

## 6. 实现路线图

### Phase 1: 基础组件（无依赖，可并行）

| 子任务 | 文件 | 工作量 | 说明 |
|--------|------|--------|------|
| A. Replay Buffer | `jax_flow/data/replay_buffer.py` | 中 | 环形 buffer + N-step + dict obs |
| B. RandomShiftsAug | `jax_flow/networks/encoders/random_shifts.py` | 小 | JAX pad + dynamic_slice |
| C. ResidualActor 修改 | `jax_flow/networks/residual_actor.py` | 小 | 添加 `add_exploration_noise()` |
| D. Value 验证 | `jax_flow/networks/value.py` | 极小 | 确认 num_ensembles=10 可用 |

### Phase 2: 环境与 Checkpoint

| 子任务 | 文件 | 工作量 | 依赖 |
|--------|------|--------|------|
| E. Residual Env Wrapper | `jax_flow/envs/residual_wrapper.py` | 中 | BC checkpoint 加载 |
| F. Checkpoint 扩展 | `jax_flow/core/checkpoint.py` | 小 | 无 |

### Phase 3: 核心 Agent

| 子任务 | 文件 | 工作量 | 依赖 |
|--------|------|--------|------|
| G. ResFiT Agent | `jax_flow/agents/IL_RLFiT/resfit_agent.py` | 大 | A, B, C, D, E, F |

### Phase 4: 训练与验证

| 子任务 | 文件 | 工作量 | 依赖 |
|--------|------|--------|------|
| H. Evaluation 适配 | `jax_flow/core/evaluation.py` | 小 | G |
| I. 训练脚本 + 配置 | `scripts/train_resfit.py` + `configs/resfit/` | 中 | G, H |
| J. 端到端验证 | - | 中 | I |

### Phase 5: 后续扩展（可选）

- 支持 Prioritized Experience Replay (PER)
- 支持 HL-Gauss / C51 distributional critic loss
- 支持 actor spatial embedding（ResFiT 中 actor 也可用 SpatialEmb）
- 支持多环境并行采集（VecEnv）
- 离线到在线 RL（ACFQL）重构，复用 replay buffer 和 env wrapper

---

## 7. 与现有代码的兼容性

### 不修改的文件

| 文件 | 原因 |
|------|------|
| `jax_flow/agents/bc_agent.py` | BC pipeline 完全独立，不受影响 |
| `jax_flow/agents/train_state.py` | TrainState/ModuleDict 通用，ResFiT 直接复用 |
| `jax_flow/flow/` | Flow matching 组件只用于 BC，RL 不使用 |
| `jax_flow/data/robomimic_dataset.py` | 数据集加载不变，offline buffer 填充在训练脚本中 |
| `jax_flow/networks/encoders/vit.py` | ViT encoder 已完成，直接复用 |
| `jax_flow/networks/spatial_emb_critic.py` | SpatialEmbCritic 已完成，直接复用 |
| `scripts/train_bc.py` | BC 训练脚本不变 |

### 需要小改的文件

| 文件 | 改动 |
|------|------|
| `jax_flow/networks/residual_actor.py` | 添加 `add_exploration_noise()` 函数 |
| `jax_flow/core/checkpoint.py` | 添加 `save/load_resfit_checkpoint()` |
| `jax_flow/networks/__init__.py` | 导出 ResidualActor |
| `jax_flow/agents/__init__.py` | 导出 ResFiTAgent |

### 新建文件

| 文件 | 说明 |
|------|------|
| `jax_flow/agents/IL_RLFiT/__init__.py` | 包初始化 |
| `jax_flow/agents/IL_RLFiT/resfit_agent.py` | ResFiT Agent 核心 |
| `jax_flow/data/replay_buffer.py` | Replay Buffer |
| `jax_flow/envs/residual_wrapper.py` | Residual Env Wrapper |
| `jax_flow/networks/encoders/random_shifts.py` | RandomShiftsAug |
| `scripts/train_resfit.py` | 训练脚本 |
| `configs/resfit/default.yaml` | 算法配置 |
| `configs/resfit/task/*.yaml` | 任务配置 |

---

## 8. 风险与注意事项

### 8.1 JAX JIT 与多 TrainState

ResFiT 的 `update_critic()` 需要对 encoder_params 和 critic_params 同时求梯度。
在 JAX 中，`jax.value_and_grad(fn, argnums=(0, 1))` 可以对多个参数求梯度，
但需要确保 loss function 的签名正确，且两组参数的 pytree 结构兼容。

### 8.2 Image 模式下的 encoder 输出分支

Image 模式下 encoder 返回 `(patch_features, prop_features)` tuple，
需要在 agent 内部正确拆分并传给 critic（SpatialEmbCritic 接收 patches + action + prop）
和 actor（接收 flattened features + base_action）。

Lowdim 模式下 encoder 返回单个 vector，critic 是 Value（接收 concat(obs, action)），
actor 接收 encoded_obs + base_action。

### 8.3 Action Chunking 与 Replay Buffer

#### ResFiT 原版的做法

ResFiT 的 RL 阶段是**完全单步**的，没有 action chunking 的概念：

- BC 策略（ACT）内部维护一个 action queue（chunk_size=50），每次 `select_action()` 只弹出一个单步动作
- `BasePolicyVecEnvWrapper` 每个 env step 都调用一次 `select_action()`，拿到单步 base_action
- RL actor 输出单步残差 `(batch, action_dim)`
- `combined = clamp(base_action + residual, -1, 1)`，单步执行
- Replay buffer 存储单步 transition `(obs, combined_action, reward, next_obs, done)`

Action chunking 被完全封装在 BC 策略内部的 queue 机制里，对 RL 训练循环透明。

#### 我们的情况

我们的 BC 策略（flow matching）没有内部 queue 机制。当前的 action chunking 通过
`ActionChunkingWrapper` 在环境层面实现：BC 一次输出 `(horizon, action_dim)` 的 chunk，
wrapper 逐步执行前 `act_steps` 步。

#### 适配方案

模仿 ResFiT 的做法：**在 ResidualEnvWrapper 内部为 BC 策略实现 action queue**。

```
ResidualEnvWrapper 内部状态：
  _base_action_queue: deque  # BC 输出的 action chunk 队列
  _current_base_action: ndarray  # 当前步的 base action

reset():
  obs = env.reset()
  bc_actions = bc_agent.eval_actions(obs)  # (1, horizon, action_dim)
  _base_action_queue = deque(bc_actions[0])  # 将 chunk 拆为单步入队
  _current_base_action = _base_action_queue.popleft()
  obs["base_action"] = _current_base_action
  return obs

step(residual_action):
  # residual_action: (action_dim,) 单步
  combined = clip(_current_base_action + residual_action, -1, 1)
  next_obs, reward, done, truncated, info = env.step(combined)  # 单步执行

  # 更新 base action：从 queue 取下一个，queue 空了就重新调用 BC
  if len(_base_action_queue) == 0 or (act_steps 步已执行完):
      bc_actions = bc_agent.eval_actions(next_obs)
      _base_action_queue = deque(bc_actions[0][:act_steps])  # 只取前 act_steps 步
  _current_base_action = _base_action_queue.popleft()

  next_obs["base_action"] = _current_base_action
  info["combined_action"] = combined
  return next_obs, reward, done, truncated, info
```

这样：
- RL actor 只输出单步残差（与 ResFiT 一致）
- Replay buffer 存储单步 transition（与 ResFiT 一致）
- BC 的 action chunking 被封装在 wrapper 内部（与 ResFiT 一致）
- 不需要 ActionChunkingWrapper（ResidualEnvWrapper 自己管理 BC 的 chunk queue）

#### Wrapper 堆叠顺序调整

```
RL 阶段（不再需要 ActionChunkingWrapper）：
  RobomimicWrapper → FrameStackWrapper → ResidualEnvWrapper
                                          ↑ 内部管理 BC action queue

BC 阶段（不变）：
  RobomimicWrapper → FrameStackWrapper → ActionChunkingWrapper
```

#### 对 4.5 节 ResidualEnvWrapper 设计的更新

ResidualEnvWrapper 需要额外管理：
- `_base_action_queue: deque` — BC chunk 拆分后的单步 action 队列
- `_replan_interval: int` — 每隔多少步重新调用 BC（= act_steps）
- `_steps_since_replan: int` — 距上次 replan 的步数
- 不再需要外层 ActionChunkingWrapper

### 8.4 Lowdim 模式下的 obs dict 格式

当前 lowdim 环境返回 ndarray obs，但 ResidualEnvWrapper 需要添加 base_action 字段，
所以会将 obs 转为 dict `{"obs": array, "base_action": array}`。
这需要 ResFiT agent 能处理这种 dict 格式。

### 8.5 BC 策略的 Frame Stacking

BC 策略使用 frame stacking（obs_steps=2），ResidualEnvWrapper 中调用 BC 时
需要确保传入的 obs 已经经过 FrameStackWrapper 处理。
wrapper 的堆叠顺序应该是：
```
RobomimicWrapper → FrameStackWrapper → ActionChunkingWrapper → ResidualEnvWrapper
```
这样 ResidualEnvWrapper 拿到的 obs 已经是 frame-stacked 的。
