"""Training script for ResFiT: Residual Fine-Tuning.

Two-phase pipeline: frozen BC policy + residual RL fine-tuning.

Usage:
    python scripts/train_resfit.py \
        bc_checkpoint=checkpoints/square_lowdim/best_model.pkl \
        task=square_lowdim

    python scripts/train_resfit.py \
        bc_checkpoint=checkpoints/square_image/best_model.pkl \
        task=square_image
"""

from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from jax_flow.agents.bc_agent import BCAgent
from jax_flow.agents.IL_RLFiT.resfit_agent import ResFiTAgent
from jax_flow.core.checkpoint import (
    load_checkpoint,
    save_resfit_checkpoint,
    cleanup_old_checkpoints,
)
from jax_flow.core.evaluation import evaluate_policy, print_evaluation_results
from jax_flow.data.resfit_replay_buffer import ResFiTReplayBuffer
from jax_flow.envs import make_robomimic_env
from jax_flow.envs.residual_wrapper import ResidualEnvWrapper


def linear_schedule(start: float, end: float, total_steps: int, step: int) -> float:
    """Linear interpolation from start to end over total_steps."""
    frac = min(step / max(total_steps, 1), 1.0)
    return start + (end - start) * frac


def merge_batches(online_batch: dict, offline_batch: dict) -> dict:
    """Merge online and offline batches by concatenation."""
    merged = {}
    for key in online_batch:
        if isinstance(online_batch[key], dict):
            merged[key] = {
                k: np.concatenate([online_batch[key][k], offline_batch[key][k]], axis=0)
                for k in online_batch[key]
            }
        else:
            merged[key] = np.concatenate(
                [online_batch[key], offline_batch[key]], axis=0
            )
    return merged


def create_bc_agent(bc_checkpoint_path: str, obs_type: str, config: dict):
    """Load frozen BC agent from checkpoint.

    Args:
        bc_checkpoint_path: Path to BC checkpoint.
        obs_type: 'lowdim' or 'image'.
        config: Task config for building example inputs.

    Returns:
        (bc_agent, bc_ckpt_dict, normalizers)
    """
    bc_ckpt = load_checkpoint(bc_checkpoint_path)
    bc_config = bc_ckpt["config"]

    # Build example inputs for agent reconstruction
    batch_size = 1
    obs_steps = bc_config.get("obs_steps", 2)
    horizon = bc_config.get("horizon", 16)
    action_dim = bc_config.get("action_dim", 7)
    obs_dim = bc_config.get("obs_dim", 20)

    if obs_type == "image":
        # Build example dict obs
        image_keys = bc_config.get("image_keys", ("agentview_image",))
        lowdim_keys = bc_config.get("lowdim_keys", ())
        ex_obs = {}
        for key in image_keys:
            ex_obs[key] = jnp.zeros((batch_size, obs_steps, 84, 84, 3))
        for key in lowdim_keys:
            ex_obs[key] = jnp.zeros((batch_size, obs_steps, 3))
    else:
        ex_obs = jnp.zeros((batch_size, obs_steps, obs_dim))

    ex_actions = jnp.zeros((batch_size, horizon, action_dim))

    # Recreate BC agent
    bc_agent = BCAgent.create(
        seed=0,
        ex_observations=ex_obs,
        ex_actions=ex_actions,
        config=bc_config,
    )

    # Restore saved params
    bc_agent = bc_agent.replace(
        rng=bc_ckpt["rng"],
        network=bc_agent.network.replace(
            params=bc_ckpt["params"],
            opt_state=bc_ckpt["opt_state"],
            step=bc_ckpt["step"],
        ),
    )

    normalizers = bc_ckpt.get("normalizers", {})
    return bc_agent, bc_ckpt, normalizers


def create_env(
    config: dict,
    normalizers: dict,
    bc_agent: BCAgent,
    dataset_path: str,
    seed: int = 0,
):
    """Create environment with residual wrapper.

    Stack: RobomimicWrapper → FrameStackWrapper → ResidualEnvWrapper
    (no ActionChunkingWrapper — BC chunking managed inside ResidualEnvWrapper)
    """
    obs_type = config.get("obs_type", "lowdim")
    is_image = obs_type == "image"

    env = make_robomimic_env(
        env_name=config["env_name"],
        dataset_path=dataset_path,
        obs_type=obs_type,
        obs_keys=tuple(config.get("obs_keys", [
            "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"
        ])),
        image_keys=tuple(config.get("image_keys", ())) if is_image else None,
        lowdim_keys=tuple(config.get("lowdim_keys", ())) if is_image else None,
        obs_normalizer=normalizers.get("obs"),
        action_normalizer=normalizers.get("action"),
        lowdim_normalizer=normalizers.get("lowdim"),
        max_episode_steps=config.get("max_episode_steps", 400),
        frame_stack=config.get("obs_steps", 2),
        act_exec_steps=None,  # No ActionChunkingWrapper
        seed=seed,
    )

    # Wrap with residual wrapper
    env = ResidualEnvWrapper(
        env=env,
        bc_agent=bc_agent,
        act_steps=config.get("act_steps", 8),
        obs_type=obs_type,
    )

    return env


def fill_offline_buffer(
    dataset_path: str,
    bc_agent: BCAgent,
    normalizers: dict,
    config: dict,
    buffer_capacity: int,
    n_step: int = 3,
    gamma: float = 0.99,
) -> ResFiTReplayBuffer:
    """Fill offline replay buffer from demo dataset + BC inference.

    Iterates through demo episodes, computes BC base_action for each timestep,
    and stores transitions with ground truth combined actions.
    """
    obs_type = config.get("obs_type", "lowdim")
    is_image = obs_type == "image"

    # Load dataset
    from jax_flow.data import make_robomimic_dataset

    dataset_kwargs = {}
    if is_image:
        dataset_kwargs["image_keys"] = tuple(config.get("image_keys", ()))
        dataset_kwargs["lowdim_keys"] = tuple(config.get("lowdim_keys", ()))

    dataset = make_robomimic_dataset(
        dataset_path=dataset_path,
        horizon=1,  # Single step for replay buffer
        obs_steps=config.get("obs_steps", 2),
        act_steps=1,
        obs_type=obs_type,
        val_ratio=0.0,
        mode="train",
        **dataset_kwargs,
    )

    action_dim = dataset.action_dim

    # Determine obs shape for buffer (without base_action)
    sample = dataset[0]
    sample_obs = sample["observations"]
    if isinstance(sample_obs, dict):
        obs_shape = {}
        for key, val in sample_obs.items():
            obs_shape[key] = val.shape
    else:
        obs_shape = {"obs": sample_obs.shape}

    # Create buffer
    buffer = ResFiTReplayBuffer(
        capacity=buffer_capacity,
        obs_shape=obs_shape,
        action_dim=action_dim,
        n_step=n_step,
        gamma=gamma,
    )

    print(f"Filling offline buffer from {len(dataset)} samples...")

    # Fill buffer
    num_added = 0
    for idx in range(len(dataset)):
        sample = dataset[idx]
        obs = sample["observations"]
        action = sample["actions"]  # (1, action_dim) or (action_dim,)
        if action.ndim > 1:
            action = action[0]

        # BC inference for base_action
        if isinstance(obs, dict):
            obs_batch = {k: jnp.array(v[np.newaxis, ...]) for k, v in obs.items()}
        else:
            obs_batch = jnp.array(obs[np.newaxis, ...])

        bc_actions = bc_agent.eval_actions(obs_batch)  # (1, horizon, action_dim)
        base_action = np.array(bc_actions[0, 0])  # First step

        # Build obs without base_action
        if isinstance(obs, dict):
            obs_clean = {k: np.array(v) for k, v in obs.items()}
        else:
            obs_clean = {"obs": np.array(obs)}

        # For next_obs, use same obs (single-step dataset doesn't have next)
        # This is approximate — offline buffer is mainly for critic warmup
        next_obs_clean = {k: v.copy() for k, v in obs_clean.items()}

        buffer._store_transition(
            obs=obs_clean,
            action=np.array(action),
            base_action=base_action,
            next_obs=next_obs_clean,
            next_base_action=base_action.copy(),
            reward=0.0,
            done=False,
            discount=gamma,
        )
        num_added += 1

        if (idx + 1) % 10000 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} samples")

    print(f"Offline buffer filled: {len(buffer)} transitions")
    return buffer


def _split_obs(obs):
    """Split observation into obs (without base_action) and base_action.

    Args:
        obs: Observation from ResidualEnvWrapper (dict with base_action key).

    Returns:
        (obs_clean, base_action): obs without base_action, and base_action array.
    """
    if isinstance(obs, dict):
        base_action = np.array(obs.get("base_action", np.zeros(1)))
        obs_clean = {k: np.array(v) for k, v in obs.items() if k != "base_action"}
        return obs_clean, base_action
    else:
        return np.array(obs), np.zeros(1)


def _to_jax_batch(batch: dict) -> dict:
    """Convert numpy batch to JAX arrays.

    ResFiTReplayBuffer already returns base_action/next_base_action as independent fields.
    """
    jax_batch = {}

    for key in ("obs", "next_obs"):
        if key in batch and isinstance(batch[key], dict):
            jax_batch[key] = {k: jnp.array(v) for k, v in batch[key].items()}
        elif key in batch:
            jax_batch[key] = jnp.array(batch[key])

    for key in ("action", "base_action", "next_base_action", "reward", "done", "discount"):
        if key in batch:
            jax_batch[key] = jnp.array(batch[key])

    return jax_batch


@hydra.main(version_base=None, config_path="../configs/resfit", config_name="default")
def main(cfg: DictConfig):
    """Main ResFiT training function."""

    print("=" * 80)
    print("ResFiT: Residual Fine-Tuning")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Resolve config
    algo = cfg.algorithm
    task_cfg = cfg.get("task", {})
    net_cfg = cfg.get("network", {})

    obs_type = task_cfg.get("obs_type", "lowdim")
    bc_checkpoint_path = cfg.bc_checkpoint

    # Set random seed
    seed = cfg.get("seed", 42)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    # Output directory (will be updated after BC config is loaded)
    output_dir = None

    # ============================================================
    # Phase 0: Initialization
    # ============================================================
    print("\n" + "=" * 80)
    print("Phase 0: Initialization")
    print("=" * 80)

    # Build unified config dict
    def _to_dict(c):
        if hasattr(c, 'items'):
            try:
                return dict(OmegaConf.to_container(c, resolve=True))
            except ValueError:
                return dict(c)
        return {}

    config = {
        **_to_dict(algo),
        **_to_dict(task_cfg),
        **_to_dict(net_cfg),
    }

    # 1. Load BC checkpoint
    print(f"\nLoading BC checkpoint: {bc_checkpoint_path}")
    bc_agent, bc_ckpt, normalizers = create_bc_agent(
        bc_checkpoint_path, obs_type, config
    )
    bc_config = bc_ckpt["config"]
    dataset_path = bc_config.get("dataset_path", "")

    # Merge BC config into config (BC config provides env/task info as defaults)
    for key in ("env_name", "obs_keys", "obs_type", "action_dim", "obs_dim",
                "horizon", "obs_steps", "image_keys", "lowdim_keys"):
        if key not in config and key in bc_config:
            config[key] = bc_config[key]
    # Derive act_steps from horizon if not set
    if "act_steps" not in config:
        config["act_steps"] = config.get("horizon", 16) // 2

    print(f"BC config: obs_type={bc_config.get('obs_type')}, "
          f"action_dim={bc_config.get('action_dim')}, "
          f"horizon={bc_config.get('horizon')}")

    # Set output directory now that we have env_name
    env_name = config.get("env_name", "unknown")
    output_dir = Path(cfg.get("checkpoint_dir", "checkpoints")) / f"{env_name}_{obs_type}_resfit"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # 2. Create environments
    print("\nCreating environments...")
    train_env = create_env(config, normalizers, bc_agent, dataset_path, seed=seed)
    eval_env = create_env(config, normalizers, bc_agent, dataset_path, seed=seed + 100)
    print("Environments created.")

    # Get action dim from env
    action_dim = train_env.action_space.shape[0]
    print(f"Action dim: {action_dim}")

    # 3. Fill offline buffer
    print("\nFilling offline replay buffer...")
    offline_buffer = fill_offline_buffer(
        dataset_path=dataset_path,
        bc_agent=bc_agent,
        normalizers=normalizers,
        config=config,
        buffer_capacity=algo.get("buffer_size", 200000),
        n_step=algo.get("n_step", 3),
        gamma=algo.get("gamma", 0.99),
    )

    # 4. Create online replay buffer
    # Get obs shape from env (obs includes base_action, we need to exclude it)
    obs_sample, _ = train_env.reset()
    if isinstance(obs_sample, dict):
        obs_shape = {k: np.array(v).shape for k, v in obs_sample.items() if k != "base_action"}
    else:
        obs_shape = np.array(obs_sample).shape

    online_buffer = ResFiTReplayBuffer(
        capacity=algo.get("buffer_size", 200000),
        obs_shape=obs_shape,
        action_dim=action_dim,
        n_step=algo.get("n_step", 3),
        gamma=algo.get("gamma", 0.99),
    )

    # 5. Create ResFiT agent
    print("\nCreating ResFiT agent...")
    # Build example obs for agent init
    if isinstance(obs_sample, dict):
        ex_obs = {k: jnp.array(np.array(v)[np.newaxis, ...]) for k, v in obs_sample.items()}
    else:
        ex_obs = {"obs": jnp.array(np.array(obs_sample)[np.newaxis, ...]),
                  "base_action": jnp.zeros((1, action_dim))}

    ex_actions = jnp.zeros((1, action_dim))

    agent = ResFiTAgent.create(
        seed=seed,
        ex_obs=ex_obs,
        ex_actions=ex_actions,
        config=config,
        bc_checkpoint_path=bc_checkpoint_path,
    )
    print("ResFiT agent created.")

    # 6. Initialize W&B
    wandb_enabled = cfg.get("wandb", {}).get("enabled", False)
    if wandb_enabled:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb.get("project", "jax_flow_resfit"),
                entity=cfg.wandb.get("entity", None),
                name=cfg.wandb.get("name", f"resfit_{env_name}_{obs_type}"),
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            print("W&B logging enabled")
        except ImportError:
            print("W&B not installed, skipping logging")
            wandb_enabled = False

    # ============================================================
    # Phase 1: Random Exploration Warmup
    # ============================================================
    learning_starts = algo.get("learning_starts", 10000)
    print(f"\n{'=' * 80}")
    print(f"Phase 1: Random Exploration ({learning_starts} steps)")
    print("=" * 80)

    obs, _ = train_env.reset()
    noise_scale = algo.get("random_action_noise_scale", 0.2)

    pbar = tqdm(total=learning_starts, desc="Exploration")
    while len(online_buffer) < learning_starts:
        # Random residual action
        residual = rng.uniform(-1, 1, size=(action_dim,)).astype(np.float32) * noise_scale

        next_obs, reward, terminated, truncated, info = train_env.step(residual)
        done = terminated or truncated

        # Extract base_action from obs/next_obs, store separately
        combined_action = info.get("combined_action", residual)
        cur_base_action = info.get("base_action", np.zeros(action_dim, dtype=np.float32))

        # Split obs: remove base_action for buffer storage
        obs_np, obs_base = _split_obs(obs)
        next_obs_np, next_obs_base = _split_obs(next_obs)

        online_buffer.add(
            obs=obs_np,
            action=np.array(combined_action),
            base_action=np.array(cur_base_action),
            next_obs=next_obs_np,
            next_base_action=np.array(next_obs_base),
            reward=float(reward),
            done=done,
        )

        if done:
            obs, _ = train_env.reset()
        else:
            obs = next_obs

        pbar.update(1)
    pbar.close()
    print(f"Online buffer size: {len(online_buffer)}")

    # ============================================================
    # Phase 2: Critic-Only Warmup
    # ============================================================
    critic_warmup_steps = algo.get("critic_warmup_steps", 10000)
    print(f"\n{'=' * 80}")
    print(f"Phase 2: Critic Warmup ({critic_warmup_steps} steps)")
    print("=" * 80)

    online_bs = algo.get("online_batch_size", 128)
    offline_bs = algo.get("offline_batch_size", 128)

    for step in tqdm(range(critic_warmup_steps), desc="Critic warmup"):
        online_batch = online_buffer.sample(online_bs, rng)
        offline_batch = offline_buffer.sample(offline_bs, rng)

        # Convert to JAX
        batch = merge_batches(online_batch, offline_batch)
        jax_batch = _to_jax_batch(batch)

        agent, info = agent.update_critic(jax_batch)

        if step % algo.get("log_interval", 1000) == 0 and step > 0:
            print(f"  Step {step}: critic_loss={float(info['critic_loss']):.4f}, "
                  f"q_mean={float(info['q_mean']):.4f}")
            if wandb_enabled:
                wandb.log({
                    "warmup/critic_loss": float(info["critic_loss"]),
                    "warmup/q_mean": float(info["q_mean"]),
                }, step=step)

    # ============================================================
    # Phase 3: Full Training
    # ============================================================
    total_timesteps = algo.get("total_timesteps", 300000)
    utd_ratio = algo.get("utd_ratio", 4)
    eval_interval = algo.get("eval_interval", 10000)
    log_interval = algo.get("log_interval", 1000)
    checkpoint_interval = algo.get("checkpoint_interval", 50000)

    print(f"\n{'=' * 80}")
    print(f"Phase 3: Full Training ({total_timesteps} env steps)")
    print("=" * 80)

    obs, _ = train_env.reset()
    global_step = 0
    best_success_rate = 0.0
    episode_reward = 0.0
    episode_length = 0
    num_episodes = 0

    pbar = tqdm(total=total_timesteps, desc="Training")

    while global_step < total_timesteps:
        # --- 1. Environment interaction ---
        stddev = linear_schedule(
            algo.get("stddev_max", 0.05),
            algo.get("stddev_min", 0.05),
            algo.get("stddev_steps", 300000),
            global_step,
        )

        # Get residual action from agent
        if isinstance(obs, dict):
            obs_batch = {k: jnp.array(np.array(v)[np.newaxis, ...]) for k, v in obs.items()}
        else:
            obs_batch = jnp.array(np.array(obs)[np.newaxis, ...])

        jax_rng = jax.random.PRNGKey(global_step)
        residual = agent.sample_actions(obs_batch, jax_rng, stddev=stddev)
        residual = np.array(residual[0])  # Unbatch

        # Progressive clipping
        prog_steps = algo.get("progressive_clipping_steps", 0)
        if prog_steps > 0:
            scale = min(1.0, global_step / prog_steps)
            residual = residual * scale

        next_obs, reward, terminated, truncated, info = train_env.step(residual)
        done = terminated or truncated

        combined_action = info.get("combined_action", residual)
        cur_base_action = info.get("base_action", np.zeros(action_dim, dtype=np.float32))

        # Store in online buffer (split obs: remove base_action)
        obs_np, _ = _split_obs(obs)
        next_obs_np, next_base = _split_obs(next_obs)

        online_buffer.add(
            obs=obs_np,
            action=np.array(combined_action),
            base_action=np.array(cur_base_action),
            next_obs=next_obs_np,
            next_base_action=np.array(next_base),
            reward=float(reward),
            done=done,
        )

        episode_reward += reward
        episode_length += 1

        if done:
            num_episodes += 1
            if wandb_enabled:
                wandb.log({
                    "train/episode_reward": episode_reward,
                    "train/episode_length": episode_length,
                    "train/num_episodes": num_episodes,
                }, step=global_step)
            episode_reward = 0.0
            episode_length = 0
            obs, _ = train_env.reset()
        else:
            obs = next_obs

        global_step += 1
        pbar.update(1)

        # --- 2. Gradient updates (UTD) ---
        actor_info = {}
        for i in range(utd_ratio):
            online_batch = online_buffer.sample(online_bs, rng)
            offline_batch = offline_buffer.sample(offline_bs, rng)
            batch = merge_batches(online_batch, offline_batch)
            jax_batch = _to_jax_batch(batch)

            # Critic update (every iteration)
            agent, critic_info = agent.update_critic(jax_batch)

            # Actor update (every utd_ratio iterations)
            if (i + 1) % utd_ratio == 0:
                agent, actor_info = agent.update_actor(jax_batch)

        # --- 3. Logging ---
        if global_step % log_interval == 0:
            log_dict = {
                "train/critic_loss": float(critic_info["critic_loss"]),
                "train/q_mean": float(critic_info["q_mean"]),
                "train/stddev": stddev,
                "train/global_step": global_step,
                "train/online_buffer_size": len(online_buffer),
            }
            if actor_info:
                log_dict["train/actor_loss"] = float(actor_info["actor_loss"])
                log_dict["train/actor_q_mean"] = float(actor_info["actor_q_mean"])

            pbar.set_postfix({
                "c_loss": f"{float(critic_info['critic_loss']):.3f}",
                "q": f"{float(critic_info['q_mean']):.2f}",
            })

            if wandb_enabled:
                wandb.log(log_dict, step=global_step)

        # --- 4. Evaluation ---
        if global_step % eval_interval == 0 and global_step > 0:
            print(f"\n[Step {global_step}] Evaluating...")
            eval_results = evaluate_policy(
                agent=agent,
                env=eval_env,
                num_episodes=algo.get("eval_episodes", 50),
                max_steps=algo.get("eval_max_steps", 500),
                save_video=wandb_enabled,
                num_videos=3,
                verbose=True,
            )
            print_evaluation_results(eval_results)

            if wandb_enabled:
                wandb.log({
                    "eval/success_rate": eval_results["success_rate"],
                    "eval/avg_length": eval_results["avg_length"],
                    "eval/avg_return": eval_results["avg_return"],
                }, step=global_step)

            # Save best model
            if eval_results["success_rate"] > best_success_rate:
                best_success_rate = eval_results["success_rate"]
                save_resfit_checkpoint(
                    checkpoint_path=output_dir / "best_model.pkl",
                    agent=agent,
                    step=global_step,
                    bc_checkpoint_path=bc_checkpoint_path,
                    normalizers=normalizers,
                    best_success_rate=best_success_rate,
                )
                print(f"New best model! Success rate: {best_success_rate:.2%}")

        # --- 5. Periodic checkpoint ---
        if global_step % checkpoint_interval == 0 and global_step > 0:
            save_resfit_checkpoint(
                checkpoint_path=output_dir / f"checkpoint_{global_step}.pkl",
                agent=agent,
                step=global_step,
                bc_checkpoint_path=bc_checkpoint_path,
                normalizers=normalizers,
            )
            cleanup_old_checkpoints(output_dir, keep_last_n=algo.get("keep_last_n", 3))

    pbar.close()

    # Final save
    save_resfit_checkpoint(
        checkpoint_path=output_dir / "final_model.pkl",
        agent=agent,
        step=global_step,
        bc_checkpoint_path=bc_checkpoint_path,
        normalizers=normalizers,
        best_success_rate=best_success_rate,
    )

    print(f"\nTraining complete! Best success rate: {best_success_rate:.2%}")
    print(f"Checkpoints saved to: {output_dir}")

    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
