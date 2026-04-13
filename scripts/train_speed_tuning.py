"""Training script for SpeedTuning: Rainbow DQN speed adjustment on frozen BC policy.

Algorithm:
  1. Load frozen BC policy from checkpoint
  2. Wrap env with SpeedTuningEnvWrapper (BC inference + interpolation + execution)
  3. Train Rainbow DQN to select discrete speed multipliers
  4. Each macro-step: speed policy picks v, BC predicts chunk, interpolate, execute k_skip steps

Usage:
    python scripts/train_speed_tuning.py task=square_lowdim \
        algorithm.bc_checkpoint=checkpoints/square_lowdim_meanflow_mlp/best_model.pkl

    python scripts/train_speed_tuning.py task=lift_image \
        algorithm.bc_checkpoint=checkpoints/lift_image_mip_unet/best_model.pkl \
        algorithm.max_speed=3.0 algorithm.speed_granularity=0.2
"""

from pathlib import Path

import hydra
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from jax_flow.agents.bc_agent import BCAgent
from jax_flow.agents.speed_tuning.interpolation import make_speed_options
from jax_flow.agents.speed_tuning.per_buffer import PrioritizedReplayBuffer
from jax_flow.agents.speed_tuning.rainbow_agent import RainbowDQNAgent
from jax_flow.agents.speed_tuning.speed_tuning_env import SpeedTuningEnvWrapper
from jax_flow.core.checkpoint import load_checkpoint
from jax_flow.data import DatasetManager, make_dataset, make_robomimic_dataset
from jax_flow.envs import make_env, make_robomimic_env


def create_base_env(cfg, normalizers, resolved_path):
    """Create base env with FrameStack only (no ActionChunking).

    SpeedTuningEnvWrapper manages action execution itself.
    """
    task_source = cfg.task.get("task_source", "robomimic")
    obs_type = cfg.task.obs_type
    is_image = obs_type == "image"
    abs_action = cfg.task.get("abs_action", False)

    common_kwargs = {
        "obs_normalizer": normalizers.get("obs"),
        "action_normalizer": normalizers.get("action"),
        "lowdim_normalizer": normalizers.get("lowdim"),
        "max_episode_steps": cfg.task.env.max_episode_steps,
        "frame_stack": cfg.task.dataset.obs_steps,
        "act_exec_steps": None,  # No ActionChunking — we handle it
        "seed": cfg.seed,
    }

    if task_source in ("pusht", "kitchen"):
        env = make_env(
            task_source=task_source,
            obs_type=obs_type if task_source == "pusht" else None,
            env_name=cfg.task.env_name if task_source == "kitchen" else None,
            **common_kwargs,
        )
    else:
        image_keys = None
        lowdim_keys = None
        if is_image:
            image_keys = tuple(cfg.task.dataset.get("image_keys", ["agentview_image"]))
            lowdim_keys = tuple(
                cfg.task.dataset.get(
                    "lowdim_keys",
                    ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                )
            )

        env = make_robomimic_env(
            env_name=cfg.task.env_name,
            dataset_path=resolved_path,
            obs_type=obs_type,
            obs_keys=tuple(
                cfg.task.dataset.get(
                    "obs_keys",
                    [
                        "robot0_eef_pos",
                        "robot0_eef_quat",
                        "robot0_gripper_qpos",
                        "object",
                    ],
                )
            ),
            image_keys=image_keys,
            lowdim_keys=lowdim_keys,
            render_offscreen=cfg.eval.get("save_video", True),
            abs_action=abs_action,
            **common_kwargs,
        )

    return env


def get_obs_shape(env):
    """Infer observation shape from environment for PER buffer."""
    obs, _ = env.reset()
    if isinstance(obs, dict):
        return {k: v.shape for k, v in obs.items()}
    return obs.shape


def evaluate_speed_policy(agent, eval_env, num_episodes=20, max_steps=500):
    """Evaluate the speed tuning policy.

    Key metrics (ordered by importance):
      1. success_rate: fraction of episodes that complete the task
      2. avg_length / avg_success_length: env steps to completion (lower = faster)
      3. avg_speed: mean speed multiplier chosen by the policy
      4. avg_return: cumulative r_ST (speed reward + task reward)
      5. speed distribution: per-episode speed stats

    Returns dict with all metrics.
    """
    successes = []
    lengths = []
    returns = []
    ep_speed_means = []
    ep_speed_maxs = []
    success_lengths = []
    failure_lengths = []

    for ep in range(num_episodes):
        obs, _ = eval_env.reset(seed=ep)
        done = False
        ep_return = 0.0
        ep_length = 0
        ep_success = False
        ep_speeds = []

        while not done and ep_length < max_steps:
            if isinstance(obs, dict):
                obs_batch = {k: np.array(v[np.newaxis, ...]) for k, v in obs.items()}
            else:
                obs_batch = obs[np.newaxis, ...]

            action = agent.eval_action(obs_batch)
            speed_idx = int(action[0])

            obs, reward, terminated, truncated, info = eval_env.step(speed_idx)
            done = terminated or truncated
            ep_return += reward
            ep_length += info.get("steps_executed", 1)
            ep_speeds.append(info.get("speed", 1.0))

            if "success" in info and bool(info["success"]):
                ep_success = True

        successes.append(ep_success)
        lengths.append(ep_length)
        returns.append(ep_return)
        ep_speed_means.append(float(np.mean(ep_speeds)) if ep_speeds else 1.0)
        ep_speed_maxs.append(float(np.max(ep_speeds)) if ep_speeds else 1.0)

        if ep_success:
            success_lengths.append(ep_length)
        else:
            failure_lengths.append(ep_length)

    results = {
        # Primary metrics
        "success_rate": float(np.mean(successes)),
        "num_successes": int(np.sum(successes)),
        "num_episodes": num_episodes,
        "avg_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        # Success/failure length breakdown
        "avg_success_length": float(np.mean(success_lengths))
        if success_lengths
        else 0.0,
        "avg_failure_length": float(np.mean(failure_lengths))
        if failure_lengths
        else 0.0,
        # Return
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        # Speed stats
        "avg_speed": float(np.mean(ep_speed_means)),
        "std_speed": float(np.std(ep_speed_means)),
        "max_speed_used": float(np.max(ep_speed_maxs)) if ep_speed_maxs else 1.0,
    }
    return results


@hydra.main(
    version_base=None,
    config_path="../configs/speed_tuning",
    config_name="default",
)
def main(cfg: DictConfig):
    """Main SpeedTuning training function."""
    print("=" * 80)
    print("SpeedTuning: Rainbow DQN Speed Adjustment")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    bc_checkpoint_path = cfg.algorithm.bc_checkpoint
    if bc_checkpoint_path is None:
        raise ValueError("algorithm.bc_checkpoint is required.")

    np.random.seed(cfg.seed)
    algo = cfg.algorithm

    # ================================================================
    # 1. Load BC checkpoint
    # ================================================================
    print("\nLoading BC checkpoint...")
    bc_ckpt = load_checkpoint(bc_checkpoint_path)
    bc_ckpt_config = bc_ckpt["config"]
    normalizers = bc_ckpt.get("normalizers", {})
    print(f"BC flow_type: {bc_ckpt_config.get('flow_type', 'unknown')}")
    print(f"BC network_type: {bc_ckpt_config.get('network_type', 'unknown')}")

    # ================================================================
    # 2. Resolve dataset path (needed for env creation)
    # ================================================================
    task_source = cfg.task.get("task_source", "robomimic")
    dataset_path = bc_ckpt_config.get("dataset_path", cfg.task.dataset.path)
    task_name = cfg.task.get("dataset_name", cfg.task.env_name.lower())
    dataset_type = cfg.task.dataset.get("dataset_type", "ph")
    obs_type = cfg.task.obs_type

    resolved_path = DatasetManager.ensure_dataset(
        dataset_path=dataset_path,
        task=task_name,
        dataset_type=dataset_type,
        obs_type=obs_type,
        source=task_source,
        auto_download=True,
    )
    if resolved_path is None:
        print("\nFailed to load dataset. Exiting.")
        return
    resolved_path = str(resolved_path)

    # ================================================================
    # 3. Create frozen BC agent
    # ================================================================
    print("\nCreating frozen BC agent...")
    # Need example data to create BCAgent
    is_image = obs_type == "image"
    dataset_kwargs = {}
    abs_action = cfg.task.get("abs_action", False)
    if abs_action:
        dataset_kwargs["abs_action"] = True
    if is_image:
        dataset_kwargs["image_keys"] = tuple(
            cfg.task.dataset.get("image_keys", ["agentview_image"])
        )
        dataset_kwargs["lowdim_keys"] = tuple(
            cfg.task.dataset.get(
                "lowdim_keys",
                ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
            )
        )
    else:
        if "obs_keys" in cfg.task:
            dataset_kwargs["obs_keys"] = tuple(cfg.task.obs_keys)

    common_ds_kwargs = {
        "horizon": cfg.task.dataset.horizon,
        "obs_steps": cfg.task.dataset.obs_steps,
        "act_steps": cfg.task.dataset.act_steps,
        "val_ratio": 0.0,
    }

    if task_source in ("pusht", "kitchen"):
        train_dataset = make_dataset(
            task_source=task_source,
            dataset_path=resolved_path,
            obs_type=obs_type,
            mode="train",
            **common_ds_kwargs,
        )
    else:
        train_dataset = make_robomimic_dataset(
            dataset_path=resolved_path,
            obs_type=obs_type,
            mode="train",
            **common_ds_kwargs,
            **dataset_kwargs,
        )

    # Get example observations
    ex_sample = train_dataset[0]
    if isinstance(ex_sample["observations"], dict):
        ex_obs = {
            k: jnp.array(v[np.newaxis, ...])
            for k, v in ex_sample["observations"].items()
        }
    else:
        ex_obs = jnp.array(ex_sample["observations"][np.newaxis, ...])
    ex_actions = jnp.array(ex_sample["actions"][np.newaxis, ...])

    # Create and restore BC agent
    bc_agent = BCAgent.create(
        seed=cfg.seed,
        ex_observations=ex_obs,
        ex_actions=ex_actions,
        config=dict(bc_ckpt_config),
    )
    # Restore weights
    bc_agent = bc_agent.replace(
        network=bc_agent.network.replace(
            params=bc_ckpt["params"],
            opt_state=bc_agent.network.opt_state,
            step=bc_ckpt.get("step", 1),
        ),
        ema_params=bc_ckpt.get("ema_params", bc_ckpt["params"]),
    )
    print("BC agent restored.")

    # ================================================================
    # 4. Create environments
    # ================================================================
    print("\nCreating environments...")
    speed_options = make_speed_options(
        max_speed=algo.get("max_speed", 2.0),
        granularity=algo.get("speed_granularity", 0.1),
    )
    num_actions = len(speed_options)
    print(f"Speed options ({num_actions}): {speed_options}")

    abs_action = cfg.task.get("abs_action", False)

    train_env = create_base_env(cfg, normalizers, resolved_path)
    train_env = SpeedTuningEnvWrapper(
        env=train_env,
        bc_agent=bc_agent,
        speed_options=speed_options,
        alpha=algo.get("alpha", 0.1),
        beta=algo.get("beta", 2.0),
        k_skip=algo.get("k_skip", 4),
        abs_action=abs_action,
    )

    eval_env = create_base_env(cfg, normalizers, resolved_path)
    eval_env = SpeedTuningEnvWrapper(
        env=eval_env,
        bc_agent=bc_agent,
        speed_options=speed_options,
        alpha=algo.get("alpha", 0.1),
        beta=algo.get("beta", 2.0),
        k_skip=algo.get("k_skip", 4),
        abs_action=abs_action,
    )

    # ================================================================
    # 5. Create Rainbow DQN agent
    # ================================================================
    print("\nCreating Rainbow DQN agent...")
    obs_shape = get_obs_shape(train_env)

    # Get example obs from env for agent init
    env_obs, _ = train_env.reset()
    if isinstance(env_obs, dict):
        env_obs_batch = {k: jnp.array(v[np.newaxis, ...]) for k, v in env_obs.items()}
    else:
        env_obs_batch = jnp.array(env_obs[np.newaxis, ...])

    agent_config = {
        "num_actions": num_actions,
        "num_atoms": algo.get("num_atoms", 51),
        "v_min": algo.get("v_min", -10.0),
        "v_max": algo.get("v_max", 10.0),
        "hidden_dims": list(algo.get("hidden_dims", [512, 512])),
        "activation": algo.get("activation", "relu"),
        "layer_norm": algo.get("layer_norm", True),
        "lr": algo.get("lr", 3e-4),
        "discount": algo.get("discount", 0.99),
        "tau": algo.get("tau", 0.005),
        "grad_clip_norm": algo.get("grad_clip_norm", 10.0),
        "freeze_encoder": algo.get("freeze_encoder", True),
    }

    agent = RainbowDQNAgent.create(
        seed=cfg.seed,
        ex_observations=env_obs_batch,
        config=agent_config,
        bc_checkpoint_path=bc_checkpoint_path,
    )
    print("Rainbow DQN agent created.")

    # ================================================================
    # 6. Create PER buffer
    # ================================================================
    per_buffer = PrioritizedReplayBuffer(
        capacity=algo.get("buffer_capacity", 100_000),
        obs_shape=obs_shape,
        per_alpha=algo.get("per_alpha", 0.6),
    )

    # ================================================================
    # 7. Setup logging and checkpointing
    # ================================================================
    bc_flow_type = bc_ckpt_config.get("flow_type", "flow_matching")
    bc_network_type = bc_ckpt_config.get("network_type", "mlp")
    experiment_name = f"speed_tuning_{cfg.task.name}_{bc_flow_type}_{bc_network_type}"
    output_dir = Path(cfg.get("checkpoint_dir", "checkpoints")) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    wandb_enabled = cfg.get("wandb", {}).get("enabled", False)
    if wandb_enabled:
        try:
            import wandb

            wandb.init(
                project=cfg.wandb.get("project", "jax_flow"),
                entity=cfg.wandb.get("entity", None),
                name=f"{experiment_name}_seed{cfg.seed}",
                config=dict(OmegaConf.to_container(cfg, resolve=True)),
            )
        except ImportError:
            wandb_enabled = False

    # ================================================================
    # 8. Training loop
    # ================================================================
    total_episodes = algo.get("total_episodes", 5000)
    batch_size = algo.get("batch_size", 256)
    learning_starts = algo.get("learning_starts", 1000)
    train_freq = algo.get("train_freq", 4)
    epsilon_start = algo.get("epsilon_start", 1.0)
    epsilon_end = algo.get("epsilon_end", 0.01)
    epsilon_decay_steps = algo.get("epsilon_decay_steps", 50000)
    per_beta_start = algo.get("per_beta_start", 0.4)
    per_beta_end = algo.get("per_beta_end", 1.0)
    log_interval = algo.get("log_interval", 100)
    eval_interval = cfg.eval.get("eval_interval", 200)
    checkpoint_interval = cfg.checkpoint.get("save_freq", 500)
    max_steps = cfg.task.env.max_episode_steps

    best_success_rate = -1.0
    global_step = 0  # total env steps (across all episodes)

    print(f"\nStarting training for {total_episodes} episodes...")
    print(f"Epsilon: {epsilon_start} -> {epsilon_end} over {epsilon_decay_steps} steps")
    print(f"PER beta: {per_beta_start} -> {per_beta_end}")
    print(f"Learning starts after {learning_starts} transitions")

    episode_returns = []
    episode_lengths = []
    episode_successes = []
    episode_speeds = []

    for episode in range(total_episodes):
        obs, _ = train_env.reset(seed=episode)
        done = False
        ep_return = 0.0
        ep_length = 0
        ep_speeds = []

        while not done and ep_length < max_steps:
            # Epsilon schedule
            epsilon = max(
                epsilon_end,
                epsilon_start
                - (epsilon_start - epsilon_end) * global_step / epsilon_decay_steps,
            )

            # Select speed
            if isinstance(obs, dict):
                obs_batch = {k: jnp.array(v[np.newaxis, ...]) for k, v in obs.items()}
            else:
                obs_batch = jnp.array(obs[np.newaxis, ...])

            agent, action = agent.select_action(obs_batch, epsilon)
            speed_idx = int(action[0])
            ep_speeds.append(speed_options[speed_idx])

            # Execute macro-step and collect per-step transitions
            next_obs, reward, terminated, truncated, info, transitions = (
                train_env.step_with_transitions(speed_idx)
            )
            done = terminated or truncated
            ep_return += reward
            ep_length += info.get("steps_executed", 1)
            global_step += info.get("steps_executed", 1)

            # Store transitions in PER buffer
            for trans in transitions:
                per_buffer.add(
                    obs=trans["obs"],
                    action=trans["action"],
                    reward=trans["reward"],
                    next_obs=trans["next_obs"],
                    done=trans["done"],
                )

            # Train
            if len(per_buffer) >= learning_starts and global_step % train_freq == 0:
                beta = min(
                    per_beta_end,
                    per_beta_start
                    + (per_beta_end - per_beta_start)
                    * global_step
                    / epsilon_decay_steps,
                )
                batch, is_weights, indices = per_buffer.sample(batch_size, beta)

                # Convert to JAX arrays
                if isinstance(batch["observations"], dict):
                    jax_obs = {
                        k: jnp.array(v) for k, v in batch["observations"].items()
                    }
                    jax_next_obs = {
                        k: jnp.array(v) for k, v in batch["next_observations"].items()
                    }
                else:
                    jax_obs = jnp.array(batch["observations"])
                    jax_next_obs = jnp.array(batch["next_observations"])

                jax_batch = {
                    "observations": jax_obs,
                    "actions": jnp.array(batch["actions"]),
                    "rewards": jnp.array(batch["rewards"]),
                    "next_observations": jax_next_obs,
                    "dones": jnp.array(batch["dones"]),
                }
                jax_is_weights = jnp.array(is_weights)

                agent, _train_info, td_errors = agent.update(jax_batch, jax_is_weights)
                per_buffer.update_priorities(indices, np.array(td_errors))

            obs = next_obs

        # Episode stats
        ep_success = (
            any(
                t.get("done", False)
                for t in transitions
                if not (terminated and not truncated)  # not a timeout
            )
            if transitions
            else False
        )
        # Check info for success flag
        ep_success = bool(info.get("success", ep_success))

        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        episode_successes.append(ep_success)
        episode_speeds.append(np.mean(ep_speeds) if ep_speeds else 1.0)

        # Logging
        if (episode + 1) % log_interval == 0:
            recent = min(log_interval, len(episode_returns))
            avg_ret = np.mean(episode_returns[-recent:])
            avg_len = np.mean(episode_lengths[-recent:])
            avg_sr = np.mean(episode_successes[-recent:])
            avg_spd = np.mean(episode_speeds[-recent:])
            print(
                f"Episode {episode + 1}/{total_episodes} | "
                f"Return: {avg_ret:.2f} | Length: {avg_len:.0f} | "
                f"Success: {avg_sr:.2%} | Speed: {avg_spd:.2f} | "
                f"Epsilon: {epsilon:.3f} | Buffer: {len(per_buffer)}"
            )
            if wandb_enabled:
                import wandb

                wandb.log(
                    {
                        "train/episode_return": avg_ret,
                        "train/episode_length": avg_len,
                        "train/success_rate": avg_sr,
                        "train/avg_speed": avg_spd,
                        "train/epsilon": epsilon,
                        "train/buffer_size": len(per_buffer),
                        "train/global_step": global_step,
                    },
                    step=episode,
                )

        # Evaluation
        if eval_interval > 0 and (episode + 1) % eval_interval == 0:
            print(f"\n[Episode {episode + 1}] Evaluating...")
            eval_results = evaluate_speed_policy(
                agent=agent,
                eval_env=eval_env,
                num_episodes=cfg.eval.get("num_episodes", 20),
                max_steps=max_steps,
            )
            print(
                f"  Success: {eval_results['success_rate']:.2%} "
                f"({eval_results['num_successes']}/{eval_results['num_episodes']}) | "
                f"Length: {eval_results['avg_length']:.0f} +/- {eval_results['std_length']:.0f} | "
                f"Success Length: {eval_results['avg_success_length']:.0f} | "
                f"Speed: {eval_results['avg_speed']:.2f} +/- {eval_results['std_speed']:.2f}"
            )

            if wandb_enabled:
                wandb.log(
                    {
                        "eval/success_rate": eval_results["success_rate"],
                        "eval/avg_length": eval_results["avg_length"],
                        "eval/std_length": eval_results["std_length"],
                        "eval/avg_success_length": eval_results["avg_success_length"],
                        "eval/avg_failure_length": eval_results["avg_failure_length"],
                        "eval/avg_return": eval_results["avg_return"],
                        "eval/avg_speed": eval_results["avg_speed"],
                        "eval/std_speed": eval_results["std_speed"],
                        "eval/max_speed_used": eval_results["max_speed_used"],
                    },
                    step=episode,
                )

            if eval_results["success_rate"] > best_success_rate:
                best_success_rate = eval_results["success_rate"]
                print(f"  New best! Success rate: {best_success_rate:.2%}")
                _save_speed_tuning_checkpoint(
                    output_dir / "best_model.pkl",
                    agent,
                    episode,
                    bc_checkpoint_path,
                    speed_options,
                    normalizers,
                    best_success_rate,
                )

        # Checkpoint
        if checkpoint_interval > 0 and (episode + 1) % checkpoint_interval == 0:
            _save_speed_tuning_checkpoint(
                output_dir / f"checkpoint_ep{episode + 1}.pkl",
                agent,
                episode,
                bc_checkpoint_path,
                speed_options,
                normalizers,
                best_success_rate,
            )

    # Final save
    _save_speed_tuning_checkpoint(
        output_dir / "final_model.pkl",
        agent,
        total_episodes,
        bc_checkpoint_path,
        speed_options,
        normalizers,
        best_success_rate,
    )

    print(f"\nTraining complete! Best success rate: {best_success_rate:.2%}")
    print(f"Checkpoints saved to: {output_dir}")

    if wandb_enabled:
        wandb.finish()


def _save_speed_tuning_checkpoint(
    path, agent, episode, bc_checkpoint_path, speed_options, normalizers, best_sr
):
    """Save SpeedTuning checkpoint."""
    import pickle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "params": agent.network.params,
        "opt_state": agent.network.opt_state,
        "step": agent.network.step,
        "config": agent.config,
        "rng": agent.rng,
        "episode": episode,
        "bc_checkpoint_path": str(bc_checkpoint_path),
        "speed_options": speed_options,
        "normalizers": normalizers,
        "best_success_rate": best_sr,
    }

    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"  Checkpoint saved to {path}")


if __name__ == "__main__":
    main()
