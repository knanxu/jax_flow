"""Training script for ACFQL: Action-Chunking Flow Q-Learning.

Two-phase pipeline: offline pretraining + online fine-tuning.

Usage:
    python scripts/train_acfql.py --config-name acfql_default task=square_lowdim
    python scripts/train_acfql.py --config-name acfql_default task=square_lowdim algorithm.actor_type=distill-ddpg
"""

from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from jax_flow.agents.offline_online.acfql_agent import ACFQLAgent
from jax_flow.agents.offline_online.utils import reward_transform
from jax_flow.core.checkpoint import cleanup_old_checkpoints, save_checkpoint
from jax_flow.core.evaluation import evaluate_policy, print_evaluation_results
from jax_flow.data import DatasetManager, make_robomimic_dataset
from jax_flow.data.replay_buffer import ReplayBuffer
from jax_flow.envs import make_robomimic_env


def fill_buffer_from_dataset(dataset, buffer, chunk_length):
    """Fill replay buffer with single-step transitions from offline dataset.

    The buffer stores single-step transitions; sample_sequence handles
    constructing action chunks at sample time.
    """
    print(f"Filling buffer from {len(dataset)} dataset samples...")
    for ep_idx in range(len(dataset.episode_obs)):
        ep_obs = dataset.episode_obs[ep_idx]
        ep_act = dataset.episode_actions[ep_idx]
        ep_len = len(ep_obs)

        for t in range(ep_len):
            obs = dataset.obs_normalizer.normalize(ep_obs[t])
            action = dataset.action_normalizer.normalize(ep_act[t])
            next_obs = dataset.obs_normalizer.normalize(ep_obs[min(t + 1, ep_len - 1)])
            done = t == ep_len - 1
            reward = 1.0 if done else 0.0  # Sparse reward (transformed later)

            buffer._store_transition(
                obs=obs.astype(np.float32),
                action=action.astype(np.float32),
                reward=reward,
                next_obs=next_obs.astype(np.float32),
                done=done,
                discount=0.0 if done else 0.99,
            )

    print(f"Buffer filled: {len(buffer)} transitions")


@hydra.main(
    version_base=None,
    config_path="../configs/offline_online",
    config_name="acfql_default",
)
def main(cfg: DictConfig):
    """Main training function."""
    print("=" * 80)
    print("ACFQL Training Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    algo = cfg.algorithm
    chunk_length = algo.chunk_length
    output_dir = Path(cfg.checkpoint_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1. Load dataset
    # ============================================================
    dataset_path = cfg.task.dataset.path
    task_name = cfg.task.get("dataset_name", cfg.task.env_name.lower())
    obs_type = cfg.task.obs_type

    resolved_path = DatasetManager.ensure_dataset(
        dataset_path=dataset_path,
        task=task_name,
        dataset_type=cfg.task.dataset.get("dataset_type", "ph"),
        obs_type=obs_type,
        source="robomimic",
        auto_download=True,
    )
    if resolved_path is None:
        print("Failed to load dataset.")
        return

    dataset_kwargs = {}
    abs_action = cfg.task.get("abs_action", False)
    if abs_action:
        dataset_kwargs["abs_action"] = True
    if obs_type == "image":
        dataset_kwargs["image_keys"] = tuple(
            cfg.task.dataset.get("image_keys", ["agentview_image"])
        )
        dataset_kwargs["lowdim_keys"] = tuple(cfg.task.dataset.get("lowdim_keys", []))
    elif "obs_keys" in cfg.task:
        dataset_kwargs["obs_keys"] = tuple(cfg.task.obs_keys)

    train_dataset = make_robomimic_dataset(
        dataset_path=str(resolved_path),
        horizon=cfg.task.dataset.horizon,
        obs_steps=cfg.task.dataset.obs_steps,
        act_steps=cfg.task.dataset.act_steps,
        obs_type=obs_type,
        val_ratio=0.0,
        mode="train",
        **dataset_kwargs,
    )

    action_dim = train_dataset.action_dim
    obs_dim = train_dataset.obs_dim
    print(
        f"Dataset: {len(train_dataset)} samples, obs_dim={obs_dim}, action_dim={action_dim}"
    )

    # ============================================================
    # 2. Create replay buffer and fill with offline data
    # ============================================================
    obs_shape = (obs_dim,) if obs_type == "lowdim" else None
    if obs_type == "image":
        raise NotImplementedError("Image obs for ACFQL buffer not yet supported")

    buffer = ReplayBuffer(
        capacity=algo.buffer_capacity,
        obs_shape=obs_shape,
        action_dim=action_dim,
        n_step=1,
        gamma=algo.discount,
    )
    fill_buffer_from_dataset(train_dataset, buffer, chunk_length)

    # ============================================================
    # 3. Create ACFQL agent
    # ============================================================
    batch_size = cfg.optimization.batch_size
    ex_obs = jnp.zeros((batch_size, obs_dim))
    ex_actions = jnp.zeros((batch_size, chunk_length, action_dim))

    agent_config = {
        # Network
        "encoder_type": cfg.get("encoder_type", "identity"),
        "encoder_hidden_dims": tuple(
            cfg.network.get("encoder_hidden_dims", [256, 256])
        ),
        "emb_dim": cfg.network.emb_dim,
        "n_blocks": cfg.network.get("n_blocks", 6),
        "expansion_factor": cfg.network.get("expansion_factor", 4),
        "dropout": cfg.network.get("dropout", 0.1),
        "timestep_embed_dim": cfg.network.get("timestep_emb_dim", 128),
        "max_freq": cfg.network.get("max_freq", 100.0),
        "network_type": cfg.network.get("network_type", "mlp"),
        # Flow
        "interp_type": cfg.flow.interp_type,
        "flow_type": cfg.flow.get("policy_type", "flow_matching"),
        "sampler_type": cfg.flow.sampler.type,
        "flow_steps": cfg.flow.sampler.num_steps,
        "sample_mode": cfg.flow.sampler.get("sample_mode", "stochastic"),
        "loss_scale": cfg.flow.loss.get("scale", 0.1),
        "t_two_step": cfg.flow.get("t_two_step", 0.9),
        "delta_t_schedule": dict(cfg.flow.get("delta_t_schedule", {})),
        "adaptive_weight": dict(cfg.flow.get("adaptive_weight", {})),
        # MeanFlow Stable parameters
        "time_steps": cfg.flow.get("time_steps", 0),
        "consistency_alpha": cfg.flow.get("consistency_alpha", 0.0),
        # Optimizer
        "lr": cfg.optimization.lr,
        "weight_decay": cfg.optimization.weight_decay,
        "schedule_type": cfg.optimization.lr_schedule.type,
        "warmup_steps": cfg.optimization.lr_schedule.warmup_steps,
        "gradient_steps": algo.offline_steps + algo.online_steps,
        # Algorithm
        "chunk_length": chunk_length,
        "action_dim": action_dim,
        "obs_dim": obs_dim,
        "horizon": chunk_length,  # For flow loss compatibility
        "actor_type": algo.actor_type,
        "actor_num_samples": algo.actor_num_samples,
        "num_qs": algo.num_qs,
        "q_agg": algo.q_agg,
        "value_hidden_dims": list(algo.value_hidden_dims),
        "critic_layer_norm": algo.critic_layer_norm,
        "tau": algo.tau,
        "discount": algo.discount,
        "alpha": algo.alpha,
        # Task info
        "env_name": cfg.task.env_name,
        "obs_type": obs_type,
    }

    agent = ACFQLAgent.create(
        seed=cfg.seed,
        ex_observations=ex_obs,
        ex_actions=ex_actions,
        config=agent_config,
    )
    print(f"ACFQL agent created: actor_type={algo.actor_type}")

    # ============================================================
    # 4. Create evaluation environment
    # ============================================================
    normalizers = train_dataset.get_normalizer()
    eval_env = make_robomimic_env(
        env_name=cfg.task.env_name,
        dataset_path=str(resolved_path),
        obs_type=obs_type,
        obs_keys=tuple(
            cfg.task.get(
                "obs_keys",
                ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
            )
        ),
        obs_normalizer=normalizers.get("obs"),
        action_normalizer=normalizers.get("action"),
        max_episode_steps=cfg.eval.max_episode_steps,
        frame_stack=cfg.task.dataset.obs_steps,
        act_exec_steps=cfg.task.dataset.act_steps,
        seed=cfg.seed + 100,
        abs_action=abs_action,
    )

    # ============================================================
    # 5. W&B init
    # ============================================================
    use_wandb = cfg.wandb.get("enabled", False)
    if use_wandb:
        import wandb

        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=list(cfg.wandb.get("tags", [])),
        )

    # ============================================================
    # 6. Phase 1: Offline pretraining
    # ============================================================
    print("\n" + "=" * 80)
    print("Phase 1: Offline Pretraining")
    print("=" * 80)

    best_success_rate = -1.0

    for step in tqdm(range(1, algo.offline_steps + 1), desc="Offline"):
        # Sample sequence batch
        seq_batch = buffer.sample_sequence(
            batch_size=batch_size,
            sequence_length=chunk_length,
            discount=algo.discount,
            rng=rng,
        )

        # Convert to JAX and apply reward transform
        jax_batch = {
            "observations": jnp.array(seq_batch["observations"]),
            "actions": jnp.array(seq_batch["actions_long"]),
            "rewards": jnp.array(reward_transform(seq_batch["rewards"])),
            "next_observations": jnp.array(seq_batch["next_observations"]),
            "masks": jnp.array(seq_batch["masks"]),
            "valid": jnp.array(seq_batch["valid"]),
        }

        agent, info = agent.update(jax_batch)

        # Logging
        if step % 1000 == 0 and use_wandb:
            log_dict = {f"train/{k}": float(v) for k, v in info.items()}
            log_dict["train/step"] = step
            wandb.log(log_dict, step=step)

        # Evaluation
        if step % cfg.eval.eval_interval == 0:
            eval_results = evaluate_policy(
                agent=agent,
                env=eval_env,
                num_episodes=cfg.eval.num_episodes,
                max_steps=cfg.eval.max_episode_steps,
                save_video=cfg.eval.save_video,
                num_videos=cfg.eval.num_videos,
            )
            print_evaluation_results(eval_results)

            if use_wandb:
                eval_log = {
                    "eval/success_rate": eval_results["success_rate"],
                    "eval/avg_return": eval_results["avg_return"],
                    "eval/avg_length": eval_results["avg_length"],
                    "eval/mean_score": eval_results["mean_score"],
                }
                wandb.log(eval_log, step=step)

            # Save best model
            if eval_results["success_rate"] > best_success_rate:
                best_success_rate = eval_results["success_rate"]
                save_checkpoint(
                    output_dir / "best_model.pkl",
                    agent,
                    step,
                    normalizers=normalizers,
                    metadata={"phase": "offline", "success_rate": best_success_rate},
                )

        # Periodic checkpoint
        if step % cfg.checkpoint.save_freq == 0:
            save_checkpoint(
                output_dir / f"checkpoint_{step}.pkl",
                agent,
                step,
                normalizers=normalizers,
            )
            cleanup_old_checkpoints(output_dir, cfg.checkpoint.keep_last_n)

    print(f"\nOffline phase complete. Best success rate: {best_success_rate:.2%}")

    # ============================================================
    # 7. Phase 2: Online fine-tuning
    # ============================================================
    if algo.online_steps > 0:
        print("\n" + "=" * 80)
        print("Phase 2: Online Fine-tuning")
        print("=" * 80)

        online_env = make_robomimic_env(
            env_name=cfg.task.env_name,
            dataset_path=str(resolved_path),
            obs_type=obs_type,
            obs_keys=tuple(
                cfg.task.get(
                    "obs_keys",
                    [
                        "robot0_eef_pos",
                        "robot0_eef_quat",
                        "robot0_gripper_qpos",
                        "object",
                    ],
                )
            ),
            obs_normalizer=normalizers.get("obs"),
            action_normalizer=normalizers.get("action"),
            max_episode_steps=cfg.eval.max_episode_steps,
            frame_stack=cfg.task.dataset.obs_steps,
            act_exec_steps=None,  # We manage chunking manually
            seed=cfg.seed,
            abs_action=abs_action,
        )

        obs, _ = online_env.reset()
        action_queue = []
        episode_reward = 0.0
        episode_count = 0
        online_step = 0

        total_step = algo.offline_steps

        for online_step in tqdm(range(1, algo.online_steps + 1), desc="Online"):
            total_step = algo.offline_steps + online_step

            # Get action
            if len(action_queue) == 0:
                if online_step <= algo.random_steps:
                    # Random exploration
                    actions = np.random.uniform(-1, 1, (1, chunk_length, action_dim))
                else:
                    obs_batch = jnp.array(obs[np.newaxis, ...])
                    sample_rng = jax.random.PRNGKey(online_step)
                    actions = np.array(agent.sample_actions(obs_batch, rng=sample_rng))

                # Queue first act_exec_steps actions
                act_steps = cfg.task.dataset.act_steps
                chunk = actions[0, :act_steps]
                action_queue = list(chunk)

            action = action_queue.pop(0)

            # Step environment
            next_obs, reward, terminated, truncated, info_env = online_env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # Store transition
            buffer.add(
                obs=obs.astype(np.float32),
                action=action.astype(np.float32),
                reward=reward,
                next_obs=next_obs.astype(np.float32),
                done=done,
            )

            obs = next_obs

            if done:
                obs, _ = online_env.reset()
                action_queue = []
                episode_count += 1
                if use_wandb:
                    wandb.log(
                        {
                            "online/episode_reward": episode_reward,
                            "online/episode_count": episode_count,
                        },
                        step=total_step,
                    )
                episode_reward = 0.0

            # Gradient updates (UTD ratio)
            if online_step > algo.random_steps:
                for _ in range(algo.utd_ratio):
                    seq_batch = buffer.sample_sequence(
                        batch_size=batch_size,
                        sequence_length=chunk_length,
                        discount=algo.discount,
                        rng=rng,
                    )
                    jax_batch = {
                        "observations": jnp.array(seq_batch["observations"]),
                        "actions": jnp.array(seq_batch["actions_long"]),
                        "rewards": jnp.array(reward_transform(seq_batch["rewards"])),
                        "next_observations": jnp.array(seq_batch["next_observations"]),
                        "masks": jnp.array(seq_batch["masks"]),
                        "valid": jnp.array(seq_batch["valid"]),
                    }
                    agent, update_info = agent.update(jax_batch)

            # Logging
            if (
                online_step % 1000 == 0
                and use_wandb
                and online_step > algo.random_steps
            ):
                log_dict = {f"train/{k}": float(v) for k, v in update_info.items()}
                log_dict["train/step"] = total_step
                wandb.log(log_dict, step=total_step)

            # Evaluation
            if online_step % cfg.eval.eval_interval == 0:
                eval_results = evaluate_policy(
                    agent=agent,
                    env=eval_env,
                    num_episodes=cfg.eval.num_episodes,
                    max_steps=cfg.eval.max_episode_steps,
                    save_video=cfg.eval.save_video,
                    num_videos=cfg.eval.num_videos,
                )
                print_evaluation_results(eval_results)

                if use_wandb:
                    wandb.log(
                        {
                            "eval/success_rate": eval_results["success_rate"],
                            "eval/avg_return": eval_results["avg_return"],
                            "eval/mean_score": eval_results["mean_score"],
                        },
                        step=total_step,
                    )

                if eval_results["success_rate"] > best_success_rate:
                    best_success_rate = eval_results["success_rate"]
                    save_checkpoint(
                        output_dir / "best_model.pkl",
                        agent,
                        total_step,
                        normalizers=normalizers,
                        metadata={
                            "phase": "online",
                            "success_rate": best_success_rate,
                        },
                    )

            # Periodic checkpoint
            if online_step % cfg.checkpoint.save_freq == 0:
                save_checkpoint(
                    output_dir / f"checkpoint_{total_step}.pkl",
                    agent,
                    total_step,
                    normalizers=normalizers,
                )
                cleanup_old_checkpoints(output_dir, cfg.checkpoint.keep_last_n)

    # Final save
    save_checkpoint(
        output_dir / "final_model.pkl",
        agent,
        total_step if algo.online_steps > 0 else algo.offline_steps,
        normalizers=normalizers,
        metadata={"best_success_rate": best_success_rate},
    )

    if use_wandb:
        wandb.finish()

    print(f"\nTraining complete. Best success rate: {best_success_rate:.2%}")


if __name__ == "__main__":
    main()
