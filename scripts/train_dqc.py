"""Training script for DQC: Decoupled Q-Chunking.

Offline RL with decoupled critic and policy chunk sizes.

Usage:
    python scripts/train_dqc.py --config-name dqc_default task=square_lowdim
    python scripts/train_dqc.py --config-name dqc_default task=square_lowdim algorithm.backup_horizon=25
"""

from pathlib import Path

import hydra
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from jax_flow.agents.offline_online.dqc_agent import DQCAgent
from jax_flow.agents.offline_online.utils import reward_transform
from jax_flow.core.checkpoint import cleanup_old_checkpoints, save_checkpoint
from jax_flow.core.evaluation import evaluate_policy, print_evaluation_results
from jax_flow.data import DatasetManager, make_robomimic_dataset
from jax_flow.data.replay_buffer import ReplayBuffer
from jax_flow.envs import make_robomimic_env


def fill_buffer_from_dataset(dataset, buffer):
    """Fill replay buffer with single-step transitions from offline dataset."""
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
            reward = 1.0 if done else 0.0

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
    config_name="dqc_default",
)
def main(cfg: DictConfig):
    """Main training function."""
    print("=" * 80)
    print("DQC Training Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    algo = cfg.algorithm
    backup_horizon = algo.backup_horizon
    policy_chunk_size = algo.policy_chunk_size
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
    if obs_type == "image":
        raise NotImplementedError("Image obs for DQC buffer not yet supported")

    obs_shape = (obs_dim,)
    buffer = ReplayBuffer(
        capacity=algo.buffer_capacity,
        obs_shape=obs_shape,
        action_dim=action_dim,
        n_step=1,
        gamma=algo.discount,
    )
    fill_buffer_from_dataset(train_dataset, buffer)

    # ============================================================
    # 3. Create DQC agent
    # ============================================================
    batch_size = cfg.optimization.batch_size
    ex_obs = jnp.zeros((batch_size, obs_dim))
    ex_actions_long = jnp.zeros((batch_size, backup_horizon, action_dim))
    ex_actions_short = jnp.zeros((batch_size, policy_chunk_size, action_dim))

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
        "gradient_steps": algo.gradient_steps,
        # Algorithm
        "backup_horizon": backup_horizon,
        "policy_chunk_size": policy_chunk_size,
        "horizon": policy_chunk_size,  # For flow loss compatibility
        "action_dim": action_dim,
        "obs_dim": obs_dim,
        "num_qs": algo.num_qs,
        "q_agg": algo.q_agg,
        "value_hidden_dims": list(algo.value_hidden_dims),
        "critic_layer_norm": algo.critic_layer_norm,
        "tau": algo.tau,
        "discount": algo.discount,
        "kappa_d": algo.kappa_d,
        "kappa_b": algo.kappa_b,
        "temperature": algo.temperature,
        "max_weight": algo.max_weight,
        "best_of_n": algo.best_of_n,
        # Task info
        "env_name": cfg.task.env_name,
        "obs_type": obs_type,
    }

    agent = DQCAgent.create(
        seed=cfg.seed,
        ex_observations=ex_obs,
        ex_actions_long=ex_actions_long,
        ex_actions_short=ex_actions_short,
        config=agent_config,
    )
    print(
        f"DQC agent created: backup_horizon={backup_horizon}, policy_chunk_size={policy_chunk_size}"
    )

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
    # 6. Offline training
    # ============================================================
    print("\n" + "=" * 80)
    print("DQC Offline Training")
    print("=" * 80)

    best_success_rate = -1.0

    for step in tqdm(range(1, algo.gradient_steps + 1), desc="Training"):
        seq_batch = buffer.sample_sequence(
            batch_size=batch_size,
            sequence_length=backup_horizon,
            discount=algo.discount,
            policy_chunk_size=policy_chunk_size,
            rng=rng,
        )

        jax_batch = {
            "observations": jnp.array(seq_batch["observations"]),
            "actions_long": jnp.array(seq_batch["actions_long"]),
            "actions_short": jnp.array(seq_batch["actions_short"]),
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
                wandb.log(
                    {
                        "eval/success_rate": eval_results["success_rate"],
                        "eval/avg_return": eval_results["avg_return"],
                        "eval/avg_length": eval_results["avg_length"],
                    },
                    step=step,
                )

            if eval_results["success_rate"] > best_success_rate:
                best_success_rate = eval_results["success_rate"]
                save_checkpoint(
                    output_dir / "best_model.pkl",
                    agent,
                    step,
                    normalizers=normalizers,
                    metadata={"success_rate": best_success_rate},
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

    # Final save
    save_checkpoint(
        output_dir / "final_model.pkl",
        agent,
        algo.gradient_steps,
        normalizers=normalizers,
        metadata={"best_success_rate": best_success_rate},
    )

    if use_wandb:
        wandb.finish()

    print(f"\nTraining complete. Best success rate: {best_success_rate:.2%}")


if __name__ == "__main__":
    main()
