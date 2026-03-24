"""Training script for Offline RL (DDPG + BC) with flow policy.

Usage:
    python scripts/train_offline_rl.py task=square_lowdim \
        algorithm.bc_checkpoint=checkpoints/square_lowdim_mip_mlp/best_model.pkl

    python scripts/train_offline_rl.py task=square_lowdim \
        algorithm.bc_checkpoint=path/to/bc/best_model.pkl \
        algorithm.gradient_steps=500000 algorithm.alpha=1.0
"""

from pathlib import Path

import hydra
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from jax_flow.agents.offline_rl_agent import OfflineRLAgent
from jax_flow.core.checkpoint import (
    cleanup_old_checkpoints,
    load_checkpoint,
    save_offline_rl_checkpoint,
)
from jax_flow.core.evaluation import evaluate_policy, print_evaluation_results
from jax_flow.data import DatasetManager, make_dataset, make_robomimic_dataset
from jax_flow.envs import make_env, make_robomimic_env


@hydra.main(
    version_base=None, config_path="../configs", config_name="offline_rl_default"
)
def main(cfg: DictConfig):
    """Main offline RL training function."""
    print("=" * 80)
    print("Offline RL: DDPG + BC on Flow Policy")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Validate BC checkpoint
    bc_checkpoint_path = cfg.algorithm.bc_checkpoint
    if bc_checkpoint_path is None:
        raise ValueError(
            "algorithm.bc_checkpoint is required. "
            "Provide path to a pretrained BC checkpoint."
        )

    # Set random seed
    np.random.seed(cfg.seed)
    rng_np = np.random.default_rng(cfg.seed)

    # ================================================================
    # 1. Load BC checkpoint to get config and normalizers
    # ================================================================
    print("\nLoading BC checkpoint...")
    bc_ckpt = load_checkpoint(bc_checkpoint_path)
    bc_config = bc_ckpt["config"]
    normalizers = bc_ckpt.get("normalizers", {})

    # ================================================================
    # 2. Resolve dataset path
    # ================================================================
    task_source = cfg.task.get("task_source", "robomimic")
    dataset_path = bc_config.get("dataset_path", cfg.task.dataset.path)
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

    dataset_path = str(resolved_path)
    is_image = obs_type == "image"
    abs_action = cfg.task.get("abs_action", False)

    # ================================================================
    # 3. Create dataset with sample_sequence support
    # ================================================================
    print("\nLoading dataset...")

    dataset_kwargs = {}
    if abs_action:
        dataset_kwargs["abs_action"] = True
    if is_image:
        image_keys = list(
            cfg.task.dataset.get("image_keys", ["agentview_image"])
        )
        lowdim_keys = list(
            cfg.task.dataset.get(
                "lowdim_keys",
                ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
            )
        )
        dataset_kwargs["image_keys"] = tuple(image_keys)
        dataset_kwargs["lowdim_keys"] = tuple(lowdim_keys)
    else:
        if "obs_keys" in cfg.task:
            dataset_kwargs["obs_keys"] = tuple(cfg.task.obs_keys)

    reward_offset = cfg.algorithm.get("reward_offset", -1.0)

    common_kwargs = {
        "horizon": cfg.task.dataset.horizon,
        "obs_steps": cfg.task.dataset.obs_steps,
        "act_steps": cfg.task.dataset.act_steps,
        "val_ratio": 0.0,
    }

    if task_source in ("pusht", "kitchen"):
        train_dataset = make_dataset(
            task_source=task_source,
            dataset_path=dataset_path,
            obs_type=obs_type,
            mode="train",
            **common_kwargs,
        )
    else:
        train_dataset = make_robomimic_dataset(
            dataset_path=dataset_path,
            obs_type=obs_type,
            mode="train",
            reward_offset=reward_offset,
            **common_kwargs,
            **dataset_kwargs,
        )

    print(f"Dataset size: {len(train_dataset)}")
    print(f"Obs dim: {train_dataset.obs_dim}, Action dim: {train_dataset.action_dim}")

    # ================================================================
    # 4. Build agent config and create OfflineRLAgent
    # ================================================================
    print("\nCreating OfflineRLAgent...")

    algo = cfg.algorithm
    agent_config = {
        # From BC config (needed for sampler/loss)
        "horizon": bc_config.get("horizon", cfg.task.dataset.horizon),
        "obs_steps": bc_config.get("obs_steps", cfg.task.dataset.obs_steps),
        "action_dim": train_dataset.action_dim,
        "obs_dim": train_dataset.obs_dim,
        "sampler_type": bc_config.get("sampler_type", "euler"),
        "flow_steps": bc_config.get("flow_steps", 10),
        "sample_mode": bc_config.get("sample_mode", "stochastic"),
        "loss_scale": bc_config.get("loss_scale", 0.1),
        "interp_type": bc_config.get("interp_type", "linear"),
        "flow_type": bc_config.get("flow_type", "flow_matching"),
        "t_two_step": bc_config.get("t_two_step", 0.9),
        "delta_t_schedule": bc_config.get("delta_t_schedule", {}),
        "adaptive_weight": bc_config.get("adaptive_weight", {}),
        "time_steps": bc_config.get("time_steps", 0),
        "consistency_alpha": bc_config.get("consistency_alpha", 0.0),
        "gradient_steps": algo.get("gradient_steps", 500000),
        # Algorithm params
        "act_exec_steps": bc_config.get(
            "act_steps", cfg.task.dataset.act_steps
        ),
        "critic_hidden_dims": list(algo.get("critic_hidden_dims", [2048, 2048, 2048])),
        "critic_activation": algo.get("critic_activation", "tanh"),
        "critic_layer_norm": algo.get("critic_layer_norm", True),
        "num_ensembles": algo.get("num_ensembles", 2),
        "tau": algo.get("tau", 0.005),
        "discount": algo.get("discount", 0.99),
        "alpha": algo.get("alpha", 1.0),
        "q_weight": algo.get("q_weight", 1.0),
        "normalize_q_loss": algo.get("normalize_q_loss", True),
        "actor_lr": algo.get("actor_lr", 1e-4),
        "critic_lr": algo.get("critic_lr", 3e-4),
        "ema_decay": algo.get("ema_decay", 0.995),
        "freeze_encoder": algo.get("freeze_encoder", True),
        "load_bc_weights": algo.get("load_bc_weights", False),
        # Task info for eval
        "dataset_path": dataset_path,
        "env_name": cfg.task.env_name,
        "obs_type": obs_type,
        "abs_action": abs_action,
    }

    if is_image:
        agent_config["image_keys"] = tuple(image_keys)
        agent_config["lowdim_keys"] = tuple(lowdim_keys)
        crop_shape = cfg.task.dataset.get("crop_shape", None)
        if crop_shape is not None:
            agent_config["crop_shape"] = tuple(crop_shape)
    else:
        agent_config["obs_keys"] = tuple(
            cfg.task.get(
                "obs_keys",
                ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
            )
        )

    # Get example batch for initialization
    batch_size = algo.get("batch_size", 256)
    ex_batch = train_dataset.sample_sequence(
        batch_size=min(batch_size, 4),
        discount=algo.get("discount", 0.99),
        rng=rng_np,
    )
    if isinstance(ex_batch["observations"], dict):
        ex_obs = {k: jnp.array(v) for k, v in ex_batch["observations"].items()}
    else:
        ex_obs = jnp.array(ex_batch["observations"])
    ex_actions = jnp.array(ex_batch["actions"])

    print(f"Example obs shape: {ex_obs.shape if not isinstance(ex_obs, dict) else {k: v.shape for k, v in ex_obs.items()}}")
    print(f"Example actions shape: {ex_actions.shape}")

    agent = OfflineRLAgent.create(
        seed=cfg.seed,
        ex_observations=ex_obs,
        ex_actions=ex_actions,
        config=agent_config,
        bc_checkpoint_path=bc_checkpoint_path,
    )
    print("OfflineRLAgent created.")

    # ================================================================
    # 5. Output directory and W&B
    # ================================================================
    env_name = cfg.task.env_name
    output_dir = Path(cfg.get("checkpoint_dir", "checkpoints")) / (
        f"{cfg.get('experiment_name', f'offline_rl_{env_name}_{obs_type}')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    wandb_enabled = cfg.get("wandb", {}).get("enabled", False)
    if wandb_enabled:
        try:
            import wandb

            wandb.init(
                project=cfg.wandb.get("project", "jax_flow"),
                entity=cfg.wandb.get("entity", None),
                name=cfg.wandb.get(
                    "name", f"offline_rl_{env_name}_{obs_type}_seed{cfg.seed}"
                ),
                config=dict(OmegaConf.to_container(cfg, resolve=True)),
            )
            print("W&B logging enabled")
        except ImportError:
            print("W&B not installed, skipping logging")
            wandb_enabled = False

    # ================================================================
    # 6. Create evaluation environment
    # ================================================================
    eval_env = None
    eval_interval = cfg.eval.get("eval_interval", 10000)
    if eval_interval > 0:
        if task_source in ("pusht", "kitchen"):
            eval_env = make_env(
                task_source=task_source,
                obs_type=obs_type if task_source == "pusht" else None,
                env_name=cfg.task.env_name if task_source == "kitchen" else None,
                obs_normalizer=normalizers.get("obs"),
                action_normalizer=normalizers.get("action"),
                lowdim_normalizer=normalizers.get("lowdim"),
                max_episode_steps=cfg.task.env.max_episode_steps,
                frame_stack=cfg.task.dataset.obs_steps,
                act_exec_steps=cfg.task.dataset.act_steps,
                seed=cfg.seed,
            )
        else:
            eval_env = make_robomimic_env(
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
                image_keys=tuple(image_keys) if is_image else None,
                lowdim_keys=tuple(lowdim_keys) if is_image else None,
                obs_normalizer=normalizers.get("obs"),
                action_normalizer=normalizers.get("action"),
                lowdim_normalizer=normalizers.get("lowdim"),
                max_episode_steps=cfg.task.env.max_episode_steps,
                frame_stack=cfg.task.dataset.obs_steps,
                act_exec_steps=cfg.task.dataset.act_steps,
                seed=cfg.seed,
                render_offscreen=cfg.eval.get("save_video", True),
                abs_action=abs_action,
            )

    # ================================================================
    # 7. Training loop
    # ================================================================
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    gradient_steps = algo.get("gradient_steps", 500000)
    bc_warmup_steps = algo.get("bc_warmup_steps", 0)
    q_weight = algo.get("q_weight", 1.0)
    log_interval = algo.get("log_interval", 1000)
    checkpoint_interval = cfg.checkpoint.get("save_freq", 50000)
    best_success_rate = -1.0
    eval_count = 0

    if bc_warmup_steps > 0:
        print(f"Mode 1: BC warmup ({bc_warmup_steps} steps) -> Offline RL")
    else:
        print(f"Mode 2: Offline RL from start (q_weight={q_weight})")

    pbar = tqdm(range(gradient_steps), desc="Training")

    for step in pbar:
        # Sample batch
        batch_np = train_dataset.sample_sequence(
            batch_size=batch_size,
            discount=algo.get("discount", 0.99),
            rng=rng_np,
        )

        # Convert to JAX
        if isinstance(batch_np["observations"], dict):
            observations = {
                k: jnp.array(v) for k, v in batch_np["observations"].items()
            }
            next_observations = {
                k: jnp.array(v) for k, v in batch_np["next_observations"].items()
            }
        else:
            observations = jnp.array(batch_np["observations"])
            next_observations = jnp.array(batch_np["next_observations"])

        batch = {
            "observations": observations,
            "actions": jnp.array(batch_np["actions"]),
            "rewards": jnp.array(batch_np["rewards"]),
            "next_observations": next_observations,
            "masks": jnp.array(batch_np["masks"]),
        }

        # Determine current q_weight: 0 during BC warmup, configured value after
        current_q_weight = 0.0 if step < bc_warmup_steps else q_weight

        # Update
        agent, info = agent.update(batch, q_weight=current_q_weight)

        # Logging
        if step % log_interval == 0:
            phase = "BC" if step < bc_warmup_steps else "RL"
            pbar.set_postfix(
                {
                    "phase": phase,
                    "loss": f"{float(info['loss']):.4f}",
                    "c_loss": f"{float(info['critic_loss']):.3f}",
                    "bc": f"{float(info['bc_loss']):.3f}",
                    "q": f"{float(info['q_mean']):.2f}",
                }
            )

            if wandb_enabled:
                log_dict = {
                    "train/loss": float(info["loss"]),
                    "train/critic_loss": float(info["critic_loss"]),
                    "train/actor_loss": float(info["actor_loss"]),
                    "train/bc_loss": float(info["bc_loss"]),
                    "train/q_loss": float(info["q_loss"]),
                    "train/q_mean": float(info["q_mean"]),
                    "train/target_q_mean": float(info["target_q_mean"]),
                    "train/grad_norm": float(info.get("grad/norm", 0)),
                    "train/q_weight": current_q_weight,
                    "train/step": step,
                }
                wandb.log(log_dict, step=step)

        # Evaluation
        if (
            eval_interval > 0
            and step % eval_interval == 0
            and step > 0
            and eval_env is not None
        ):
            print(f"\n[Step {step}] Evaluating...")
            eval_count += 1

            eval_results = evaluate_policy(
                agent=agent,
                env=eval_env,
                num_episodes=cfg.eval.get("num_episodes", 50),
                max_steps=cfg.task.env.max_episode_steps,
                save_video=cfg.eval.get("save_video", True) and wandb_enabled,
                num_videos=cfg.eval.get("num_videos", 3),
                verbose=True,
            )
            print_evaluation_results(eval_results)

            if wandb_enabled:
                eval_log = {
                    "eval/success_rate": eval_results["success_rate"],
                    "eval/avg_length": eval_results["avg_length"],
                    "eval/avg_return": eval_results["avg_return"],
                    "eval/mean_score": eval_results["mean_score"],
                }
                for k in range(1, 8):
                    pk = f"p{k}"
                    if pk in eval_results:
                        eval_log[f"eval/{pk}"] = eval_results[pk]
                wandb.log(eval_log, step=step)

                if (
                    cfg.wandb.get("log_videos", True)
                    and "videos" in eval_results
                    and eval_count % cfg.wandb.get("log_video_interval", 1) == 0
                ):
                    for i, video_frames in enumerate(eval_results["videos"]):
                        video_t = np.transpose(video_frames, (0, 3, 1, 2))
                        wandb.log(
                            {f"eval/video_{i}": wandb.Video(video_t, fps=30, format="mp4")},
                            step=step,
                        )

            # Save best model
            if eval_results["success_rate"] > best_success_rate:
                best_success_rate = eval_results["success_rate"]
                save_offline_rl_checkpoint(
                    checkpoint_path=output_dir / "best_model.pkl",
                    agent=agent,
                    step=step,
                    bc_checkpoint_path=bc_checkpoint_path,
                    normalizers=normalizers,
                    best_success_rate=best_success_rate,
                )
                print(f"New best! Success rate: {best_success_rate:.2%}")

        # Periodic checkpoint
        if checkpoint_interval > 0 and step % checkpoint_interval == 0 and step > 0:
            save_offline_rl_checkpoint(
                checkpoint_path=output_dir / f"checkpoint_{step}.pkl",
                agent=agent,
                step=step,
                bc_checkpoint_path=bc_checkpoint_path,
                normalizers=normalizers,
            )
            cleanup_old_checkpoints(
                output_dir, keep_last_n=cfg.checkpoint.get("keep_last_n", 3)
            )

    pbar.close()

    # Final save
    save_offline_rl_checkpoint(
        checkpoint_path=output_dir / "final_model.pkl",
        agent=agent,
        step=gradient_steps,
        bc_checkpoint_path=bc_checkpoint_path,
        normalizers=normalizers,
        best_success_rate=best_success_rate,
    )

    if eval_env is not None:
        eval_env.close()

    print(f"\nTraining complete! Best success rate: {best_success_rate:.2%}")
    print(f"Checkpoints saved to: {output_dir}")

    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
