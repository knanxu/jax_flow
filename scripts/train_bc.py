"""Training script for Behavior Cloning with Flow Matching.

Usage:
    # Train with default config (lift lowdim)
    python scripts/train_bc.py

    # Train with specific task
    python scripts/train_bc.py task=lift_lowdim

    # Override parameters
    python scripts/train_bc.py task=lift_lowdim optimization.lr=1e-3 optimization.batch_size=128

    # Train with image observations
    python scripts/train_bc.py task=lift_image network=resnet
"""

from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from jax_flow.agents.bc_agent import BCAgent
from jax_flow.core.checkpoint import cleanup_old_checkpoints, save_checkpoint
from jax_flow.core.evaluation import evaluate_policy, print_evaluation_results
from jax_flow.data import DatasetManager, make_dataset, make_robomimic_dataset
from jax_flow.envs import make_env, make_robomimic_env


def create_dataloader(dataset, batch_size, shuffle=True):
    """Create a simple dataloader iterator."""
    indices = np.arange(len(dataset))

    while True:
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]

            # Collect batch
            obs_batch = []
            act_batch = []
            for idx in batch_indices:
                sample = dataset[idx]
                obs_batch.append(sample["observations"])
                act_batch.append(sample["actions"])

            # Stack into batch — handle dict obs (image) vs array obs (lowdim)
            first_obs = obs_batch[0]
            if isinstance(first_obs, dict):
                observations = {
                    k: jnp.array(np.stack([o[k] for o in obs_batch])) for k in first_obs
                }
            else:
                observations = jnp.array(np.stack(obs_batch))
            actions = jnp.array(np.stack(act_batch))

            batch = {
                "observations": observations,
                "actions": actions,
            }

            yield batch


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    # Merge task-level optimization overrides into top-level optimization
    # Priority: optimization/default.yaml < task.optimization
    # To override from CLI, use: task.optimization.batch_size=512
    if "optimization" in cfg.task:
        cfg.optimization = OmegaConf.merge(cfg.optimization, cfg.task.optimization)

    # Print config
    print("=" * 80)
    print("Training Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Set random seed
    np.random.seed(cfg.seed)
    rng = jax.random.PRNGKey(cfg.seed)

    # Create output directory
    output_dir = Path(cfg.checkpoint_dir) / cfg.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Load dataset
    print("\n" + "=" * 80)
    print("Loading Dataset")
    print("=" * 80)

    # Determine task source
    task_source = cfg.task.get("task_source", "robomimic")

    # Ensure dataset exists (auto-download if missing)
    dataset_path = cfg.task.dataset.path
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
        print("\n✗ Failed to load dataset. Exiting.")
        return

    # Use resolved path
    dataset_path = str(resolved_path)

    is_image = cfg.task.obs_type == "image"
    abs_action = cfg.task.get("abs_action", False)

    # Get image/lowdim keys from config
    dataset_kwargs = {}
    if abs_action:
        dataset_kwargs["abs_action"] = True
    if is_image:
        image_keys = list(cfg.task.dataset.get("image_keys", ["agentview_image"]))
        lowdim_keys = list(
            cfg.task.dataset.get(
                "lowdim_keys",
                [
                    "robot0_eef_pos",
                    "robot0_eef_quat",
                    "robot0_gripper_qpos",
                ],
            )
        )
        dataset_kwargs["image_keys"] = tuple(image_keys)
        dataset_kwargs["lowdim_keys"] = tuple(lowdim_keys)
    else:
        # Pass obs_keys from task config for lowdim mode
        if "obs_keys" in cfg.task:
            dataset_kwargs["obs_keys"] = tuple(cfg.task.obs_keys)

    # Create datasets via dispatch
    common_kwargs = {
        "horizon": cfg.task.dataset.horizon,
        "obs_steps": cfg.task.dataset.obs_steps,
        "act_steps": cfg.task.dataset.act_steps,
        "val_ratio": 0.02,
    }

    if task_source in ("pusht", "kitchen"):
        train_dataset = make_dataset(
            task_source=task_source,
            dataset_path=dataset_path,
            obs_type=obs_type,
            mode="train",
            **common_kwargs,
        )
        val_dataset = make_dataset(
            task_source=task_source,
            dataset_path=dataset_path,
            obs_type=obs_type,
            mode="val",
            **common_kwargs,
        )
    else:
        train_dataset = make_robomimic_dataset(
            dataset_path=dataset_path,
            obs_type=cfg.task.obs_type,
            mode="train",
            **common_kwargs,
            **dataset_kwargs,
        )
        val_dataset = make_robomimic_dataset(
            dataset_path=dataset_path,
            obs_type=cfg.task.obs_type,
            mode="val",
            **common_kwargs,
            **dataset_kwargs,
        )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    if is_image:
        print(f"Image keys: {image_keys}")
        print(f"Lowdim keys: {lowdim_keys}")
        print(f"Lowdim obs dim: {train_dataset.obs_dim}")
    else:
        print(f"Observation dim: {train_dataset.obs_dim}")
    print(f"Action dim: {train_dataset.action_dim}")

    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=cfg.optimization.batch_size,
        shuffle=True,
    )

    # Create agent
    print("\n" + "=" * 80)
    print("Creating Agent")
    print("=" * 80)

    # Get example batch for initialization
    example_batch = next(train_loader)
    ex_observations = example_batch["observations"]
    ex_actions = example_batch["actions"]

    if isinstance(ex_observations, dict):
        print("Example observations (dict):")
        for k, v in ex_observations.items():
            print(f"  {k}: {v.shape}")
    else:
        print(f"Example observations shape: {ex_observations.shape}")
    print(f"Example actions shape: {ex_actions.shape}")

    # Build agent config
    agent_config = {
        "horizon": cfg.task.dataset.horizon,
        "obs_steps": cfg.task.dataset.obs_steps,
        "action_dim": train_dataset.action_dim,
        "obs_dim": train_dataset.obs_dim,
        "encoder_type": cfg.get("encoder_type", "mlp"),
        "encoder_hidden_dims": tuple(
            cfg.network.get("encoder_hidden_dims", [256, 256])
        ),
        "emb_dim": cfg.network.emb_dim,
        # MLP residual block params
        "n_blocks": cfg.network.get("n_blocks", 6),
        "expansion_factor": cfg.network.get("expansion_factor", 4),
        "dropout": cfg.network.get("dropout", 0.1),
        "timestep_embed_dim": cfg.network.get("timestep_emb_dim", 128),
        "max_freq": cfg.network.get("max_freq", 100.0),
        # SmallMLP params
        "hidden_dims": tuple(cfg.network.get("hidden_dims", [512, 512, 512, 512])),
        "layer_norm": cfg.network.get("layer_norm", True),
        "network_type": cfg.network.get("network_type", "mlp"),
        "lr": cfg.optimization.lr,
        "weight_decay": cfg.optimization.weight_decay,
        "schedule_type": cfg.optimization.lr_schedule.type,
        "warmup_steps": cfg.optimization.lr_schedule.warmup_steps,
        "gradient_steps": cfg.optimization.gradient_steps,
        "grad_clip_norm": cfg.optimization.get("grad_clip_norm", 10.0),
        "b1": cfg.optimization.optimizer_kwargs.get("b1", 0.95),
        "b2": cfg.optimization.optimizer_kwargs.get("b2", 0.999),
        "interp_type": cfg.flow.interp_type,
        "flow_type": cfg.flow.get("policy_type", "flow_matching"),
        "sampler_type": cfg.flow.sampler.type,
        "flow_steps": cfg.flow.sampler.num_steps,
        "sample_mode": cfg.flow.sampler.get("sample_mode", "stochastic"),
        "loss_scale": cfg.flow.loss.get("scale", 0.1),
        # EMA
        "ema_type": cfg.optimization.get("ema_type", "fixed"),
        "ema_decay": cfg.optimization.get("ema_decay", 0.995),
        "ema_inv_gamma": cfg.optimization.get("ema_inv_gamma", 1.0),
        "ema_power": cfg.optimization.get("ema_power", 0.75),
        "ema_min_value": cfg.optimization.get("ema_min_value", 0.0),
        "ema_max_value": cfg.optimization.get("ema_max_value", 0.9999),
        # MIP t_two_step
        "t_two_step": cfg.flow.get("t_two_step", 0.9),
        # Delta-t schedule for MeanFlow
        "delta_t_schedule": dict(cfg.flow.get("delta_t_schedule", {})),
        # Adaptive weighting for MeanFlow
        "adaptive_weight": dict(cfg.flow.get("adaptive_weight", {})),
        # MeanFlow Stable parameters
        "time_steps": cfg.flow.get("time_steps", 0),
        "consistency_alpha": cfg.flow.get("consistency_alpha", 0.0),
        # Drift policy parameters
        "gen_per_label": cfg.flow.get("gen_per_label", 8),
        "drift_temperatures": list(cfg.flow.get("drift_temperatures", [0.02, 0.05, 0.2])),
        # Store task info for evaluation
        "dataset_path": str(resolved_path),
        "env_name": cfg.task.env_name,
        "obs_type": cfg.task.obs_type,
        "abs_action": abs_action,
    }

    # Image mode: auto-set encoder and pass keys
    if is_image:
        agent_config["encoder_type"] = "multi_image"
        agent_config["image_keys"] = tuple(image_keys)
        agent_config["lowdim_keys"] = tuple(lowdim_keys)
        crop_shape = cfg.task.dataset.get("crop_shape", None)
        if crop_shape is not None:
            agent_config["crop_shape"] = tuple(crop_shape)
    else:
        # Store obs_keys for lowdim mode
        agent_config["obs_keys"] = tuple(
            cfg.task.get(
                "obs_keys",
                ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
            )
        )

    rng, agent_rng = jax.random.split(rng)
    agent = BCAgent.create(
        seed=int(agent_rng[0]),
        ex_observations=ex_observations,
        ex_actions=ex_actions,
        config=agent_config,
    )

    print("Agent created successfully!")
    print(f"Network step: {agent.network.step}")

    # Initialize W&B
    if cfg.wandb.enabled:
        try:
            import wandb

            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.name,
                config=dict(OmegaConf.to_container(cfg, resolve=True)),
                tags=list(cfg.wandb.tags),
            )
            print("\n✓ W&B logging enabled")
        except ImportError:
            print("\n✗ W&B not installed, skipping logging")
            cfg.wandb.enabled = False

    # Prepare normalizers for checkpoint saving
    normalizers: dict = {
        "action": train_dataset.action_normalizer,
    }
    if is_image:
        # Image mode: save lowdim normalizer if exists
        lowdim_norm = getattr(train_dataset, "lowdim_normalizer", None)
        if lowdim_norm is not None:
            normalizers["lowdim"] = lowdim_norm
    else:
        # Lowdim mode: save obs normalizer
        obs_norm = getattr(train_dataset, "obs_normalizer", None)
        if obs_norm is not None:
            normalizers["obs"] = obs_norm

    # Training loop
    print("\n" + "=" * 80)
    print("Training")
    print("=" * 80)

    best_val_loss = float("inf")
    best_success_rate = -1.0
    train_losses = []
    eval_count = 0

    # Create evaluation environment once (avoid repeated OpenGL context create/destroy → segfault)
    eval_env = None
    if cfg.eval.eval_interval > 0:
        if task_source in ("pusht", "kitchen"):
            eval_env = make_env(
                task_source=task_source,
                obs_type=obs_type if task_source == "pusht" else None,
                env_name=cfg.task.env_name if task_source == "kitchen" else None,
                obs_normalizer=normalizers.get("obs"),
                action_normalizer=normalizers["action"],
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
                obs_type=cfg.task.obs_type,
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
                action_normalizer=normalizers["action"],
                lowdim_normalizer=normalizers.get("lowdim"),
                max_episode_steps=cfg.task.env.max_episode_steps,
                frame_stack=cfg.task.dataset.obs_steps,
                act_exec_steps=cfg.task.dataset.act_steps,
                seed=cfg.seed,
                render_offscreen=cfg.eval.save_video or cfg.eval.render,
                abs_action=abs_action,
            )

    pbar = tqdm(range(cfg.optimization.gradient_steps), desc="Training")

    for step in pbar:
        # Get batch
        batch = next(train_loader)

        # Update agent
        agent, info = agent.update(batch)

        # Log
        train_losses.append(float(info["loss"]))

        if step % cfg.optimization.log_freq == 0:
            avg_loss = np.mean(train_losses[-cfg.optimization.log_freq :])
            pbar.set_postfix(
                {
                    "loss": f"{avg_loss:.4f}",
                    "grad_norm": f"{float(info['grad/norm']):.4f}",
                }
            )

            # Log to W&B
            if cfg.wandb.enabled:
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/grad_norm": float(info["grad/norm"]),
                        "train/step": step,
                        "train/lr": float(info.get("lr", cfg.optimization.lr)),
                        **{
                            f"train/{k}": float(info[k])
                            for k in ("flow_matching_loss", "mf_term", "delta_t", "drift_scale")
                            if k in info
                        },
                    },
                    step=step,
                )

        # Validation
        if step % cfg.optimization.eval_freq == 0 and step > 0:
            print(f"\n[Step {step}] Running validation...")

            val_losses = []
            for _ in range(max(1, min(100, len(val_dataset) // cfg.optimization.batch_size))):
                # Sample once, use both obs and actions from same batch
                val_sample = val_dataset.sample_batch(cfg.optimization.batch_size)
                val_obs = val_sample["observations"]
                val_act = val_sample["actions"]

                # Convert to jax arrays (handle dict obs)
                if isinstance(val_obs, dict):
                    val_obs = {k: jnp.array(v) for k, v in val_obs.items()}
                else:
                    val_obs = jnp.array(val_obs)
                val_batch = {
                    "observations": val_obs,
                    "actions": jnp.array(val_act),
                }

                # Compute loss without updating (use dropout rng for consistency)
                val_rng = jax.random.PRNGKey(step)
                val_dropout_rng, val_crop_rng = jax.random.split(val_rng)
                val_rngs = {"dropout": val_dropout_rng, "crop": val_crop_rng}

                def val_loss_fn(params):
                    def encode(obs, training=False, rngs=None):
                        if rngs is None:
                            rngs = val_rngs
                        return agent.network(
                            obs,
                            training=training,
                            name="encoder",
                            params=params,
                            rngs=rngs,
                        )

                    def flow_net(at, s, t, cond, training=False):
                        return agent.network(
                            at,
                            s,
                            t,
                            cond,
                            training=training,
                            name="flow",
                            params=params,
                            rngs=val_rngs,
                        )

                    return agent.loss_fn_type(
                        network=flow_net,
                        encoder=encode,
                        interpolant=agent.interpolant,
                        batch=val_batch,
                        rng=agent.rng,
                        config=agent.config,
                        step=agent.network.step,
                    )

                loss, _ = val_loss_fn(agent.network.params)
                val_losses.append(float(loss))

            avg_val_loss = np.mean(val_losses)
            print(f"Validation loss: {avg_val_loss:.4f}")

            # Log to W&B
            if cfg.wandb.enabled:
                wandb.log(
                    {
                        "val/loss": avg_val_loss,
                    },
                    step=step,
                )

            # Track best val loss (but save to separate file)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint_path = output_dir / "best_val_loss_model.pkl"
                print(f"New best val loss: {avg_val_loss:.4f} — saving to {checkpoint_path}")
                save_checkpoint(
                    checkpoint_path=checkpoint_path,
                    agent=agent,
                    step=step,
                    normalizers=normalizers,
                    best_val_loss=float(best_val_loss),
                )

        # Environment evaluation
        if cfg.eval.eval_interval > 0 and step % cfg.eval.eval_interval == 0 and step > 0 and eval_env is not None:
            print(f"\n[Step {step}] Running environment evaluation...")
            eval_count += 1

            # Run evaluation
            eval_results = evaluate_policy(
                agent=agent,
                env=eval_env,
                num_episodes=cfg.eval.num_episodes,
                max_steps=cfg.task.env.max_episode_steps,
                render=cfg.eval.render,
                save_video=cfg.eval.save_video,
                num_videos=cfg.eval.num_videos,
                verbose=True,
            )

            # Print results
            print_evaluation_results(eval_results)

            # Log to W&B
            if cfg.wandb.enabled:
                eval_log = {
                    "eval/success_rate": eval_results["success_rate"],
                    "eval/num_successes": eval_results["num_successes"],
                    "eval/avg_length": eval_results["avg_length"],
                    "eval/std_length": eval_results["std_length"],
                    "eval/avg_return": eval_results["avg_return"],
                    "eval/std_return": eval_results["std_return"],
                    "eval/mean_score": eval_results["mean_score"],
                }
                if "action_mean" in eval_results:
                    eval_log["eval/action_mean"] = eval_results["action_mean"]
                    eval_log["eval/action_std"] = eval_results["action_std"]
                    eval_log["eval/action_abs_mean"] = eval_results["action_abs_mean"]
                    eval_log["eval/action_clip_ratio"] = eval_results[
                        "action_clip_ratio"
                    ]
                # Kitchen p1-p7 metrics
                for k in range(1, 8):
                    pk = f"p{k}"
                    if pk in eval_results:
                        eval_log[f"eval/{pk}"] = eval_results[pk]
                wandb.log(eval_log, step=step)

                # Upload videos
                if (
                    cfg.wandb.log_videos
                    and "videos" in eval_results
                    and eval_count % cfg.wandb.log_video_interval == 0
                ):
                    for i, video_frames in enumerate(eval_results["videos"]):
                        # video_frames: (T, H, W, C) in uint8
                        video_frames_transposed = np.transpose(
                            video_frames, (0, 3, 1, 2)
                        )  # (T, C, H, W)
                        wandb.log(
                            {
                                f"eval/video_{i}": wandb.Video(
                                    video_frames_transposed,
                                    fps=cfg.eval.video_fps,
                                    format="mp4",
                                )
                            },
                            step=step,
                        )

            # Save best model based on eval success_rate (or mean_score)
            eval_score = eval_results["success_rate"]
            if eval_score > best_success_rate:
                best_success_rate = eval_score
                best_checkpoint_path = output_dir / "best_model.pkl"
                print(f"New best success rate: {eval_score:.2%} — saving to {best_checkpoint_path}")
                save_checkpoint(
                    checkpoint_path=best_checkpoint_path,
                    agent=agent,
                    step=step,
                    normalizers=normalizers,
                    best_val_loss=float(best_val_loss)
                    if best_val_loss != float("inf")
                    else None,
                )

        # Save checkpoint
        if step % cfg.checkpoint.save_freq == 0 and step > 0:
            checkpoint_path = output_dir / f"checkpoint_{step}.pkl"
            print(f"\nSaving checkpoint to {checkpoint_path}")
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                agent=agent,
                step=step,
                normalizers=normalizers,
                best_val_loss=float(best_val_loss)
                if best_val_loss != float("inf")
                else None,
            )

            # Cleanup old checkpoints
            if cfg.checkpoint.keep_last_n > 0:
                cleanup_old_checkpoints(
                    output_dir, keep_last_n=cfg.checkpoint.keep_last_n
                )

    # Close evaluation environment
    if eval_env is not None:
        eval_env.close()

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Final train loss: {np.mean(train_losses[-100:]):.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
