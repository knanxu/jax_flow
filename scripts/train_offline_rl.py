"""Training script for Offline RL (DDPG + BC) with flow policy.

Two-phase training:
  Phase 1: BCAgent trains actor (100% identical to train_bc.py) + CriticState trains critic
  Phase 2: DDPGBCAgent jointly optimizes actor (Q+BC loss) and critic

Usage:
    # Mode 1: BC warmup -> RL
    python scripts/train_offline_rl.py task=pusht_image \
        algorithm.bc_checkpoint=checkpoints/pusht_image_mip_unet/best_model.pkl \
        algorithm.bc_warmup_steps=150000 algorithm.rl_steps=1000000

    # Mode 2: Direct RL (no BC warmup)
    python scripts/train_offline_rl.py task=square_lowdim \
        algorithm.bc_checkpoint=checkpoints/square_lowdim_mip_mlp/best_model.pkl \
        algorithm.bc_warmup_steps=0 algorithm.rl_steps=500000
"""

from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from jax_flow.agents.bc_agent import BCAgent
from jax_flow.agents.critic_state import CriticState
from jax_flow.agents.offline_online.ddpg_bc_agent import DDPGBCAgent
from jax_flow.core.checkpoint import (
    cleanup_old_checkpoints,
    load_checkpoint,
    save_checkpoint,
    save_ddpg_bc_checkpoint,
)
from jax_flow.core.evaluation import evaluate_policy, print_evaluation_results
from jax_flow.data import DatasetManager, make_dataset, make_robomimic_dataset
from jax_flow.envs import make_env, make_robomimic_env


def create_bc_config(bc_ckpt_config):
    """Build BCAgent config from BC checkpoint config.

    NOTE: gradient_steps is NOT overridden — it controls both the LR cosine
    schedule and the MeanFlow delta_t progressive schedule.  Changing it to
    bc_warmup_steps compresses the delta_t ramp and degrades MeanFlow training.
    """
    return dict(bc_ckpt_config)


def create_rl_config(bc_ckpt_config, algo_cfg, task_cfg):
    """Build DDPGBCAgent config for RL phase."""
    critic_encoder_type = algo_cfg.get("critic_encoder_type", "none")
    return {
        # From BC config (needed for sampler/loss)
        "horizon": bc_ckpt_config.get("horizon", task_cfg.dataset.horizon),
        "obs_steps": bc_ckpt_config.get("obs_steps", task_cfg.dataset.obs_steps),
        "action_dim": bc_ckpt_config.get("action_dim"),
        "obs_dim": bc_ckpt_config.get("obs_dim"),
        "sampler_type": bc_ckpt_config.get("sampler_type", "euler"),
        "flow_steps": bc_ckpt_config.get("flow_steps", 10),
        "sample_mode": bc_ckpt_config.get("sample_mode", "stochastic"),
        "loss_scale": bc_ckpt_config.get("loss_scale", 0.1),
        "interp_type": bc_ckpt_config.get("interp_type", "linear"),
        "flow_type": bc_ckpt_config.get("flow_type", "flow_matching"),
        "t_two_step": bc_ckpt_config.get("t_two_step", 0.9),
        "delta_t_schedule": bc_ckpt_config.get("delta_t_schedule", {}),
        "adaptive_weight": bc_ckpt_config.get("adaptive_weight", {}),
        "time_steps": bc_ckpt_config.get("time_steps", 0),
        "consistency_alpha": bc_ckpt_config.get("consistency_alpha", 0.0),
        # RL params
        "act_exec_steps": bc_ckpt_config.get("act_steps", task_cfg.dataset.act_steps),
        "critic_hidden_dims": list(algo_cfg.get("critic_hidden_dims", [2048, 2048, 2048])),
        "critic_activation": algo_cfg.get("critic_activation", "tanh"),
        "critic_layer_norm": algo_cfg.get("critic_layer_norm", True),
        "num_ensembles": algo_cfg.get("num_ensembles", 2),
        "tau": algo_cfg.get("tau", 0.005),
        "discount": algo_cfg.get("discount", 0.99),
        "alpha": algo_cfg.get("alpha", 1.0),
        "q_weight": algo_cfg.get("q_weight", 1.0),
        "normalize_q_loss": algo_cfg.get("normalize_q_loss", True),
        "actor_lr": algo_cfg.get("actor_lr", 1e-4),
        "critic_lr": algo_cfg.get("critic_lr", 3e-4),
        "grad_clip_norm": bc_ckpt_config.get("grad_clip_norm", 10.0),
        "freeze_encoder": algo_cfg.get("freeze_encoder", True),
        # Critic encoder
        "critic_encoder_type": critic_encoder_type,
        "critic_has_encoder": critic_encoder_type != "none",
    }


def create_critic_config(algo_cfg, bc_ckpt_config):
    """Build CriticState config for BC warmup phase."""
    critic_encoder_type = algo_cfg.get("critic_encoder_type", "none")
    config = {
        "critic_hidden_dims": list(algo_cfg.get("critic_hidden_dims", [2048, 2048, 2048])),
        "critic_activation": algo_cfg.get("critic_activation", "tanh"),
        "critic_layer_norm": algo_cfg.get("critic_layer_norm", True),
        "num_ensembles": algo_cfg.get("num_ensembles", 2),
        "critic_lr": algo_cfg.get("critic_lr", 3e-4),
        "grad_clip_norm": bc_ckpt_config.get("grad_clip_norm", 10.0),
        "tau": algo_cfg.get("tau", 0.005),
        "discount": algo_cfg.get("discount", 0.99),
        "act_exec_steps": bc_ckpt_config.get("act_steps", 8),
        # Critic encoder
        "critic_encoder_type": critic_encoder_type,
    }
    # Pass BC encoder architecture params when critic needs its own encoder
    if critic_encoder_type != "none":
        config.update({
            "encoder_type": bc_ckpt_config.get("encoder_type", "mlp"),
            "encoder_hidden_dims": list(bc_ckpt_config.get("encoder_hidden_dims", [256, 256])),
            "emb_dim": bc_ckpt_config.get("emb_dim", 256),
            "image_keys": list(bc_ckpt_config.get("image_keys", ["agentview_image"])),
            "lowdim_keys": list(bc_ckpt_config.get(
                "lowdim_keys", ["robot0_eef_pos", "robot0_gripper_qpos"]
            )),
            "crop_shape": bc_ckpt_config.get("crop_shape", None),
        })
    return config


@hydra.main(
    version_base=None, config_path="../configs/offline_online", config_name="ddpg_bc_default"
)
def main(cfg: DictConfig):
    """Main offline RL training function."""
    print("=" * 80)
    print("Offline RL: DDPG + BC on Flow Policy")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Validate
    bc_checkpoint_path = cfg.algorithm.bc_checkpoint
    if bc_checkpoint_path is None:
        raise ValueError("algorithm.bc_checkpoint is required.")

    np.random.seed(cfg.seed)
    rng_np = np.random.default_rng(cfg.seed)

    # ================================================================
    # 1. Load BC checkpoint config and normalizers
    # ================================================================
    print("\nLoading BC checkpoint...")
    bc_ckpt = load_checkpoint(bc_checkpoint_path)
    bc_ckpt_config = bc_ckpt["config"]
    normalizers = bc_ckpt.get("normalizers", {})

    print(f"BC checkpoint flow_type: {bc_ckpt_config.get('flow_type', 'unknown')}")
    print(f"BC checkpoint network_type: {bc_ckpt_config.get('network_type', 'unknown')}")
    print(f"BC checkpoint gradient_steps: {bc_ckpt_config.get('gradient_steps', 'unknown')}")
    print(f"(NOTE: the Hydra config printed above shows defaults, not the actual BC policy config)")

    # ================================================================
    # 2. Resolve dataset
    # ================================================================
    algo = cfg.algorithm
    task_source = cfg.task.get("task_source", "robomimic")
    dataset_path = bc_ckpt_config.get("dataset_path", cfg.task.dataset.path)
    task_name = cfg.task.get("dataset_name", cfg.task.env_name.lower())
    dataset_type = cfg.task.dataset.get("dataset_type", "ph")
    obs_type = cfg.task.obs_type
    is_image = obs_type == "image"
    abs_action = cfg.task.get("abs_action", False)

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

    # ================================================================
    # 3. Create dataset
    # ================================================================
    print("\nLoading dataset...")
    dataset_kwargs = {}
    if abs_action:
        dataset_kwargs["abs_action"] = True
    if is_image:
        image_keys = list(cfg.task.dataset.get("image_keys", ["agentview_image"]))
        lowdim_keys = list(cfg.task.dataset.get(
            "lowdim_keys", ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
        ))
        dataset_kwargs["image_keys"] = tuple(image_keys)
        dataset_kwargs["lowdim_keys"] = tuple(lowdim_keys)
    else:
        if "obs_keys" in cfg.task:
            dataset_kwargs["obs_keys"] = tuple(cfg.task.obs_keys)

    reward_offset = algo.get("reward_offset", -1.0)
    common_kwargs = {
        "horizon": cfg.task.dataset.horizon,
        "obs_steps": cfg.task.dataset.obs_steps,
        "act_steps": cfg.task.dataset.act_steps,
        "val_ratio": 0.0,
    }

    if task_source in ("pusht", "kitchen"):
        train_dataset = make_dataset(
            task_source=task_source, dataset_path=dataset_path,
            obs_type=obs_type, mode="train", **common_kwargs,
        )
    else:
        train_dataset = make_robomimic_dataset(
            dataset_path=dataset_path, obs_type=obs_type, mode="train",
            reward_offset=reward_offset, **common_kwargs, **dataset_kwargs,
        )

    print(f"Dataset size: {len(train_dataset)}")

    # ================================================================
    # 4. Get example data
    # ================================================================
    batch_size = algo.get("batch_size", 64)
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

    # ================================================================
    # 5. Output directory and W&B
    # ================================================================
    # Use flow_type and network_type from BC checkpoint (not Hydra defaults)
    bc_flow_type = bc_ckpt_config.get("flow_type", cfg.flow.name)
    bc_network_type = bc_ckpt_config.get("network_type", cfg.network.name)
    experiment_name = f"offline_rl_{cfg.task.name}_{bc_flow_type}_{bc_network_type}"
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
    # 6. Create eval environment
    # ================================================================
    eval_env = None
    eval_interval = cfg.eval.get("eval_interval", 10000)
    if eval_interval > 0:
        eval_kwargs = {
            "obs_normalizer": normalizers.get("obs"),
            "action_normalizer": normalizers.get("action"),
            "lowdim_normalizer": normalizers.get("lowdim"),
            "max_episode_steps": cfg.task.env.max_episode_steps,
            "frame_stack": cfg.task.dataset.obs_steps,
            "act_exec_steps": cfg.task.dataset.act_steps,
            "seed": cfg.seed,
        }
        if task_source in ("pusht", "kitchen"):
            eval_env = make_env(
                task_source=task_source,
                obs_type=obs_type if task_source == "pusht" else None,
                env_name=cfg.task.env_name if task_source == "kitchen" else None,
                **eval_kwargs,
            )
        else:
            eval_env = make_robomimic_env(
                env_name=cfg.task.env_name,
                dataset_path=resolved_path,
                obs_type=obs_type,
                obs_keys=tuple(cfg.task.dataset.get("obs_keys", [
                    "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object",
                ])),
                image_keys=tuple(image_keys) if is_image else None,
                lowdim_keys=tuple(lowdim_keys) if is_image else None,
                render_offscreen=cfg.eval.get("save_video", True),
                abs_action=abs_action,
                **eval_kwargs,
            )

    # ================================================================
    # Helper: sample batch
    # ================================================================
    act_exec_steps = algo.get("act_exec_steps", cfg.task.dataset.act_steps)

    def sample_batch():
        batch_np = train_dataset.sample_sequence(
            batch_size=batch_size,
            discount=algo.get("discount", 0.99),
            rng=rng_np,
            reward_offset=reward_offset,
            act_exec_steps=act_exec_steps,
        )
        if isinstance(batch_np["observations"], dict):
            obs = {k: jnp.array(v) for k, v in batch_np["observations"].items()}
            next_obs = {k: jnp.array(v) for k, v in batch_np["next_observations"].items()}
        else:
            obs = jnp.array(batch_np["observations"])
            next_obs = jnp.array(batch_np["next_observations"])
        return {
            "observations": obs,
            "actions": jnp.array(batch_np["actions"]),
            "rewards": jnp.array(batch_np["rewards"]),
            "next_observations": next_obs,
            "masks": jnp.array(batch_np["masks"]),
        }

    # ================================================================
    # Helper: evaluate
    # ================================================================
    log_interval = algo.get("log_interval", 1000)
    checkpoint_interval = cfg.checkpoint.get("save_freq", 50000)
    best_success_rate = -1.0
    eval_count = 0
    global_step = 0

    def run_eval(agent, step_label):
        nonlocal best_success_rate, eval_count
        if eval_env is None or eval_interval <= 0:
            return
        eval_count += 1
        print(f"\n[Step {step_label}] Evaluating...")
        eval_results = evaluate_policy(
            agent=agent, env=eval_env,
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
            wandb.log(eval_log, step=step_label)

        if eval_results["success_rate"] > best_success_rate:
            best_success_rate = eval_results["success_rate"]
            print(f"New best! Success rate: {best_success_rate:.2%}")
        return eval_results

    # ================================================================
    # 7. Phase 1: BC warmup
    # ================================================================
    bc_warmup_steps = algo.get("bc_warmup_steps", 0)
    rl_steps = algo.get("rl_steps", 500000)

    if bc_warmup_steps > 0:
        print("\n" + "=" * 80)
        print(f"Phase 1: BC Warmup ({bc_warmup_steps} steps)")
        print("=" * 80)

        # Create BCAgent (100% identical to train_bc.py)
        bc_config = create_bc_config(bc_ckpt_config)
        bc_agent = BCAgent.create(
            seed=cfg.seed,
            ex_observations=ex_obs,
            ex_actions=ex_actions,
            config=bc_config,
        )
        print("BCAgent created.")

        # Create CriticState
        # Need ex_cond from BCAgent's encoder
        ex_cond = bc_agent.network(
            ex_obs, training=False, name="encoder",
            params=bc_agent.network.params, rngs={},
        )
        critic_config = create_critic_config(algo, bc_ckpt_config)
        critic_state = CriticState.create(
            seed=cfg.seed + 1,
            ex_cond=ex_cond,
            ex_actions=ex_actions,
            config=critic_config,
            ex_observations=ex_obs,
        )
        print("CriticState created.")

        # BC training loop
        pbar = tqdm(range(bc_warmup_steps), desc="BC Warmup")
        for step in pbar:
            batch = sample_batch()

            # Update BCAgent (identical to BCAgent.update)
            bc_batch = {"observations": batch["observations"], "actions": batch["actions"]}
            bc_agent, bc_info = bc_agent.update(bc_batch)

            # Update CriticState
            critic_state, critic_info = critic_state.update(batch, bc_agent)

            global_step = step

            if step % log_interval == 0:
                pbar.set_postfix({
                    "phase": "BC",
                    "bc_loss": f"{float(bc_info['loss']):.4f}",
                    "c_loss": f"{float(critic_info['critic_loss']):.3f}",
                    "q": f"{float(critic_info['q_mean']):.2f}",
                })
                if wandb_enabled:
                    wandb.log({
                        "train/bc_loss": float(bc_info["loss"]),
                        "train/critic_loss": float(critic_info["critic_loss"]),
                        "train/q_mean": float(critic_info["q_mean"]),
                        "train/target_q_mean": float(critic_info["target_q_mean"]),
                        "train/phase": 0,
                        "train/step": step,
                    }, step=step)

            # Eval (uses BCAgent with EMA params)
            if eval_interval > 0 and step % eval_interval == 0 and step > 0:
                eval_results = run_eval(bc_agent, step)
                if eval_results and eval_results["success_rate"] >= best_success_rate:
                    save_checkpoint(
                        output_dir / "best_bc_model.pkl",
                        bc_agent, step, normalizers=normalizers,
                        best_val_loss=best_success_rate,
                    )

            if checkpoint_interval > 0 and step % checkpoint_interval == 0 and step > 0:
                save_checkpoint(
                    output_dir / f"bc_checkpoint_{step}.pkl",
                    bc_agent, step, normalizers=normalizers,
                )
                cleanup_old_checkpoints(
                    output_dir, keep_last_n=cfg.checkpoint.get("keep_last_n", 3),
                )

        pbar.close()
        print(f"\nBC warmup complete. Best success rate: {best_success_rate:.2%}")

        # Create DDPGBCAgent from BCAgent + CriticState
        print("\nCreating DDPGBCAgent from BC + Critic...")
        rl_config = create_rl_config(bc_ckpt_config, algo, cfg.task)
        rl_agent = DDPGBCAgent.create_from_bc(
            seed=cfg.seed + 2,
            bc_agent=bc_agent,
            critic_state=critic_state,
            config=rl_config,
        )
        print("DDPGBCAgent created from BC warmup.")

    else:
        # Mode 2: Direct RL (no BC warmup)
        print("\n" + "=" * 80)
        print("Mode 2: Direct Offline RL (no BC warmup)")
        print("=" * 80)

        rl_config = create_rl_config(bc_ckpt_config, algo, cfg.task)
        rl_agent = DDPGBCAgent.create(
            seed=cfg.seed,
            ex_observations=ex_obs,
            ex_actions=ex_actions,
            config=rl_config,
            bc_checkpoint_path=bc_checkpoint_path,
        )
        print("DDPGBCAgent created.")

    # ================================================================
    # 8. Phase 2: RL training
    # ================================================================
    print("\n" + "=" * 80)
    print(f"Phase 2: Offline RL ({rl_steps} steps)")
    print("=" * 80)

    pbar = tqdm(range(rl_steps), desc="Offline RL")
    for step in pbar:
        batch = sample_batch()
        rl_agent, info = rl_agent.update(batch)

        rl_step = bc_warmup_steps + step
        global_step = rl_step

        if step % log_interval == 0:
            pbar.set_postfix({
                "phase": "RL",
                "loss": f"{float(info['loss']):.4f}",
                "c_loss": f"{float(info['critic_loss']):.3f}",
                "bc": f"{float(info['bc_loss']):.3f}",
                "q": f"{float(info['q_mean']):.2f}",
            })
            if wandb_enabled:
                wandb.log({
                    "train/loss": float(info["loss"]),
                    "train/critic_loss": float(info["critic_loss"]),
                    "train/actor_loss": float(info["actor_loss"]),
                    "train/bc_loss": float(info["bc_loss"]),
                    "train/q_loss": float(info["q_loss"]),
                    "train/q_loss_raw": float(info["q_loss_raw"]),
                    "train/q_mean": float(info["q_mean"]),
                    "train/target_q_mean": float(info["target_q_mean"]),
                    "train/q_actor_mean": float(info["q_actor_mean"]),
                    "train/phase": 1,
                    "train/step": rl_step,
                }, step=rl_step)

        if eval_interval > 0 and step % eval_interval == 0 and step > 0:
            eval_results = run_eval(rl_agent, rl_step)
            if eval_results and eval_results["success_rate"] >= best_success_rate:
                save_ddpg_bc_checkpoint(
                    output_dir / "best_model.pkl",
                    rl_agent, rl_step,
                    bc_checkpoint_path=bc_checkpoint_path,
                    normalizers=normalizers,
                    best_success_rate=best_success_rate,
                )

        if checkpoint_interval > 0 and step % checkpoint_interval == 0 and step > 0:
            save_ddpg_bc_checkpoint(
                output_dir / f"rl_checkpoint_{rl_step}.pkl",
                rl_agent, rl_step,
                bc_checkpoint_path=bc_checkpoint_path,
                normalizers=normalizers,
            )
            cleanup_old_checkpoints(
                output_dir, keep_last_n=cfg.checkpoint.get("keep_last_n", 3),
            )

    pbar.close()

    # Final save
    save_ddpg_bc_checkpoint(
        output_dir / "final_model.pkl",
        rl_agent, global_step,
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
