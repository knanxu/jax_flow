"""Evaluate BC policy at fixed speed multipliers.

Sweeps a range of fixed speeds and reports success rate / avg length for each,
so you can see the success-rate ceiling before training an RL speed policy.

Usage:
    python scripts/eval_fixed_speed.py task=square_lowdim \
        algorithm.bc_checkpoint=checkpoints/square_lowdim_meanflow_mlp/best_model.pkl

    # Custom speed range
    python scripts/eval_fixed_speed.py task=square_lowdim \
        algorithm.bc_checkpoint=path/to/best_model.pkl \
        algorithm.max_speed=4.0 algorithm.speed_granularity=0.5

    # More eval episodes
    python scripts/eval_fixed_speed.py task=square_lowdim \
        algorithm.bc_checkpoint=path/to/best_model.pkl \
        eval.num_episodes=50
"""

import os

_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_strict_conv_algorithm_picker" not in _xla_flags:
    os.environ["XLA_FLAGS"] = _xla_flags + " --xla_gpu_strict_conv_algorithm_picker=false"

import hydra
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

from jax_flow.agents.bc_agent import BCAgent
from jax_flow.agents.speed_tuning.interpolation import make_speed_options
from jax_flow.agents.speed_tuning.speed_tuning_env import SpeedTuningEnvWrapper
from jax_flow.core.checkpoint import load_checkpoint, restore_agent
from jax_flow.data import DatasetManager, make_dataset, make_robomimic_dataset
from jax_flow.envs import make_env, make_robomimic_env


def create_base_env(cfg, normalizers, resolved_path, abs_action=False):
    """Create base env with FrameStack only (no ActionChunking)."""
    task_source = cfg.task.get("task_source", "robomimic")
    obs_type = cfg.task.obs_type
    is_image = obs_type == "image"

    common_kwargs = {
        "obs_normalizer": normalizers.get("obs"),
        "action_normalizer": normalizers.get("action"),
        "lowdim_normalizer": normalizers.get("lowdim"),
        "max_episode_steps": cfg.task.env.max_episode_steps,
        "frame_stack": cfg.task.dataset.obs_steps,
        "act_exec_steps": None,
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
            render_offscreen=False,
            abs_action=abs_action,
            **common_kwargs,
        )

    return env


def eval_fixed_speed(env, num_episodes, max_steps):
    """Run episodes with a fixed speed (already set in the wrapper)."""
    successes = []
    lengths = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_length = 0
        ep_success = False

        while not done and ep_length < max_steps:
            # Pass 0 as action — wrapper has single speed_option
            obs, reward, terminated, truncated, info = env.step(0)
            done = terminated or truncated
            ep_length += info.get("steps_executed", 1)

            if "success" in info and bool(info["success"]):
                ep_success = True

        successes.append(ep_success)
        lengths.append(ep_length)

    success_rate = float(np.mean(successes))
    avg_length = float(np.mean(lengths))
    success_lengths = [l for s, l in zip(successes, lengths) if s]
    avg_success_length = float(np.mean(success_lengths)) if success_lengths else 0.0

    return {
        "success_rate": success_rate,
        "num_successes": int(np.sum(successes)),
        "avg_length": avg_length,
        "avg_success_length": avg_success_length,
    }


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="speed_tuning_config",
)
def main(cfg: DictConfig):
    print("=" * 80)
    print("Fixed Speed Evaluation Sweep")
    print("=" * 80)

    bc_checkpoint_path = cfg.algorithm.bc_checkpoint
    if bc_checkpoint_path is None:
        raise ValueError("algorithm.bc_checkpoint is required.")

    np.random.seed(cfg.seed)
    algo = cfg.algorithm

    # Load BC checkpoint
    bc_ckpt = load_checkpoint(bc_checkpoint_path)
    bc_ckpt_config = bc_ckpt["config"]
    normalizers = bc_ckpt.get("normalizers", {})

    # Resolve dataset path
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
        print("Failed to load dataset. Exiting.")
        return
    resolved_path = str(resolved_path)

    # Create BC agent
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

    ex_sample = train_dataset[0]
    if isinstance(ex_sample["observations"], dict):
        ex_obs = {
            k: jnp.array(v[np.newaxis, ...])
            for k, v in ex_sample["observations"].items()
        }
    else:
        ex_obs = jnp.array(ex_sample["observations"][np.newaxis, ...])

    bc_action_dim = bc_ckpt_config.get("action_dim", ex_sample["actions"].shape[-1])
    bc_horizon = bc_ckpt_config.get("horizon", 16)
    ex_actions = jnp.zeros((1, bc_horizon, bc_action_dim))

    # Use restore_agent for correct loading (restores rng, ema_params, opt_state)
    bc_agent, _ = restore_agent(bc_checkpoint_path, BCAgent, ex_obs, ex_actions)

    # Build speed list to sweep
    speed_options = make_speed_options(
        max_speed=algo.get("max_speed", 2.0),
        granularity=algo.get("speed_granularity", 0.1),
    )
    bc_abs_action = bc_ckpt_config.get("abs_action", False)
    num_episodes = cfg.eval.get("num_episodes", 20)
    max_steps = cfg.task.env.max_episode_steps
    k_skip = algo.get("k_skip", 4)

    print(f"Task: {cfg.task.name}")
    print(f"BC checkpoint: {bc_checkpoint_path}")
    print(f"Speeds to test: {speed_options}")
    print(f"Episodes per speed: {num_episodes}")
    print(f"k_skip: {k_skip}")
    print("=" * 80)

    # Sweep
    results_table = []
    for speed in speed_options:
        env = create_base_env(cfg, normalizers, resolved_path, abs_action=bc_abs_action)
        env = SpeedTuningEnvWrapper(
            env=env,
            bc_agent=bc_agent,
            speed_options=[speed],  # single speed
            alpha=0.0,  # no speed reward — pure task eval
            beta=0.0,
            k_skip=k_skip,
            abs_action=bc_abs_action,
        )
        # Force the wrapper to always use this speed
        env._current_speed = speed

        res = eval_fixed_speed(env, num_episodes, max_steps)
        results_table.append((speed, res))

        print(
            f"  speed={speed:.2f}  |  "
            f"success={res['success_rate']:.2%} ({res['num_successes']}/{num_episodes})  |  "
            f"avg_len={res['avg_length']:.0f}  |  "
            f"avg_success_len={res['avg_success_length']:.0f}"
        )

    # Summary table
    print("\n" + "=" * 80)
    print(f"{'Speed':>6}  {'Success%':>9}  {'Avg Len':>8}  {'Success Len':>12}")
    print("-" * 42)
    for speed, res in results_table:
        print(
            f"{speed:>6.2f}  {res['success_rate']:>8.1%}  "
            f"{res['avg_length']:>8.0f}  {res['avg_success_length']:>12.0f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
