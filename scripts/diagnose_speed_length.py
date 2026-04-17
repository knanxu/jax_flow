"""Diagnose why high speed doesn't reduce episode length.

This script analyzes:
1. Speed trajectory vs episode length for successful episodes
2. Interpolation behavior at different speeds
3. Action magnitude and delta distribution
4. Temporal coverage of action chunks

Usage:
    python scripts/diagnose_speed_length.py task=square_lowdim \
        algorithm.bc_checkpoint=path/to/bc/best_model.pkl \
        algorithm.speed_checkpoint=path/to/speed/best_model.pkl
"""

import os

_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_strict_conv_algorithm_picker" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (
        _xla_flags + " --xla_gpu_strict_conv_algorithm_picker=false"
    )

from pathlib import Path

import hydra
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig

from jax_flow.agents.bc_agent import BCAgent
from jax_flow.agents.speed_tuning.interpolation import make_speed_options, temporal_interpolate
from jax_flow.agents.speed_tuning.rainbow_agent import RainbowDQNAgent
from jax_flow.agents.speed_tuning.speed_tuning_env import SpeedTuningEnvWrapper
from jax_flow.core.checkpoint import load_checkpoint, restore_agent
from jax_flow.data import DatasetManager, make_dataset, make_robomimic_dataset
from jax_flow.envs import make_env, make_robomimic_env


def _obs_to_batch(obs):
    """Convert single obs to batched obs."""
    if isinstance(obs, dict):
        return {k: np.array(v[np.newaxis, ...]) for k, v in obs.items()}
    return obs[np.newaxis, ...]


def create_base_env(cfg, normalizers, resolved_path, bc_ckpt_config, abs_action=False):
    """Create base env with FrameStack only (no ActionChunking)."""
    task_source = cfg.task.get("task_source", "robomimic")
    obs_type = bc_ckpt_config.get("obs_type", cfg.task.obs_type)
    is_image = obs_type == "image"

    common_kwargs = {
        "obs_normalizer": normalizers.get("obs"),
        "action_normalizer": normalizers.get("action"),
        "lowdim_normalizer": normalizers.get("lowdim"),
        "max_episode_steps": cfg.task.env.max_episode_steps,
        "frame_stack": bc_ckpt_config.get("obs_steps", cfg.task.dataset.obs_steps),
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
        if is_image:
            image_keys = tuple(bc_ckpt_config.get("image_keys", ("agentview_image",)))
            lowdim_keys = tuple(
                bc_ckpt_config.get(
                    "lowdim_keys",
                    ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
                )
            )
            obs_keys = None
        else:
            image_keys = None
            lowdim_keys = None
            obs_keys = tuple(
                bc_ckpt_config.get(
                    "obs_keys",
                    ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"),
                )
            )

        env = make_robomimic_env(
            env_name=bc_ckpt_config.get("env_name", cfg.task.env_name),
            dataset_path=resolved_path,
            obs_type=obs_type,
            obs_keys=obs_keys,
            image_keys=image_keys,
            lowdim_keys=lowdim_keys,
            render_offscreen=True,
            abs_action=abs_action,
            **common_kwargs,
        )

    return env


def restore_speed_agent(speed_ckpt_path, env, bc_checkpoint_path):
    """Restore RainbowDQNAgent from SpeedTuning checkpoint."""
    speed_ckpt = load_checkpoint(speed_ckpt_path)

    obs, _ = env.reset()
    env_obs_batch = _obs_to_batch(obs)
    if isinstance(env_obs_batch, dict):
        env_obs_batch = {k: jnp.array(v) for k, v in env_obs_batch.items()}
    else:
        env_obs_batch = jnp.array(env_obs_batch)

    agent = RainbowDQNAgent.create(
        seed=0,
        ex_observations=env_obs_batch,
        config=speed_ckpt["config"],
        bc_checkpoint_path=bc_checkpoint_path,
    )
    agent = agent.replace(
        network=agent.network.replace(
            params=speed_ckpt["params"],
            opt_state=speed_ckpt["opt_state"],
            step=speed_ckpt["step"],
        ),
        rng=speed_ckpt["rng"],
    )
    return agent, speed_ckpt


def analyze_episode_detailed(speed_agent, env, bc_agent, action_normalizer, seed, max_steps):
    """Run one episode and collect detailed diagnostics.

    Returns:
        dict with trajectory, action_chunks, interpolated_actions, speeds, etc.
    """
    obs, _ = env.reset(seed=seed)
    done = False
    ep_length = 0

    # Collect data
    macro_steps = []
    speeds = []
    action_chunks = []
    interpolated_actions = []
    steps_executed_list = []

    macro_step_idx = 0
    while not done and ep_length < max_steps:
        obs_batch = _obs_to_batch(obs)
        if isinstance(obs_batch, dict):
            obs_batch = {k: jnp.array(v) for k, v in obs_batch.items()}
        else:
            obs_batch = jnp.array(obs_batch)

        # Get speed decision
        action = speed_agent.eval_action(obs_batch)
        speed_idx = int(action[0])
        speed = env.speed_options[speed_idx]

        # Get BC action chunk (before interpolation)
        bc_actions = bc_agent.eval_actions(obs_batch)
        action_chunk = np.array(bc_actions[0])  # (horizon, action_dim)

        # Manually interpolate to see what happens
        if action_normalizer is not None:
            chunk_raw = action_normalizer.unnormalize(action_chunk)
        else:
            chunk_raw = action_chunk

        rot6d_slices = env._rot6d_slices
        interpolated = temporal_interpolate(
            chunk_raw, speed, env.k_skip, rot6d_slices=rot6d_slices
        )

        # Execute
        obs, _reward, terminated, truncated, info = env.step(speed_idx)
        done = terminated or truncated
        steps_executed = info.get("steps_executed", 1)

        # Record
        macro_steps.append(macro_step_idx)
        speeds.append(speed)
        action_chunks.append(action_chunk)
        interpolated_actions.append(interpolated)
        steps_executed_list.append(steps_executed)

        ep_length += steps_executed
        macro_step_idx += 1

        if "success" in info and bool(info["success"]):
            break

    return {
        "macro_steps": macro_steps,
        "speeds": speeds,
        "action_chunks": action_chunks,
        "interpolated_actions": interpolated_actions,
        "steps_executed": steps_executed_list,
        "total_length": ep_length,
        "success": info.get("success", False),
    }


def plot_diagnostics(results_list, output_dir):
    """Generate diagnostic plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter successful episodes
    success_results = [r for r in results_list if r["success"]]
    if not success_results:
        print("No successful episodes to analyze.")
        return

    print(f"\nAnalyzing {len(success_results)} successful episodes...")

    # Figure 1: Speed vs steps_executed per macro-step
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Speed trajectory
    ax = axes[0, 0]
    for r in success_results[:5]:  # Plot first 5
        ax.plot(r["macro_steps"], r["speeds"], alpha=0.6, marker="o", markersize=3)
    ax.set_xlabel("Macro-step Index")
    ax.set_ylabel("Speed Multiplier")
    ax.set_title("Speed Trajectory (first 5 successful episodes)")
    ax.grid(True, alpha=0.3)

    # Plot 2: Steps executed per macro-step
    ax = axes[0, 1]
    for r in success_results[:5]:
        ax.plot(r["macro_steps"], r["steps_executed"], alpha=0.6, marker="o", markersize=3)
    ax.axhline(y=4, color="red", linestyle="--", label="k_skip=4")
    ax.set_xlabel("Macro-step Index")
    ax.set_ylabel("Steps Executed")
    ax.set_title("Steps Executed per Macro-step")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Speed vs steps_executed correlation
    ax = axes[1, 0]
    all_speeds = []
    all_steps = []
    for r in success_results:
        all_speeds.extend(r["speeds"])
        all_steps.extend(r["steps_executed"])
    ax.scatter(all_speeds, all_steps, alpha=0.3, s=10)
    ax.set_xlabel("Speed Multiplier")
    ax.set_ylabel("Steps Executed")
    ax.set_title("Speed vs Steps Executed (all macro-steps)")
    ax.grid(True, alpha=0.3)

    # Plot 4: Episode length vs average speed
    ax = axes[1, 1]
    avg_speeds = [np.mean(r["speeds"]) for r in success_results]
    lengths = [r["total_length"] for r in success_results]
    ax.scatter(avg_speeds, lengths, alpha=0.6, s=30)
    ax.set_xlabel("Average Speed")
    ax.set_ylabel("Episode Length (steps)")
    ax.set_title("Episode Length vs Average Speed")
    ax.grid(True, alpha=0.3)

    # Add trend line
    if len(avg_speeds) > 1:
        z = np.polyfit(avg_speeds, lengths, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(avg_speeds), max(avg_speeds), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f"y={z[0]:.1f}x+{z[1]:.1f}")
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_dir / "speed_diagnostics.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved {output_dir / 'speed_diagnostics.png'}")

    # Figure 2: Action magnitude analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Analyze first successful episode in detail
    r = success_results[0]

    # Plot 1: Action chunk norms
    ax = axes[0]
    chunk_norms = [np.linalg.norm(chunk, axis=-1).mean() for chunk in r["action_chunks"]]
    ax.plot(r["macro_steps"], chunk_norms, marker="o", markersize=4)
    ax.set_xlabel("Macro-step Index")
    ax.set_ylabel("Mean Action Norm")
    ax.set_title("BC Action Chunk Magnitude (first success)")
    ax.grid(True, alpha=0.3)

    # Plot 2: Interpolated action norms
    ax = axes[1]
    interp_norms = [np.linalg.norm(interp, axis=-1).mean() for interp in r["interpolated_actions"]]
    ax.plot(r["macro_steps"], interp_norms, marker="o", markersize=4, color="orange")
    ax.set_xlabel("Macro-step Index")
    ax.set_ylabel("Mean Action Norm")
    ax.set_title("Interpolated Action Magnitude (first success)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "action_magnitude.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved {output_dir / 'action_magnitude.png'}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics (successful episodes)")
    print("=" * 80)
    print(f"Number of successful episodes: {len(success_results)}")
    print(f"Average episode length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
    print(f"Average speed: {np.mean(avg_speeds):.2f} ± {np.std(avg_speeds):.2f}")
    print(f"Average macro-steps: {np.mean([len(r['macro_steps']) for r in success_results]):.1f}")

    # Analyze steps_executed distribution
    all_steps_flat = [s for r in success_results for s in r["steps_executed"]]
    print(f"\nSteps executed per macro-step:")
    print(f"  Mean: {np.mean(all_steps_flat):.2f}")
    print(f"  Std: {np.std(all_steps_flat):.2f}")
    print(f"  Min: {np.min(all_steps_flat)}")
    print(f"  Max: {np.max(all_steps_flat)}")
    print(f"  % of macro-steps with < k_skip steps: {100 * np.mean(np.array(all_steps_flat) < 4):.1f}%")

    # Correlation analysis
    if len(avg_speeds) > 1:
        corr = np.corrcoef(avg_speeds, lengths)[0, 1]
        print(f"\nCorrelation (avg_speed, episode_length): {corr:.3f}")
        if abs(corr) < 0.3:
            print("  ⚠ Weak correlation! Speed is not strongly affecting episode length.")

    print("=" * 80)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="speed_tuning_config",
)
def main(cfg: DictConfig):
    """Main diagnostic function."""
    print("=" * 80)
    print("Speed-Length Diagnostic Analysis")
    print("=" * 80)

    bc_checkpoint_path = cfg.algorithm.bc_checkpoint
    speed_checkpoint_path = cfg.algorithm.get("speed_checkpoint")

    if bc_checkpoint_path is None:
        raise ValueError("algorithm.bc_checkpoint is required.")
    if speed_checkpoint_path is None:
        raise ValueError("algorithm.speed_checkpoint is required.")

    np.random.seed(cfg.seed)

    # Load BC checkpoint
    print("\nLoading BC checkpoint...")
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
        print("\nFailed to load dataset. Exiting.")
        return
    resolved_path = str(resolved_path)

    # Create BC agent
    print("\nCreating BC agent...")
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

    bc_agent, _ = restore_agent(bc_checkpoint_path, BCAgent, ex_obs, ex_actions)
    print("BC agent restored.")

    # Setup parameters
    speed_options = make_speed_options(
        max_speed=cfg.algorithm.get("max_speed", 2.0),
        granularity=cfg.algorithm.get("speed_granularity", 0.1),
    )
    max_steps = cfg.task.env.max_episode_steps
    k_skip = cfg.algorithm.get("k_skip", 4)
    bc_abs_action = bc_ckpt_config.get("abs_action", False)
    action_normalizer = normalizers.get("action")

    # Create speed env
    print("\nCreating SpeedTuning environment...")
    speed_env = create_base_env(
        cfg, normalizers, resolved_path, bc_ckpt_config, abs_action=bc_abs_action,
    )
    speed_env = SpeedTuningEnvWrapper(
        env=speed_env, bc_agent=bc_agent, speed_options=speed_options,
        alpha=0.0, beta=0.0, k_skip=k_skip,
        abs_action=bc_abs_action, action_normalizer=action_normalizer,
    )

    # Restore speed agent
    print("Restoring SpeedTuning agent...")
    speed_agent, _ = restore_speed_agent(
        speed_checkpoint_path, speed_env, bc_checkpoint_path,
    )

    # Run detailed analysis on multiple episodes
    num_episodes = cfg.eval.get("num_episodes", 20)
    print(f"\nRunning detailed analysis on {num_episodes} episodes...")

    results_list = []
    for ep in range(num_episodes):
        result = analyze_episode_detailed(
            speed_agent, speed_env, bc_agent, action_normalizer, ep, max_steps
        )
        results_list.append(result)

        if (ep + 1) % 5 == 0:
            success_count = sum(1 for r in results_list if r["success"])
            print(f"  Episode {ep + 1}/{num_episodes} | Success: {success_count}/{ep + 1}")

    # Generate plots
    output_dir = Path(cfg.get("checkpoint_dir", "checkpoints")) / "speed_diagnostics"
    print(f"\nGenerating diagnostic plots...")
    plot_diagnostics(results_list, output_dir)

    print(f"\n✓ Diagnostics saved to {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
