"""Compare BC baseline vs SpeedTuning policy.

Evaluates:
  1. Baseline BC (speed=1.0) success rate and episode length
  2. SpeedTuning (learned speed) success rate, episode length, speed trajectories
  3. Fixed-speed sweep: success rate vs fixed speed multiplier

Outputs matplotlib plots and optional rollout videos.

Usage:
    python scripts/eval_speed_comparison.py task=square_lowdim \
        algorithm.bc_checkpoint=path/to/bc/best_model.pkl \
        algorithm.speed_checkpoint=path/to/speed/best_model.pkl

    # Custom eval episodes and speed range
    python scripts/eval_speed_comparison.py task=square_lowdim \
        algorithm.bc_checkpoint=path/to/bc/best_model.pkl \
        algorithm.speed_checkpoint=path/to/speed/best_model.pkl \
        eval.num_episodes=50 algorithm.max_speed=3.0
"""

import json
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
from jax_flow.agents.speed_tuning.interpolation import make_speed_options
from jax_flow.agents.speed_tuning.rainbow_agent import RainbowDQNAgent
from jax_flow.agents.speed_tuning.speed_tuning_env import SpeedTuningEnvWrapper
from jax_flow.core.checkpoint import load_checkpoint, restore_agent
from jax_flow.data import DatasetManager, make_dataset, make_robomimic_dataset
from jax_flow.envs import make_env, make_robomimic_env


# ------------------------------------------------------------------
# Environment helpers (from eval_fixed_speed.py)
# ------------------------------------------------------------------

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


def create_baseline_env(cfg, normalizers, resolved_path, bc_ckpt_config, abs_action=False):
    """Create env with FrameStack + ActionChunking for baseline BC eval."""
    task_source = cfg.task.get("task_source", "robomimic")
    obs_type = bc_ckpt_config.get("obs_type", cfg.task.obs_type)
    is_image = obs_type == "image"
    act_steps = bc_ckpt_config.get("act_steps", cfg.task.dataset.act_steps)

    common_kwargs = {
        "obs_normalizer": normalizers.get("obs"),
        "action_normalizer": normalizers.get("action"),
        "lowdim_normalizer": normalizers.get("lowdim"),
        "max_episode_steps": cfg.task.env.max_episode_steps,
        "frame_stack": bc_ckpt_config.get("obs_steps", cfg.task.dataset.obs_steps),
        "act_exec_steps": act_steps,
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


def _obs_to_batch(obs):
    """Convert single obs to batched obs for agent inference."""
    if isinstance(obs, dict):
        return {k: np.array(v[np.newaxis, ...]) for k, v in obs.items()}
    return obs[np.newaxis, ...]


# ------------------------------------------------------------------
# Agent restoration
# ------------------------------------------------------------------

def restore_speed_agent(speed_ckpt_path, env, bc_checkpoint_path):
    """Restore RainbowDQNAgent from SpeedTuning checkpoint.

    Args:
        speed_ckpt_path: Path to SpeedTuning checkpoint.
        env: SpeedTuningEnvWrapper (for example obs).
        bc_checkpoint_path: Path to BC checkpoint (for encoder init).

    Returns:
        (agent, speed_ckpt) tuple.
    """
    speed_ckpt = load_checkpoint(speed_ckpt_path)

    # Get example obs from env
    obs, _ = env.reset()
    env_obs_batch = _obs_to_batch(obs)
    if isinstance(env_obs_batch, dict):
        env_obs_batch = {k: jnp.array(v) for k, v in env_obs_batch.items()}
    else:
        env_obs_batch = jnp.array(env_obs_batch)

    # Create agent with saved config, then restore weights
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


# ------------------------------------------------------------------
# Evaluation routines
# ------------------------------------------------------------------

def eval_baseline(bc_agent, env, num_episodes, max_steps):
    """Evaluate baseline BC policy."""
    from jax_flow.core.evaluation import rollout_episode

    successes = []
    lengths = []
    returns = []

    for ep in range(num_episodes):
        result = rollout_episode(
            agent=bc_agent, env=env, max_steps=max_steps,
            save_video=False, seed=ep,
        )
        successes.append(result["success"])
        lengths.append(result["length"])
        returns.append(result["return"])

        if (ep + 1) % 10 == 0:
            print(f"  Baseline {ep + 1}/{num_episodes} | SR: {np.mean(successes):.2%}")

    success_lengths = [l for s, l in zip(successes, lengths) if s]
    return {
        "success_rate": float(np.mean(successes)),
        "num_successes": int(np.sum(successes)),
        "num_episodes": num_episodes,
        "avg_length": float(np.mean(lengths)),
        "avg_success_length": float(np.mean(success_lengths)) if success_lengths else 0.0,
        "avg_return": float(np.mean(returns)),
        "successes": successes,
    }


def eval_speedtuning(speed_agent, env, num_episodes, max_steps):
    """Evaluate SpeedTuning policy. Record speed trajectories for successful eps."""
    successes = []
    lengths = []
    returns = []
    all_speeds = []
    success_trajectories = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        ep_return = 0.0
        ep_length = 0
        ep_success = False
        trajectory = []

        while not done and ep_length < max_steps:
            obs_batch = _obs_to_batch(obs)
            if isinstance(obs_batch, dict):
                obs_batch = {k: jnp.array(v) for k, v in obs_batch.items()}
            else:
                obs_batch = jnp.array(obs_batch)

            action = speed_agent.eval_action(obs_batch)
            speed_idx = int(action[0])

            obs, reward, terminated, truncated, info = env.step(speed_idx)
            done = terminated or truncated
            steps_executed = info.get("steps_executed", 1)
            speed = info.get("speed", 1.0)

            ep_return += reward
            ep_length += steps_executed
            trajectory.append((ep_length, speed))

            if "success" in info and bool(info["success"]):
                ep_success = True

        successes.append(ep_success)
        lengths.append(ep_length)
        returns.append(ep_return)
        ep_speeds = [s for _, s in trajectory]
        all_speeds.append(float(np.mean(ep_speeds)) if ep_speeds else 1.0)

        if ep_success:
            success_trajectories.append(trajectory)

        if (ep + 1) % 10 == 0:
            print(
                f"  SpeedTuning {ep + 1}/{num_episodes} | "
                f"SR: {np.mean(successes):.2%} | "
                f"avg_speed: {np.mean(all_speeds):.2f}"
            )

    success_lengths = [l for s, l in zip(successes, lengths) if s]
    return {
        "success_rate": float(np.mean(successes)),
        "num_successes": int(np.sum(successes)),
        "num_episodes": num_episodes,
        "avg_length": float(np.mean(lengths)),
        "avg_success_length": float(np.mean(success_lengths)) if success_lengths else 0.0,
        "avg_return": float(np.mean(returns)),
        "avg_speed": float(np.mean(all_speeds)),
        "success_trajectories": success_trajectories,
        "successes": successes,
    }


def record_paired_video(bc_agent, speed_agent, baseline_env, speed_env, seed, max_steps, fps=30):
    """Record a single paired comparison video for a specific seed.

    Returns:
        dict with baseline_frames, speedtuning_frames, baseline_length, speedtuning_length, trajectory
    """
    from jax_flow.core.evaluation import rollout_episode

    # Record baseline
    baseline_result = rollout_episode(
        agent=bc_agent, env=baseline_env, max_steps=max_steps,
        save_video=True, seed=seed,
    )

    # Record speedtuning
    obs, _ = speed_env.reset(seed=seed)
    done = False
    ep_length = 0
    trajectory = []
    frames = []

    while not done and ep_length < max_steps:
        obs_batch = _obs_to_batch(obs)
        if isinstance(obs_batch, dict):
            obs_batch = {k: jnp.array(v) for k, v in obs_batch.items()}
        else:
            obs_batch = jnp.array(obs_batch)

        action = speed_agent.eval_action(obs_batch)
        speed_idx = int(action[0])

        obs, reward, terminated, truncated, info = speed_env.step(speed_idx)
        done = terminated or truncated
        steps_executed = info.get("steps_executed", 1)
        speed = info.get("speed", 1.0)

        ep_length += steps_executed
        trajectory.append((ep_length, speed))

        frame = speed_env.render()
        if frame is not None:
            frames.append(frame)

    return {
        "baseline_frames": baseline_result.get("frames", []),
        "speedtuning_frames": frames,
        "baseline_length": baseline_result["length"],
        "speedtuning_length": ep_length,
        "trajectory": trajectory,
    }


def eval_fixed_speed_sweep(bc_agent, cfg, normalizers, resolved_path, bc_ckpt_config,
                           speed_options, num_episodes, max_steps, k_skip):
    """Sweep fixed speeds and return (speed, success_rate) pairs."""
    bc_abs_action = bc_ckpt_config.get("abs_action", False)
    action_normalizer = normalizers.get("action")
    results = []

    for speed in speed_options:
        env = create_base_env(
            cfg, normalizers, resolved_path, bc_ckpt_config, abs_action=bc_abs_action,
        )
        env = SpeedTuningEnvWrapper(
            env=env, bc_agent=bc_agent, speed_options=[speed],
            alpha=0.0, beta=0.0, k_skip=k_skip,
            abs_action=bc_abs_action, action_normalizer=action_normalizer,
        )
        env._current_speed = speed

        ep_successes = []
        for ep in range(num_episodes):
            obs, _ = env.reset(seed=ep)
            done = False
            ep_length = 0
            ep_success = False
            while not done and ep_length < max_steps:
                obs, _reward, terminated, truncated, info = env.step(0)
                done = terminated or truncated
                ep_length += info.get("steps_executed", 1)
                if "success" in info and bool(info["success"]):
                    ep_success = True
            ep_successes.append(ep_success)

        sr = float(np.mean(ep_successes))
        results.append((speed, sr))
        print(f"  speed={speed:.2f} | success={sr:.2%} ({int(np.sum(ep_successes))}/{num_episodes})")

    return results


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_results(output_dir, baseline_results, speed_results, fixed_sweep_results):
    """Generate comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Speed trajectories (successful episodes only)
    if speed_results["success_trajectories"]:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot individual trajectories (semi-transparent)
        for traj in speed_results["success_trajectories"]:
            steps, speeds = zip(*traj)
            ax.plot(steps, speeds, alpha=0.3, color="tab:blue", linewidth=1)

        # Compute and plot mean trajectory
        max_len = max(len(t) for t in speed_results["success_trajectories"])
        speed_matrix = np.full((len(speed_results["success_trajectories"]), max_len), np.nan)
        for i, traj in enumerate(speed_results["success_trajectories"]):
            steps, speeds = zip(*traj)
            for j, (step, speed) in enumerate(traj):
                if j < max_len:
                    speed_matrix[i, j] = speed

        mean_speeds = np.nanmean(speed_matrix, axis=0)
        mean_steps = np.arange(1, len(mean_speeds) + 1)
        ax.plot(mean_steps, mean_speeds, color="tab:blue", linewidth=3, label="SpeedTuning (mean)")

        # Baseline reference line
        ax.axhline(y=1.0, color="tab:orange", linestyle="--", linewidth=2, label="Baseline (speed=1.0)")

        ax.set_xlabel("Cumulative Environment Steps", fontsize=12)
        ax.set_ylabel("Speed Multiplier", fontsize=12)
        ax.set_title("Speed Trajectories (Successful Episodes)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / "speed_trajectories.png", dpi=150)
        plt.close(fig)
        print(f"  ✓ Saved {output_dir / 'speed_trajectories.png'}")

    # Figure 2: Fixed speed vs success rate
    if fixed_sweep_results:
        fig, ax = plt.subplots(figsize=(10, 6))

        speeds, success_rates = zip(*fixed_sweep_results)
        ax.plot(speeds, success_rates, marker="o", linewidth=2, markersize=6, color="tab:green")

        # Mark baseline (speed=1.0)
        baseline_sr = baseline_results["success_rate"]
        ax.axvline(x=1.0, color="tab:orange", linestyle="--", linewidth=2, alpha=0.7)
        ax.scatter([1.0], [baseline_sr], color="tab:orange", s=100, zorder=5,
                   label=f"Baseline (speed=1.0, SR={baseline_sr:.1%})")

        # Mark SpeedTuning average speed
        if "avg_speed" in speed_results:
            avg_speed = speed_results["avg_speed"]
            speed_sr = speed_results["success_rate"]
            ax.axvline(x=avg_speed, color="tab:blue", linestyle="--", linewidth=2, alpha=0.7)
            ax.scatter([avg_speed], [speed_sr], color="tab:blue", s=100, zorder=5,
                       label=f"SpeedTuning (avg speed={avg_speed:.2f}, SR={speed_sr:.1%})")

        ax.set_xlabel("Fixed Speed Multiplier", fontsize=12)
        ax.set_ylabel("Success Rate", fontsize=12)
        ax.set_title("Success Rate vs Fixed Speed", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(output_dir / "fixed_speed_success_rate.png", dpi=150)
        plt.close(fig)
        print(f"  ✓ Saved {output_dir / 'fixed_speed_success_rate.png'}")

    # Figure 3: Comparison summary
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Success rate
    ax = axes[0]
    methods = ["Baseline", "SpeedTuning"]
    srs = [baseline_results["success_rate"], speed_results["success_rate"]]
    bars = ax.bar(methods, srs, color=["tab:orange", "tab:blue"], alpha=0.8)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_title("Success Rate", fontsize=13, fontweight="bold")
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis="y")
    for bar, sr in zip(bars, srs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                f"{sr:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Average episode length (successful episodes)
    ax = axes[1]
    lengths = [
        baseline_results["avg_success_length"],
        speed_results["avg_success_length"],
    ]
    bars = ax.bar(methods, lengths, color=["tab:orange", "tab:blue"], alpha=0.8)
    ax.set_ylabel("Episode Length (steps)", fontsize=12)
    ax.set_title("Avg Success Episode Length", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, length in zip(bars, lengths):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, height + max(lengths) * 0.02,
                    f"{length:.0f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Average speed
    ax = axes[2]
    speeds_plot = [1.0, speed_results.get("avg_speed", 1.0)]
    bars = ax.bar(methods, speeds_plot, color=["tab:orange", "tab:blue"], alpha=0.8)
    ax.set_ylabel("Speed Multiplier", fontsize=12)
    ax.set_title("Average Speed", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, speed in zip(bars, speeds_plot):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05,
                f"{speed:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_dir / "comparison_summary.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved {output_dir / 'comparison_summary.png'}")


def save_videos(output_dir, video_data, seed, fps=30):
    """Save paired side-by-side comparison video for a single episode.

    Args:
        output_dir: Output directory path
        video_data: Dict with baseline_frames, speedtuning_frames, baseline_length, speedtuning_length
        seed: Episode seed number
        fps: Video frame rate
    """
    video_dir = Path(output_dir) / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    try:
        import imageio
    except ImportError:
        print("  ✗ imageio not installed, skipping video saving")
        return

    b_frames = video_data["baseline_frames"]
    s_frames = video_data["speedtuning_frames"]
    b_len = video_data["baseline_length"]
    s_len = video_data["speedtuning_length"]

    if not b_frames or not s_frames:
        print("  ✗ No frames to save")
        return

    # Save individual videos
    imageio.mimsave(video_dir / f"seed{seed}_baseline.mp4", b_frames, fps=fps)
    imageio.mimsave(video_dir / f"seed{seed}_speedtuning.mp4", s_frames, fps=fps)
    print(f"  ✓ Saved individual videos: seed{seed}_baseline.mp4, seed{seed}_speedtuning.mp4")

    # Build side-by-side comparison video
    h, w = b_frames[0].shape[:2]
    gap = 4
    label_h = 30
    canvas_w = w * 2 + gap
    canvas_h = h + label_h

    max_frames = max(len(b_frames), len(s_frames))
    combined = []

    for i in range(max_frames):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Left panel: baseline
        if i < len(b_frames):
            canvas[label_h:label_h + h, :w] = b_frames[i]
        elif len(b_frames) > 0:
            canvas[label_h:label_h + h, :w] = b_frames[-1]

        # Gap
        canvas[:, w:w + gap] = 128

        # Right panel: speedtuning
        if i < len(s_frames):
            canvas[label_h:label_h + h, w + gap:] = s_frames[i]
        elif len(s_frames) > 0:
            canvas[label_h:label_h + h, w + gap:] = s_frames[-1]

        # Label area background
        canvas[:label_h, :w] = 40
        canvas[:label_h, w + gap:] = 40

        combined.append(canvas)

    path = video_dir / f"seed{seed}_comparison.mp4"
    imageio.mimsave(path, combined, fps=fps)
    print(
        f"  ✓ Saved comparison video: seed{seed}_comparison.mp4 "
        f"(baseline {b_len} steps vs speedtuning {s_len} steps, "
        f"{b_len / max(s_len, 1):.2f}x)"
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="speed_tuning_config",
)
def main(cfg: DictConfig):
    """Main comparison evaluation."""
    print("=" * 80)
    print("SpeedTuning Comparison Evaluation")
    print("=" * 80)

    bc_checkpoint_path = cfg.algorithm.bc_checkpoint
    speed_checkpoint_path = cfg.algorithm.get("speed_checkpoint")

    if bc_checkpoint_path is None:
        raise ValueError("algorithm.bc_checkpoint is required.")
    if speed_checkpoint_path is None:
        raise ValueError("algorithm.speed_checkpoint is required.")

    np.random.seed(cfg.seed)

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
    # 2. Resolve dataset path
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
    # 3. Create BC agent
    # ================================================================
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

    # ================================================================
    # 4. Setup parameters
    # ================================================================
    speed_options = make_speed_options(
        max_speed=cfg.algorithm.get("max_speed", 2.0),
        granularity=cfg.algorithm.get("speed_granularity", 0.1),
    )
    num_episodes = cfg.eval.get("num_episodes", 20)
    max_steps = cfg.task.env.max_episode_steps
    k_skip = cfg.algorithm.get("k_skip", 4)
    bc_abs_action = bc_ckpt_config.get("abs_action", False)
    action_normalizer = normalizers.get("action")

    print(f"\nTask: {cfg.task.name}")
    print(f"Speed options: {speed_options}")
    print(f"Eval episodes: {num_episodes}")
    print(f"k_skip: {k_skip}")
    print("=" * 80)

    # ================================================================
    # 5. Evaluate Baseline BC
    # ================================================================
    print("\n[1/4] Evaluating Baseline BC (speed=1.0)...")
    baseline_env = create_baseline_env(
        cfg, normalizers, resolved_path, bc_ckpt_config, abs_action=bc_abs_action,
    )
    baseline_results = eval_baseline(
        bc_agent, baseline_env, num_episodes, max_steps,
    )
    print(f"  Baseline SR: {baseline_results['success_rate']:.2%} "
          f"({baseline_results['num_successes']}/{num_episodes})")
    print(f"  Avg success length: {baseline_results['avg_success_length']:.0f}")

    # ================================================================
    # 6. Evaluate SpeedTuning
    # ================================================================
    print("\n[2/4] Evaluating SpeedTuning...")
    speed_env = create_base_env(
        cfg, normalizers, resolved_path, bc_ckpt_config, abs_action=bc_abs_action,
    )
    speed_env = SpeedTuningEnvWrapper(
        env=speed_env, bc_agent=bc_agent, speed_options=speed_options,
        alpha=0.0, beta=0.0, k_skip=k_skip,
        abs_action=bc_abs_action, action_normalizer=action_normalizer,
    )
    speed_agent, _ = restore_speed_agent(
        speed_checkpoint_path, speed_env, bc_checkpoint_path,
    )
    speed_results = eval_speedtuning(
        speed_agent, speed_env, num_episodes, max_steps,
    )
    print(f"  SpeedTuning SR: {speed_results['success_rate']:.2%} "
          f"({speed_results['num_successes']}/{num_episodes})")
    print(f"  Avg success length: {speed_results['avg_success_length']:.0f}")
    print(f"  Avg speed: {speed_results['avg_speed']:.2f}")

    # ================================================================
    # 7. Record paired video for one successful episode
    # ================================================================
    print("\n[3/5] Recording paired comparison video...")
    # Find seeds where both succeeded
    baseline_success_seeds = [i for i, s in enumerate(baseline_results["successes"]) if s]
    speed_success_seeds = [i for i, s in enumerate(speed_results["successes"]) if s]
    paired_seeds = sorted(set(baseline_success_seeds) & set(speed_success_seeds))

    if paired_seeds:
        # Randomly pick one
        chosen_seed = np.random.choice(paired_seeds)
        print(f"  Recording seed {chosen_seed} (both succeeded)...")

        video_data = record_paired_video(
            bc_agent, speed_agent, baseline_env, speed_env,
            chosen_seed, max_steps, fps=cfg.eval.get("video_fps", 30),
        )
        print(f"  ✓ Baseline: {video_data['baseline_length']} steps")
        print(f"  ✓ SpeedTuning: {video_data['speedtuning_length']} steps "
              f"({video_data['baseline_length'] / max(video_data['speedtuning_length'], 1):.2f}x)")
    else:
        print("  ✗ No episodes where both policies succeeded. Skipping video.")
        video_data = None

    # ================================================================
    # 8. Fixed speed sweep
    # ================================================================
    print("\n[4/5] Fixed speed sweep...")
    fixed_sweep_results = eval_fixed_speed_sweep(
        bc_agent, cfg, normalizers, resolved_path, bc_ckpt_config,
        speed_options, num_episodes, max_steps, k_skip,
    )

    # ================================================================
    # 9. Plot and save
    # ================================================================
    print("\n[5/5] Generating plots...")
    output_dir = Path(cfg.get("checkpoint_dir", "checkpoints")) / "eval_comparison"
    plot_results(output_dir, baseline_results, speed_results, fixed_sweep_results)

    if video_data is not None:
        print("\nSaving paired comparison video...")
        save_videos(output_dir, video_data, chosen_seed,
                    fps=cfg.eval.get("video_fps", 30))

    # Save raw results
    results_json = {
        "baseline": {
            "success_rate": baseline_results["success_rate"],
            "num_successes": baseline_results["num_successes"],
            "avg_success_length": baseline_results["avg_success_length"],
        },
        "speedtuning": {
            "success_rate": speed_results["success_rate"],
            "num_successes": speed_results["num_successes"],
            "avg_success_length": speed_results["avg_success_length"],
            "avg_speed": speed_results["avg_speed"],
        },
        "fixed_sweep": [{"speed": s, "success_rate": sr} for s, sr in fixed_sweep_results],
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n✓ Results saved to {output_dir / 'results.json'}")

    print("\n" + "=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    print(f"Baseline:     SR={baseline_results['success_rate']:.2%}, "
          f"Len={baseline_results['avg_success_length']:.0f}, Speed=1.00")
    print(f"SpeedTuning:  SR={speed_results['success_rate']:.2%}, "
          f"Len={speed_results['avg_success_length']:.0f}, "
          f"Speed={speed_results['avg_speed']:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
