"""Test action distribution of a trained flow policy.

Given a fixed observation, sample actions with many different initial noises
to visualize:
1. Whether different noises converge to similar actions (unimodal) or diverge (multimodal)
2. The ODE trajectory: how noise evolves through flow steps t=0 -> t=1

Usage:
    python scripts/test_action_distribution.py --checkpoint path/to/best_model.pkl
    python scripts/test_action_distribution.py --checkpoint path/to/best_model.pkl --num_samples 500 --obs_index 100
"""

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from jax_flow.agents.bc_agent import BCAgent
from jax_flow.core.checkpoint import restore_agent
from jax_flow.core.utils import get_batch_size
from jax_flow.flow.samplers import get_sampler


def euler_sampler_with_trajectory(network, encoder, observations, num_steps, rng, config):
    """Euler sampler that returns intermediate states at each ODE step.

    Returns:
        final_actions: (batch, horizon, action_dim)
        trajectory: (num_steps+1, batch, horizon, action_dim) — states at t=0,dt,2dt,...,1
    """
    batch_size = get_batch_size(observations)
    horizon = config.get("horizon", 10)
    action_dim = config.get("action_dim", 7)

    cond = encoder(observations, training=False, rngs={})
    x = jax.random.normal(rng, (batch_size, horizon, action_dim))

    trajectory = [np.array(x)]  # t=0: pure noise

    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = jnp.full((batch_size,), step * dt)
        velocity = network(x, t, t, cond, training=False)
        x = x + velocity * dt
        trajectory.append(np.array(x))

    return np.array(x), np.stack(trajectory, axis=0)  # (num_steps+1, batch, horizon, action_dim)


def _infer_task_source(config):
    """Infer task_source from checkpoint config."""
    task_source = config.get("task_source")
    if task_source:
        return task_source
    env_name = config.get("env_name", "")
    if env_name in ("PushT", "pusht"):
        return "pusht"
    if env_name in ("kitchen",):
        return "kitchen"
    return "robomimic"


def load_obs_from_dataset(config, normalizers, obs_index=0):
    """Load a single observation from the dataset."""
    task_source = _infer_task_source(config)
    obs_steps = config.get("obs_steps", 2)
    horizon = config.get("horizon", 16)

    if task_source == "pusht":
        from jax_flow.data.pusht_dataset import PushTDataset, download_pusht_dataset
        dataset_path = config.get("dataset_path")
        if dataset_path is None:
            dataset_path = download_pusht_dataset()
        ds = PushTDataset(
            dataset_path=dataset_path,
            obs_type=config.get("obs_type", "state"),
            horizon=horizon,
            obs_steps=obs_steps,
        )
    elif task_source == "kitchen":
        from jax_flow.data.kitchen_dataset import KitchenDataset, download_kitchen_dataset
        dataset_path = config.get("dataset_path")
        if dataset_path is None:
            dataset_path = download_kitchen_dataset()
        ds = KitchenDataset(dataset_path=dataset_path, horizon=horizon, obs_steps=obs_steps)
    else:
        from jax_flow.data.robomimic_dataset import RobomimicDataset
        ds = RobomimicDataset(
            dataset_path=config["dataset_path"],
            horizon=horizon, obs_steps=obs_steps,
            obs_keys=tuple(config.get("obs_keys", (
                "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object",
            ))),
            abs_action=config.get("abs_action", False),
        )

    sample = ds[obs_index]
    obs = sample["observations"]

    # For non-dict obs, apply normalizer and add batch dim
    if isinstance(obs, dict):
        # Image obs (e.g. PushT image): dataset already normalizes internally
        obs = jax.tree_util.tree_map(lambda x: jnp.array(x)[None], obs)
    else:
        obs_norm = normalizers.get("obs")
        if obs_norm is not None:
            obs = obs_norm.normalize(obs)
        obs = jnp.array(obs)[None]

    gt_action = sample["actions"]
    action_norm = normalizers.get("action")
    if action_norm is not None:
        gt_action = action_norm.normalize(gt_action)

    return obs, np.array(gt_action)


def sample_with_trajectory(agent, obs, num_samples):
    """Sample actions and record ODE trajectory for visualization."""
    if isinstance(obs, dict):
        obs_tiled = {k: jnp.tile(v, (num_samples,) + (1,) * (v.ndim - 1)) for k, v in obs.items()}
    else:
        obs_tiled = jnp.tile(obs, (num_samples,) + (1,) * (obs.ndim - 1))

    rng = jax.random.PRNGKey(42)
    params = agent.ema_params

    def encode(o, training=False, rngs=None):
        return agent.network(o, training=training, name="encoder", params=params, rngs=rngs or {})

    def flow_net(at, s, t, cond, training=False):
        return agent.network(at, s, t, cond, training=training, name="flow", params=params)

    sampler_type = agent.config.get("sampler_type", "euler")
    num_steps = agent.config.get("flow_steps", 10)

    if sampler_type in ("euler", "heun"):
        actions, trajectory = euler_sampler_with_trajectory(
            flow_net, encode, obs_tiled, num_steps, rng, agent.config,
        )
    else:
        # For non-Euler samplers (mip, meanflow), just get final actions
        sampler = get_sampler(sampler_type)
        actions = np.array(sampler(flow_net, encode, obs_tiled, num_steps, rng, agent.config))
        trajectory = None

    return actions, trajectory


def plot_all(actions, trajectory, gt_action, num_steps, save_dir):
    """Generate all visualization plots."""
    num_samples, horizon, action_dim = actions.shape
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ===== Figure 1: Per-dim histogram at t=0 (first horizon step) =====
    first_step = actions[:, 0, :]  # (num_samples, action_dim)
    n_cols = min(action_dim, 4)
    n_rows = (action_dim + n_cols - 1) // n_cols

    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows * n_cols == 1:
        axes1 = np.array([axes1])
    axes1 = np.array(axes1).flatten()
    fig1.suptitle(f"Output Action Distribution (horizon t=0, {num_samples} noise samples)", fontsize=13)

    for d in range(action_dim):
        ax = axes1[d]
        vals = first_step[:, d]
        ax.hist(vals, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
        if gt_action is not None:
            ax.axvline(gt_action[0, d], color="red", linewidth=2, label="GT")
            ax.legend(fontsize=8)
        ax.set_title(f"dim {d}  (std={vals.std():.4f})", fontsize=10)
    for d in range(action_dim, len(axes1)):
        axes1[d].set_visible(False)
    fig1.tight_layout()
    p1 = save_dir / "histogram.png"
    fig1.savefig(p1, dpi=150)
    print(f"Saved: {p1}")

    # ===== Figure 2: ODE trajectory convergence (the key plot) =====
    if trajectory is not None:
        # trajectory: (num_steps+1, num_samples, horizon, action_dim)
        # Pick horizon step 0, show dim 0 and dim 1
        n_show = min(num_samples, 50)  # show subset for clarity
        T = trajectory.shape[0]  # num_steps + 1
        t_values = np.linspace(0, 1, T)

        # Plot 2a: ODE trajectories in 1D (dim 0)
        fig2a, ax2a = plt.subplots(figsize=(10, 5))
        for i in range(n_show):
            ax2a.plot(t_values, trajectory[:, i, 0, 0], alpha=0.3, color="steelblue", linewidth=0.8)
        if gt_action is not None:
            ax2a.axhline(gt_action[0, 0], color="red", linewidth=2, linestyle="--", label="GT")
            ax2a.legend()
        ax2a.set_xlabel("ODE time (t=0: noise, t=1: action)", fontsize=12)
        ax2a.set_ylabel("action dim 0", fontsize=12)
        ax2a.set_title(f"ODE Trajectories: Noise → Action (dim 0, {n_show} samples)", fontsize=13)
        ax2a.grid(True, alpha=0.3)
        p2a = save_dir / "ode_trajectory_1d.png"
        fig2a.tight_layout()
        fig2a.savefig(p2a, dpi=150)
        print(f"Saved: {p2a}")

        # Plot 2b: ODE trajectories in 2D (dim 0 vs dim 1)
        if action_dim >= 2:
            fig2b, ax2b = plt.subplots(figsize=(8, 7))
            cmap = plt.get_cmap("coolwarm")
            for i in range(n_show):
                # Color by ODE time
                for s in range(T - 1):
                    color = cmap(t_values[s])
                    ax2b.plot(
                        trajectory[s:s+2, i, 0, 0], trajectory[s:s+2, i, 0, 1],
                        color=color, alpha=0.4, linewidth=0.8,
                    )
                # Mark start (noise) and end (action)
                ax2b.scatter(trajectory[0, i, 0, 0], trajectory[0, i, 0, 1],
                             color="blue", s=10, alpha=0.3, zorder=5)
                ax2b.scatter(trajectory[-1, i, 0, 0], trajectory[-1, i, 0, 1],
                             color="red", s=15, alpha=0.5, zorder=5)
            if gt_action is not None:
                ax2b.scatter(gt_action[0, 0], gt_action[0, 1], color="black",
                             s=100, marker="*", zorder=10, label="GT")
                ax2b.legend(fontsize=10)
            ax2b.set_xlabel("action dim 0", fontsize=12)
            ax2b.set_ylabel("action dim 1", fontsize=12)
            ax2b.set_title(f"ODE Flow: Noise(blue) → Action(red) in 2D", fontsize=13)
            # Add colorbar for time
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
            sm.set_array([])
            cbar = fig2b.colorbar(sm, ax=ax2b, label="ODE time t")
            p2b = save_dir / "ode_trajectory_2d.png"
            fig2b.tight_layout()
            fig2b.savefig(p2b, dpi=150)
            print(f"Saved: {p2b}")

        # Plot 2c: Std collapse across ODE steps
        # trajectory: (T, num_samples, horizon, action_dim) -> std over samples at each t
        traj_std = trajectory[:, :, 0, :].std(axis=1)  # (T, action_dim)
        fig2c, ax2c = plt.subplots(figsize=(10, 4))
        for d in range(action_dim):
            ax2c.plot(t_values, traj_std[:, d], marker="o", markersize=3, label=f"dim {d}")
        ax2c.set_xlabel("ODE time (t=0: noise, t=1: action)", fontsize=12)
        ax2c.set_ylabel("std across samples", fontsize=12)
        ax2c.set_title("Action Std Collapse during ODE Integration", fontsize=13)
        ax2c.legend(fontsize=7, ncol=min(action_dim, 5))
        ax2c.grid(True, alpha=0.3)
        p2c = save_dir / "ode_std_collapse.png"
        fig2c.tight_layout()
        fig2c.savefig(p2c, dpi=150)
        print(f"Saved: {p2c}")

    # ===== Figure 3: 2D scatter of final actions =====
    if action_dim >= 2:
        fig3, ax3 = plt.subplots(figsize=(7, 6))
        ax3.scatter(actions[:, 0, 0], actions[:, 0, 1], alpha=0.4, s=12, color="steelblue")
        if gt_action is not None:
            ax3.scatter(gt_action[0, 0], gt_action[0, 1], color="red", s=100, marker="*", zorder=10, label="GT")
            ax3.legend()
        ax3.set_xlabel("action dim 0")
        ax3.set_ylabel("action dim 1")
        ax3.set_title(f"Final Action Scatter (dim0 vs dim1, {num_samples} samples)")
        ax3.grid(True, alpha=0.3)
        p3 = save_dir / "action_scatter.png"
        fig3.tight_layout()
        fig3.savefig(p3, dpi=150)
        print(f"Saved: {p3}")

    # ===== Summary =====
    print("\n=== Action Distribution Summary ===")
    print(f"Samples: {num_samples}, Horizon: {horizon}, Action dim: {action_dim}")
    std_all = actions[:, 0, :].std(axis=0)
    mean_all = actions[:, 0, :].mean(axis=0)
    print(f"First horizon step (t_horizon=0):")
    print(f"  Mean: {mean_all}")
    print(f"  Std:  {std_all}")
    print(f"  Min std: {std_all.min():.6f}, Max std: {std_all.max():.6f}, Mean std: {std_all.mean():.6f}")
    if trajectory is not None:
        noise_std = trajectory[0, :, 0, :].std(axis=0).mean()
        final_std = trajectory[-1, :, 0, :].std(axis=0).mean()
        print(f"\nODE convergence:")
        print(f"  Noise std (t=0): {noise_std:.4f}")
        print(f"  Action std (t=1): {final_std:.6f}")
        print(f"  Compression ratio: {noise_std / max(final_std, 1e-8):.1f}x")

    plt.close("all")


def main():
    parser = argparse.ArgumentParser(description="Test action distribution of flow policy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=200, help="Number of noise samples")
    parser.add_argument("--obs_index", type=int, default=0, help="Observation index in dataset")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for plots")
    args = parser.parse_args()

    # Load checkpoint
    print("Loading checkpoint...")
    ckpt_path = Path(args.checkpoint)
    with open(ckpt_path, "rb") as f:
        checkpoint = pickle.load(f)
    config = checkpoint["config"]
    normalizers = checkpoint.get("normalizers", {})

    obs_steps = config.get("obs_steps", 2)
    horizon = config.get("horizon", 16)
    action_dim = config.get("action_dim", 7)
    obs_type = config.get("obs_type", "lowdim")

    print(f"Config: horizon={horizon}, action_dim={action_dim}, obs_steps={obs_steps}")
    print(f"Sampler: {config.get('sampler_type', 'euler')}, flow_steps={config.get('flow_steps', 10)}")

    # Create example tensors for agent restoration
    if obs_type == "image":
        image_keys = config.get("image_keys", ("agentview_image",))
        ex_obs = {key: jnp.zeros((1, obs_steps, 84, 84, 3)) for key in image_keys}
        lowdim_dim = config.get("obs_dim")
        if lowdim_dim and config.get("lowdim_keys"):
            ex_obs["lowdim"] = jnp.zeros((1, obs_steps, lowdim_dim))
    else:
        ex_obs = jnp.zeros((1, obs_steps, config.get("obs_dim", 10)))
    ex_actions = jnp.zeros((1, horizon, action_dim))

    # Restore agent
    print("Restoring agent...")
    agent, _ = restore_agent(args.checkpoint, BCAgent, ex_obs, ex_actions)
    print(f"Agent restored (step {checkpoint.get('training_step', 'unknown')})")

    # Load observation + GT action
    print(f"\nLoading observation (index={args.obs_index})...")
    obs, gt_action = load_obs_from_dataset(config, normalizers, obs_index=args.obs_index)
    print(f"Obs shape: {obs.shape if not isinstance(obs, dict) else {k: v.shape for k, v in obs.items()}}")
    print(f"GT action shape: {gt_action.shape}")

    # Sample with trajectory
    print(f"\nSampling {args.num_samples} actions with different noises...")
    actions, trajectory = sample_with_trajectory(agent, obs, args.num_samples)
    print(f"Actions shape: {actions.shape}")
    if trajectory is not None:
        print(f"Trajectory shape: {trajectory.shape} (ODE steps + 1, samples, horizon, action_dim)")

    # Plot
    save_dir = args.output_dir or str(ckpt_path.parent / "action_distribution")
    num_steps = config.get("flow_steps", 10)
    plot_all(actions, trajectory, gt_action, num_steps, save_dir)


if __name__ == "__main__":
    main()
