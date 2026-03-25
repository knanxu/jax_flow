"""Visualize sampled action trajectories from a trained Push-T policy.

Given a checkpoint and initial states, samples multiple trajectories from the policy
and draws them on the Push-T scene to visualize multi-modal behavior.

Usage:
    # Basic usage with random states
    python scripts/visualize_pusht_trajectories.py --checkpoint path/to/best_model.pkl

    # Specify number of states and trajectories per state
    python scripts/visualize_pusht_trajectories.py --checkpoint path/to/best_model.pkl \
        --num_states 4 --num_trajectories 50

    # Use specific seeds for reproducible states
    python scripts/visualize_pusht_trajectories.py --checkpoint path/to/best_model.pkl \
        --state_seeds 0 1 2 3

    # Rollout trajectories in the environment (simulate physics)
    python scripts/visualize_pusht_trajectories.py --checkpoint path/to/best_model.pkl --rollout
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util

from jax_flow.agents.bc_agent import BCAgent
from jax_flow.core.checkpoint import load_checkpoint, restore_agent
from jax_flow.envs.pusht.pusht_env import PushTEnv
from jax_flow.envs.pusht.pusht_image_env import PushTImageEnv


def render_scene(env):
    """Render the Push-T scene at full 512x512 resolution."""
    canvas = pygame.Surface((env.window_size, env.window_size))
    canvas.fill((255, 255, 255))
    env.screen = canvas

    from jax_flow.envs.pusht.pymunk_override import DrawOptions
    draw_options = DrawOptions(canvas)

    # Draw goal
    goal_body = env._get_goal_pose_body(env.goal_pose)
    for shape in env.block.shapes:
        goal_points = [
            pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface)
            for v in shape.get_vertices()
        ]
        goal_points += [goal_points[0]]
        pygame.draw.polygon(canvas, env.goal_color, goal_points)

    # Draw block and agent
    env.space.debug_draw(draw_options)

    img = np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    return img


def get_normalized_obs(env, obs_normalizer, lowdim_normalizer, obs_steps, obs_type="state"):
    """Get normalized, frame-stacked observation from current env state."""
    if obs_type == "image":
        # Use PushTImageEnv's _get_obs which returns {"image": (C,H,W), "agent_pos": (2,)}
        raw_obs = env._get_obs()
        # image: (C,H,W) -> (H,W,C) float32 [0,1]
        img = raw_obs["image"]
        if img.ndim == 3 and img.shape[0] in (1, 3):
            img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.float32)
        # agent_pos: normalize with lowdim_normalizer
        agent_pos = raw_obs["agent_pos"].astype(np.float32)
        if lowdim_normalizer is not None:
            agent_pos = lowdim_normalizer.normalize(agent_pos)
        # Stack obs_steps copies
        obs = {
            "image": np.stack([img] * obs_steps, axis=0),       # (obs_steps, H, W, C)
            "agent_pos": np.stack([agent_pos] * obs_steps, axis=0),  # (obs_steps, 2)
        }
        return obs
    else:
        raw_obs = env._get_obs().astype(np.float32)
        if obs_normalizer is not None:
            obs = obs_normalizer.normalize(raw_obs)
        else:
            obs = raw_obs
        obs_stacked = np.stack([obs] * obs_steps, axis=0)  # (obs_steps, obs_dim)
        return obs_stacked


def sample_trajectories(agent, obs_batch, num_trajectories, action_normalizer):
    """Sample multiple action trajectories for the same observation.

    Returns unnormalized trajectories in world coordinates [0, 512].
    Shape: (num_trajectories, horizon, 2)
    """
    # Tile obs to batch all trajectories in one forward pass
    obs_tiled = jax.tree.map(
        lambda x: jnp.tile(x, (num_trajectories,) + (1,) * (x.ndim - 1)),
        obs_batch,
    )  # (N, obs_steps, obs_dim)

    # Single batched call with a random key
    rng = jax.random.PRNGKey(42)
    actions = agent.sample_actions(obs_tiled, rng=rng)  # (N, horizon, 2)
    actions = np.array(actions)

    # Unnormalize to world coordinates
    if action_normalizer is not None:
        orig_shape = actions.shape
        actions_flat = actions.reshape(-1, orig_shape[-1])
        actions_flat = action_normalizer.unnormalize(actions_flat)
        actions = actions_flat.reshape(orig_shape)

    return actions


def rollout_trajectories(env, agent, obs_normalizer, action_normalizer,
                         lowdim_normalizer, obs_steps, act_steps,
                         num_trajectories, max_steps, initial_state,
                         obs_type="state"):
    """Rollout trajectories by actually stepping the environment.

    Returns list of agent position traces, each (T, 2) in world coords.
    """
    traces = []
    for i in range(num_trajectories):
        # Reset to same initial state
        env.reset()
        env._set_state(initial_state)

        positions = [np.array(env.agent.position)]
        obs_history = []

        for step in range(max_steps):
            obs = get_normalized_obs(env, obs_normalizer, lowdim_normalizer, 1, obs_type)
            # obs is either (1, obs_dim) or dict with (1, ...) values for single frame
            if isinstance(obs, dict):
                # Extract single frame, append to history
                single = {k: v[0] for k, v in obs.items()}
            else:
                single = obs[0]
            obs_history.append(single)

            # Build frame-stacked obs
            if len(obs_history) < obs_steps:
                frames = [obs_history[0]] * (obs_steps - len(obs_history)) + obs_history
            else:
                frames = obs_history[-obs_steps:]

            if isinstance(frames[0], dict):
                stacked = {k: np.stack([f[k] for f in frames], axis=0) for k in frames[0]}
                obs_batch = jax.tree.map(lambda x: jnp.array(x)[None], stacked)
            else:
                stacked = np.stack(frames, axis=0)
                obs_batch = jnp.array(stacked)[None]

            # Sample actions
            rng = jax.random.PRNGKey(i * 1000 + step)
            actions = agent.sample_actions(obs_batch, rng=rng)
            actions = np.array(actions[0])  # (horizon, 2)

            # Unnormalize
            if action_normalizer is not None:
                actions = action_normalizer.unnormalize(actions)

            # Execute act_steps actions
            for a_idx in range(min(act_steps, len(actions))):
                action = actions[a_idx]
                action = np.clip(action, 0, 512)
                env.step(action)
                positions.append(np.array(env.agent.position))

        traces.append(np.array(positions))
    return traces


def plot_trajectories_on_scene(scene_img, trajectories, agent_pos, title="",
                               output_path=None, is_rollout=False):
    """Plot action trajectories overlaid on the Push-T scene.

    Args:
        scene_img: (512, 512, 3) RGB image.
        trajectories: If not rollout: (N, horizon, 2) world-coord target positions.
                      If rollout: list of (T, 2) agent position traces.
        agent_pos: (2,) current agent position in world coords.
        title: Plot title.
        output_path: If set, save figure to this path.
        is_rollout: Whether trajectories are rollout traces.
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(scene_img, extent=(0, 512, 512, 0))

    n = len(trajectories)
    cmap = plt.get_cmap("plasma")
    colors = cmap(np.linspace(0.1, 0.9, n))

    for i, traj in enumerate(trajectories):
        if is_rollout:
            ax.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.4, linewidth=1.0)
        else:
            # Draw from agent position through the action waypoints
            pts = np.concatenate([[agent_pos], traj], axis=0)
            ax.plot(pts[:, 0], pts[:, 1], color=colors[i], alpha=0.4, linewidth=1.0)

    # Draw agent position
    ax.plot(agent_pos[0], agent_pos[1], 'o', color='royalblue', markersize=8,
            markeredgecolor='white', markeredgewidth=1.5, zorder=10)

    ax.set_xlim(0, 512)
    ax.set_ylim(512, 0)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=12)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize Push-T policy trajectories")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--num_states", type=int, default=4, help="Number of initial states")
    parser.add_argument("--num_trajectories", type=int, default=50,
                        help="Trajectories per state")
    parser.add_argument("--state_seeds", type=int, nargs="+", default=None,
                        help="Specific seeds for initial states")
    parser.add_argument("--rollout", action="store_true",
                        help="Rollout in env instead of open-loop visualization")
    parser.add_argument("--rollout_steps", type=int, default=30,
                        help="Max env steps per rollout")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: next to checkpoint)")
    args = parser.parse_args()

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    normalizers = checkpoint.get("normalizers", {})

    obs_normalizer = normalizers.get("obs")
    action_normalizer = normalizers.get("action")
    lowdim_normalizer = normalizers.get("lowdim")
    obs_steps = config.get("obs_steps", 2)
    act_steps = config.get("act_steps", 8)
    horizon = config.get("horizon", 16)
    action_dim = config.get("action_dim", 2)
    obs_type = config.get("obs_type", "state")

    # Restore agent
    print("Restoring agent...")
    if obs_type == "image":
        image_keys = config.get("image_keys", ("image",))
        lowdim_keys = config.get("lowdim_keys", ())
        ex_observations = {}
        for key in image_keys:
            ex_observations[key] = jnp.zeros((1, obs_steps, 96, 96, 3))
        # Infer lowdim dim from normalizer
        lowdim_dim = config.get("obs_dim")
        if lowdim_dim is None and lowdim_normalizer is not None:
            if hasattr(lowdim_normalizer, "min_val"):
                lowdim_dim = lowdim_normalizer.min_val.shape[-1]
        if lowdim_dim is not None and lowdim_dim > 0 and lowdim_keys:
            ex_observations["lowdim"] = jnp.zeros((1, obs_steps, lowdim_dim))
    else:
        obs_dim = config.get("obs_dim", 5)
        ex_observations = jnp.zeros((1, obs_steps, obs_dim))
    ex_actions = jnp.zeros((1, horizon, action_dim))
    agent, _ = restore_agent(args.checkpoint, BCAgent, ex_observations, ex_actions)
    print(f"Agent restored (step {checkpoint.get('training_step', '?')})")

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.checkpoint).parent / "trajectory_vis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pygame (needed for rendering)
    pygame.init()

    # Determine state seeds
    if args.state_seeds is not None:
        state_seeds = args.state_seeds
    else:
        state_seeds = list(range(args.num_states))

    # Create env (use image env for image obs_type)
    if obs_type == "image":
        env = PushTImageEnv(legacy=False, render_size=96)
    else:
        env = PushTEnv(legacy=False, render_size=512, render_action=False)
    # Separate env for high-res scene rendering
    render_env = PushTEnv(legacy=False, render_size=512, render_action=False)

    for idx, seed in enumerate(state_seeds):
        print(f"\nState {idx+1}/{len(state_seeds)} (seed={seed})")

        # Reset to random state
        env.reset(seed=seed)
        agent_pos = np.array(env.agent.position)
        state = np.array(
            list(env.agent.position) + list(env.block.position) + [env.block.angle]
        )

        print(f"  Agent: ({agent_pos[0]:.0f}, {agent_pos[1]:.0f}), "
              f"Block: ({env.block.position[0]:.0f}, {env.block.position[1]:.0f}), "
              f"Angle: {env.block.angle:.2f}")

        # Render scene (use render_env for high-res)
        render_env.reset(seed=seed)
        scene_img = render_scene(render_env)

        if args.rollout:
            # Rollout trajectories
            traces = rollout_trajectories(
                env, agent, obs_normalizer, action_normalizer, lowdim_normalizer,
                obs_steps, act_steps, args.num_trajectories,
                args.rollout_steps, state, obs_type,
            )
            plot_trajectories_on_scene(
                scene_img, traces, agent_pos,
                title=f"Rollout (seed={seed}, n={args.num_trajectories})",
                output_path=output_dir / f"rollout_seed{seed}.png",
                is_rollout=True,
            )
        else:
            # Open-loop: sample action chunks from current obs
            obs_stacked = get_normalized_obs(env, obs_normalizer, lowdim_normalizer, obs_steps, obs_type)
            # Add batch dim
            obs_batch = jax.tree.map(lambda x: jnp.array(x)[None], obs_stacked)

            trajectories = sample_trajectories(
                agent, obs_batch, args.num_trajectories, action_normalizer
            )
            plot_trajectories_on_scene(
                scene_img, trajectories, agent_pos,
                title=f"Action Chunks (seed={seed}, n={args.num_trajectories})",
                output_path=output_dir / f"actions_seed{seed}.png",
            )

    pygame.quit()
    print(f"\nDone! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
