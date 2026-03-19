"""Evaluation script for trained BC policies.

Usage:
    # Evaluate a checkpoint
    python scripts/eval_bc.py --checkpoint checkpoints/square_lowdim_meanflow_mlp/best_model.pkl

    # Evaluate with custom settings
    python scripts/eval_bc.py --checkpoint path/to/checkpoint.pkl --num_episodes 100 --save_video

    # Evaluate with rendering
    python scripts/eval_bc.py --checkpoint path/to/checkpoint.pkl --render
"""

import argparse
from pathlib import Path

import numpy as np

from jax_flow.agents.bc_agent import BCAgent
from jax_flow.core.checkpoint import load_checkpoint
from jax_flow.core.evaluation import evaluate_policy, print_evaluation_results


# Tasks that use pusht/kitchen backends
PUSHT_ENV_NAMES = {"PushT", "pusht"}
KITCHEN_ENV_NAMES = {"kitchen"}


def _infer_task_source(config):
    """Infer task_source from checkpoint config."""
    task_source = config.get("task_source")
    if task_source:
        return task_source
    env_name = config.get("env_name", "")
    if env_name in PUSHT_ENV_NAMES:
        return "pusht"
    if env_name in KITCHEN_ENV_NAMES:
        return "kitchen"
    return "robomimic"


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained BC policy")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=50, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Maximum steps per episode (auto-detect from config)"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render episodes in real-time"
    )
    parser.add_argument(
        "--save_video", action="store_true", help="Save videos of episodes"
    )
    parser.add_argument(
        "--num_videos", type=int, default=3, help="Number of videos to save"
    )
    parser.add_argument("--video_fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory for videos"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("BC Policy Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Seed: {args.seed}")
    print("=" * 80)

    # Set random seed
    np.random.seed(args.seed)

    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    normalizers = checkpoint.get("normalizers", {})

    # Infer task source
    task_source = _infer_task_source(config)
    obs_type = config.get("obs_type", "lowdim")
    obs_steps = config.get("obs_steps", 2)
    act_steps = config.get("act_steps", 8)
    horizon = config.get("horizon", 16)
    action_dim = config.get("action_dim", 7)

    # Default max_steps per task source
    if args.max_steps is None:
        if task_source == "pusht":
            args.max_steps = 300
        elif task_source == "kitchen":
            args.max_steps = 280
        else:
            args.max_steps = 400

    print(f"Task source: {task_source}, obs_type: {obs_type}")

    # Create example observations for agent restoration
    print("Restoring agent...")
    import jax.numpy as jnp

    if obs_type == "image":
        image_keys = config.get("image_keys", ("agentview_image",))
        lowdim_keys = config.get("lowdim_keys", ())
        ex_observations = {}
        for key in image_keys:
            ex_observations[key] = jnp.zeros((1, obs_steps, 84, 84, 3))
        # Infer lowdim dim from normalizer or config
        lowdim_dim = config.get("obs_dim")
        if lowdim_dim is None and "lowdim" in normalizers:
            # Infer from normalizer shape
            norm = normalizers["lowdim"]
            if hasattr(norm, "data_min"):
                lowdim_dim = norm.data_min.shape[-1]
            elif hasattr(norm, "min_val"):
                lowdim_dim = norm.min_val.shape[-1]
        if lowdim_dim is not None and lowdim_dim > 0 and lowdim_keys:
            ex_observations["lowdim"] = jnp.zeros((1, obs_steps, lowdim_dim))
    else:
        obs_dim = config.get("obs_dim", 10)
        ex_observations = jnp.zeros((1, obs_steps, obs_dim))

    ex_actions = jnp.zeros((1, horizon, action_dim))

    # Restore agent
    from jax_flow.core.checkpoint import restore_agent

    agent, _ = restore_agent(args.checkpoint, BCAgent, ex_observations, ex_actions)
    print(f"✓ Agent restored (step {checkpoint.get('training_step', 'unknown')})")

    # Create evaluation environment
    print("\nCreating evaluation environment...")

    if task_source in ("pusht", "kitchen"):
        from jax_flow.envs import make_env

        env_kwargs = {
            "task_source": task_source,
            "obs_normalizer": normalizers.get("obs"),
            "action_normalizer": normalizers.get("action"),
            "lowdim_normalizer": normalizers.get("lowdim"),
            "max_episode_steps": args.max_steps,
            "frame_stack": obs_steps,
            "act_exec_steps": act_steps,
            "seed": args.seed,
        }
        if task_source == "pusht":
            env_kwargs["obs_type"] = obs_type
        elif task_source == "kitchen":
            env_kwargs["env_name"] = config.get("env_name", "kitchen")

        eval_env = make_env(**env_kwargs)
    else:
        from jax_flow.envs import make_robomimic_env

        dataset_path = config.get("dataset_path")
        if dataset_path is None:
            print("\n✗ Error: dataset_path not found in checkpoint config")
            return

        env_name = config.get("env_name", "lift")
        abs_action = config.get("abs_action", False)

        if obs_type == "image":
            image_keys_env = tuple(config.get("image_keys", ("agentview_image",)))
            lowdim_keys_env = tuple(
                config.get(
                    "lowdim_keys",
                    ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
                )
            )
            obs_keys_env = None
        else:
            image_keys_env = None
            lowdim_keys_env = None
            obs_keys_env = tuple(
                config.get(
                    "obs_keys",
                    ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"),
                )
            )

        eval_env = make_robomimic_env(
            env_name=env_name,
            dataset_path=dataset_path,
            obs_type=obs_type,
            obs_keys=obs_keys_env,
            image_keys=image_keys_env,
            lowdim_keys=lowdim_keys_env,
            obs_normalizer=normalizers.get("obs"),
            action_normalizer=normalizers.get("action"),
            lowdim_normalizer=normalizers.get("lowdim"),
            max_episode_steps=args.max_steps,
            frame_stack=obs_steps,
            act_exec_steps=act_steps,
            seed=args.seed,
            render_offscreen=args.save_video or args.render,
            abs_action=abs_action,
        )

    print(f"✓ Environment created: {config.get('env_name', 'unknown')} ({obs_type})")

    # Run evaluation
    print("\n" + "=" * 80)
    print("Running Evaluation")
    print("=" * 80)

    eval_results = evaluate_policy(
        agent=agent,
        env=eval_env,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        render=args.render,
        save_video=args.save_video,
        num_videos=args.num_videos,
        verbose=True,
    )

    # Print results
    print_evaluation_results(eval_results)

    # Save videos if requested
    if args.save_video and "videos" in eval_results:
        output_dir = (
            Path(args.output_dir)
            if args.output_dir
            else Path(args.checkpoint).parent / "eval_videos"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving videos to {output_dir}...")
        try:
            import imageio

            for i, video_frames in enumerate(eval_results["videos"]):
                video_path = output_dir / f"episode_{i}.mp4"
                imageio.mimsave(video_path, video_frames, fps=args.video_fps)
                print(f"  ✓ Saved {video_path}")
        except ImportError:
            print("  ✗ imageio not installed, skipping video saving")
            print("  Install with: pip install imageio[ffmpeg]")

    # Cleanup
    if hasattr(eval_env, "close"):
        eval_env.close()


if __name__ == "__main__":
    main()
