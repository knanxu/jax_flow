"""Evaluation utilities for policy rollout."""

import numpy as np
from typing import Any, Dict


def rollout_episode(
    agent: Any,
    env: Any,
    max_steps: int = 500,
    render: bool = False,
    save_video: bool = False,
) -> Dict[str, Any]:
    """Rollout a single episode.

    Args:
        agent: Agent with eval_actions method.
        env: Gymnasium environment.
        max_steps: Maximum episode steps.
        render: Whether to render in real-time.
        save_video: Whether to save video frames.

    Returns:
        Dict containing episode statistics and optional video frames.
    """
    obs, info = env.reset()
    done = False
    episode_length = 0
    episode_return = 0.0
    success = False
    frames = [] if save_video else None

    while not done and episode_length < max_steps:
        # Preprocess observation for agent
        if isinstance(obs, dict):
            # Dict obs (image mode): add batch dimension
            obs_batch = {k: v[np.newaxis, ...] for k, v in obs.items()}
        else:
            # Array obs (lowdim mode): add batch dimension
            obs_batch = obs[np.newaxis, ...]

        # Check if we need to replan (for ActionChunkingWrapper)
        needs_replan = getattr(env, "needs_replan", lambda: True)()

        if needs_replan:
            # Sample action sequence from agent
            actions = agent.eval_actions(obs_batch)  # (1, horizon, action_dim)
            action_seq = np.array(actions[0])  # (horizon, action_dim)
        else:
            # Use buffered actions
            action_seq = None

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action_seq)
        done = terminated or truncated
        episode_length += 1
        episode_return += reward

        # Check success
        if "success" in info:
            success = bool(info["success"])

        # Save frame
        if save_video:
            frame = env.render()
            frames.append(frame)

        # Render
        if render:
            env.render()

    result = {
        "length": episode_length,
        "return": episode_return,
        "success": success,
    }

    if save_video and frames:
        result["frames"] = np.array(frames)  # (T, H, W, C)

    return result


def evaluate_policy(
    agent: Any,
    env: Any,
    num_episodes: int = 50,
    max_steps: int = 500,
    render: bool = False,
    save_video: bool = False,
    num_videos: int = 3,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate policy over multiple episodes.

    Args:
        agent: Agent with eval_actions method.
        env: Gymnasium environment.
        num_episodes: Number of episodes to evaluate.
        max_steps: Maximum steps per episode.
        render: Whether to render in real-time.
        save_video: Whether to save videos.
        num_videos: Number of videos to save (first N episodes).
        verbose: Whether to print progress.

    Returns:
        Dict containing evaluation statistics and optional videos.
    """
    episode_lengths = []
    episode_returns = []
    successes = []
    videos = []

    for i in range(num_episodes):
        # Save video for first num_videos episodes
        save_this_video = save_video and i < num_videos

        result = rollout_episode(
            agent=agent,
            env=env,
            max_steps=max_steps,
            render=render,
            save_video=save_this_video,
        )

        episode_lengths.append(result["length"])
        episode_returns.append(result["return"])
        successes.append(result["success"])

        if save_this_video and "frames" in result:
            videos.append(result["frames"])

        if verbose and (i + 1) % 10 == 0:
            current_success_rate = np.mean(successes)
            print(f"  Episode {i + 1}/{num_episodes} | Success rate: {current_success_rate:.2%}")

    # Compute statistics
    success_rate = np.mean(successes)
    avg_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    avg_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)

    results = {
        "success_rate": float(success_rate),
        "num_successes": int(np.sum(successes)),
        "num_episodes": num_episodes,
        "avg_length": float(avg_length),
        "std_length": float(std_length),
        "avg_return": float(avg_return),
        "std_return": float(std_return),
        "episode_lengths": episode_lengths,
        "episode_returns": episode_returns,
        "successes": successes,
    }

    if videos:
        results["videos"] = videos

    return results


def print_evaluation_results(results: Dict[str, Any]):
    """Print evaluation results in a formatted way.

    Args:
        results: Results dict from evaluate_policy.
    """
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Episodes: {results['num_episodes']}")
    print(f"Success Rate: {results['success_rate']:.2%} ({results['num_successes']}/{results['num_episodes']})")
    print(f"Avg Episode Length: {results['avg_length']:.1f} ± {results['std_length']:.1f}")
    print(f"Avg Return: {results['avg_return']:.2f} ± {results['std_return']:.2f}")
    print("=" * 80)
