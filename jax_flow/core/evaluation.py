"""Evaluation utilities for policy rollout."""

from typing import Any

import numpy as np


def rollout_episode(
    agent: Any,
    env: Any,
    max_steps: int = 500,
    render: bool = False,
    save_video: bool = False,
    seed: int | None = None,
) -> dict[str, Any]:
    """Rollout a single episode."""
    if seed is not None:
        obs, info = env.reset(seed=seed)
    else:
        obs, info = env.reset()
    done = False
    episode_length = 0
    episode_return = 0.0
    max_reward = 0.0
    success = False
    result_num_completed = 0
    frames = [] if save_video else None
    action_list = []

    while not done and episode_length < max_steps:
        if isinstance(obs, dict):
            obs_batch = {k: np.array(v[np.newaxis, ...]) for k, v in obs.items()}
        else:
            obs_batch = obs[np.newaxis, ...]

        needs_replan = getattr(env, "needs_replan", lambda: True)()

        if needs_replan:
            actions = agent.eval_actions(obs_batch)
            action_seq = np.array(actions[0])
            action_list.append(action_seq)
        else:
            action_seq = None

        # For residual agents: action_seq is (action_dim,) single step
        # For BC agents: action_seq is (horizon, action_dim) chunk or None
        obs, reward, terminated, truncated, info = env.step(action_seq)
        done = terminated or truncated
        episode_length += 1
        episode_return += reward
        max_reward = max(max_reward, float(reward))

        if "success" in info and bool(info["success"]):
            success = True

        # Track Kitchen completed tasks (highest seen during episode)
        if "num_completed" in info:
            num_completed = int(info["num_completed"])
            if num_completed > result_num_completed:
                result_num_completed = num_completed

        if save_video:
            frame = env.render()
            frames.append(frame)

        if render:
            env.render()

    result = {
        "length": episode_length,
        "return": episode_return,
        "success": success,
        "max_reward": max_reward,
    }

    if result_num_completed > 0:
        result["num_completed"] = result_num_completed

    if action_list:
        all_actions = np.concatenate(action_list, axis=0)
        result["action_mean"] = float(np.mean(all_actions))
        result["action_std"] = float(np.std(all_actions))
        result["action_abs_mean"] = float(np.mean(np.abs(all_actions)))
        result["action_clip_ratio"] = float(np.mean(np.abs(all_actions) > 0.99))

    if save_video and frames:
        result["frames"] = np.array(frames)

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
) -> dict[str, Any]:
    """Evaluate policy over multiple episodes."""
    episode_lengths = []
    episode_returns = []
    successes = []
    max_rewards = []
    num_completed_list = []
    videos = []
    action_means = []
    action_stds = []
    action_abs_means = []
    action_clip_ratios = []

    for i in range(num_episodes):
        save_this_video = save_video and i < num_videos

        result = rollout_episode(
            agent=agent,
            env=env,
            max_steps=max_steps,
            render=render,
            save_video=save_this_video,
            seed=i,
        )

        episode_lengths.append(result["length"])
        episode_returns.append(result["return"])
        successes.append(result["success"])
        max_rewards.append(result["max_reward"])

        if "num_completed" in result:
            num_completed_list.append(result["num_completed"])

        if "action_mean" in result:
            action_means.append(result["action_mean"])
            action_stds.append(result["action_std"])
            action_abs_means.append(result["action_abs_mean"])
            action_clip_ratios.append(result["action_clip_ratio"])

        if save_this_video and "frames" in result:
            videos.append(result["frames"])

        if verbose and (i + 1) % 10 == 0:
            current_success_rate = np.mean(successes)
            print(
                f"  Episode {i + 1}/{num_episodes} | Success rate: {current_success_rate:.2%}"
            )

    results = {
        "success_rate": float(np.mean(successes)),
        "num_successes": int(np.sum(successes)),
        "num_episodes": num_episodes,
        "avg_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "avg_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_score": float(np.mean(max_rewards)),
        "episode_lengths": episode_lengths,
        "episode_returns": episode_returns,
        "successes": successes,
    }

    if action_means:
        results["action_mean"] = float(np.mean(action_means))
        results["action_std"] = float(np.mean(action_stds))
        results["action_abs_mean"] = float(np.mean(action_abs_means))
        results["action_clip_ratio"] = float(np.mean(action_clip_ratios))

    if videos:
        results["videos"] = videos

    # Kitchen p1-p7 metrics: fraction of episodes completing >= k subtasks
    if num_completed_list:
        for k in range(1, 8):
            results[f"p{k}"] = float(np.mean([n >= k for n in num_completed_list]))
        results["avg_num_completed"] = float(np.mean(num_completed_list))

    return results


def print_evaluation_results(results: dict[str, Any]):
    """Print evaluation results."""
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(
        f"Episodes: {results['num_episodes']}"
    )
    print(
        f"Success Rate: {results['success_rate']:.2%} ({results['num_successes']}/{results['num_episodes']})"
    )
    print(
        f"Avg Episode Length: {results['avg_length']:.1f} ± {results['std_length']:.1f}"
    )
    print(f"Avg Return: {results['avg_return']:.2f} ± {results['std_return']:.2f}")
    if "mean_score" in results:
        print(f"Mean Score: {results['mean_score']:.4f}")
    if "action_mean" in results:
        print(
            f"Action stats: mean={results['action_mean']:.3f}, std={results['action_std']:.3f}, "
            f"|mean|={results['action_abs_mean']:.3f}, clip_ratio={results['action_clip_ratio']:.3f}"
        )
    # Kitchen p1-p7
    if "p1" in results:
        print(f"Avg completed subtasks: {results['avg_num_completed']:.2f}")
        pk_str = " | ".join(f"p{k}={results[f'p{k}']:.2%}" for k in range(1, 8))
        print(f"  {pk_str}")
    print("=" * 80)
