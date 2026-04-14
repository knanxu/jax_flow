"""Diagnostic: compare BC eval actions between normal pipeline and SpeedTuning pipeline.

Runs both pipelines on the same env reset seed and prints action stats to find divergence.

Usage:
    python scripts/debug_speed_tuning.py task=square_lowdim \
        algorithm.bc_checkpoint=path/to/best_model.pkl
"""

import os

_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_strict_conv_algorithm_picker" not in _xla_flags:
    os.environ["XLA_FLAGS"] = _xla_flags + " --xla_gpu_strict_conv_algorithm_picker=false"

import hydra
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig

from jax_flow.agents.bc_agent import BCAgent
from jax_flow.agents.speed_tuning.interpolation import temporal_interpolate
from jax_flow.agents.speed_tuning.speed_tuning_env import SpeedTuningEnvWrapper
from jax_flow.core.checkpoint import load_checkpoint, restore_agent
from jax_flow.core.evaluation import rollout_episode
from jax_flow.data import DatasetManager, make_dataset, make_robomimic_dataset
from jax_flow.envs import make_env, make_robomimic_env


def create_base_env(cfg, normalizers, resolved_path, abs_action=False):
    """Same as train_speed_tuning.py — FrameStack only, no ActionChunking."""
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
        return make_env(
            task_source=task_source,
            obs_type=obs_type if task_source == "pusht" else None,
            env_name=cfg.task.env_name if task_source == "kitchen" else None,
            **common_kwargs,
        )

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

    return make_robomimic_env(
        env_name=cfg.task.env_name,
        dataset_path=resolved_path,
        obs_type=obs_type,
        obs_keys=tuple(
            cfg.task.dataset.get(
                "obs_keys",
                ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
            )
        ),
        image_keys=image_keys,
        lowdim_keys=lowdim_keys,
        render_offscreen=False,
        abs_action=abs_action,
        **common_kwargs,
    )


def create_normal_eval_env(cfg, normalizers, resolved_path, abs_action=False):
    """Normal BC eval env — FrameStack + ActionChunking."""
    task_source = cfg.task.get("task_source", "robomimic")
    obs_type = cfg.task.obs_type
    is_image = obs_type == "image"

    common_kwargs = {
        "obs_normalizer": normalizers.get("obs"),
        "action_normalizer": normalizers.get("action"),
        "lowdim_normalizer": normalizers.get("lowdim"),
        "max_episode_steps": cfg.task.env.max_episode_steps,
        "frame_stack": cfg.task.dataset.obs_steps,
        "act_exec_steps": cfg.task.dataset.act_steps,
        "seed": cfg.seed,
    }

    if task_source in ("pusht", "kitchen"):
        return make_env(
            task_source=task_source,
            obs_type=obs_type if task_source == "pusht" else None,
            env_name=cfg.task.env_name if task_source == "kitchen" else None,
            **common_kwargs,
        )

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

    return make_robomimic_env(
        env_name=cfg.task.env_name,
        dataset_path=resolved_path,
        obs_type=obs_type,
        obs_keys=tuple(
            cfg.task.dataset.get(
                "obs_keys",
                ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
            )
        ),
        image_keys=image_keys,
        lowdim_keys=lowdim_keys,
        render_offscreen=False,
        abs_action=abs_action,
        **common_kwargs,
    )


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="speed_tuning_config",
)
def main(cfg: DictConfig):
    bc_checkpoint_path = cfg.algorithm.bc_checkpoint
    if bc_checkpoint_path is None:
        raise ValueError("algorithm.bc_checkpoint is required.")

    np.random.seed(cfg.seed)

    # Load checkpoint
    bc_ckpt = load_checkpoint(bc_checkpoint_path)
    bc_ckpt_config = bc_ckpt["config"]
    normalizers = bc_ckpt.get("normalizers", {})
    bc_abs_action = bc_ckpt_config.get("abs_action", False)

    # Resolve dataset
    task_source = cfg.task.get("task_source", "robomimic")
    dataset_path = bc_ckpt_config.get("dataset_path", cfg.task.dataset.path)
    task_name = cfg.task.get("dataset_name", cfg.task.env_name.lower())
    dataset_type = cfg.task.dataset.get("dataset_type", "ph")
    obs_type = cfg.task.obs_type

    resolved_path = DatasetManager.ensure_dataset(
        dataset_path=dataset_path, task=task_name, dataset_type=dataset_type,
        obs_type=obs_type, source=task_source, auto_download=True,
    )
    resolved_path = str(resolved_path)

    # Create dataset for example obs
    dataset_kwargs = {}
    if bc_abs_action:
        dataset_kwargs["abs_action"] = True
    if obs_type != "image" and "obs_keys" in cfg.task:
        dataset_kwargs["obs_keys"] = tuple(cfg.task.obs_keys)

    common_ds_kwargs = {
        "horizon": cfg.task.dataset.horizon,
        "obs_steps": cfg.task.dataset.obs_steps,
        "act_steps": cfg.task.dataset.act_steps,
        "val_ratio": 0.0,
    }

    if task_source in ("pusht", "kitchen"):
        train_dataset = make_dataset(
            task_source=task_source, dataset_path=resolved_path,
            obs_type=obs_type, mode="train", **common_ds_kwargs,
        )
    else:
        train_dataset = make_robomimic_dataset(
            dataset_path=resolved_path, obs_type=obs_type,
            mode="train", **common_ds_kwargs, **dataset_kwargs,
        )

    ex_sample = train_dataset[0]
    if isinstance(ex_sample["observations"], dict):
        ex_obs = {k: jnp.array(v[np.newaxis, ...]) for k, v in ex_sample["observations"].items()}
    else:
        ex_obs = jnp.array(ex_sample["observations"][np.newaxis, ...])

    bc_action_dim = bc_ckpt_config.get("action_dim", ex_sample["actions"].shape[-1])
    bc_horizon = bc_ckpt_config.get("horizon", 16)
    ex_actions = jnp.zeros((1, bc_horizon, bc_action_dim))

    bc_agent, _ = restore_agent(bc_checkpoint_path, BCAgent, ex_obs, ex_actions)

    k_skip = cfg.algorithm.get("k_skip", 4)
    act_steps = cfg.task.dataset.act_steps

    print("=" * 80)
    print("DIAGNOSTIC: Comparing normal BC eval vs SpeedTuning pipeline")
    print(f"  bc_abs_action={bc_abs_action}, action_dim={bc_action_dim}, horizon={bc_horizon}")
    print(f"  act_steps={act_steps}, k_skip={k_skip}")
    print("=" * 80)

    # ================================================================
    # Test 1: Normal BC eval (known working)
    # ================================================================
    print("\n--- Test 1: Normal BC eval (FrameStack + ActionChunking) ---")
    normal_env = create_normal_eval_env(cfg, normalizers, resolved_path, abs_action=bc_abs_action)
    for ep in range(3):
        result = rollout_episode(agent=bc_agent, env=normal_env, max_steps=400, seed=ep)
        print(f"  Episode {ep}: success={result['success']}, length={result['length']}")

    # ================================================================
    # Test 2: SpeedTuning pipeline at speed=1.0
    # ================================================================
    print("\n--- Test 2: SpeedTuning pipeline (speed=1.0, k_skip={}) ---".format(k_skip))
    for ep in range(3):
        st_env = create_base_env(cfg, normalizers, resolved_path, abs_action=bc_abs_action)
        st_env = SpeedTuningEnvWrapper(
            env=st_env, bc_agent=bc_agent, speed_options=[1.0],
            alpha=0.0, beta=0.0, k_skip=k_skip, abs_action=bc_abs_action,
        )
        obs, _ = st_env.reset(seed=ep)
        done = False
        ep_length = 0
        ep_success = False
        while not done and ep_length < 400:
            obs, reward, terminated, truncated, info = st_env.step(0)
            done = terminated or truncated
            ep_length += info.get("steps_executed", 1)
            if "success" in info and bool(info["success"]):
                ep_success = True
        print(f"  Episode {ep}: success={ep_success}, length={ep_length}")

    # ================================================================
    # Test 3: Compare first action chunk
    # ================================================================
    print("\n--- Test 3: Compare first action chunk ---")
    # Get obs from normal env
    normal_env2 = create_normal_eval_env(cfg, normalizers, resolved_path, abs_action=bc_abs_action)
    obs_normal, _ = normal_env2.reset(seed=0)
    if isinstance(obs_normal, dict):
        obs_batch_normal = {k: np.array(v[np.newaxis, ...]) for k, v in obs_normal.items()}
    else:
        obs_batch_normal = obs_normal[np.newaxis, ...]

    # Get obs from speed tuning env
    st_env2 = create_base_env(cfg, normalizers, resolved_path, abs_action=bc_abs_action)
    st_env2_wrapped = SpeedTuningEnvWrapper(
        env=st_env2, bc_agent=bc_agent, speed_options=[1.0],
        alpha=0.0, beta=0.0, k_skip=k_skip, abs_action=bc_abs_action,
    )
    obs_st, _ = st_env2_wrapped.reset(seed=0)
    if isinstance(obs_st, dict):
        obs_batch_st = {k: np.array(v[np.newaxis, ...]) for k, v in obs_st.items()}
    else:
        obs_batch_st = obs_st[np.newaxis, ...]

    # Compare obs
    if isinstance(obs_batch_normal, dict):
        for k in obs_batch_normal:
            diff = np.max(np.abs(np.array(obs_batch_normal[k]) - np.array(obs_batch_st[k])))
            print(f"  Obs key '{k}' max diff: {diff:.6e}")
    else:
        diff = np.max(np.abs(np.array(obs_batch_normal) - np.array(obs_batch_st)))
        print(f"  Obs max diff: {diff:.6e}")

    # Compare BC action output
    actions_normal = bc_agent.eval_actions(obs_batch_normal)
    actions_st = bc_agent.eval_actions(obs_batch_st)
    chunk_normal = np.array(actions_normal[0])
    chunk_st = np.array(actions_st[0])
    print(f"  Action chunk shape (normal): {chunk_normal.shape}")
    print(f"  Action chunk shape (st):     {chunk_st.shape}")
    print(f"  Action chunk max diff: {np.max(np.abs(chunk_normal - chunk_st)):.6e}")
    print(f"  Action chunk[0] (normal): {chunk_normal[0][:5]}...")
    print(f"  Action chunk[0] (st):     {chunk_st[0][:5]}...")

    # Compare interpolated actions
    interp = temporal_interpolate(chunk_st, speed=1.0, k_skip=k_skip)
    print(f"\n  Interpolated shape (speed=1.0, k_skip={k_skip}): {interp.shape}")
    print(f"  Original chunk[:k_skip] vs interpolated max diff: {np.max(np.abs(chunk_st[:k_skip] - interp)):.6e}")
    print(f"  Interpolated[0]: {interp[0][:5]}...")
    print(f"  Original[0]:     {chunk_st[0][:5]}...")

    # ================================================================
    # Test 4: Manual step-by-step execution on base env (no wrapper)
    # ================================================================
    print("\n--- Test 4: Manual step-by-step on base env (no wrapper) ---")
    base_env = create_base_env(cfg, normalizers, resolved_path, abs_action=bc_abs_action)
    obs, _ = base_env.reset(seed=0)
    if isinstance(obs, dict):
        obs_b = {k: np.array(v[np.newaxis, ...]) for k, v in obs.items()}
    else:
        obs_b = obs[np.newaxis, ...]

    actions = bc_agent.eval_actions(obs_b)
    chunk = np.array(actions[0])
    print(f"  BC chunk shape: {chunk.shape}, range: [{chunk.min():.3f}, {chunk.max():.3f}]")

    ep_success = False
    ep_length = 0
    done = False
    # Execute first k_skip actions manually
    for i in range(min(k_skip, len(chunk))):
        obs, reward, terminated, truncated, info = base_env.step(chunk[i])
        done = terminated or truncated
        ep_length += 1
        if "success" in info and bool(info["success"]):
            ep_success = True
        if done:
            break

    # Continue with replan loop
    while not done and ep_length < 400:
        if isinstance(obs, dict):
            obs_b = {k: np.array(v[np.newaxis, ...]) for k, v in obs.items()}
        else:
            obs_b = obs[np.newaxis, ...]
        actions = bc_agent.eval_actions(obs_b)
        chunk = np.array(actions[0])
        for i in range(min(k_skip, len(chunk))):
            obs, reward, terminated, truncated, info = base_env.step(chunk[i])
            done = terminated or truncated
            ep_length += 1
            if "success" in info and bool(info["success"]):
                ep_success = True
            if done:
                break

    print(f"  Manual execution: success={ep_success}, length={ep_length}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
