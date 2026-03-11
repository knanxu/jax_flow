"""Checkpoint saving and loading utilities."""

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    agent: Any,
    step: int,
    normalizers: Optional[Dict[str, Any]] = None,
    best_val_loss: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Save checkpoint to disk.

    Args:
        checkpoint_path: Path to save checkpoint.
        agent: BCAgent instance.
        step: Current training step.
        normalizers: Dict of normalizers (obs, action, lowdim).
        best_val_loss: Best validation loss so far.
        metadata: Additional metadata to save.
    """
    checkpoint_path = Path(checkpoint_path) if isinstance(checkpoint_path, str) else checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Build checkpoint dict - only save serializable parts
    # We save params and opt_state separately, not the full TrainState
    checkpoint = {
        "params": agent.network.params,
        "opt_state": agent.network.opt_state,
        "step": agent.network.step,
        "config": agent.config,
        "rng": agent.rng,
        "ema_params": agent.ema_params,
        "training_step": step,
    }

    if normalizers is not None:
        checkpoint["normalizers"] = normalizers

    if best_val_loss is not None:
        checkpoint["best_val_loss"] = best_val_loss

    if metadata is not None:
        checkpoint["metadata"] = metadata

    # Save with pickle
    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"✓ Checkpoint saved to {checkpoint_path}")


def load_checkpoint(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """Load checkpoint from disk.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        Checkpoint dict containing params, opt_state, config, normalizers, etc.
    """
    checkpoint_path = Path(checkpoint_path) if isinstance(checkpoint_path, str) else checkpoint_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)

    print(f"✓ Checkpoint loaded from {checkpoint_path}")
    return checkpoint


def restore_agent(checkpoint_path: Union[str, Path], agent_class: Any, ex_observations: Any, ex_actions: Any) -> Any:
    """Restore agent from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        agent_class: Agent class (e.g., BCAgent).
        ex_observations: Example observations for initialization.
        ex_actions: Example actions for initialization.

    Returns:
        Restored agent instance and checkpoint dict.
    """
    checkpoint = load_checkpoint(checkpoint_path)

    # Recreate agent with saved config
    agent = agent_class.create(
        seed=0,  # Seed doesn't matter, we'll replace the state
        ex_observations=ex_observations,
        ex_actions=ex_actions,
        config=checkpoint["config"],
    )

    # Restore network state
    restored_fields = {
        "rng": checkpoint["rng"],
        "network": agent.network.replace(
            params=checkpoint["params"],
            opt_state=checkpoint["opt_state"],
            step=checkpoint["step"],
        ),
    }
    # Restore EMA params if available, otherwise fall back to params
    restored_fields["ema_params"] = checkpoint.get("ema_params", checkpoint["params"])

    agent = agent.replace(**restored_fields)

    return agent, checkpoint


def cleanup_old_checkpoints(checkpoint_dir: Union[str, Path], keep_last_n: int = 3):
    """Remove old checkpoints, keeping only the last N.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        keep_last_n: Number of recent checkpoints to keep.
    """
    checkpoint_dir = Path(checkpoint_dir) if isinstance(checkpoint_dir, str) else checkpoint_dir

    if not checkpoint_dir.exists():
        return

    # Find all checkpoint files (exclude best_model.pkl and latest.pkl)
    checkpoint_files = sorted(
        [f for f in checkpoint_dir.glob("checkpoint_*.pkl")],
        key=lambda x: int(x.stem.split("_")[1]),
    )

    # Remove old checkpoints
    if len(checkpoint_files) > keep_last_n:
        for old_checkpoint in checkpoint_files[:-keep_last_n]:
            old_checkpoint.unlink()
            print(f"✓ Removed old checkpoint: {old_checkpoint.name}")


def save_resfit_checkpoint(
    checkpoint_path: Union[str, Path],
    agent: Any,
    step: int,
    bc_checkpoint_path: str = "",
    normalizers: Optional[Dict[str, Any]] = None,
    best_success_rate: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Save ResFiT checkpoint with multiple TrainStates.

    Args:
        checkpoint_path: Path to save checkpoint.
        agent: ResFiTAgent instance.
        step: Current training step.
        bc_checkpoint_path: Path to BC checkpoint (for restoring BC policy).
        normalizers: Dict of normalizers.
        best_success_rate: Best evaluation success rate so far.
        metadata: Additional metadata.
    """
    checkpoint_path = Path(checkpoint_path) if isinstance(checkpoint_path, str) else checkpoint_path
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        # Encoder
        "encoder_params": agent.encoder_state.params,
        "encoder_opt_state": agent.encoder_state.opt_state,
        "encoder_step": agent.encoder_state.step,
        # Critic
        "critic_params": agent.critic_state.params,
        "critic_opt_state": agent.critic_state.opt_state,
        "critic_step": agent.critic_state.step,
        # Actor
        "actor_params": agent.actor_state.params,
        "actor_opt_state": agent.actor_state.opt_state,
        "actor_step": agent.actor_state.step,
        # Targets
        "target_critic_params": agent.target_critic_params,
        "target_actor_params": agent.target_actor_params,
        # Agent state
        "rng": agent.rng,
        "config": agent.config,
        "training_step": step,
        "bc_checkpoint_path": bc_checkpoint_path,
    }

    if normalizers is not None:
        checkpoint["normalizers"] = normalizers
    if best_success_rate is not None:
        checkpoint["best_success_rate"] = best_success_rate
    if metadata is not None:
        checkpoint["metadata"] = metadata

    with open(checkpoint_path, "wb") as f:
        pickle.dump(checkpoint, f)

    print(f"Checkpoint saved to {checkpoint_path}")


def load_resfit_checkpoint(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """Load ResFiT checkpoint from disk.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        Checkpoint dict with encoder/actor/critic params, targets, config, etc.
    """
    return load_checkpoint(checkpoint_path)
