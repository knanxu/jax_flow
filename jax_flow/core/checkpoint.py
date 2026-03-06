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
    agent = agent.replace(
        rng=checkpoint["rng"],
        network=agent.network.replace(
            params=checkpoint["params"],
            opt_state=checkpoint["opt_state"],
            step=checkpoint["step"],
        ),
    )

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
