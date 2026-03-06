"""Test checkpoint and evaluation functionality."""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jax_flow.agents.bc_agent import BCAgent
from jax_flow.core.checkpoint import save_checkpoint, load_checkpoint, cleanup_old_checkpoints


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    print("Testing checkpoint save/load...")

    # Create a simple agent
    rng = jax.random.PRNGKey(42)

    # Example observations and actions
    batch_size = 4
    obs_steps = 2
    obs_dim = 10
    horizon = 16
    action_dim = 7

    ex_observations = jnp.zeros((batch_size, obs_steps, obs_dim))
    ex_actions = jnp.zeros((batch_size, horizon, action_dim))

    config = {
        "horizon": horizon,
        "obs_steps": obs_steps,
        "action_dim": action_dim,
        "obs_dim": obs_dim,
        "encoder_type": "mlp",
        "encoder_hidden_dims": (128, 128),
        "emb_dim": 128,
        "hidden_dims": (256, 256),
        "activation": "gelu",
        "layer_norm": False,
        "network_type": "mlp",
        "lr": 1e-4,
        "weight_decay": 0.0,
        "schedule_type": "constant",
        "warmup_steps": 0,
        "gradient_steps": 1000,
        "interp_type": "linear",
        "flow_type": "flow_matching",
        "sampler_type": "euler",
        "flow_steps": 10,
    }

    agent = BCAgent.create(
        seed=42,
        ex_observations=ex_observations,
        ex_actions=ex_actions,
        config=config,
    )

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Save checkpoint
        checkpoint_path = tmpdir / "test_checkpoint.pkl"
        save_checkpoint(
            checkpoint_path=checkpoint_path,
            agent=agent,
            step=1000,
            normalizers={"action": None, "obs": None},
            best_val_loss=0.5,
        )

        assert checkpoint_path.exists(), "Checkpoint file not created"
        print(f"  ✓ Checkpoint saved to {checkpoint_path}")

        # Load checkpoint
        loaded_checkpoint = load_checkpoint(checkpoint_path)

        assert "params" in loaded_checkpoint
        assert "opt_state" in loaded_checkpoint
        assert "config" in loaded_checkpoint
        assert "normalizers" in loaded_checkpoint
        assert "step" in loaded_checkpoint
        assert "training_step" in loaded_checkpoint
        assert "best_val_loss" in loaded_checkpoint

        assert loaded_checkpoint["training_step"] == 1000
        assert loaded_checkpoint["best_val_loss"] == 0.5

        print("  ✓ Checkpoint loaded successfully")

        # Test cleanup
        for i in range(5):
            ckpt_path = tmpdir / f"checkpoint_{i * 1000}.pkl"
            save_checkpoint(
                checkpoint_path=ckpt_path,
                agent=agent,
                step=i * 1000,
                normalizers={},
            )

        # Should have 5 checkpoints
        checkpoints = list(tmpdir.glob("checkpoint_*.pkl"))
        assert len(checkpoints) == 5, f"Expected 5 checkpoints, got {len(checkpoints)}"
        print(f"  ✓ Created {len(checkpoints)} checkpoints")

        # Cleanup, keep last 2
        cleanup_old_checkpoints(tmpdir, keep_last_n=2)

        checkpoints = list(tmpdir.glob("checkpoint_*.pkl"))
        assert len(checkpoints) == 2, f"Expected 2 checkpoints after cleanup, got {len(checkpoints)}"
        print(f"  ✓ Cleanup successful, kept {len(checkpoints)} checkpoints")

        # Verify the kept checkpoints are the latest ones
        kept_steps = sorted([int(p.stem.split("_")[1]) for p in checkpoints])
        assert kept_steps == [3000, 4000], f"Expected [3000, 4000], got {kept_steps}"
        print(f"  ✓ Kept correct checkpoints: {kept_steps}")

    print("✓ All checkpoint tests passed!\n")


def test_agent_sample_actions():
    """Test agent action sampling."""
    print("Testing agent action sampling...")

    rng = jax.random.PRNGKey(42)

    # Create agent
    batch_size = 2
    obs_steps = 2
    obs_dim = 10
    horizon = 16
    action_dim = 7

    ex_observations = jnp.zeros((batch_size, obs_steps, obs_dim))
    ex_actions = jnp.zeros((batch_size, horizon, action_dim))

    config = {
        "horizon": horizon,
        "obs_steps": obs_steps,
        "action_dim": action_dim,
        "obs_dim": obs_dim,
        "encoder_type": "mlp",
        "encoder_hidden_dims": (128, 128),
        "emb_dim": 128,
        "hidden_dims": (256, 256),
        "activation": "gelu",
        "layer_norm": False,
        "network_type": "mlp",
        "lr": 1e-4,
        "weight_decay": 0.0,
        "schedule_type": "constant",
        "warmup_steps": 0,
        "gradient_steps": 1000,
        "interp_type": "linear",
        "flow_type": "flow_matching",
        "sampler_type": "euler",
        "flow_steps": 10,
    }

    agent = BCAgent.create(
        seed=42,
        ex_observations=ex_observations,
        ex_actions=ex_actions,
        config=config,
    )

    # Sample actions
    test_obs = jnp.ones((1, obs_steps, obs_dim))
    actions = agent.sample_actions(test_obs)

    assert actions.shape == (1, horizon, action_dim), f"Expected shape (1, {horizon}, {action_dim}), got {actions.shape}"
    assert jnp.all(jnp.abs(actions) <= 1.0), "Actions should be clipped to [-1, 1]"

    print(f"  ✓ Sampled actions shape: {actions.shape}")
    print(f"  ✓ Actions range: [{float(jnp.min(actions)):.3f}, {float(jnp.max(actions)):.3f}]")
    print("✓ Action sampling test passed!\n")


if __name__ == "__main__":
    print("=" * 80)
    print("Running Checkpoint and Evaluation Tests")
    print("=" * 80 + "\n")

    test_checkpoint_save_load()
    test_agent_sample_actions()

    print("=" * 80)
    print("All tests passed!")
    print("=" * 80)
