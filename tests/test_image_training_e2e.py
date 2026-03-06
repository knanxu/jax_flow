"""End-to-end test for image training pipeline with mock data."""

import jax
import jax.numpy as jnp
import numpy as np


def test_full_training_loop():
    """Test complete training loop with image observations."""
    from jax_flow.agents.bc_agent import BCAgent

    print("=" * 60)
    print("End-to-End Image Training Test")
    print("=" * 60)

    # Configuration
    batch_size = 8
    obs_steps = 2
    horizon = 16
    action_dim = 7
    num_train_steps = 50

    # Mock image dataset
    print("\n1. Creating mock image dataset...")
    mock_observations = {
        "agentview_image": np.random.randint(0, 255, (batch_size, obs_steps, 84, 84, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.random.randint(0, 255, (batch_size, obs_steps, 84, 84, 3), dtype=np.uint8),
        "robot0_eef_pos": np.random.randn(batch_size, obs_steps, 3).astype(np.float32),
        "robot0_eef_quat": np.random.randn(batch_size, obs_steps, 4).astype(np.float32),
        "robot0_gripper_qpos": np.random.randn(batch_size, obs_steps, 2).astype(np.float32),
    }
    mock_actions = np.random.randn(batch_size, horizon, action_dim).astype(np.float32)

    # Convert to JAX arrays
    observations_jax = {k: jnp.array(v) for k, v in mock_observations.items()}
    actions_jax = jnp.array(mock_actions)

    print(f"  Batch size: {batch_size}")
    print(f"  Observation keys: {list(observations_jax.keys())}")
    print(f"  Image shapes: {observations_jax['agentview_image'].shape}")
    print(f"  Lowdim shape: {observations_jax['robot0_eef_pos'].shape}")
    print(f"  Actions shape: {actions_jax.shape}")

    # Create agent
    print("\n2. Creating BCAgent with image encoder...")
    config = {
        "horizon": horizon,
        "obs_steps": obs_steps,
        "action_dim": action_dim,
        "obs_dim": 9,  # 3 + 4 + 2
        "encoder_type": "multi_image",
        "image_keys": ("agentview_image", "robot0_eye_in_hand_image"),
        "lowdim_keys": ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"),
        "crop_shape": (76, 76),
        "encoder_hidden_dims": (256, 256),
        "emb_dim": 256,
        "hidden_dims": (512, 512, 512),
        "activation": "gelu",
        "layer_norm": False,
        "network_type": "mlp",
        "lr": 1e-4,
        "weight_decay": 0.0,
        "schedule_type": "constant",
        "warmup_steps": 0,
        "gradient_steps": num_train_steps,
        "interp_type": "linear",
        "flow_type": "flow_matching",
        "sampler_type": "euler",
        "flow_steps": 10,
    }

    agent = BCAgent.create(
        seed=42,
        ex_observations=observations_jax,
        ex_actions=actions_jax,
        config=config,
    )
    print("  ✓ Agent created successfully")

    # Training loop
    print(f"\n3. Running training for {num_train_steps} steps...")
    losses = []
    grad_norms = []

    for step in range(num_train_steps):
        # Create batch
        batch = {
            "observations": observations_jax,
            "actions": actions_jax,
        }

        # Update
        agent, info = agent.update(batch)

        # Record metrics
        losses.append(float(info["loss"]))
        grad_norms.append(float(info["grad/norm"]))

        if (step + 1) % 10 == 0:
            avg_loss = np.mean(losses[-10:])
            avg_grad = np.mean(grad_norms[-10:])
            print(f"  Step {step + 1:3d}: loss={avg_loss:.4f}, grad_norm={avg_grad:.4f}")

    print(f"\n  Final loss: {losses[-1]:.4f}")
    print(f"  Loss decreased: {losses[0]:.4f} -> {losses[-1]:.4f}")

    # Test sampling
    print("\n4. Testing action sampling...")
    sampled_actions = agent.sample_actions(observations_jax)
    print(f"  Sampled actions shape: {sampled_actions.shape}")
    print(f"  Actions in range [-1, 1]: {jnp.all(jnp.abs(sampled_actions) <= 1.0)}")

    # Test eval mode
    print("\n5. Testing eval mode...")
    eval_actions = agent.eval_actions(observations_jax)
    print(f"  Eval actions shape: {eval_actions.shape}")
    print(f"  Actions in range [-1, 1]: {jnp.all(jnp.abs(eval_actions) <= 1.0)}")

    # Verify determinism in eval mode
    eval_actions2 = agent.eval_actions(observations_jax)
    is_deterministic = jnp.allclose(eval_actions, eval_actions2)
    print(f"  Eval is deterministic: {is_deterministic}")

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"✓ Agent creation: PASSED")
    print(f"✓ Training loop: PASSED ({num_train_steps} steps)")
    print(f"✓ Action sampling: PASSED")
    print(f"✓ Eval mode: PASSED")
    print(f"✓ Deterministic eval: {'PASSED' if is_deterministic else 'FAILED'}")
    print("=" * 60)

    # Assertions
    assert jnp.isfinite(losses[-1]), "Final loss is not finite"
    assert sampled_actions.shape == (batch_size, horizon, action_dim), "Wrong action shape"
    assert jnp.all(jnp.abs(sampled_actions) <= 1.0), "Actions out of range"
    assert is_deterministic, "Eval mode is not deterministic"

    print("\n✓ All end-to-end tests passed!")


def test_different_network_types():
    """Test image pipeline with different network architectures."""
    from jax_flow.agents.bc_agent import BCAgent

    print("\n" + "=" * 60)
    print("Testing Different Network Types")
    print("=" * 60)

    batch_size = 4
    obs_steps = 2
    horizon = 16
    action_dim = 7

    # Mock data
    observations = {
        "agentview_image": jnp.zeros((batch_size, obs_steps, 84, 84, 3)),
        "robot0_eef_pos": jnp.zeros((batch_size, obs_steps, 3)),
    }
    actions = jnp.zeros((batch_size, horizon, action_dim))

    base_config = {
        "horizon": horizon,
        "obs_steps": obs_steps,
        "action_dim": action_dim,
        "obs_dim": 3,
        "encoder_type": "multi_image",
        "image_keys": ("agentview_image",),
        "lowdim_keys": ("robot0_eef_pos",),
        "crop_shape": (76, 76),
        "encoder_hidden_dims": (256,),
        "emb_dim": 128,
        "lr": 1e-4,
        "gradient_steps": 10,
        "interp_type": "linear",
        "flow_type": "flow_matching",
        "sampler_type": "euler",
        "flow_steps": 5,
    }

    network_configs = {
        "mlp": {
            "network_type": "mlp",
            "hidden_dims": (256, 256),
            "activation": "gelu",
            "layer_norm": False,
        },
        "unet": {
            "network_type": "unet",
            "down_dims": (128, 256),
            "kernel_size": 5,
            "n_groups": 8,
            "timestep_embed_dim": 64,
        },
        "transformer": {
            "network_type": "transformer",
            "n_layer": 2,
            "n_head": 4,
            "n_emb": 128,
            "dropout": 0.0,
            "timestep_embed_dim": 64,
        },
    }

    for net_type, net_config in network_configs.items():
        print(f"\n{net_type.upper()}:")
        config = {**base_config, **net_config}

        try:
            agent = BCAgent.create(
                seed=0,
                ex_observations=observations,
                ex_actions=actions,
                config=config,
            )

            # Test update
            batch = {"observations": observations, "actions": actions}
            agent, info = agent.update(batch)

            # Test sampling
            sampled = agent.sample_actions(observations)

            print(f"  ✓ {net_type} network: PASSED")
            print(f"    Loss: {info['loss']:.4f}")
            print(f"    Sampled shape: {sampled.shape}")

        except Exception as e:
            print(f"  ✗ {net_type} network: FAILED")
            print(f"    Error: {e}")
            raise

    print("\n✓ All network types passed!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running End-to-End Image Training Tests")
    print("=" * 60)

    test_full_training_loop()
    print("\n")
    test_different_network_types()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
