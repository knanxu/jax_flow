"""Integration test for image training pipeline."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_image_pipeline_integration():
    """Test complete image pipeline without real dataset."""
    from jax_flow.agents.bc_agent import BCAgent
    from jax_flow.core.utils import get_batch_size

    # Mock image observations (dict format)
    batch_size = 4
    obs_steps = 2
    horizon = 16
    action_dim = 7

    # Create mock dict observations
    mock_observations = {
        "agentview_image": jnp.zeros((batch_size, obs_steps, 84, 84, 3)),
        "robot0_eye_in_hand_image": jnp.zeros((batch_size, obs_steps, 84, 84, 3)),
        "robot0_eef_pos": jnp.zeros((batch_size, obs_steps, 3)),
        "robot0_eef_quat": jnp.zeros((batch_size, obs_steps, 4)),
        "robot0_gripper_qpos": jnp.zeros((batch_size, obs_steps, 2)),
    }
    mock_actions = jnp.zeros((batch_size, horizon, action_dim))

    # Test get_batch_size with dict obs
    assert get_batch_size(mock_observations) == batch_size

    # Create agent config
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
        "gradient_steps": 1000,
        "interp_type": "linear",
        "flow_type": "flow_matching",
        "sampler_type": "euler",
        "flow_steps": 10,
    }

    # Create agent
    agent = BCAgent.create(
        seed=0,
        ex_observations=mock_observations,
        ex_actions=mock_actions,
        config=config,
    )

    # Test update
    batch = {
        "observations": mock_observations,
        "actions": mock_actions,
    }
    agent, info = agent.update(batch)

    # Check loss is finite
    assert jnp.isfinite(info["loss"])
    assert info["loss"] > 0

    # Test sampling
    sampled_actions = agent.sample_actions(mock_observations)
    assert sampled_actions.shape == (batch_size, horizon, action_dim)
    assert jnp.all(jnp.abs(sampled_actions) <= 1.0)

    print("✓ Image pipeline integration test passed!")
    print(f"  Loss: {info['loss']:.4f}")
    print(f"  Sampled actions shape: {sampled_actions.shape}")


def test_dataloader_dict_obs():
    """Test dataloader handles dict observations correctly."""
    import numpy as np

    # Mock dataset samples
    obs_batch = []
    act_batch = []

    for _ in range(4):
        obs = {
            "agentview_image": np.zeros((2, 84, 84, 3)),
            "robot0_eef_pos": np.zeros((2, 3)),
        }
        act = np.zeros((16, 7))
        obs_batch.append(obs)
        act_batch.append(act)

    # Stack like in train_bc.py
    first_obs = obs_batch[0]
    if isinstance(first_obs, dict):
        observations = {
            k: jnp.array(np.stack([o[k] for o in obs_batch]))
            for k in first_obs
        }
    else:
        observations = jnp.array(np.stack(obs_batch))

    actions = jnp.array(np.stack(act_batch))

    # Check shapes
    assert isinstance(observations, dict)
    assert observations["agentview_image"].shape == (4, 2, 84, 84, 3)
    assert observations["robot0_eef_pos"].shape == (4, 2, 3)
    assert actions.shape == (4, 16, 7)

    print("✓ Dataloader dict obs test passed!")


def test_losses_with_dict_obs():
    """Test loss functions work with dict observations."""
    from jax_flow.flow.losses import flow_loss
    from jax_flow.flow.interpolant import Interpolant
    from jax_flow.networks.encoders import create_encoder
    from jax_flow.networks.mlp import MLP

    # Mock dict observations
    observations = {
        "agentview_image": jnp.zeros((4, 2, 84, 84, 3)),
        "robot0_eef_pos": jnp.zeros((4, 2, 3)),
    }
    actions = jnp.zeros((4, 16, 7))

    # Create encoder
    encoder_def = create_encoder(
        encoder_type="multi_image",
        hidden_dims=(256, 256),
        output_dim=256,
        image_keys=("agentview_image",),
        lowdim_keys=("robot0_eef_pos",),
        crop_shape=(76, 76),
    )

    rng = jax.random.PRNGKey(0)
    rng, enc_rng, crop_rng = jax.random.split(rng, 3)
    encoder_params = encoder_def.init(
        {"params": enc_rng, "crop": crop_rng}, observations
    )

    def encode(obs, training=False, rngs=None):
        if rngs is None:
            rngs = {}
        return encoder_def.apply(encoder_params, obs, training=training, rngs=rngs)

    # Create flow network
    cond = encode(observations)
    flow_def = MLP(
        action_dim=7,
        hidden_dims=(512, 512, 512),
        cond_dim=cond.shape[-1],
        activation="gelu",
        layer_norm=False,
    )

    rng, flow_rng = jax.random.split(rng)
    flow_params = flow_def.init(
        flow_rng,
        jnp.zeros((4, 16, 7)),
        jnp.zeros((4,)),
        jnp.zeros((4,)),
        cond,
    )

    def flow_net(at, s, t, cond, training=False):
        return flow_def.apply(flow_params, at, s, t, cond, training=training)

    # Test loss
    interpolant = Interpolant("linear")
    batch = {"observations": observations, "actions": actions}
    config = {"horizon": 16, "obs_steps": 2}

    rng, loss_rng = jax.random.split(rng)
    loss, info = flow_loss(
        network=flow_net,
        encoder=encode,
        interpolant=interpolant,
        batch=batch,
        rng=loss_rng,
        config=config,
    )

    assert jnp.isfinite(loss)
    assert loss > 0

    print("✓ Loss with dict obs test passed!")
    print(f"  Loss: {loss:.4f}")


def test_samplers_with_dict_obs():
    """Test samplers work with dict observations."""
    from jax_flow.flow.samplers import euler_sampler
    from jax_flow.networks.encoders import create_encoder
    from jax_flow.networks.mlp import MLP

    # Mock dict observations
    observations = {
        "agentview_image": jnp.zeros((4, 2, 84, 84, 3)),
        "robot0_eef_pos": jnp.zeros((4, 2, 3)),
    }

    # Create encoder
    encoder_def = create_encoder(
        encoder_type="multi_image",
        hidden_dims=(256, 256),
        output_dim=256,
        image_keys=("agentview_image",),
        lowdim_keys=("robot0_eef_pos",),
        crop_shape=(76, 76),
    )

    rng = jax.random.PRNGKey(0)
    rng, enc_rng, crop_rng = jax.random.split(rng, 3)
    encoder_params = encoder_def.init(
        {"params": enc_rng, "crop": crop_rng}, observations
    )

    def encode(obs, training=False, rngs=None):
        if rngs is None:
            rngs = {}
        return encoder_def.apply(encoder_params, obs, training=training, rngs=rngs)

    # Create flow network
    cond = encode(observations)
    flow_def = MLP(
        action_dim=7,
        hidden_dims=(512, 512, 512),
        cond_dim=cond.shape[-1],
        activation="gelu",
        layer_norm=False,
    )

    rng, flow_rng = jax.random.split(rng)
    flow_params = flow_def.init(
        flow_rng,
        jnp.zeros((4, 16, 7)),
        jnp.zeros((4,)),
        jnp.zeros((4,)),
        cond,
    )

    def flow_net(at, s, t, cond, training=False):
        return flow_def.apply(flow_params, at, s, t, cond, training=training)

    # Test sampler
    config = {"horizon": 16, "obs_steps": 2, "action_dim": 7}
    rng, sample_rng = jax.random.split(rng)

    actions = euler_sampler(
        network=flow_net,
        encoder=encode,
        observations=observations,
        num_steps=10,
        rng=sample_rng,
        config=config,
    )

    assert actions.shape == (4, 16, 7)

    print("✓ Sampler with dict obs test passed!")
    print(f"  Sampled actions shape: {actions.shape}")


if __name__ == "__main__":
    print("Running image pipeline integration tests...\n")

    test_dataloader_dict_obs()
    print()

    test_losses_with_dict_obs()
    print()

    test_samplers_with_dict_obs()
    print()

    test_image_pipeline_integration()
    print()

    print("=" * 60)
    print("All image pipeline tests passed! ✓")
    print("=" * 60)
