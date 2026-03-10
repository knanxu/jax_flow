"""Behavior Cloning agent with flow matching."""

from typing import Any

import flax
import jax
import jax.numpy as jnp

from jax_flow.agents.train_state import ModuleDict, TrainState, nonpytree_field
from jax_flow.core.utils import get_batch_size
from jax_flow.flow.interpolant import Interpolant
from jax_flow.flow.losses import get_loss_fn
from jax_flow.flow.samplers import get_sampler


class BCAgent(flax.struct.PyTreeNode):
    """Behavior Cloning agent using flow matching.

    Immutable PyTreeNode agent following qc project's design pattern.
    Single TrainState with ModuleDict manages all networks.
    """

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()
    interpolant: Any = nonpytree_field()
    loss_fn_type: Any = nonpytree_field()

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new BCAgent.

        Args:
            seed: Random seed.
            ex_observations: Example observations for initialization.
                Shape: (batch, obs_steps, obs_dim) or (batch, obs_dim)
            ex_actions: Example actions for initialization.
                Shape: (batch, horizon, action_dim)
            config: Configuration dict.

        Returns:
            New BCAgent instance.
        """
        from jax_flow.core.utils import create_optimizer
        from jax_flow.networks.encoders import create_encoder
        from jax_flow.networks.mlp import MLP
        from jax_flow.networks.unet import ConditionalUnet1D
        from jax_flow.networks.transformer import TransformerForFlow

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        action_dim = ex_actions.shape[-1]
        horizon = config.get("horizon", 10)

        # Create encoder
        encoder_def = create_encoder(
            encoder_type=config.get("encoder_type", "mlp"),
            hidden_dims=tuple(config.get("encoder_hidden_dims", (256, 256))),
            output_dim=config.get("emb_dim", 256),
            image_keys=tuple(config.get("image_keys", ("agentview_image",))),
            lowdim_keys=tuple(
                config.get("lowdim_keys", ("robot0_eef_pos", "robot0_gripper_qpos"))
            ),
            crop_shape=config.get("crop_shape", None),
        )

        # Create flow network based on network_type
        network_type = config.get("network_type", "mlp")
        if network_type == "mlp":
            flow_def = MLP(
                action_dim=action_dim,
                emb_dim=config.get("emb_dim", 512),
                n_blocks=config.get("n_blocks", 6),
                expansion_factor=config.get("expansion_factor", 4),
                dropout=config.get("dropout", 0.1),
                timestep_embed_dim=config.get("timestep_embed_dim", 128),
                max_freq=config.get("max_freq", 100.0),
            )
        elif network_type == "unet":
            flow_def = ConditionalUnet1D(
                action_dim=action_dim,
                down_dims=tuple(config.get("down_dims", (256, 512, 1024))),
                kernel_size=config.get("kernel_size", 5),
                n_groups=config.get("n_groups", 8),
                cond_dim=config.get("emb_dim", 256),
                timestep_embed_dim=config.get("timestep_embed_dim", 256),
                cond_predict_scale=config.get("cond_predict_scale", False),
            )
        elif network_type == "transformer":
            flow_def = TransformerForFlow(
                action_dim=action_dim,
                n_layer=config.get("n_layer", 4),
                n_head=config.get("n_head", 4),
                n_emb=config.get("n_emb", 256),
                cond_dim=config.get("emb_dim", 256),
                dropout=config.get("dropout", 0.1),
                timestep_embed_dim=config.get("timestep_embed_dim", 128),
            )
        else:
            raise ValueError(f"Unknown network type: {network_type}")

        # Initialize encoder first to get actual cond_dim
        rng, enc_rng, crop_init_rng = jax.random.split(rng, 3)
        encoder_params = encoder_def.init(
            {"params": enc_rng, "crop": crop_init_rng}, ex_observations
        )
        ex_cond = encoder_def.apply(encoder_params, ex_observations)
        actual_cond_dim = ex_cond.shape[-1]

        # Rebuild flow network with actual cond_dim
        if network_type == "mlp":
            flow_def = MLP(
                action_dim=action_dim,
                emb_dim=config.get("emb_dim", 512),
                n_blocks=config.get("n_blocks", 6),
                expansion_factor=config.get("expansion_factor", 4),
                dropout=config.get("dropout", 0.1),
                timestep_embed_dim=config.get("timestep_embed_dim", 128),
                max_freq=config.get("max_freq", 100.0),
            )
        elif network_type == "unet":
            flow_def = ConditionalUnet1D(
                action_dim=action_dim,
                down_dims=tuple(config.get("down_dims", (256, 512, 1024))),
                kernel_size=config.get("kernel_size", 5),
                n_groups=config.get("n_groups", 8),
                cond_dim=actual_cond_dim,
                timestep_embed_dim=config.get("timestep_embed_dim", 256),
                cond_predict_scale=config.get("cond_predict_scale", False),
            )
        elif network_type == "transformer":
            flow_def = TransformerForFlow(
                action_dim=action_dim,
                n_layer=config.get("n_layer", 4),
                n_head=config.get("n_head", 4),
                n_emb=config.get("n_emb", 256),
                cond_dim=actual_cond_dim,
                dropout=config.get("dropout", 0.1),
                timestep_embed_dim=config.get("timestep_embed_dim", 128),
            )

        # Build ModuleDict with example inputs
        batch_size = get_batch_size(ex_observations)
        ex_times = jnp.zeros((batch_size,))
        ex_actions_seq = jnp.zeros((batch_size, horizon, action_dim))

        networks = {
            "encoder": encoder_def,
            "flow": flow_def,
        }
        network_args = {
            "encoder": (ex_observations,),
            "flow": (ex_actions_seq, ex_times, ex_times, ex_cond),
        }

        network_def = ModuleDict(networks)
        rng, crop_md_rng = jax.random.split(rng)
        network_params = network_def.init(
            {"params": init_rng, "crop": crop_md_rng}, **network_args
        )["params"]

        # Create optimizer
        tx = create_optimizer(
            lr=config.get("lr", 3e-4),
            weight_decay=config.get("weight_decay", 0.0),
            schedule_type=config.get("schedule_type", "constant"),
            warmup_steps=config.get("warmup_steps", 0),
            total_steps=config.get("gradient_steps", 100000),
        )

        network = TrainState.create(network_def, network_params, tx=tx)

        # Create interpolant and loss function
        interpolant = Interpolant(config.get("interp_type", "linear"))
        loss_fn_type = get_loss_fn(config.get("flow_type", "flow_matching"))

        return cls(
            rng=rng,
            network=network,
            config=dict(config),
            interpolant=interpolant,
            loss_fn_type=loss_fn_type,
        )

    @jax.jit
    def update(self, batch):
        """Update agent with a batch of data.

        Args:
            batch: Dict with 'observations' and 'actions'.

        Returns:
            Tuple of (new_agent, info_dict).
        """
        new_rng, rng, dropout_rng, crop_rng = jax.random.split(self.rng, 4)

        def loss_fn(params):
            # Get encoder and flow network with gradient flow
            dropout_rngs = {"dropout": dropout_rng, "crop": crop_rng}

            def encode(obs, training=True, rngs=None):
                if rngs is None:
                    rngs = dropout_rngs
                return self.network(
                    obs,
                    training=training,
                    name="encoder",
                    params=params,
                    rngs=rngs,
                )

            def flow_net(at, s, t, cond, training=True):
                return self.network(
                    at,
                    s,
                    t,
                    cond,
                    training=training,
                    name="flow",
                    params=params,
                    rngs=dropout_rngs,
                )

            return self.loss_fn_type(
                network=flow_net,
                encoder=encode,
                interpolant=self.interpolant,
                batch=batch,
                rng=rng,
                config=self.config,
                step=self.network.step,
            )

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, rng=None):
        """Sample actions from the flow policy.

        Args:
            observations: Observations. Shape: (batch, obs_steps, obs_dim)
            rng: Optional random key. If None, uses agent's rng.

        Returns:
            Sampled actions. Shape: (batch, horizon, action_dim)
        """
        if rng is None:
            rng = self.rng

        sampler_type = self.config.get("sampler_type", "euler")
        num_steps = self.config.get("flow_steps", 10)

        def encode(obs, training=False, rngs=None):
            if rngs is None:
                rngs = {}
            return self.network(obs, training=training, name="encoder", rngs=rngs)

        def flow_net(at, s, t, cond, training=False):
            return self.network(at, s, t, cond, training=training, name="flow")

        sampler = get_sampler(sampler_type)

        actions = sampler(
            network=flow_net,
            encoder=encode,
            observations=observations,
            num_steps=num_steps,
            rng=rng,
            config=self.config,
        )

        return jnp.clip(actions, -1, 1)

    @jax.jit
    def eval_actions(self, observations):
        """Deterministic action sampling for evaluation.

        Uses a fixed rng key for reproducibility.

        Args:
            observations: Observations. Shape: (batch, obs_steps, obs_dim)

        Returns:
            Actions. Shape: (batch, horizon, action_dim)
        """
        eval_rng = jax.random.PRNGKey(0)
        return self.sample_actions(observations, rng=eval_rng)
