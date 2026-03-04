"""Behavior Cloning agent with flow matching."""

from typing import Any

import flax
import jax
import jax.numpy as jnp

from jax_flow.agents.train_state import ModuleDict, TrainState, nonpytree_field
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
                Shape: (batch, obs_steps, obs_dim)
            ex_actions: Example actions for initialization.
                Shape: (batch, action_dim)
            config: Configuration dict (ml_collections.ConfigDict).

        Returns:
            New BCAgent instance.
        """
        from jax_flow.core.utils import create_optimizer
        from jax_flow.networks.encoders import create_encoder
        from jax_flow.networks.mlp import MLP

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        action_dim = ex_actions.shape[-1]
        horizon = config.get("horizon", 10)
        full_action_dim = (
            action_dim * horizon if config.get("action_chunking", False) else action_dim
        )

        # Create encoder
        encoder_def = create_encoder(
            encoder_type=config.get("encoder_type", "mlp"),
            hidden_dims=tuple(config.get("encoder_hidden_dims", (256, 256))),
            output_dim=config.get("emb_dim", 256),
        )

        # Create flow network
        flow_def = MLP(
            action_dim=full_action_dim,
            hidden_dims=tuple(config.get("hidden_dims", (512, 512, 512))),
            cond_dim=config.get("emb_dim", 256),
            activation=config.get("activation", "gelu"),
            layer_norm=config.get("layer_norm", False),
        )

        # Build ModuleDict
        ex_times = jnp.zeros((ex_observations.shape[0], 1))
        ex_cond = jnp.zeros((ex_observations.shape[0], config.get("emb_dim", 256)))
        ex_flat_actions = jnp.zeros((ex_observations.shape[0], full_action_dim))

        networks = {
            "encoder": encoder_def,
            "flow": flow_def,
        }
        network_args = {
            "encoder": (ex_observations,),
            "flow": (ex_flat_actions, ex_times, ex_times, ex_cond),
        }

        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)["params"]

        # Create optimizer
        tx = create_optimizer(
            lr=config.get("lr", 3e-4),
            weight_decay=config.get("weight_decay", 0.0),
            schedule_type=config.get("schedule_type", "constant"),
            warmup_steps=config.get("warmup_steps", 0),
            total_steps=config.get("gradient_steps", 100000),
        )

        network = TrainState.create(network_def, network_params, tx=tx)

        # Create interpolant and loss function (avoid recreating in hot path)
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
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(params):
            # Get encoder and flow network with gradient flow
            def encode(obs):
                return self.network(obs, name="encoder", params=params)

            def flow_net(x, s, t, cond, training=True):
                return self.network(
                    x, s, t, cond, training=training, name="flow", params=params
                )

            return self.loss_fn_type(
                network=flow_net,
                encoder=encode,
                interpolant=self.interpolant,
                batch=batch,
                rng=rng,
                config=self.config,
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

        def encode(obs):
            return self.network(obs, name="encoder")

        def flow_net(x, s, t, cond, training=False):
            return self.network(x, s, t, cond, training=training, name="flow")

        sampler = get_sampler(sampler_type)

        if sampler_type == "mip":
            actions = sampler(
                network=flow_net,
                encoder=encode,
                observations=observations,
                rng=rng,
                config=self.config,
            )
        else:
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
