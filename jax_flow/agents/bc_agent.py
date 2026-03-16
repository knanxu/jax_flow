"""Behavior Cloning agent with flow matching."""

import logging
from typing import Any

import flax
import jax
import jax.numpy as jnp

from jax_flow.agents.train_state import ModuleDict, TrainState, nonpytree_field
from jax_flow.core.utils import create_optimizer, get_batch_size
from jax_flow.flow.interpolant import Interpolant
from jax_flow.flow.losses import get_loss_fn
from jax_flow.flow.samplers import get_sampler

logger = logging.getLogger(__name__)


def _ema_update(ema_params, new_params, decay):
    """Update EMA parameters: ema = decay * ema + (1 - decay) * new."""
    return jax.tree_util.tree_map(
        lambda e, n: e * decay + n * (1.0 - decay), ema_params, new_params
    )


def _create_flow_def(network_type, action_dim, config, cond_dim=None):
    """Create flow network definition based on network_type.

    Args:
        network_type: One of 'mlp', 'small_mlp', 'unet', 'transformer'.
        action_dim: Action dimension.
        config: Configuration dict.
        cond_dim: Conditioning dimension (used by unet/transformer, ignored by mlp/small_mlp).

    Returns:
        Flow network module definition.
    """
    from jax_flow.networks.mlp import MLP, SmallMLP
    from jax_flow.networks.transformer import TransformerForFlow
    from jax_flow.networks.unet import ConditionalUnet1D

    if network_type == "mlp":
        return MLP(
            action_dim=action_dim,
            emb_dim=config.get("emb_dim", 512),
            n_blocks=config.get("n_blocks", 6),
            expansion_factor=config.get("expansion_factor", 4),
            dropout=config.get("dropout", 0.1),
            timestep_embed_dim=config.get("timestep_embed_dim", 128),
            max_freq=config.get("max_freq", 100.0),
        )
    elif network_type == "small_mlp":
        return SmallMLP(
            action_dim=action_dim,
            hidden_dims=tuple(config.get("hidden_dims", (512, 512, 512, 512))),
            timestep_embed_dim=config.get("timestep_embed_dim", 128),
            max_freq=config.get("max_freq", 100.0),
            layer_norm=config.get("layer_norm", True),
        )
    elif network_type == "unet":
        return ConditionalUnet1D(
            action_dim=action_dim,
            cond_dim=cond_dim or config.get("emb_dim", 256),
            model_dim=config.get("model_dim", 256),
            emb_dim=config.get("emb_dim", 256),
            kernel_size=config.get("kernel_size", 5),
            n_groups=config.get("n_groups", 8),
            cond_predict_scale=config.get("cond_predict_scale", True),
            dim_mult=tuple(config.get("dim_mult", (1, 2, 2))),
        )
    elif network_type == "transformer":
        return TransformerForFlow(
            action_dim=action_dim,
            n_layer=config.get("n_layer", 4),
            n_head=config.get("n_head", 4),
            n_emb=config.get("n_emb", 256),
            cond_dim=cond_dim or config.get("emb_dim", 256),
            dropout=config.get("dropout", 0.1),
            timestep_embed_dim=config.get("timestep_embed_dim", 128),
        )
    else:
        raise ValueError(f"Unknown network type: {network_type}")


class BCAgent(flax.struct.PyTreeNode):
    """Behavior Cloning agent using flow matching.

    Immutable PyTreeNode agent following qc project's design pattern.
    Single TrainState with ModuleDict manages all networks.
    """

    rng: Any
    network: TrainState
    ema_params: Any  # EMA copy of network params, used for inference
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
        from jax_flow.networks.encoders import create_encoder

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

        # Create flow network (initial, may be rebuilt with actual cond_dim)
        network_type = config.get("network_type", "mlp")
        flow_def = _create_flow_def(network_type, action_dim, config)

        # Initialize encoder first to get actual cond_dim
        rng, enc_rng, crop_init_rng = jax.random.split(rng, 3)
        encoder_params = encoder_def.init(
            {"params": enc_rng, "crop": crop_init_rng}, ex_observations
        )
        ex_cond = encoder_def.apply(encoder_params, ex_observations)
        actual_cond_dim = ex_cond.shape[-1]

        # Rebuild flow network with actual cond_dim (matters for unet/transformer)
        flow_def = _create_flow_def(
            network_type, action_dim, config, cond_dim=actual_cond_dim
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
        all_variables = network_def.init(
            {"params": init_rng, "crop": crop_md_rng}, **network_args
        )
        network_params = flax.core.unfreeze(all_variables["params"])
        batch_stats = (
            flax.core.unfreeze(all_variables["batch_stats"])
            if "batch_stats" in all_variables
            else None
        )

        network_params = flax.core.freeze(network_params)

        # Create optimizer (unified LR for all params, matching Diffusion Policy)
        grad_clip = config.get("grad_clip_norm", 0.0)
        tx = create_optimizer(
            lr=config.get("lr", 3e-4),
            weight_decay=config.get("weight_decay", 0.0),
            schedule_type=config.get("schedule_type", "constant"),
            warmup_steps=config.get("warmup_steps", 0),
            total_steps=config.get("gradient_steps", 100000),
            grad_clip_norm=grad_clip,
            b1=config.get("b1", 0.95),
            b2=config.get("b2", 0.999),
        )

        extra_vars = {"batch_stats": batch_stats} if batch_stats is not None else None
        network = TrainState.create(
            network_def, network_params, tx=tx, extra_variables=extra_vars
        )

        # Create interpolant and loss function
        interpolant = Interpolant(config.get("interp_type", "linear"))
        loss_fn_type = get_loss_fn(config.get("flow_type", "flow_matching"))

        # Initialize EMA params (frozen params are immutable, safe to share)
        ema_params = network_params

        return cls(
            rng=rng,
            network=network,
            ema_params=ema_params,
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

        # Update EMA params
        ema_decay = self.config.get("ema_decay", 0.995)
        new_ema_params = _ema_update(self.ema_params, new_network.params, ema_decay)

        return self.replace(
            network=new_network, ema_params=new_ema_params, rng=new_rng
        ), info

    @jax.jit
    def sample_actions(self, observations, rng=None, use_ema=True):
        """Sample actions from the flow policy.

        Args:
            observations: Observations. Shape: (batch, obs_steps, obs_dim)
            rng: Optional random key. If None, uses agent's rng.
            use_ema: Whether to use EMA params for inference (default True).

        Returns:
            Sampled actions. Shape: (batch, horizon, action_dim)
        """
        if rng is None:
            rng = self.rng

        sampler_type = self.config.get("sampler_type", "euler")
        num_steps = self.config.get("flow_steps", 10)

        # Use EMA params for inference (much-ado / Diffusion Policy pattern)
        params = self.ema_params if use_ema else None

        def encode(obs, training=False, rngs=None):
            if rngs is None:
                rngs = {}
            return self.network(
                obs, training=training, name="encoder", params=params, rngs=rngs
            )

        def flow_net(at, s, t, cond, training=False):
            return self.network(
                at, s, t, cond, training=training, name="flow", params=params
            )

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
