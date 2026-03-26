"""Offline RL Agent: DDPG + BC constraint on flow policy.

Two-phase training:
  Phase 1 (BC warmup): BCAgent trains actor, CriticState trains critic in parallel.
  Phase 2 (RL): OfflineRLAgent optimizes actor (Q+BC loss) and critic (TD loss).

Follows the ACFQL grad_params pattern: separate critic_loss / actor_loss methods,
each selectively routing gradients via params=grad_params.
"""

import logging
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from jax_flow.agents.bc_agent import _create_flow_def
from jax_flow.agents.train_state import ModuleDict, TrainState, nonpytree_field
from jax_flow.core.checkpoint import load_checkpoint
from jax_flow.core.utils import (
    create_optimizer,
    get_batch_size,
    make_offline_rl_param_labels,
)
from jax_flow.flow.interpolant import Interpolant
from jax_flow.flow.losses import get_loss_fn
from jax_flow.flow.samplers import get_sampler
from jax_flow.networks.value import Value

logger = logging.getLogger(__name__)


class OfflineRLAgent(flax.struct.PyTreeNode):
    """Offline RL agent for the RL phase.

    Single ModuleDict TrainState: encoder + flow + critic + target_critic.
    Uses the grad_params pattern for clean gradient routing:
      - critic_loss: only critic receives grad_params
      - actor_loss: only flow (+ optionally encoder) receives grad_params
    No EMA — eval uses current params directly.
    """

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()
    interpolant: Any = nonpytree_field()
    loss_fn_type: Any = nonpytree_field()

    # ================================================================
    # Creation
    # ================================================================

    @classmethod
    def create_from_bc(cls, seed, bc_agent, critic_state, config):
        """Create OfflineRLAgent from trained BCAgent + CriticState.

        Args:
            seed: Random seed.
            bc_agent: Trained BCAgent (provides encoder + flow params and defs).
            critic_state: Trained CriticState (provides critic + target_critic params).
            config: RL phase config dict.

        Returns:
            New OfflineRLAgent.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        # Extract module defs from BCAgent's ModuleDict
        bc_modules = bc_agent.network.model_def.modules
        encoder_def = bc_modules["encoder"]
        flow_def = bc_modules["flow"]

        # Extract module defs from CriticState's ModuleDict
        critic_modules = critic_state.network.model_def.modules
        critic_def = critic_modules["critic"]
        target_critic_def = critic_modules["target_critic"]

        # Build combined ModuleDict
        network_def = ModuleDict({
            "encoder": encoder_def,
            "flow": flow_def,
            "critic": critic_def,
            "target_critic": target_critic_def,
        })

        # Assemble params from both sources
        # Use EMA params from BCAgent (best BC weights) for encoder + flow
        bc_params = flax.core.unfreeze(bc_agent.ema_params)
        critic_params = flax.core.unfreeze(critic_state.network.params)

        params = {
            "modules_encoder": bc_params["modules_encoder"],
            "modules_flow": bc_params["modules_flow"],
            "modules_critic": critic_params["modules_critic"],
            "modules_target_critic": critic_params["modules_target_critic"],
        }
        params = flax.core.freeze(params)

        # Build optimizer: encoder frozen, flow + critic trainable
        actor_lr = config.get("actor_lr", 1e-4)
        critic_lr = config.get("critic_lr", 3e-4)
        grad_clip = config.get("grad_clip_norm", 10.0)
        freeze_encoder = config.get("freeze_encoder", True)

        param_labels = make_offline_rl_param_labels(
            params, freeze_encoder=freeze_encoder
        )
        tx = optax.multi_transform(
            {
                "frozen": optax.set_to_zero(),
                "actor": create_optimizer(lr=actor_lr, grad_clip_norm=grad_clip),
                "critic": create_optimizer(lr=critic_lr, grad_clip_norm=grad_clip),
            },
            param_labels,
        )

        # Handle batch_stats if present in BCAgent
        extra_vars = None
        if bc_agent.network.extra_variables is not None:
            extra_vars = bc_agent.network.extra_variables

        network = TrainState.create(
            network_def, params, tx=tx, extra_variables=extra_vars
        )

        # Interpolant and loss from BCAgent config
        bc_config = bc_agent.config
        interpolant = Interpolant(bc_config.get("interp_type", "linear"))
        loss_fn_type = get_loss_fn(bc_config.get("flow_type", "flow_matching"))

        return cls(
            rng=rng,
            network=network,
            config=dict(config),
            interpolant=interpolant,
            loss_fn_type=loss_fn_type,
        )

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, bc_checkpoint_path=None):
        """Create OfflineRLAgent from scratch (mode 2: direct RL, no BC warmup).

        Args:
            seed: Random seed.
            ex_observations: Example observations for init.
            ex_actions: Example actions (batch, horizon, action_dim).
            config: Algorithm config dict.
            bc_checkpoint_path: Path to BC checkpoint (for network architecture).

        Returns:
            New OfflineRLAgent.
        """
        from jax_flow.networks.encoders import create_encoder

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        # Load BC checkpoint for network config
        load_bc_weights = config.get("load_bc_weights", False)
        bc_params = None
        if bc_checkpoint_path is not None:
            bc_ckpt = load_checkpoint(bc_checkpoint_path)
            bc_config = bc_ckpt["config"]
            if load_bc_weights:
                bc_params = bc_ckpt["params"]
        else:
            bc_config = config

        action_dim = ex_actions.shape[-1]
        horizon = config.get("horizon", bc_config.get("horizon", 16))

        # Rebuild encoder def
        encoder_def = create_encoder(
            encoder_type=bc_config.get("encoder_type", "mlp"),
            hidden_dims=tuple(bc_config.get("encoder_hidden_dims", (256, 256))),
            output_dim=bc_config.get("emb_dim", 256),
            image_keys=tuple(bc_config.get("image_keys", ("agentview_image",))),
            lowdim_keys=tuple(
                bc_config.get("lowdim_keys", ("robot0_eef_pos", "robot0_gripper_qpos"))
            ),
            crop_shape=bc_config.get("crop_shape", None),
        )

        # Init encoder to get cond_dim
        rng, enc_rng, crop_rng = jax.random.split(rng, 3)
        encoder_params_init = encoder_def.init(
            {"params": enc_rng, "crop": crop_rng}, ex_observations
        )
        ex_cond = encoder_def.apply(encoder_params_init, ex_observations)
        actual_cond_dim = ex_cond.shape[-1]

        network_type = bc_config.get("network_type", "mlp")
        flow_def = _create_flow_def(
            network_type, action_dim, bc_config, cond_dim=actual_cond_dim
        )

        # Critic defs — optionally with independent encoder
        critic_hidden_dims = tuple(config.get("critic_hidden_dims", [512, 1024, 512]))
        critic_encoder_type = config.get("critic_encoder_type", "none")
        critic_has_encoder = critic_encoder_type != "none"
        critic_kwargs = {
            "hidden_dims": critic_hidden_dims,
            "layer_norm": config.get("critic_layer_norm", True),
            "num_ensembles": config.get("num_ensembles", 2),
            "activation": config.get("critic_activation", "tanh"),
        }

        critic_encoder_def = None
        target_critic_encoder_def = None
        if critic_has_encoder:
            critic_encoder_def = create_encoder(
                encoder_type=bc_config.get("encoder_type", "mlp"),
                hidden_dims=tuple(bc_config.get("encoder_hidden_dims", (256, 256))),
                output_dim=bc_config.get("emb_dim", 256),
                image_keys=tuple(bc_config.get("image_keys", ("agentview_image",))),
                lowdim_keys=tuple(
                    bc_config.get("lowdim_keys", ("robot0_eef_pos", "robot0_gripper_qpos"))
                ),
                crop_shape=bc_config.get("crop_shape", None),
            )
            target_critic_encoder_def = create_encoder(
                encoder_type=bc_config.get("encoder_type", "mlp"),
                hidden_dims=tuple(bc_config.get("encoder_hidden_dims", (256, 256))),
                output_dim=bc_config.get("emb_dim", 256),
                image_keys=tuple(bc_config.get("image_keys", ("agentview_image",))),
                lowdim_keys=tuple(
                    bc_config.get("lowdim_keys", ("robot0_eef_pos", "robot0_gripper_qpos"))
                ),
                crop_shape=bc_config.get("crop_shape", None),
            )

        critic_def = Value(encoder=critic_encoder_def, **critic_kwargs)
        target_critic_def = Value(encoder=target_critic_encoder_def, **critic_kwargs)

        # Build ModuleDict
        batch_size = get_batch_size(ex_observations)
        ex_times = jnp.zeros((batch_size,))
        ex_actions_seq = jnp.zeros((batch_size, horizon, action_dim))
        act_exec_steps = config.get("act_exec_steps", bc_config.get("act_steps", 8))
        ex_critic_act = jnp.zeros((batch_size, act_exec_steps * action_dim))

        # Critic input: raw obs if it has encoder, else encoded cond
        ex_critic_obs = ex_observations if critic_has_encoder else ex_cond

        networks = {
            "encoder": encoder_def,
            "flow": flow_def,
            "critic": critic_def,
            "target_critic": target_critic_def,
        }
        network_args = {
            "encoder": (ex_observations,),
            "flow": (ex_actions_seq, ex_times, ex_times, ex_cond),
            "critic": (ex_critic_obs, ex_critic_act),
            "target_critic": (ex_critic_obs, ex_critic_act),
        }

        network_def = ModuleDict(networks)
        rng, crop_md_rng = jax.random.split(rng)
        all_variables = network_def.init(
            {"params": init_rng, "crop": crop_md_rng}, **network_args
        )
        network_params = flax.core.unfreeze(all_variables["params"])

        if bc_params is not None:
            for key in network_params:
                if key in bc_params:
                    network_params[key] = bc_params[key]

        network_params["modules_target_critic"] = jax.tree_util.tree_map(
            lambda x: x.copy(), network_params["modules_critic"]
        )
        network_params = flax.core.freeze(network_params)

        # Optimizer
        actor_lr = config.get("actor_lr", 1e-4)
        critic_lr = config.get("critic_lr", 3e-4)
        grad_clip = config.get("grad_clip_norm", 10.0)
        freeze_encoder = config.get("freeze_encoder", True)

        param_labels = make_offline_rl_param_labels(
            network_params, freeze_encoder=freeze_encoder
        )
        tx = optax.multi_transform(
            {
                "frozen": optax.set_to_zero(),
                "actor": create_optimizer(lr=actor_lr, grad_clip_norm=grad_clip),
                "critic": create_optimizer(lr=critic_lr, grad_clip_norm=grad_clip),
            },
            param_labels,
        )

        batch_stats = (
            flax.core.unfreeze(all_variables["batch_stats"])
            if "batch_stats" in all_variables
            else None
        )
        extra_vars = {"batch_stats": batch_stats} if batch_stats is not None else None

        network = TrainState.create(
            network_def, network_params, tx=tx, extra_variables=extra_vars
        )

        interpolant = Interpolant(bc_config.get("interp_type", "linear"))
        loss_fn_type = get_loss_fn(bc_config.get("flow_type", "flow_matching"))

        return cls(
            rng=rng,
            network=network,
            config=dict(config),
            interpolant=interpolant,
            loss_fn_type=loss_fn_type,
        )

    # ================================================================
    # Loss functions (grad_params pattern)
    # ================================================================

    def critic_loss(self, batch, grad_params, rng):
        """Critic TD loss. Only critic uses grad_params.

        Gradient routing:
          - encoder: frozen (uses self.network stored params)
          - flow/sampler: frozen (uses self.network stored params)
          - target_critic: frozen (uses self.network stored params)
          - critic: params=grad_params (receives gradients, including critic encoder if present)
        """
        config = self.config
        discount = config.get("discount", 0.99)
        act_exec_steps = config.get("act_exec_steps", 8)
        critic_has_encoder = config.get("critic_has_encoder", False)

        rng, sample_rng = jax.random.split(rng)

        # Sample next actions from frozen flow policy (always needs actor encoder)
        sampler_type = config.get("sampler_type", "euler")
        num_steps = config.get("flow_steps", 10)
        sampler = get_sampler(sampler_type)

        def flow_net_frozen(at, s, t, cond_v, training=False):
            return self.network(at, s, t, cond_v, training=training, name="flow")

        def encode_frozen(obs, training=False, rngs=None):
            if rngs is None:
                rngs = {}
            return self.network(obs, training=training, name="encoder", rngs=rngs)

        next_action_chunk = sampler(
            network=flow_net_frozen,
            encoder=encode_frozen,
            observations=batch["next_observations"],
            num_steps=num_steps,
            rng=sample_rng,
            config=config,
        )
        next_action_chunk = jnp.clip(next_action_chunk, -1, 1)
        next_action_exec = next_action_chunk[:, :act_exec_steps, :]
        next_action_flat = next_action_exec.reshape(next_action_exec.shape[0], -1)

        # Determine critic observation input
        if critic_has_encoder:
            # Critic has its own encoder — pass raw observations
            critic_obs = batch["observations"]
            critic_next_obs = batch["next_observations"]
        else:
            # Encode obs with shared actor encoder — frozen (no grad_params)
            critic_obs = self.network(
                batch["observations"], training=False, name="encoder", rngs={},
            )
            critic_next_obs = self.network(
                batch["next_observations"], training=False, name="encoder", rngs={},
            )

        # Target Q — frozen (no grad_params)
        next_qs = self.network(critic_next_obs, next_action_flat, name="target_critic")
        next_q = jnp.min(next_qs, axis=0)
        discount_factor = discount ** act_exec_steps
        target_q = batch["rewards"] + discount_factor * batch["masks"] * next_q

        # Current Q — WITH grad_params (only gradient path)
        action_exec = batch["actions"][:, :act_exec_steps, :]
        action_flat = action_exec.reshape(action_exec.shape[0], -1)
        qs = self.network(
            critic_obs, action_flat, name="critic", params=grad_params,
            training=critic_has_encoder,
        )
        critic_loss = jnp.mean((qs - target_q[None, :]) ** 2)

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": jnp.mean(qs),
            "q_max": jnp.max(qs),
            "q_min": jnp.min(qs),
            "target_q_mean": jnp.mean(target_q),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Actor loss = alpha * bc_loss + q_weight * q_loss.

        Gradient routing:
          - encoder (BC path): grad_params if not freeze_encoder, else frozen
          - flow (BC path): params=grad_params (receives gradients)
          - encoder (Q path): frozen (no grad_params)
          - flow (Q path): params=grad_params (differentiable sampling)
          - critic (Q path): frozen (evaluator, no grad_params)
        """
        config = self.config
        alpha = config.get("alpha", 1.0)
        q_weight = config.get("q_weight", 1.0)
        normalize_q = config.get("normalize_q_loss", True)
        freeze_encoder = config.get("freeze_encoder", True)
        act_exec_steps = config.get("act_exec_steps", 8)

        rng, bc_rng, sample_rng, dropout_rng, crop_rng = jax.random.split(rng, 5)
        dropout_rngs = {"dropout": dropout_rng, "crop": crop_rng}

        # --- BC loss ---
        def encode_for_bc(obs, training=True, rngs=None):
            if rngs is None:
                rngs = dropout_rngs
            if freeze_encoder:
                return self.network(
                    obs, training=training, name="encoder", rngs=rngs,
                )
            else:
                return self.network(
                    obs, training=training, name="encoder",
                    params=grad_params, rngs=rngs,
                )

        def flow_net_for_bc(at, s, t, cond, training=True):
            return self.network(
                at, s, t, cond, training=training, name="flow",
                params=grad_params, rngs=dropout_rngs,
            )

        bc_loss, bc_info = self.loss_fn_type(
            network=flow_net_for_bc,
            encoder=encode_for_bc,
            interpolant=self.interpolant,
            batch=batch,
            rng=bc_rng,
            config=config,
            step=self.network.step,
        )

        # --- Q loss ---
        sampler_type = config.get("sampler_type", "euler")
        num_steps = config.get("flow_steps", 10)
        sampler = get_sampler(sampler_type)

        # Encoder frozen for Q path
        def encode_for_q(obs, training=False, rngs=None):
            if rngs is None:
                rngs = {}
            return self.network(obs, training=training, name="encoder", rngs=rngs)

        # Flow with grad_params for differentiable action sampling
        def flow_net_for_q(at, s, t, cond_v, training=False):
            return self.network(
                at, s, t, cond_v, training=training, name="flow",
                params=grad_params,
            )

        action_chunk = sampler(
            network=flow_net_for_q,
            encoder=encode_for_q,
            observations=batch["observations"],
            num_steps=num_steps,
            rng=sample_rng,
            config=config,
        )
        action_chunk = jnp.clip(action_chunk, -1, 1)
        action_exec = action_chunk[:, :act_exec_steps, :]
        action_flat = action_exec.reshape(action_exec.shape[0], -1)

        # Critic as frozen evaluator (no grad_params)
        critic_has_encoder = config.get("critic_has_encoder", False)
        if critic_has_encoder:
            critic_obs_for_q = batch["observations"]
        else:
            critic_obs_for_q = self.network(
                batch["observations"], training=False, name="encoder", rngs={},
            )
        qs_actor = self.network(critic_obs_for_q, action_flat, name="critic")
        q_actor = jnp.mean(qs_actor, axis=0)
        q_loss_raw = -jnp.mean(q_actor)

        if normalize_q:
            lam = jax.lax.stop_gradient(
                1.0 / jnp.maximum(jnp.mean(jnp.abs(q_actor)), 1e-8)
            )
            q_loss = lam * q_loss_raw
        else:
            q_loss = q_loss_raw

        actor_loss = alpha * bc_loss + q_weight * q_loss

        return actor_loss, {
            "actor_loss": actor_loss,
            "bc_loss": bc_loss,
            "q_loss": q_loss,
            "q_loss_raw": q_loss_raw,
            "q_actor_mean": jnp.mean(q_actor),
            **{f"bc/{k}": v for k, v in bc_info.items() if k != "loss"},
        }

    def total_loss(self, batch, grad_params, rng):
        """Combined critic + actor loss."""
        rng, critic_rng, actor_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)

        loss = critic_loss + actor_loss

        info = {"loss": loss, **critic_info, **actor_info}
        return loss, info

    # ================================================================
    # Target update
    # ================================================================

    def target_update(self, network):
        """Polyak update: target_critic <- tau * critic + (1-tau) * target_critic.

        Returns new TrainState with updated target params.
        """
        tau = self.config.get("tau", 0.005)
        new_params = flax.core.unfreeze(network.params)
        new_params["modules_target_critic"] = jax.tree_util.tree_map(
            lambda p, tp: tau * p + (1 - tau) * tp,
            new_params["modules_critic"],
            new_params["modules_target_critic"],
        )
        return network.replace(params=flax.core.freeze(new_params))

    # ================================================================
    # Update
    # ================================================================

    @staticmethod
    def _update(agent, batch):
        """Static update method (compatible with jax.lax.scan)."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = agent.target_update(new_network)

        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        """JIT-compiled update."""
        return self._update(self, batch)

    # ================================================================
    # Inference
    # ================================================================

    @jax.jit
    def sample_actions(self, observations, rng=None):
        """Sample actions using current params (no EMA)."""
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
        """Deterministic action sampling for evaluation."""
        return self.sample_actions(observations, rng=jax.random.PRNGKey(0))
