"""Offline RL Agent: DDPG + BC constraint on flow policy.

Two-phase training:
  Phase 1 (BC warmup): BCAgent trains actor, CriticState trains critic in parallel.
  Phase 2 (RL): OfflineRLAgent jointly optimizes actor (Q+BC loss) and critic.

OfflineRLAgent is created from BCAgent params + pretrained critic after BC phase.
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
from jax_flow.core.utils import create_optimizer, get_batch_size, make_offline_rl_param_labels
from jax_flow.flow.interpolant import Interpolant
from jax_flow.flow.losses import get_loss_fn
from jax_flow.flow.samplers import get_sampler
from jax_flow.networks.value import Value

logger = logging.getLogger(__name__)


# ============================================================
# CriticState: standalone critic for BC warmup phase
# ============================================================

class CriticState(flax.struct.PyTreeNode):
    """Standalone critic + target_critic for BC warmup phase.

    Updated each step alongside BCAgent. Uses BCAgent's encoder (frozen)
    to encode observations, and BCAgent's flow policy to sample next_actions.
    """

    rng: Any
    network: TrainState  # ModuleDict with critic + target_critic
    config: Any = nonpytree_field()

    @classmethod
    def create(cls, seed, ex_cond, ex_actions, config):
        """Create CriticState.

        Args:
            seed: Random seed.
            ex_cond: Example encoded observations (B, cond_dim).
            ex_actions: Example actions (B, horizon, action_dim).
            config: Config dict with critic_hidden_dims, critic_lr, etc.

        Returns:
            New CriticState.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        act_exec_steps = config.get("act_exec_steps", 8)
        action_dim = ex_actions.shape[-1]
        batch_size = ex_cond.shape[0]

        critic_hidden_dims = tuple(config.get("critic_hidden_dims", [2048, 2048, 2048]))
        critic_def = Value(
            hidden_dims=critic_hidden_dims,
            layer_norm=config.get("critic_layer_norm", True),
            num_ensembles=config.get("num_ensembles", 2),
            activation=config.get("critic_activation", "tanh"),
        )
        target_critic_def = Value(
            hidden_dims=critic_hidden_dims,
            layer_norm=config.get("critic_layer_norm", True),
            num_ensembles=config.get("num_ensembles", 2),
            activation=config.get("critic_activation", "tanh"),
        )

        ex_critic_act = jnp.zeros((batch_size, act_exec_steps * action_dim))
        networks = {"critic": critic_def, "target_critic": target_critic_def}
        network_args = {
            "critic": (ex_cond, ex_critic_act),
            "target_critic": (ex_cond, ex_critic_act),
        }

        network_def = ModuleDict(networks)
        all_variables = network_def.init({"params": init_rng}, **network_args)
        params = flax.core.unfreeze(all_variables["params"])

        # target_critic = copy of critic
        params["modules_target_critic"] = jax.tree_util.tree_map(
            lambda x: x.copy(), params["modules_critic"]
        )
        params = flax.core.freeze(params)

        # Optimizer: critic updates, target_critic frozen (Polyak only)
        critic_lr = config.get("critic_lr", 3e-4)
        grad_clip = config.get("grad_clip_norm", 0.0)

        def label_fn(path, _):
            key_str = "/".join(str(getattr(k, "key", "")) for k in path)
            if "modules_target_critic" in key_str:
                return "frozen"
            return "critic"

        param_labels = jax.tree_util.tree_map_with_path(label_fn, params)
        tx = optax.multi_transform(
            {
                "frozen": optax.set_to_zero(),
                "critic": create_optimizer(lr=critic_lr, grad_clip_norm=grad_clip),
            },
            param_labels,
        )

        network = TrainState.create(network_def, params, tx=tx)

        return cls(rng=rng, network=network, config=dict(config))

    @jax.jit
    def update(self, batch, bc_agent):
        """Update critic with TD loss, using BCAgent as frozen actor.

        Args:
            batch: Dict with observations, actions, rewards, next_observations, masks.
            bc_agent: BCAgent instance (used for encoding obs and sampling next_actions).

        Returns:
            (new_critic_state, info_dict)
        """
        new_rng, rng, sample_rng = jax.random.split(self.rng, 3)
        config = self.config
        tau = config.get("tau", 0.005)
        discount = config.get("discount", 0.99)
        act_exec_steps = config.get("act_exec_steps", 8)

        def critic_loss_fn(params):
            # Encode obs with BCAgent's encoder (no grad)
            cond = jax.lax.stop_gradient(
                bc_agent.network(
                    batch["observations"], training=False, name="encoder",
                    params=bc_agent.ema_params, rngs={},
                )
            )
            next_cond = jax.lax.stop_gradient(
                bc_agent.network(
                    batch["next_observations"], training=False, name="encoder",
                    params=bc_agent.ema_params, rngs={},
                )
            )

            # Sample next_actions from BCAgent's flow policy (no grad)
            next_action_chunk = jax.lax.stop_gradient(
                bc_agent.sample_actions(batch["next_observations"], rng=sample_rng)
            )
            next_action_chunk = jnp.clip(next_action_chunk, -1, 1)
            next_action_exec = next_action_chunk[:, :act_exec_steps, :]
            next_action_flat = next_action_exec.reshape(next_action_exec.shape[0], -1)

            # Target Q
            next_qs = self.network(
                next_cond, next_action_flat, name="target_critic"
            )
            next_q = jnp.min(next_qs, axis=0)
            discount_factor = discount ** act_exec_steps
            target_q = jax.lax.stop_gradient(
                batch["rewards"] + discount_factor * batch["masks"] * next_q
            )

            # Current Q
            action_exec = batch["actions"][:, :act_exec_steps, :]
            action_flat = action_exec.reshape(action_exec.shape[0], -1)
            qs = self.network(
                cond, action_flat, name="critic", params=params
            )
            critic_loss = jnp.mean((qs - target_q[None, :]) ** 2)

            return critic_loss, {
                "critic_loss": critic_loss,
                "q_mean": jnp.mean(qs),
                "q_max": jnp.max(qs),
                "q_min": jnp.min(qs),
                "target_q_mean": jnp.mean(target_q),
            }

        new_network, info = self.network.apply_loss_fn(loss_fn=critic_loss_fn)

        # Target critic Polyak update
        new_params = flax.core.unfreeze(new_network.params)
        new_params["modules_target_critic"] = jax.tree_util.tree_map(
            lambda p, tp: tau * p + (1 - tau) * tp,
            new_params["modules_critic"],
            new_params["modules_target_critic"],
        )
        new_network = new_network.replace(params=flax.core.freeze(new_params))

        return self.replace(network=new_network, rng=new_rng), info


# ============================================================
# OfflineRLAgent: RL phase only
# ============================================================

class OfflineRLAgent(flax.struct.PyTreeNode):
    """Offline RL agent for the RL phase.

    Created from BCAgent params + CriticState params after BC warmup.
    Single ModuleDict TrainState: encoder + flow + critic + target_critic.
    No EMA — eval uses current params directly.
    """

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()
    interpolant: Any = nonpytree_field()
    loss_fn_type: Any = nonpytree_field()

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
                "actor": create_optimizer(
                    lr=actor_lr,
                    grad_clip_norm=grad_clip,
                ),
                "critic": create_optimizer(
                    lr=critic_lr,
                    grad_clip_norm=grad_clip,
                ),
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

        # Critic defs
        critic_hidden_dims = tuple(config.get("critic_hidden_dims", [2048, 2048, 2048]))
        critic_def = Value(
            hidden_dims=critic_hidden_dims,
            layer_norm=config.get("critic_layer_norm", True),
            num_ensembles=config.get("num_ensembles", 2),
            activation=config.get("critic_activation", "tanh"),
        )
        target_critic_def = Value(
            hidden_dims=critic_hidden_dims,
            layer_norm=config.get("critic_layer_norm", True),
            num_ensembles=config.get("num_ensembles", 2),
            activation=config.get("critic_activation", "tanh"),
        )

        # Build ModuleDict
        batch_size = get_batch_size(ex_observations)
        ex_times = jnp.zeros((batch_size,))
        ex_actions_seq = jnp.zeros((batch_size, horizon, action_dim))
        act_exec_steps = config.get("act_exec_steps", bc_config.get("act_steps", 8))
        ex_critic_act = jnp.zeros((batch_size, act_exec_steps * action_dim))

        networks = {
            "encoder": encoder_def,
            "flow": flow_def,
            "critic": critic_def,
            "target_critic": target_critic_def,
        }
        network_args = {
            "encoder": (ex_observations,),
            "flow": (ex_actions_seq, ex_times, ex_times, ex_cond),
            "critic": (ex_cond, ex_critic_act),
            "target_critic": (ex_cond, ex_critic_act),
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

    @jax.jit
    def update(self, batch):
        """RL phase update: actor (Q+BC loss) + critic (TD loss) + target Polyak.

        Args:
            batch: Dict with observations, actions, rewards, next_observations, masks.

        Returns:
            (new_agent, info_dict)
        """
        new_rng, rng, dropout_rng, crop_rng, sample_rng = jax.random.split(
            self.rng, 5
        )
        config = self.config
        tau = config.get("tau", 0.005)
        alpha = config.get("alpha", 1.0)
        q_weight = config.get("q_weight", 1.0)
        discount = config.get("discount", 0.99)
        act_exec_steps = config.get("act_exec_steps", 8)
        normalize_q = config.get("normalize_q_loss", True)

        def total_loss_fn(params):
            dropout_rngs = {"dropout": dropout_rng, "crop": crop_rng}

            def encode(obs, training=False, rngs=None, p=None):
                if rngs is None:
                    rngs = dropout_rngs
                if p is None:
                    p = params
                return self.network(
                    obs, training=training, name="encoder", params=p, rngs=rngs
                )

            def flow_net(at, s, t, cond, training=True, p=None):
                if p is None:
                    p = params
                return self.network(
                    at, s, t, cond, training=training, name="flow",
                    params=p, rngs=dropout_rngs,
                )

            # ============================================================
            # Critic loss
            # ============================================================
            cond = jax.lax.stop_gradient(
                encode(batch["observations"], training=False)
            )
            next_cond = jax.lax.stop_gradient(
                encode(batch["next_observations"], training=False)
            )

            # Next action from current flow policy (no grad for critic)
            sampler_type = config.get("sampler_type", "euler")
            num_steps = config.get("flow_steps", 10)
            sampler = get_sampler(sampler_type)

            def encode_no_grad(obs, training=False, rngs=None):
                if rngs is None:
                    rngs = {}
                return self.network(obs, training=training, name="encoder", rngs=rngs)

            def flow_net_no_grad(at, s, t, cond_v, training=False):
                return self.network(at, s, t, cond_v, training=training, name="flow")

            next_action_chunk = sampler(
                network=flow_net_no_grad,
                encoder=encode_no_grad,
                observations=batch["next_observations"],
                num_steps=num_steps,
                rng=sample_rng,
                config=config,
            )
            next_action_chunk = jnp.clip(next_action_chunk, -1, 1)
            next_action_exec = next_action_chunk[:, :act_exec_steps, :]
            next_action_flat = next_action_exec.reshape(next_action_exec.shape[0], -1)

            # Target Q
            next_qs = self.network(
                next_cond, next_action_flat, name="target_critic"
            )
            next_q = jnp.min(next_qs, axis=0)
            discount_factor = discount ** act_exec_steps
            target_q = jax.lax.stop_gradient(
                batch["rewards"] + discount_factor * batch["masks"] * next_q
            )

            # Current Q
            action_exec = batch["actions"][:, :act_exec_steps, :]
            action_flat = action_exec.reshape(action_exec.shape[0], -1)
            qs = self.network(
                cond, action_flat, name="critic", params=params
            )
            critic_loss = jnp.mean((qs - target_q[None, :]) ** 2)

            # ============================================================
            # Actor loss (BC + Q)
            # ============================================================
            def flow_net_with_grad(at, s, t, cond_v, training=True):
                return flow_net(at, s, t, cond_v, training=training, p=params)

            def encode_with_grad(obs, training=True, rngs=None):
                return encode(obs, training=training, rngs=rngs, p=params)

            bc_loss, bc_info = self.loss_fn_type(
                network=flow_net_with_grad,
                encoder=encode_with_grad,
                interpolant=self.interpolant,
                batch=batch,
                rng=rng,
                config=config,
                step=self.network.step,
            )

            # Q loss through differentiable sampler
            def flow_net_diff(at, s, t, cond_v, training=False):
                return flow_net(at, s, t, cond_v, training=training, p=params)

            def encode_diff(obs, training=False, rngs=None):
                if rngs is None:
                    rngs = {}
                return jax.lax.stop_gradient(
                    encode(obs, training=training, rngs=rngs, p=params)
                )

            action_chunk_diff = sampler(
                network=flow_net_diff,
                encoder=encode_diff,
                observations=batch["observations"],
                num_steps=num_steps,
                rng=sample_rng,
                config=config,
            )
            action_chunk_diff = jnp.clip(action_chunk_diff, -1, 1)
            action_exec_diff = action_chunk_diff[:, :act_exec_steps, :]
            action_flat_diff = action_exec_diff.reshape(action_exec_diff.shape[0], -1)

            qs_actor = self.network(
                cond, action_flat_diff, name="critic"
            )
            q_actor = jnp.mean(qs_actor, axis=0)
            q_loss = -jnp.mean(q_actor)

            if normalize_q:
                lam = jax.lax.stop_gradient(
                    1.0 / jnp.maximum(jnp.mean(jnp.abs(q_actor)), 1e-8)
                )
                q_loss = lam * q_loss

            actor_loss = q_weight * q_loss + alpha * bc_loss

            # ============================================================
            # Total loss
            # ============================================================
            total_loss = critic_loss + actor_loss

            info = {
                "loss": total_loss,
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "bc_loss": bc_loss,
                "q_loss": q_loss,
                "q_mean": jnp.mean(qs),
                "target_q_mean": jnp.mean(target_q),
                **{f"bc/{k}": v for k, v in bc_info.items() if k != "loss"},
            }
            return total_loss, info

        new_network, info = self.network.apply_loss_fn(loss_fn=total_loss_fn)

        # Target critic Polyak update
        new_params = flax.core.unfreeze(new_network.params)
        new_params["modules_target_critic"] = jax.tree_util.tree_map(
            lambda p, tp: tau * p + (1 - tau) * tp,
            new_params["modules_critic"],
            new_params["modules_target_critic"],
        )
        new_network = new_network.replace(params=flax.core.freeze(new_params))

        return self.replace(network=new_network, rng=new_rng), info

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
