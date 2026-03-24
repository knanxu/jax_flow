"""Offline RL Agent: DDPG + BC constraint on flow policy.

Loads a pretrained BC checkpoint (frozen encoder + flow network),
adds a critic, and jointly optimizes flow policy with Q loss + BC loss.
Single ModuleDict TrainState with optax.multi_transform for per-module LR routing.
"""

import logging
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from jax_flow.agents.bc_agent import _create_flow_def, _ema_update
from jax_flow.agents.train_state import ModuleDict, TrainState, nonpytree_field
from jax_flow.core.checkpoint import load_checkpoint
from jax_flow.core.utils import get_batch_size, make_offline_rl_param_labels
from jax_flow.flow.interpolant import Interpolant
from jax_flow.flow.losses import get_loss_fn
from jax_flow.flow.samplers import get_sampler
from jax_flow.networks.value import Value

logger = logging.getLogger(__name__)


class OfflineRLAgent(flax.struct.PyTreeNode):
    """Offline RL agent: DDPG + BC constraint on flow policy.

    Single TrainState with ModuleDict containing:
    - encoder (frozen): observation encoder from BC
    - flow (actor_lr): flow policy from BC, fine-tuned with Q + BC loss
    - critic (critic_lr): new Q-function ensemble
    - target_critic (frozen): EMA target for critic
    """

    rng: Any
    network: TrainState
    ema_params: Any  # EMA of flow params for eval
    config: Any = nonpytree_field()
    interpolant: Any = nonpytree_field()
    loss_fn_type: Any = nonpytree_field()

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, bc_checkpoint_path=None):
        """Create OfflineRLAgent.

        Uses bc_checkpoint_path to obtain network architecture config.
        If load_bc_weights=True (default False), also loads pretrained params.

        Args:
            seed: Random seed.
            ex_observations: Example observations for init.
            ex_actions: Example actions (batch, horizon, action_dim).
            config: Algorithm config dict.
            bc_checkpoint_path: Path to BC checkpoint (for network config).
                If None, uses config directly for network architecture.

        Returns:
            New OfflineRLAgent instance.
        """
        from jax_flow.networks.encoders import create_encoder

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        # Load BC checkpoint for network config (and optionally weights)
        load_bc_weights = config.get("load_bc_weights", False)
        bc_params = None
        if bc_checkpoint_path is not None:
            bc_ckpt = load_checkpoint(bc_checkpoint_path)
            bc_config = bc_ckpt["config"]
            if load_bc_weights:
                bc_params = bc_ckpt["params"]
        else:
            # No BC checkpoint — use config directly as network config
            bc_config = config

        action_dim = ex_actions.shape[-1]
        horizon = config.get("horizon", bc_config.get("horizon", 16))

        # --- Rebuild encoder def (same as BC) ---
        encoder_def = create_encoder(
            encoder_type=bc_config.get("encoder_type", "mlp"),
            hidden_dims=tuple(bc_config.get("encoder_hidden_dims", (256, 256))),
            output_dim=bc_config.get("emb_dim", 256),
            image_keys=tuple(bc_config.get("image_keys", ("agentview_image",))),
            lowdim_keys=tuple(
                bc_config.get(
                    "lowdim_keys", ("robot0_eef_pos", "robot0_gripper_qpos")
                )
            ),
            crop_shape=bc_config.get("crop_shape", None),
        )

        # --- Rebuild flow def (same as BC) ---
        # Init encoder to get actual cond_dim
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

        # --- Create critic and target_critic ---
        critic_hidden_dims = tuple(
            config.get("critic_hidden_dims", [2048, 2048, 2048])
        )
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

        # --- Build ModuleDict ---
        batch_size = get_batch_size(ex_observations)
        ex_times = jnp.zeros((batch_size,))
        ex_actions_seq = jnp.zeros((batch_size, horizon, action_dim))

        # Critic input: cond + flattened executed actions
        act_exec_steps = config.get(
            "act_exec_steps", bc_config.get("act_steps", 8)
        )
        ex_critic_obs = ex_cond  # (batch, cond_dim)
        ex_critic_act = jnp.zeros(
            (batch_size, act_exec_steps * action_dim)
        )  # flattened

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

        # --- Optionally load pretrained encoder/flow params from BC checkpoint ---
        if bc_params is not None:
            for key in network_params:
                if key in bc_params:
                    network_params[key] = bc_params[key]

        # --- Copy critic params to target_critic ---
        network_params["modules_target_critic"] = jax.tree_util.tree_map(
            lambda x: x.copy(), network_params["modules_critic"]
        )

        network_params = flax.core.freeze(network_params)

        # --- Build multi_transform optimizer ---
        actor_lr = config.get("actor_lr", 1e-4)
        critic_lr = config.get("critic_lr", 3e-4)
        freeze_encoder = config.get("freeze_encoder", True)

        param_labels = make_offline_rl_param_labels(
            network_params, freeze_encoder=freeze_encoder
        )
        tx = optax.multi_transform(
            {
                "frozen": optax.set_to_zero(),
                "actor": optax.adam(learning_rate=actor_lr),
                "critic": optax.adam(learning_rate=critic_lr),
            },
            param_labels,
        )

        # Handle batch_stats if present
        batch_stats = (
            flax.core.unfreeze(all_variables["batch_stats"])
            if "batch_stats" in all_variables
            else None
        )
        extra_vars = (
            {"batch_stats": batch_stats} if batch_stats is not None else None
        )

        network = TrainState.create(
            network_def, network_params, tx=tx, extra_variables=extra_vars
        )

        # --- Interpolant and loss ---
        interpolant = Interpolant(bc_config.get("interp_type", "linear"))
        loss_fn_type = get_loss_fn(bc_config.get("flow_type", "flow_matching"))

        # EMA params for flow (init from BC)
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
    def update(self, batch, q_weight=1.0):
        """Update agent with a batch from sample_sequence.

        Args:
            batch: Dict with keys:
                observations: (B, obs_steps, obs_dim)
                actions: (B, horizon, action_dim) — full chunk for BC loss
                rewards: (B,) — cumulative discounted reward
                next_observations: (B, obs_steps, obs_dim)
                masks: (B,) — 0 if terminal, 1 otherwise
            q_weight: Q loss weight. 0.0 = pure BC, 1.0 = full offline RL.
                      Static Python float — JIT recompiles per distinct value.

        Returns:
            (new_agent, info_dict)
        """
        new_rng, rng, dropout_rng, crop_rng, sample_rng = jax.random.split(
            self.rng, 5
        )
        config = self.config
        tau = config.get("tau", 0.005)
        alpha = config.get("alpha", 1.0)
        discount = config.get("discount", 0.99)
        act_exec_steps = config.get("act_exec_steps", 8)
        normalize_q = config.get("normalize_q_loss", True)

        def total_loss_fn(params):
            dropout_rngs = {"dropout": dropout_rng, "crop": crop_rng}

            # --- Helper closures ---
            def encode(obs, training=False, rngs=None, p=None):
                """Encode obs. Always uses frozen encoder (no grad)."""
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

            def critic_fn(obs_cond, action_flat, p=None):
                if p is None:
                    p = params
                return self.network(
                    obs_cond, action_flat, name="critic", params=p
                )

            def target_critic_fn(obs_cond, action_flat):
                return self.network(
                    obs_cond, action_flat, name="target_critic"
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

            # Next action from current flow policy (no grad through flow for critic)
            sampler_type = config.get("sampler_type", "euler")
            num_steps = config.get("flow_steps", 10)
            sampler = get_sampler(sampler_type)

            def encode_no_grad(obs, training=False, rngs=None):
                if rngs is None:
                    rngs = {}
                return self.network(
                    obs, training=training, name="encoder", rngs=rngs
                )

            def flow_net_no_grad(at, s, t, cond_v, training=False):
                return self.network(
                    at, s, t, cond_v, training=training, name="flow"
                )

            next_action_chunk = sampler(
                network=flow_net_no_grad,
                encoder=encode_no_grad,
                observations=batch["next_observations"],
                num_steps=num_steps,
                rng=sample_rng,
                config=config,
            )  # (B, horizon, action_dim)
            next_action_chunk = jnp.clip(next_action_chunk, -1, 1)
            next_action_exec = next_action_chunk[:, :act_exec_steps, :]
            next_action_flat = next_action_exec.reshape(
                next_action_exec.shape[0], -1
            )

            # Target Q
            next_qs = target_critic_fn(
                next_cond, next_action_flat
            )  # (num_ensembles, B)
            next_q = jnp.min(next_qs, axis=0)  # double Q: (B,)
            discount_factor = discount ** act_exec_steps
            target_q = jax.lax.stop_gradient(
                batch["rewards"] + discount_factor * batch["masks"] * next_q
            )

            # Current Q
            action_exec = batch["actions"][:, :act_exec_steps, :]
            action_flat = action_exec.reshape(action_exec.shape[0], -1)
            qs = critic_fn(cond, action_flat, p=params)  # (num_ensembles, B)
            critic_loss = jnp.mean((qs - target_q[None, :]) ** 2)

            # ============================================================
            # Actor loss (BC + Q)
            # ============================================================
            # BC loss — uses full action chunk
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

            # Q loss — uses executed action from differentiable sampler
            # We need gradients through the flow network for actor update
            def flow_net_diff(at, s, t, cond_v, training=False):
                return flow_net(at, s, t, cond_v, training=training, p=params)

            def encode_diff(obs, training=False, rngs=None):
                if rngs is None:
                    rngs = {}
                # Encoder is frozen, stop_gradient
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
            action_flat_diff = action_exec_diff.reshape(
                action_exec_diff.shape[0], -1
            )

            # Q value (no grad through critic)
            qs_actor = self.network(
                cond, action_flat_diff, name="critic"
            )  # (num_ensembles, B)
            q_actor = jnp.mean(qs_actor, axis=0)  # (B,)
            q_loss = -jnp.mean(q_actor)

            # Normalize Q loss (MeanFlowQL style)
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
                **{
                    f"bc/{k}": v
                    for k, v in bc_info.items()
                    if k != "loss"
                },
            }
            return total_loss, info

        new_network, info = self.network.apply_loss_fn(loss_fn=total_loss_fn)

        # --- Target critic soft update ---
        new_params = flax.core.unfreeze(new_network.params)
        new_params["modules_target_critic"] = jax.tree_util.tree_map(
            lambda p, tp: tau * p + (1 - tau) * tp,
            new_params["modules_critic"],
            new_params["modules_target_critic"],
        )
        new_network = new_network.replace(
            params=flax.core.freeze(new_params)
        )

        # --- EMA update for flow params (for eval) ---
        ema_decay = self.config.get("ema_decay", 0.995)
        new_ema_params = _ema_update(
            self.ema_params, new_network.params, ema_decay
        )

        return self.replace(
            network=new_network, ema_params=new_ema_params, rng=new_rng
        ), info

    @jax.jit
    def sample_actions(self, observations, rng=None, use_ema=True):
        """Sample actions from the flow policy.

        Args:
            observations: (batch, obs_steps, obs_dim) or dict.
            rng: Optional random key.
            use_ema: Whether to use EMA params.

        Returns:
            Actions (batch, horizon, action_dim).
        """
        if rng is None:
            rng = self.rng

        sampler_type = self.config.get("sampler_type", "euler")
        num_steps = self.config.get("flow_steps", 10)
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
        """Deterministic action sampling for evaluation."""
        eval_rng = jax.random.PRNGKey(0)
        return self.sample_actions(observations, rng=eval_rng)
