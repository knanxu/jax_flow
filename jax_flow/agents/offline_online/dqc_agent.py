"""DQC: Decoupled Q-Chunking agent.

Offline RL with decoupled critic and policy chunk sizes.
Reference: https://github.com/ColinQiyangLi/dqc

Key design:
- chunk_critic evaluates long action chunks (backup_horizon)
- action_critic evaluates short action chunks (policy_chunk_size)
- value function V(s) for bootstrapping
- Expectile regression for implicit maximization (IQL-style)
- Actor trained with advantage-weighted BC flow loss
- Inference via best-of-N with action_critic
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp

from jax_flow.agents.offline_online.utils import (
    aggregate_q,
    expectile_loss,
    flatten_action_chunk,
)
from jax_flow.agents.train_state import ModuleDict, TrainState, nonpytree_field
from jax_flow.core.utils import get_batch_size
from jax_flow.flow.interpolant import Interpolant
from jax_flow.flow.losses import get_loss_fn
from jax_flow.flow.samplers import get_sampler


class DQCAgent(flax.struct.PyTreeNode):
    """Decoupled Q-Chunking agent.

    Networks (all in single ModuleDict):
    - encoder: observation encoder
    - actor_bc: flow matching policy (generates short chunks)
    - chunk_critic: Q(s, a_{1:H}) for long chunks, ensemble
    - action_critic: Q(s, a_{1:h}) for short chunks, ensemble
    - target_action_critic: Polyak-averaged target
    - value: V(s), single head
    """

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()
    interpolant: Any = nonpytree_field()
    loss_fn_type: Any = nonpytree_field()

    @classmethod
    def create(cls, seed, ex_observations, ex_actions_long, ex_actions_short, config):
        """Create a new DQCAgent.

        Args:
            seed: Random seed.
            ex_observations: Example observations.
            ex_actions_long: Example long action chunks (batch, backup_horizon, action_dim).
            ex_actions_short: Example short action chunks (batch, policy_chunk_size, action_dim).
            config: Configuration dict.

        Returns:
            New DQCAgent instance.
        """
        from jax_flow.core.utils import create_optimizer
        from jax_flow.networks.encoders import create_encoder
        from jax_flow.networks.mlp import MLP
        from jax_flow.networks.transformer import TransformerForFlow
        from jax_flow.networks.unet import ConditionalUnet1D
        from jax_flow.networks.value import Value

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        action_dim = ex_actions_long.shape[-1]
        backup_horizon = config.get("backup_horizon", 16)
        policy_chunk_size = config.get("policy_chunk_size", 4)

        # ============================================================
        # 1. Create encoder
        # ============================================================
        encoder_def = create_encoder(
            encoder_type=config.get("encoder_type", "identity"),
            hidden_dims=tuple(config.get("encoder_hidden_dims", (256, 256))),
            output_dim=config.get("emb_dim", 256),
            image_keys=tuple(config.get("image_keys", ("agentview_image",))),
            lowdim_keys=tuple(
                config.get("lowdim_keys", ("robot0_eef_pos", "robot0_gripper_qpos"))
            ),
            crop_shape=config.get("crop_shape", None),
        )

        rng, enc_rng, crop_init_rng = jax.random.split(rng, 3)
        encoder_params = encoder_def.init(
            {"params": enc_rng, "crop": crop_init_rng}, ex_observations
        )
        ex_cond = encoder_def.apply(encoder_params, ex_observations)
        actual_cond_dim = ex_cond.shape[-1]

        # ============================================================
        # 2. Create actor (flow network for short chunks)
        # ============================================================
        network_type = config.get("network_type", "mlp")
        if network_type == "mlp":
            actor_def = MLP(
                action_dim=action_dim,
                emb_dim=config.get("emb_dim", 512),
                n_blocks=config.get("n_blocks", 6),
                expansion_factor=config.get("expansion_factor", 4),
                dropout=config.get("dropout", 0.1),
                timestep_embed_dim=config.get("timestep_embed_dim", 128),
                max_freq=config.get("max_freq", 100.0),
            )
        elif network_type == "unet":
            actor_def = ConditionalUnet1D(
                action_dim=action_dim,
                cond_dim=actual_cond_dim,
                model_dim=config.get("model_dim", 256),
                emb_dim=config.get("emb_dim", 256),
                kernel_size=config.get("kernel_size", 5),
                n_groups=config.get("n_groups", 8),
                cond_predict_scale=config.get("cond_predict_scale", True),
                dim_mult=tuple(config.get("dim_mult", (1, 2, 2))),
            )
        elif network_type == "transformer":
            actor_def = TransformerForFlow(
                action_dim=action_dim,
                n_layer=config.get("n_layer", 4),
                n_head=config.get("n_head", 4),
                n_emb=config.get("n_emb", 256),
                cond_dim=actual_cond_dim,
                dropout=config.get("dropout", 0.1),
                timestep_embed_dim=config.get("timestep_embed_dim", 128),
            )
        else:
            raise ValueError(f"Unknown network type: {network_type}")

        # ============================================================
        # 3. Create critics and value function
        # ============================================================
        critic_kwargs = dict(
            hidden_dims=tuple(
                config.get("value_hidden_dims", (512, 512, 512, 512))
            ),
            layer_norm=config.get("critic_layer_norm", True),
            num_ensembles=config.get("num_qs", 2),
        )

        chunk_critic_def = Value(**critic_kwargs)
        action_critic_def = Value(**critic_kwargs)
        value_def = Value(
            hidden_dims=tuple(
                config.get("value_hidden_dims", (512, 512, 512, 512))
            ),
            layer_norm=config.get("critic_layer_norm", True),
            num_ensembles=1,
        )

        # ============================================================
        # 4. Build ModuleDict
        # ============================================================
        batch_size = get_batch_size(ex_observations)
        ex_times = jnp.zeros((batch_size,))
        ex_short_seq = jnp.zeros((batch_size, policy_chunk_size, action_dim))
        ex_flat_long = jnp.zeros((batch_size, backup_horizon * action_dim))
        ex_flat_short = jnp.zeros((batch_size, policy_chunk_size * action_dim))

        networks = {
            "encoder": encoder_def,
            "actor_bc": actor_def,
            "chunk_critic": chunk_critic_def,
            "action_critic": action_critic_def,
            "target_action_critic": copy.deepcopy(action_critic_def),
            "value": value_def,
        }
        network_args = {
            "encoder": (ex_observations,),
            "actor_bc": (ex_short_seq, ex_times, ex_times, ex_cond),
            "chunk_critic": (ex_cond, ex_flat_long),
            "action_critic": (ex_cond, ex_flat_short),
            "target_action_critic": (ex_cond, ex_flat_short),
            "value": (ex_cond,),  # V(s), no actions
        }

        network_def = ModuleDict(networks)
        rng, crop_md_rng = jax.random.split(rng)
        network_params = network_def.init(
            {"params": init_rng, "crop": crop_md_rng}, **network_args
        )["params"]

        # Copy action_critic params to target
        network_params = dict(network_params)
        network_params["modules_target_action_critic"] = (
            network_params["modules_action_critic"]
        )
        network_params = flax.core.freeze(network_params)

        # ============================================================
        # 5. Create optimizer and TrainState
        # ============================================================
        tx = create_optimizer(
            lr=config.get("lr", 3e-4),
            weight_decay=config.get("weight_decay", 0.0),
            schedule_type=config.get("schedule_type", "constant"),
            warmup_steps=config.get("warmup_steps", 0),
            total_steps=config.get("gradient_steps", 1000000),
        )

        network = TrainState.create(network_def, network_params, tx=tx)

        interpolant = Interpolant(config.get("interp_type", "linear"))
        loss_fn_type = get_loss_fn(config.get("flow_type", "flow_matching"))

        return cls(
            rng=rng,
            network=network,
            config=dict(config),
            interpolant=interpolant,
            loss_fn_type=loss_fn_type,
        )

    def _target_update(self, network):
        """Polyak average update for target action critic."""
        tau = self.config.get("tau", 0.005)
        new_target = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            network.params["modules_action_critic"],
            network.params["modules_target_action_critic"],
        )
        updated_params = flax.core.unfreeze(network.params)
        updated_params["modules_target_action_critic"] = new_target
        return network.replace(params=flax.core.freeze(updated_params))

    def _sample_flow_actions(self, observations, rng, params=None):
        """Sample short action chunks via flow integration."""
        policy_chunk_size = self.config.get("policy_chunk_size", 4)
        action_dim = self.config["action_dim"]
        sampler_type = self.config.get("sampler_type", "euler")
        num_steps = self.config.get("flow_steps", 10)

        def encode(obs, training=False, rngs=None):
            if rngs is None:
                rngs = {}
            return self.network(obs, training=training, name="encoder",
                                params=params, rngs=rngs)

        def flow_net(at, s, t, cond, training=False):
            return self.network(at, s, t, cond, training=training,
                                name="actor_bc", params=params)

        sampler = get_sampler(sampler_type)
        actions = sampler(
            network=flow_net,
            encoder=encode,
            observations=observations,
            num_steps=num_steps,
            rng=rng,
            config={
                "horizon": policy_chunk_size,
                "action_dim": action_dim,
                "sample_mode": self.config.get("sample_mode", "stochastic"),
                "t_two_step": self.config.get("t_two_step", 0.9),
            },
        )
        return jnp.clip(actions, -1, 1)

    @jax.jit
    def update(self, batch):
        """Update agent with a batch of data.

        Args:
            batch: Dict with keys:
                - observations: first-step obs
                - actions_long: (batch, backup_horizon, action_dim)
                - actions_short: (batch, policy_chunk_size, action_dim)
                - rewards: cumulative discounted reward (batch, 1)
                - next_observations: last-step next_obs
                - masks: 1 - done (batch, 1)
                - valid: sequence validity (batch, 1)

        Returns:
            Tuple of (new_agent, info_dict).
        """
        new_rng, rng, dropout_rng, crop_rng = jax.random.split(self.rng, 4)
        discount = self.config.get("discount", 0.99)
        backup_horizon = self.config.get("backup_horizon", 16)
        kappa_d = self.config.get("kappa_d", 0.7)
        kappa_b = self.config.get("kappa_b", 0.7)
        temperature = self.config.get("temperature", 1.0)
        max_weight = self.config.get("max_weight", 100.0)

        actions_long = batch["actions_long"]
        actions_short = batch["actions_short"]
        valid = batch.get("valid", jnp.ones((actions_long.shape[0], 1)))
        valid_flat = valid.squeeze(-1)

        def loss_fn(params):
            dropout_rngs = {"dropout": dropout_rng, "crop": crop_rng}

            def encode(obs, training=True, rngs=None):
                if rngs is None:
                    rngs = dropout_rngs
                return self.network(
                    obs, training=training, name="encoder",
                    params=params, rngs=rngs,
                )

            def actor_flow_net(at, s, t, cond, training=True):
                return self.network(
                    at, s, t, cond, training=training,
                    name="actor_bc", params=params, rngs=dropout_rngs,
                )

            cond = encode(batch["observations"], training=False, rngs={})
            flat_long = flatten_action_chunk(actions_long)
            flat_short = flatten_action_chunk(actions_short)

            # ========== 1. chunk_critic_loss: MSE TD ==========
            q_chunk = self.network(
                cond, flat_long, name="chunk_critic", params=params
            )  # (num_qs, batch)

            # Bootstrap from V(s')
            next_cond = encode(
                batch["next_observations"], training=False, rngs={}
            )
            v_next = self.network(
                next_cond, name="value"
            )  # (batch,) since num_ensembles=1

            chunk_discount = discount ** backup_horizon
            rewards = batch["rewards"].squeeze(-1)
            masks = batch["masks"].squeeze(-1)
            td_target = rewards + chunk_discount * masks * v_next
            td_target = jax.lax.stop_gradient(td_target)

            chunk_critic_loss = jnp.mean(
                valid_flat[None, :] * (q_chunk - td_target[None, :]) ** 2
            )

            # ========== 2. action_critic_loss: expectile regression ==========
            q_action = self.network(
                cond, flat_short, name="action_critic", params=params
            )
            q_action_agg = aggregate_q(
                q_action, self.config.get("q_agg", "mean")
            )

            # Target from chunk_critic (no grad)
            q_chunk_target = aggregate_q(
                self.network(cond, flat_long, name="chunk_critic"),
                self.config.get("q_agg", "mean"),
            )
            q_chunk_target = jax.lax.stop_gradient(q_chunk_target)

            action_critic_loss = expectile_loss(
                q_action_agg, q_chunk_target, expectile=kappa_d
            )

            # ========== 3. value_loss: expectile regression ==========
            v_pred = self.network(
                cond, name="value", params=params
            )  # (batch,)

            # Target from target_action_critic
            target_q_action = aggregate_q(
                self.network(
                    cond, flat_short, name="target_action_critic"
                ),
                self.config.get("q_agg", "mean"),
            )
            target_q_action = jax.lax.stop_gradient(target_q_action)

            value_loss = expectile_loss(
                v_pred, target_q_action, expectile=kappa_b
            )

            # ========== 4. actor_loss: advantage-weighted BC flow ==========
            # IQL advantage weights
            advantage = jax.lax.stop_gradient(q_chunk_target - v_pred)
            weights = jnp.exp(advantage / temperature)
            weights = jnp.minimum(weights, max_weight)
            weights = jax.lax.stop_gradient(weights)
            # Normalize weights
            weights = weights / jnp.mean(weights)

            actor_batch = {
                "observations": batch["observations"],
                "actions": actions_short,
                "sample_weights": weights,
            }

            # Use actor config with policy_chunk_size as horizon
            actor_config = dict(self.config)
            actor_config["horizon"] = self.config.get("policy_chunk_size", 4)

            actor_loss, actor_info = self.loss_fn_type(
                network=actor_flow_net,
                encoder=encode,
                interpolant=self.interpolant,
                batch=actor_batch,
                rng=rng,
                config=actor_config,
                step=self.network.step,
            )

            # ========== Total loss ==========
            total_loss = (
                chunk_critic_loss + action_critic_loss
                + value_loss + actor_loss
            )

            info = {
                "loss": total_loss,
                "chunk_critic_loss": chunk_critic_loss,
                "action_critic_loss": action_critic_loss,
                "value_loss": value_loss,
                "actor_loss": actor_loss,
                "q_chunk_mean": jnp.mean(q_chunk),
                "q_action_mean": jnp.mean(q_action),
                "v_mean": jnp.mean(v_pred),
                "advantage_mean": jnp.mean(advantage),
                "weight_mean": jnp.mean(weights),
                "weight_max": jnp.max(weights),
            }
            for k, v in actor_info.items():
                if k != "loss":
                    info[f"actor/{k}"] = v

            return total_loss, info

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = self._target_update(new_network)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, rng=None):
        """Sample short action chunks via best-of-N.

        Args:
            observations: (batch, ...) observations.
            rng: Optional PRNG key.

        Returns:
            actions: (batch, policy_chunk_size, action_dim), clipped to [-1, 1].
        """
        if rng is None:
            rng = self.rng

        n_samples = self.config.get("best_of_n", 32)
        policy_chunk_size = self.config.get("policy_chunk_size", 4)
        action_dim = self.config["action_dim"]
        batch_size = get_batch_size(observations)

        cond = self.network(
            observations, training=False, name="encoder", rngs={}
        )

        # Generate N short action chunks
        all_actions = []
        for i in range(n_samples):
            rng_i = jax.random.fold_in(rng, i)
            acts = self._sample_flow_actions(observations, rng_i)
            all_actions.append(acts)

        all_actions = jnp.stack(all_actions, axis=0)  # (N, batch, chunk, dim)

        # Evaluate with action_critic
        cond_exp = jnp.broadcast_to(
            cond[None, :, :], (n_samples, batch_size, cond.shape[-1])
        )
        flat_actions = all_actions.reshape(
            n_samples * batch_size, policy_chunk_size * action_dim
        )
        cond_flat = cond_exp.reshape(n_samples * batch_size, -1)

        q_all = self.network(
            cond_flat, flat_actions, name="action_critic"
        )
        q_agg = aggregate_q(q_all, self.config.get("q_agg", "mean"))
        q_agg = q_agg.reshape(n_samples, batch_size)

        best_idx = jnp.argmax(q_agg, axis=0)
        best_actions = all_actions[best_idx, jnp.arange(batch_size)]

        return best_actions

    @jax.jit
    def eval_actions(self, observations):
        """Deterministic action sampling for evaluation."""
        eval_rng = jax.random.PRNGKey(0)
        return self.sample_actions(observations, rng=eval_rng)
