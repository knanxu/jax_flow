"""ACFQL: Action-Chunking Flow Q-Learning agent.

Offline-to-online RL with flow matching policy and action chunking.
Reference: https://github.com/ColinQiyangLi/qc

Key design:
- Single ModuleDict + single TrainState (same pattern as BCAgent)
- Critic Q(s, a_{1:H}) evaluates flattened action chunks
- Actor uses existing flow losses (flow/mip/mf/imf) and samplers (euler/heun/mip/meanflow)
- Two inference modes: best-of-N and distill-ddpg
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp

from jax_flow.agents.offline_online.utils import (
    aggregate_q,
    flatten_action_chunk,
)
from jax_flow.agents.train_state import ModuleDict, TrainState, nonpytree_field
from jax_flow.core.utils import get_batch_size
from jax_flow.flow.interpolant import Interpolant
from jax_flow.flow.losses import get_loss_fn
from jax_flow.flow.samplers import get_sampler


class ACFQLAgent(flax.struct.PyTreeNode):
    """Action-Chunking Flow Q-Learning agent.

    Manages all networks via a single ModuleDict + TrainState:
    - encoder: observation encoder
    - actor_bc_flow: multi-step flow matching policy
    - actor_onestep_flow: one-step distilled policy (distill-ddpg mode only)
    - critic: Q(s, a_chunk) ensemble
    - target_critic: Polyak-averaged target Q
    """

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()
    interpolant: Any = nonpytree_field()
    loss_fn_type: Any = nonpytree_field()

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new ACFQLAgent.

        Args:
            seed: Random seed.
            ex_observations: Example observations for initialization.
            ex_actions: Example actions (batch, chunk_length, action_dim).
            config: Configuration dict.

        Returns:
            New ACFQLAgent instance.
        """
        from jax_flow.core.utils import create_optimizer
        from jax_flow.networks.encoders import create_encoder
        from jax_flow.networks.mlp import MLP
        from jax_flow.networks.transformer import TransformerForFlow
        from jax_flow.networks.unet import ConditionalUnet1D
        from jax_flow.networks.value import Value

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        action_dim = ex_actions.shape[-1]
        chunk_length = config.get("chunk_length", 5)

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
        # 2. Create flow network (actor_bc_flow)
        # ============================================================
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
                cond_dim=actual_cond_dim,
                model_dim=config.get("model_dim", 256),
                emb_dim=config.get("emb_dim", 256),
                kernel_size=config.get("kernel_size", 5),
                n_groups=config.get("n_groups", 8),
                cond_predict_scale=config.get("cond_predict_scale", True),
                dim_mult=tuple(config.get("dim_mult", (1, 2, 4))),
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
        else:
            raise ValueError(f"Unknown network type: {network_type}")

        # ============================================================
        # 3. Create critic
        # ============================================================
        critic_def = Value(
            hidden_dims=tuple(
                config.get("value_hidden_dims", (512, 512, 512, 512))
            ),
            layer_norm=config.get("critic_layer_norm", True),
            num_ensembles=config.get("num_qs", 2),
        )

        # ============================================================
        # 4. Build ModuleDict
        # ============================================================
        batch_size = get_batch_size(ex_observations)
        ex_times = jnp.zeros((batch_size,))
        ex_actions_seq = jnp.zeros((batch_size, chunk_length, action_dim))
        ex_flat_actions = jnp.zeros((batch_size, chunk_length * action_dim))

        networks = {
            "encoder": encoder_def,
            "actor_bc_flow": flow_def,
            "critic": critic_def,
            "target_critic": copy.deepcopy(critic_def),
        }
        network_args = {
            "encoder": (ex_observations,),
            "actor_bc_flow": (ex_actions_seq, ex_times, ex_times, ex_cond),
            "critic": (ex_cond, ex_flat_actions),
            "target_critic": (ex_cond, ex_flat_actions),
        }

        # Optionally add one-step actor for distill-ddpg mode
        actor_type = config.get("actor_type", "best-of-n")
        if actor_type == "distill-ddpg":
            onestep_def = copy.deepcopy(flow_def)
            networks["actor_onestep_flow"] = onestep_def
            network_args["actor_onestep_flow"] = (
                ex_actions_seq, ex_times, ex_times, ex_cond,
            )

        network_def = ModuleDict(networks)
        rng, crop_md_rng = jax.random.split(rng)
        network_params = network_def.init(
            {"params": init_rng, "crop": crop_md_rng}, **network_args
        )["params"]

        # Copy critic params to target_critic
        network_params = dict(network_params)
        network_params["modules_target_critic"] = network_params["modules_critic"]
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

    def _target_update(self, network):
        """Polyak average update for target critic."""
        tau = self.config.get("tau", 0.005)
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            network.params["modules_critic"],
            network.params["modules_target_critic"],
        )
        updated_params = flax.core.unfreeze(network.params)
        updated_params["modules_target_critic"] = new_target_params
        return network.replace(params=flax.core.freeze(updated_params))

    def _sample_flow_actions(self, observations, rng, params=None):
        """Sample action chunks via flow integration (no grad).

        Args:
            observations: (batch, ...) observations.
            rng: PRNG key.
            params: If None, uses stored params.

        Returns:
            actions: (batch, chunk_length, action_dim)
        """
        chunk_length = self.config.get("chunk_length", 5)
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
                                name="actor_bc_flow", params=params)

        sampler = get_sampler(sampler_type)
        actions = sampler(
            network=flow_net,
            encoder=encode,
            observations=observations,
            num_steps=num_steps,
            rng=rng,
            config={
                "horizon": chunk_length,
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
                - actions (or actions_long): (batch, chunk_length, action_dim)
                - rewards: cumulative discounted reward (batch, 1)
                - next_observations: last-step next_obs
                - masks: 1 - done (batch, 1)
                - valid: sequence validity (batch, 1)

        Returns:
            Tuple of (new_agent, info_dict).
        """
        new_rng, rng, dropout_rng, crop_rng, sample_rng = jax.random.split(
            self.rng, 5
        )
        chunk_length = self.config.get("chunk_length", 5)
        action_dim = self.config["action_dim"]
        discount = self.config.get("discount", 0.99)
        alpha = self.config.get("alpha", 100.0)
        actor_type = self.config.get("actor_type", "best-of-n")

        # Resolve action key (support both "actions" and "actions_long")
        if "actions_long" in batch:
            actions = batch["actions_long"]
        else:
            actions = batch["actions"]

        # Validity mask for weighting losses
        valid = batch.get("valid", jnp.ones((actions.shape[0], 1)))
        valid_flat = valid.squeeze(-1)  # (batch,)

        def loss_fn(params):
            dropout_rngs = {"dropout": dropout_rng, "crop": crop_rng}

            # --- Encoder closures ---
            def encode(obs, training=True, rngs=None):
                if rngs is None:
                    rngs = dropout_rngs
                return self.network(
                    obs, training=training, name="encoder",
                    params=params, rngs=rngs,
                )

            def flow_net(at, s, t, cond, training=True):
                return self.network(
                    at, s, t, cond, training=training,
                    name="actor_bc_flow", params=params, rngs=dropout_rngs,
                )

            # ========== 1. BC flow loss (reuse existing loss functions) ==========
            flow_batch = {
                "observations": batch["observations"],
                "actions": actions,
                "sample_weights": valid_flat,
            }
            bc_loss, bc_info = self.loss_fn_type(
                network=flow_net,
                encoder=encode,
                interpolant=self.interpolant,
                batch=flow_batch,
                rng=rng,
                config=self.config,
                step=self.network.step,
            )

            # ========== 2. Critic loss ==========
            cond = encode(batch["observations"], training=False, rngs={})
            flat_actions = flatten_action_chunk(actions)
            q_all = self.network(
                cond, flat_actions, name="critic", params=params
            )  # (num_qs, batch)

            # Sample next actions from flow policy (no grad)
            rng_next = jax.random.fold_in(sample_rng, self.network.step)
            next_actions = self._sample_flow_actions(
                batch["next_observations"], rng_next
            )
            next_flat = flatten_action_chunk(next_actions)
            next_cond = encode(batch["next_observations"], training=False, rngs={})

            # Target Q
            target_q_all = self.network(
                next_cond, next_flat, name="target_critic"
            )  # (num_qs, batch)
            target_q = aggregate_q(
                target_q_all, self.config.get("q_agg", "mean")
            )

            # TD target with chunk-level discount
            chunk_discount = discount ** chunk_length
            rewards = batch["rewards"].squeeze(-1)
            masks = batch["masks"].squeeze(-1)
            td_target = rewards + chunk_discount * masks * target_q
            td_target = jax.lax.stop_gradient(td_target)

            # Critic MSE loss (masked by validity)
            critic_loss = jnp.mean(
                valid_flat[None, :] * (q_all - td_target[None, :]) ** 2
            )

            # ========== 3. Distill + Q loss (distill-ddpg mode) ==========
            distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())

            if actor_type == "distill-ddpg":
                rng_distill = jax.random.fold_in(rng, 1)
                rng_noise, rng_flow = jax.random.split(rng_distill)

                batch_size = actions.shape[0]
                noises = jax.random.normal(
                    rng_noise, (batch_size, chunk_length, action_dim)
                )

                # Multi-step flow actions (no grad through bc_flow)
                flow_actions = self._sample_flow_actions(
                    batch["observations"], rng_flow
                )

                # One-step prediction (with grad)
                onestep_cond = encode(
                    batch["observations"], training=False, rngs={}
                )
                s_zero = jnp.zeros((batch_size,))
                t_one = jnp.ones((batch_size,))
                onestep_pred = self.network(
                    noises, s_zero, t_one, onestep_cond,
                    training=True, name="actor_onestep_flow",
                    params=params, rngs=dropout_rngs,
                )
                # For MLP: output is velocity, action = noise + velocity
                onestep_actions = noises + onestep_pred

                # Distillation loss
                distill_loss = jnp.mean(
                    (onestep_actions - jax.lax.stop_gradient(flow_actions)) ** 2
                )

                # Q loss through one-step actor (no grad through critic)
                onestep_flat = flatten_action_chunk(
                    jnp.clip(onestep_actions, -1, 1)
                )
                q_onestep = self.network(
                    onestep_cond, onestep_flat, name="critic"
                )
                q_loss = -jnp.mean(aggregate_q(q_onestep, "mean"))

            # ========== Total loss ==========
            total_loss = critic_loss + bc_loss + alpha * distill_loss + q_loss

            info = {
                "loss": total_loss,
                "critic_loss": critic_loss,
                "bc_loss": bc_loss,
                "distill_loss": distill_loss,
                "q_loss": q_loss,
                "q_mean": jnp.mean(q_all),
                "q_max": jnp.max(q_all),
                "q_min": jnp.min(q_all),
                "td_target_mean": jnp.mean(td_target),
            }
            # Merge bc_info
            for k, v in bc_info.items():
                if k != "loss":
                    info[f"bc/{k}"] = v

            return total_loss, info

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        # Polyak update target critic
        new_network = self._target_update(new_network)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, rng=None):
        """Sample action chunks for environment interaction.

        Args:
            observations: (batch, ...) observations.
            rng: Optional PRNG key.

        Returns:
            actions: (batch, chunk_length, action_dim), clipped to [-1, 1].
        """
        if rng is None:
            rng = self.rng

        actor_type = self.config.get("actor_type", "best-of-n")

        if actor_type == "distill-ddpg":
            return self._sample_distill_ddpg(observations, rng)
        elif actor_type == "best-of-n":
            return self._sample_best_of_n(observations, rng)
        else:
            raise ValueError(f"Unknown actor_type: {actor_type}")

    def _sample_distill_ddpg(self, observations, rng):
        """Sample via one-step distilled actor."""
        chunk_length = self.config.get("chunk_length", 5)
        action_dim = self.config["action_dim"]
        batch_size = get_batch_size(observations)

        cond = self.network(observations, training=False, name="encoder", rngs={})

        noises = jax.random.normal(
            rng, (batch_size, chunk_length, action_dim)
        )
        s_zero = jnp.zeros((batch_size,))
        t_one = jnp.ones((batch_size,))
        velocity = self.network(
            noises, s_zero, t_one, cond,
            training=False, name="actor_onestep_flow",
        )
        actions = noises + velocity
        return jnp.clip(actions, -1, 1)

    def _sample_best_of_n(self, observations, rng):
        """Sample via best-of-N: generate N candidates, pick highest Q."""
        n_samples = self.config.get("actor_num_samples", 32)
        chunk_length = self.config.get("chunk_length", 5)
        action_dim = self.config["action_dim"]
        batch_size = get_batch_size(observations)

        cond = self.network(observations, training=False, name="encoder", rngs={})

        # Generate N action chunks using vmap for efficiency
        # Split rng into N keys
        rngs = jax.random.split(rng, n_samples)

        # Vectorize _sample_flow_actions over the N samples
        # observations needs to be broadcast to (N, batch, ...)
        obs_expanded = jax.tree_util.tree_map(
            lambda x: jnp.broadcast_to(x[None, ...], (n_samples,) + x.shape),
            observations
        )

        # vmap over the first axis (N samples)
        sample_fn = jax.vmap(
            lambda r, o: self._sample_flow_actions(o, r),
            in_axes=(0, 0),
            out_axes=0
        )
        all_actions = sample_fn(rngs, obs_expanded)  # (N, batch, chunk_length, action_dim)

        # Evaluate Q for each candidate
        # Expand cond: (batch, cond_dim) -> (N, batch, cond_dim)
        cond_exp = jnp.broadcast_to(
            cond[None, :, :], (n_samples, batch_size, cond.shape[-1])
        )

        # Flatten for critic: (N * batch, chunk_length * action_dim)
        flat_actions = all_actions.reshape(
            n_samples * batch_size, chunk_length * action_dim
        )
        cond_flat = cond_exp.reshape(n_samples * batch_size, -1)

        q_all = self.network(
            cond_flat, flat_actions, name="critic"
        )  # (num_qs, N * batch)
        q_agg = aggregate_q(q_all, self.config.get("q_agg", "mean"))
        q_agg = q_agg.reshape(n_samples, batch_size)  # (N, batch)

        # Select best per batch element
        best_idx = jnp.argmax(q_agg, axis=0)  # (batch,)
        best_actions = all_actions[best_idx, jnp.arange(batch_size)]

        return best_actions

    @jax.jit
    def eval_actions(self, observations):
        """Deterministic action sampling for evaluation."""
        eval_rng = jax.random.PRNGKey(0)
        return self.sample_actions(observations, rng=eval_rng)
