"""ResFiT: Residual Fine-Tuning Agent.

Implements the ResFiT algorithm for offline-to-online RL fine-tuning.
Key features:
- 3 independent TrainStates (encoder, actor, critic) with separate optimizers
- RED-Q ensemble (10 Q-heads) for critic
- TD3-style deterministic policy gradient
- Encoder only updated from critic loss (stop_gradient in actor loss)
- 50/50 online/offline data mixing
"""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from jax_flow.agents.train_state import TrainState, nonpytree_field
from jax_flow.networks.encoders import create_encoder
from jax_flow.networks.residual_actor import ResidualActor, add_exploration_noise
from jax_flow.networks.spatial_emb_critic import SpatialEmbCritic, ensemble_mean_q, redq_q_value
from jax_flow.networks.value import Value

# Re-export for type checking (used in create())
__all__ = ["ResFiTAgent"]


def _get_actor_features_image(patch_features, prop_features, base_action, critic_params, config):
    """Helper to compute actor features from image observations.

    Extracts spatial embedding from critic trunk.
    """
    from jax_flow.networks.spatial_emb_critic import SpatialEmbTrunk

    trunk = SpatialEmbTrunk(spatial_emb_dim=config.get("spatial_emb_dim", 1024))
    trunk_params = {
        k: v for k, v in critic_params.items() if k.startswith("SpatialEmbTrunk")
    }
    dummy_action = jnp.zeros_like(base_action)
    z = trunk.apply({"params": trunk_params}, patch_features, dummy_action, prop_features)

    # Extract spatial_emb + prop (remove dummy action)
    spatial_emb_dim = config.get("spatial_emb_dim", 1024)
    action_dim = base_action.shape[-1]
    actor_feat = jnp.concatenate(
        [z[:, :spatial_emb_dim], z[:, spatial_emb_dim + action_dim :]], axis=-1
    )
    return actor_feat


def _clip_grads(grads, max_norm: float):
    """Clip gradients by global norm."""
    grad_norm = jnp.sqrt(
        sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(grads))
    )
    scale = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))
    return jax.tree.map(lambda g: g * scale, grads)


class ResFiTAgent(flax.struct.PyTreeNode):
    """ResFiT: Residual Fine-Tuning Agent.

    Manages 3 independent TrainStates (each with its own optimizer):
    - encoder_state: ViT/MLP encoder (updated only from critic loss)
    - critic_state: SpatialEmbCritic or Value (RED-Q ensemble)
    - actor_state: ResidualActor

    Plus 2 target networks (Polyak-averaged, no optimizer):
    - target_critic_params
    - target_actor_params
    """

    rng: Any
    encoder_state: TrainState
    critic_state: TrainState
    actor_state: TrainState
    target_critic_params: Any
    target_actor_params: Any
    config: Any = nonpytree_field()

    @classmethod
    def create(
        cls,
        seed: int,
        ex_obs: dict,
        ex_actions: jnp.ndarray,
        config: dict,
        bc_checkpoint_path: str | None = None,
    ):
        """Create a new ResFiT agent.

        Args:
            seed: Random seed.
            ex_obs: Example observation dict (with base_action field).
                Lowdim: {"obs": (batch, obs_steps, obs_dim), "base_action": (batch, action_dim)}
                Image: {"image_key": (batch, obs_steps, H, W, C), ..., "base_action": (batch, action_dim)}
            ex_actions: Example actions (batch, action_dim).
            config: Configuration dict.
            bc_checkpoint_path: Optional path to BC checkpoint for encoder initialization.

        Returns:
            New ResFiTAgent instance.
        """
        rng = jax.random.PRNGKey(seed)
        rng, enc_rng, crit_rng, act_rng = jax.random.split(rng, 4)

        obs_type = config.get("obs_type", "lowdim")
        action_dim = ex_actions.shape[-1]
        batch_size = ex_actions.shape[0]

        # Extract base_action from obs
        base_action = ex_obs.pop("base_action")

        # ============================================================
        # 1. Create Encoder
        # ============================================================
        if obs_type == "image":
            encoder_def = create_encoder(
                encoder_type="multi_image",
                image_keys=tuple(config.get("image_keys", ("agentview_image",))),
                lowdim_keys=tuple(config.get("lowdim_keys", ())),
                image_backbone=config.get("image_backbone", "vit"),
                vit_embed_dim=config.get("vit_embed_dim", 128),
                vit_num_heads=config.get("vit_num_heads", 4),
                vit_depth=config.get("vit_depth", 1),
                vit_ffn_mult=config.get("vit_ffn_mult", 4),
                return_patches=True,  # For SpatialEmbCritic
                crop_shape=config.get("crop_shape", None),
            )
        else:
            # Lowdim mode: obs is already flattened by FrameStack
            encoder_def = create_encoder(
                encoder_type=config.get("encoder_type", "mlp"),
                hidden_dims=tuple(config.get("encoder_hidden_dims", (256, 256))),
                output_dim=config.get("encoder_output_dim", 256),
            )

        # Initialize encoder
        enc_init_rng, crop_rng = jax.random.split(enc_rng)
        if obs_type == "image":
            encoder_variables = encoder_def.init(
                {"params": enc_init_rng, "crop": crop_rng}, ex_obs, training=False
            )
        else:
            encoder_variables = encoder_def.init(
                {"params": enc_init_rng}, ex_obs["obs"], training=False
            )
        encoder_params = encoder_variables["params"]

        # Optional: load encoder weights from BC checkpoint
        if bc_checkpoint_path and config.get("init_encoder_from_bc", False):
            from jax_flow.core.checkpoint import load_checkpoint

            bc_ckpt = load_checkpoint(bc_checkpoint_path)
            # Extract BC encoder params (assuming ModuleDict structure)
            if "encoder" in bc_ckpt["params"]:
                encoder_params = bc_ckpt["params"]["encoder"]
                print("✓ Initialized encoder from BC checkpoint")

        # Create encoder TrainState
        encoder_tx = optax.adamw(
            learning_rate=config.get("critic_lr", 1e-4),
            weight_decay=config.get("weight_decay", 0.0),
        )
        encoder_state = TrainState.create(encoder_def, encoder_params, tx=encoder_tx)

        # Get encoder output for critic/actor initialization
        if obs_type == "image":
            encoder_out = encoder_state(ex_obs, training=False)
            patch_features, prop_features = encoder_out
            num_patches = patch_features.shape[1]
            embed_dim = patch_features.shape[2]
            prop_dim = prop_features.shape[1]
        else:
            encoder_out = encoder_state(ex_obs["obs"], training=False)
            feat_dim = encoder_out.shape[-1]

        # ============================================================
        # 2. Create Critic
        # ============================================================
        if obs_type == "image":
            critic_def = SpatialEmbCritic(
                spatial_emb_dim=config.get("spatial_emb_dim", 1024),
                hidden_dim=config.get("critic_hidden_dim", 1024),
                num_layers=config.get("critic_num_layers", 2),
                num_q=config.get("num_q", 10),
                layer_norm=config.get("critic_layer_norm", True),
            )
            # Initialize with example inputs
            ex_patch = jnp.zeros((batch_size, num_patches, embed_dim))
            ex_prop = jnp.zeros((batch_size, prop_dim))
            critic_params = critic_def.init(
                {"params": crit_rng}, ex_patch, ex_actions, ex_prop
            )["params"]
        else:
            critic_def = Value(
                hidden_dims=tuple(
                    [config.get("critic_hidden_dim", 1024)]
                    * config.get("critic_num_layers", 2)
                ),
                layer_norm=config.get("critic_layer_norm", True),
                num_ensembles=config.get("num_q", 10),
                activation="relu",
            )
            critic_params = critic_def.init(
                {"params": crit_rng}, encoder_out, ex_actions
            )["params"]

        critic_tx = optax.adamw(
            learning_rate=config.get("critic_lr", 1e-4),
            weight_decay=config.get("weight_decay", 0.0),
        )
        critic_state = TrainState.create(critic_def, critic_params, tx=critic_tx)

        # ============================================================
        # 3. Create Actor
        # ============================================================
        actor_def = ResidualActor(
            action_dim=action_dim,
            hidden_dim=config.get("actor_hidden_dim", 1024),
            num_layers=config.get("actor_num_layers", 2),
            action_scale=config.get("action_scale", 0.1),
            layer_norm=config.get("actor_layer_norm", True),
        )

        # Actor input: features + base_action
        if obs_type == "image":
            # For image mode, actor gets flattened spatial embedding + prop + base_action
            # We'll compute this in the forward pass, but for init we need the size
            # SpatialEmbTrunk output: (batch, spatial_emb_dim + action_dim + prop_dim)
            spatial_emb_dim = config.get("spatial_emb_dim", 1024)
            actor_feat_dim = spatial_emb_dim + prop_dim
        else:
            actor_feat_dim = feat_dim

        ex_actor_feat = jnp.zeros((batch_size, actor_feat_dim))
        actor_params = actor_def.init({"params": act_rng}, ex_actor_feat, base_action)["params"]

        actor_tx = optax.adamw(
            learning_rate=config.get("actor_lr", 1e-6),
            weight_decay=config.get("weight_decay", 0.0),
        )
        actor_state = TrainState.create(actor_def, actor_params, tx=actor_tx)

        # ============================================================
        # 4. Initialize Target Networks
        # ============================================================
        target_critic_params = critic_params
        target_actor_params = actor_params

        return cls(
            rng=rng,
            encoder_state=encoder_state,
            critic_state=critic_state,
            actor_state=actor_state,
            target_critic_params=target_critic_params,
            target_actor_params=target_actor_params,
            config=dict(config),
        )

    @jax.jit
    def update_critic(self, batch: dict):
        """Update encoder + critic.

        Args:
            batch: Dict with keys:
                - obs: Observation dict (without base_action)
                - action: Combined action (batch, action_dim)
                - reward: Reward (batch, 1)
                - next_obs: Next observation dict (without base_action)
                - done: Done flag (batch, 1)
                - discount: Discount factor (batch, 1)
                - base_action: Base action for current obs (batch, action_dim)
                - next_base_action: Base action for next obs (batch, action_dim)

        Returns:
            (new_agent, info_dict)
        """
        new_rng, noise_rng, redq_rng = jax.random.split(self.rng, 3)
        obs_type = self.config.get("obs_type", "lowdim")
        tau = self.config.get("tau", 0.005)
        grad_clip_norm = self.config.get("grad_clip_norm", 1.0)

        def critic_loss_fn(encoder_params, critic_params):
            # 1. Encode observations
            if obs_type == "image":
                features = self.encoder_state(
                    batch["obs"], params=encoder_params, training=True
                )
                patch_features, prop_features = features
                next_features = self.encoder_state(
                    batch["next_obs"], params=encoder_params, training=True
                )
                next_patch_features, next_prop_features = next_features
                # Stop gradient on next features
                next_patch_features = jax.lax.stop_gradient(next_patch_features)
                next_prop_features = jax.lax.stop_gradient(next_prop_features)
            else:
                features = self.encoder_state(
                    batch["obs"]["obs"], params=encoder_params, training=True
                )
                next_features = self.encoder_state(
                    batch["next_obs"]["obs"], params=encoder_params, training=True
                )
                next_features = jax.lax.stop_gradient(next_features)

            # 2. Target actor → next residual (no grad)
            if obs_type == "image":
                next_actor_feat = _get_actor_features_image(
                    next_patch_features, next_prop_features,
                    batch["next_base_action"],
                    self.target_critic_params, self.config,
                )
            else:
                next_actor_feat = next_features

            next_residual = self.actor_state(
                next_actor_feat,
                batch["next_base_action"],
                params=self.target_actor_params,
            )
            # Add exploration noise (TD3 style)
            next_residual = add_exploration_noise(
                next_residual,
                noise_rng,
                stddev=self.config.get("stddev_max", 0.05),
                clip=self.config.get("stddev_clip", 0.3),
            )
            next_combined = jnp.clip(
                batch["next_base_action"] + next_residual, -1.0, 1.0
            )

            # 3. Target Q
            if obs_type == "image":
                target_q_all = self.critic_state(
                    next_patch_features,
                    next_combined,
                    next_prop_features,
                    params=self.target_critic_params,
                )
            else:
                target_q_all = self.critic_state(
                    next_features, next_combined, params=self.target_critic_params
                )

            # RED-Q: randomly select min_q_heads and take min
            target_q = redq_q_value(
                target_q_all, redq_rng, min_q_heads=self.config.get("min_q_heads", 2)
            )
            target_q = batch["reward"].squeeze(-1) + batch["discount"].squeeze(
                -1
            ) * target_q
            target_q = jax.lax.stop_gradient(target_q)

            # 4. Current Q
            if obs_type == "image":
                q_all = self.critic_state(
                    patch_features, batch["action"], prop_features, params=critic_params
                )
            else:
                q_all = self.critic_state(features, batch["action"], params=critic_params)

            # MSE loss over all ensemble heads
            critic_loss = jnp.mean((q_all - target_q[None, :]) ** 2)

            info = {
                "critic_loss": critic_loss,
                "q_mean": jnp.mean(q_all),
                "q_std": jnp.std(q_all),
                "target_q_mean": jnp.mean(target_q),
            }
            return critic_loss, info

        # Compute gradients for encoder and critic
        (loss, info), (enc_grads, crit_grads) = jax.value_and_grad(
            critic_loss_fn, argnums=(0, 1), has_aux=True
        )(self.encoder_state.params, self.critic_state.params)

        # Gradient clipping
        enc_grads = _clip_grads(enc_grads, grad_clip_norm)
        crit_grads = _clip_grads(crit_grads, grad_clip_norm)

        # Apply gradients
        new_encoder = self.encoder_state.apply_gradients(enc_grads)
        new_critic = self.critic_state.apply_gradients(crit_grads)

        # Polyak update target critic
        new_target_critic = jax.tree.map(
            lambda p, tp: tau * p + (1 - tau) * tp,
            new_critic.params,
            self.target_critic_params,
        )

        return (
            self.replace(
                encoder_state=new_encoder,
                critic_state=new_critic,
                target_critic_params=new_target_critic,
                rng=new_rng,
            ),
            info,
        )

    @jax.jit
    def update_actor(self, batch: dict):
        """Update actor (encoder not updated).

        Args:
            batch: Same as update_critic.

        Returns:
            (new_agent, info_dict)
        """
        new_rng = jax.random.split(self.rng)[0]
        obs_type = self.config.get("obs_type", "lowdim")
        tau = self.config.get("tau", 0.005)

        def actor_loss_fn(actor_params):
            # Encode observations (stop gradient)
            if obs_type == "image":
                features = self.encoder_state(batch["obs"], training=False)
                patch_features, prop_features = features
                patch_features = jax.lax.stop_gradient(patch_features)
                prop_features = jax.lax.stop_gradient(prop_features)

                actor_feat = _get_actor_features_image(
                    patch_features, prop_features,
                    batch["base_action"],
                    self.critic_state.params, self.config,
                )
            else:
                features = self.encoder_state(batch["obs"]["obs"], training=False)
                features = jax.lax.stop_gradient(features)
                actor_feat = features

            # Actor forward
            residual = self.actor_state(actor_feat, batch["base_action"], params=actor_params)
            combined = jnp.clip(batch["base_action"] + residual, -1.0, 1.0)

            # Q-value (mean over ensemble)
            if obs_type == "image":
                q_all = self.critic_state(patch_features, combined, prop_features)
            else:
                q_all = self.critic_state(features, combined)

            q = ensemble_mean_q(q_all)

            # Actor loss: maximize Q
            actor_loss = -jnp.mean(q)

            info = {"actor_loss": actor_loss, "actor_q_mean": jnp.mean(q)}
            return actor_loss, info

        (loss, info), grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
            self.actor_state.params
        )

        # Apply gradients
        new_actor = self.actor_state.apply_gradients(grads)

        # Polyak update target actor
        new_target_actor = jax.tree.map(
            lambda p, tp: tau * p + (1 - tau) * tp,
            new_actor.params,
            self.target_actor_params,
        )

        return (
            self.replace(
                actor_state=new_actor,
                target_actor_params=new_target_actor,
                rng=new_rng,
            ),
            info,
        )

    @jax.jit
    def sample_actions(self, obs: dict, rng: jax.Array, stddev: float = 0.0):
        """Sample actions for environment interaction.

        Args:
            obs: Observation dict (with base_action field).
            rng: PRNG key.
            stddev: Exploration noise standard deviation (0 = deterministic).

        Returns:
            Residual action (action_dim,).
        """
        obs_type = self.config.get("obs_type", "lowdim")
        base_action = obs["base_action"]
        # Build encoder input without base_action
        enc_obs = {k: v for k, v in obs.items() if k != "base_action"}

        # Encode
        if obs_type == "image":
            features = self.encoder_state(enc_obs, training=False)
            patch_features, prop_features = features
            actor_feat = _get_actor_features_image(
                patch_features, prop_features, base_action,
                self.critic_state.params, self.config,
            )
        else:
            features = self.encoder_state(enc_obs["obs"], training=False)
            actor_feat = features

        # Actor forward
        residual = self.actor_state(actor_feat, base_action)

        # Always add noise (stddev=0 means no effect)
        noise = jax.random.normal(rng, residual.shape) * stddev
        noise = jnp.clip(noise, -self.config.get("stddev_clip", 0.3),
                         self.config.get("stddev_clip", 0.3))
        residual = jnp.clip(residual + noise, -1.0, 1.0)

        return residual

    @jax.jit
    def eval_actions(self, obs: dict):
        """Deterministic action sampling for evaluation.

        Args:
            obs: Observation dict (with base_action field).

        Returns:
            Residual action (action_dim,).
        """
        eval_rng = jax.random.PRNGKey(0)
        return self.sample_actions(obs, rng=eval_rng, stddev=0.0)
