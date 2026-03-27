"""CriticState: standalone critic for BC warmup phase.

Updated each step alongside BCAgent. Uses BCAgent's encoder (frozen)
to encode observations, and BCAgent's flow policy to sample next_actions.

When critic_encoder_type != "none", critic embeds its own independent encoder
so it can learn observation representations suited for Q-value estimation.
"""

from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from jax_flow.agents.train_state import ModuleDict, TrainState, nonpytree_field
from jax_flow.core.utils import create_optimizer, get_batch_size
from jax_flow.networks.value import Value


class CriticState(flax.struct.PyTreeNode):
    """Standalone critic + target_critic for BC warmup phase.

    Updated each step alongside BCAgent. Uses BCAgent's encoder (frozen)
    to encode observations, and BCAgent's flow policy to sample next_actions.

    When critic_has_encoder=True, critic has its own encoder and receives
    raw observations directly instead of pre-encoded cond.
    """

    rng: Any
    network: TrainState  # ModuleDict with critic + target_critic
    config: Any = nonpytree_field()

    @classmethod
    def create(cls, seed, ex_cond, ex_actions, config, ex_observations=None):
        """Create CriticState.

        Args:
            seed: Random seed.
            ex_cond: Example encoded observations (B, cond_dim).
            ex_actions: Example actions (B, horizon, action_dim).
            config: Config dict with critic_hidden_dims, critic_lr, etc.
            ex_observations: Example raw observations (needed when critic_encoder_type != "none").

        Returns:
            New CriticState.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        act_exec_steps = config.get("act_exec_steps", 8)
        action_dim = ex_actions.shape[-1]
        critic_encoder_type = config.get("critic_encoder_type", "none")
        critic_has_encoder = critic_encoder_type != "none"

        critic_hidden_dims = tuple(config.get("critic_hidden_dims", [2048, 2048, 2048]))
        critic_kwargs = {
            "hidden_dims": critic_hidden_dims,
            "layer_norm": config.get("critic_layer_norm", True),
            "num_ensembles": config.get("num_ensembles", 2),
            "activation": config.get("critic_activation", "tanh"),
        }

        # Build encoder for critic if requested
        encoder_def = None
        if critic_has_encoder:
            from jax_flow.networks.encoders import create_encoder

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

        critic_def = Value(encoder=encoder_def, **critic_kwargs)
        # target_critic needs its own encoder instance for independent params
        target_encoder_def = None
        if critic_has_encoder:
            from jax_flow.networks.encoders import create_encoder

            target_encoder_def = create_encoder(
                encoder_type=config.get("encoder_type", "mlp"),
                hidden_dims=tuple(config.get("encoder_hidden_dims", (256, 256))),
                output_dim=config.get("emb_dim", 256),
                image_keys=tuple(config.get("image_keys", ("agentview_image",))),
                lowdim_keys=tuple(
                    config.get("lowdim_keys", ("robot0_eef_pos", "robot0_gripper_qpos"))
                ),
                crop_shape=config.get("crop_shape", None),
            )
        target_critic_def = Value(encoder=target_encoder_def, **critic_kwargs)

        # Example inputs depend on whether critic has its own encoder
        if critic_has_encoder:
            assert ex_observations is not None, (
                "ex_observations required when critic_encoder_type != 'none'"
            )
            batch_size = get_batch_size(ex_observations)
            ex_critic_obs = ex_observations
        else:
            batch_size = ex_cond.shape[0]
            ex_critic_obs = ex_cond

        ex_critic_act = jnp.zeros((batch_size, act_exec_steps * action_dim))
        networks = {"critic": critic_def, "target_critic": target_critic_def}
        network_args = {
            "critic": (ex_critic_obs, ex_critic_act),
            "target_critic": (ex_critic_obs, ex_critic_act),
        }

        network_def = ModuleDict(networks)
        rng, crop_rng = jax.random.split(rng)
        all_variables = network_def.init(
            {"params": init_rng, "crop": crop_rng}, **network_args
        )
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

        # Store critic_has_encoder flag in config
        final_config = dict(config)
        final_config["critic_has_encoder"] = critic_has_encoder

        return cls(rng=rng, network=network, config=final_config)

    @jax.jit
    def update(self, batch, bc_agent):
        """Update critic with TD loss, using BCAgent as frozen actor.

        Args:
            batch: Dict with observations, actions, rewards, next_observations, masks.
            bc_agent: BCAgent instance (used for encoding obs and sampling next_actions).

        Returns:
            (new_critic_state, info_dict)
        """
        new_rng, rng, sample_rng, crop_rng, crop_rng2 = jax.random.split(self.rng, 5)
        config = self.config
        tau = config.get("tau", 0.005)
        discount = config.get("discount", 0.99)
        act_exec_steps = config.get("act_exec_steps", 8)
        critic_has_encoder = config.get("critic_has_encoder", False)

        def critic_loss_fn(params):
            # Determine critic observation input
            if critic_has_encoder:
                # Critic has its own encoder — pass raw observations
                critic_obs = batch["observations"]
                critic_next_obs = batch["next_observations"]
            else:
                # Encode obs with BCAgent's encoder (no grad)
                critic_obs = jax.lax.stop_gradient(
                    bc_agent.network(
                        batch["observations"], training=False, name="encoder",
                        params=bc_agent.ema_params, rngs={},
                    )
                )
                critic_next_obs = jax.lax.stop_gradient(
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
                critic_next_obs, next_action_flat, name="target_critic",
                training=critic_has_encoder, rngs={"crop": crop_rng},
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
                critic_obs, action_flat, name="critic", params=params,
                training=critic_has_encoder, rngs={"crop": crop_rng2},
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
