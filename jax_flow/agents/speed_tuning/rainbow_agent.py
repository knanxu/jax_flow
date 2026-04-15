"""Rainbow DQN agent for speed tuning.

Components: Double Q-learning, C51 distributional Q, Dueling architecture.
PER is managed externally (buffer in training script).
"""

import flax
import jax
import jax.numpy as jnp

from jax_flow.agents.speed_tuning.networks import DuelingC51Network
from jax_flow.agents.train_state import ModuleDict, TrainState, nonpytree_field
from jax_flow.core.checkpoint import load_checkpoint
from jax_flow.core.utils import create_optimizer, get_batch_size


class RainbowDQNAgent(flax.struct.PyTreeNode):
    """Rainbow DQN agent for learning discrete speed multipliers.

    Uses the same encoder as the frozen BC policy to encode observations,
    then feeds encoded features to a DuelingC51Network.

    Fields:
        rng: PRNG key.
        network: TrainState (ModuleDict: encoder + q_network + target_q_network).
        config: Non-pytree config dict.
    """

    rng: jnp.ndarray
    network: TrainState
    config: dict = nonpytree_field()

    @classmethod
    def create(cls, seed, ex_observations, config, bc_checkpoint_path):
        """Create RainbowDQNAgent from BC checkpoint.

        Loads the BC encoder (frozen) and creates Q-networks.

        Args:
            seed: Random seed.
            ex_observations: Example observations for init.
            config: Algorithm config dict.
            bc_checkpoint_path: Path to trained BC checkpoint.

        Returns:
            New RainbowDQNAgent.
        """
        from jax_flow.networks.encoders import create_encoder

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        # Load BC checkpoint for encoder architecture
        bc_ckpt = load_checkpoint(bc_checkpoint_path)
        bc_config = bc_ckpt["config"]
        bc_params = bc_ckpt.get("ema_params", bc_ckpt["params"])

        # Rebuild encoder from BC config
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
        cond_dim = ex_cond.shape[-1]

        # Q-network defs
        num_actions = config.get("num_actions", 11)
        num_atoms = config.get("num_atoms", 51)
        q_def = DuelingC51Network(
            num_actions=num_actions,
            num_atoms=num_atoms,
            hidden_dims=tuple(config.get("hidden_dims", (512, 512))),
            activation=config.get("activation", "relu"),
            layer_norm=config.get("layer_norm", True),
        )
        target_q_def = DuelingC51Network(
            num_actions=num_actions,
            num_atoms=num_atoms,
            hidden_dims=tuple(config.get("hidden_dims", (512, 512))),
            activation=config.get("activation", "relu"),
            layer_norm=config.get("layer_norm", True),
        )

        # Build ModuleDict
        batch_size = get_batch_size(ex_observations)
        ex_cond_input = jnp.zeros((batch_size, cond_dim))

        networks = {
            "encoder": encoder_def,
            "q_network": q_def,
            "target_q_network": target_q_def,
        }
        network_args = {
            "encoder": (ex_observations,),
            "q_network": (ex_cond_input,),
            "target_q_network": (ex_cond_input,),
        }

        network_def = ModuleDict(networks)
        rng, crop_md_rng = jax.random.split(rng)
        all_variables = network_def.init(
            {"params": init_rng, "crop": crop_md_rng}, **network_args
        )
        network_params = flax.core.unfreeze(all_variables["params"])

        # Load BC encoder weights (frozen)
        bc_encoder_key = "modules_encoder"
        if bc_encoder_key in bc_params:
            network_params[bc_encoder_key] = bc_params[bc_encoder_key]

        # Copy q_network params to target
        network_params["modules_target_q_network"] = jax.tree_util.tree_map(
            lambda x: x.copy(), network_params["modules_q_network"]
        )
        network_params = flax.core.freeze(network_params)

        # Optimizer: encoder frozen, q_network trainable, target frozen
        lr = config.get("lr", 3e-4)
        grad_clip = config.get("grad_clip_norm", 10.0)

        param_labels = jax.tree_util.tree_map_with_path(
            lambda path, _: _label_param(path), network_params
        )
        import optax

        tx = optax.multi_transform(
            {
                "frozen": optax.set_to_zero(),
                "q_network": create_optimizer(lr=lr, grad_clip_norm=grad_clip),
            },
            param_labels,
        )

        extra_vars = None
        if "batch_stats" in all_variables:
            extra_vars = {
                "batch_stats": flax.core.unfreeze(all_variables["batch_stats"])
            }

        network = TrainState.create(
            network_def, network_params, tx=tx, extra_variables=extra_vars
        )

        return cls(
            rng=rng,
            network=network,
            config=dict(config),
        )

    @jax.jit
    def select_action(self, observations, epsilon):
        """Epsilon-greedy action selection.

        Args:
            observations: Batched observations (1, obs_steps, obs_dim) or dict.
            epsilon: Exploration rate.

        Returns:
            (new_agent, action) where action is (1,) int array.
        """
        new_rng, rng, eps_rng = jax.random.split(self.rng, 3)

        # Encode observations (frozen encoder)
        cond = self.network(observations, training=False, name="encoder", rngs={})

        # Q-values from atom distribution
        logits = self.network(cond, name="q_network")  # (1, num_actions, num_atoms)
        probs = jax.nn.softmax(logits, axis=-1)
        support = self._get_support()
        q_values = jnp.sum(probs * support[None, None, :], axis=-1)  # (1, num_actions)

        # Epsilon-greedy
        greedy_action = jnp.argmax(q_values, axis=-1)  # (1,)
        random_action = jax.random.randint(
            eps_rng, shape=(1,), minval=0, maxval=self.config["num_actions"]
        )
        use_random = jax.random.uniform(rng, shape=(1,)) < epsilon
        action = jnp.where(use_random, random_action, greedy_action)

        return self.replace(rng=new_rng), action

    @jax.jit
    def eval_action(self, observations):
        """Greedy action selection for evaluation (no exploration)."""
        cond = self.network(observations, training=False, name="encoder", rngs={})
        logits = self.network(cond, name="q_network")
        probs = jax.nn.softmax(logits, axis=-1)
        support = self._get_support()
        q_values = jnp.sum(probs * support[None, None, :], axis=-1)
        return jnp.argmax(q_values, axis=-1)  # (batch,)

    @jax.jit
    def update(self, batch, is_weights):
        """Update Q-network with C51 distributional TD loss.

        Uses Double Q-learning: online selects action, target evaluates.

        Args:
            batch: Dict with observations, actions, rewards, next_observations, dones.
            is_weights: (batch_size,) PER importance sampling weights.

        Returns:
            (new_agent, info_dict, td_errors).
        """
        new_rng, rng = jax.random.split(self.rng)

        config = self.config
        discount = config.get("discount", 0.99)
        num_atoms = config.get("num_atoms", 51)
        v_min = config.get("v_min", -10.0)
        v_max = config.get("v_max", 10.0)
        support = self._get_support()
        delta_z = (v_max - v_min) / (num_atoms - 1)

        def loss_fn(params):
            # Current Q distribution
            cond = self.network(
                batch["observations"], training=False, name="encoder", rngs={}
            )
            logits = self.network(cond, name="q_network", params=params)
            log_probs = jax.nn.log_softmax(logits, axis=-1)  # (B, A, atoms)

            actions = batch["actions"].astype(jnp.int32)  # (B,)
            batch_size = actions.shape[0]
            batch_idx = jnp.arange(batch_size)
            chosen_log_probs = log_probs[batch_idx, actions]  # (B, atoms)

            # Next Q distribution — Double Q-learning
            next_cond = self.network(
                batch["next_observations"], training=False, name="encoder", rngs={}
            )
            # Online network selects action
            next_logits_online = self.network(
                next_cond, name="q_network", params=params
            )
            next_probs_online = jax.nn.softmax(next_logits_online, axis=-1)
            next_q_online = jnp.sum(next_probs_online * support[None, None, :], axis=-1)
            next_actions = jnp.argmax(next_q_online, axis=-1)  # (B,)

            # Target network evaluates
            next_logits_target = self.network(next_cond, name="target_q_network")
            next_probs_target = jax.nn.softmax(next_logits_target, axis=-1)
            next_dist = next_probs_target[batch_idx, next_actions]  # (B, atoms)

            # Bellman projection
            rewards = batch["rewards"][:, None]  # (B, 1)
            dones = batch["dones"][:, None]  # (B, 1)
            tz = rewards + discount * (1.0 - dones) * support[None, :]  # (B, atoms)
            tz = jnp.clip(tz, v_min, v_max)
            b = (tz - v_min) / delta_z  # (B, atoms)
            lo = jnp.floor(b).astype(jnp.int32)
            hi = jnp.ceil(b).astype(jnp.int32)
            lo = jnp.clip(lo, 0, num_atoms - 1)
            hi = jnp.clip(hi, 0, num_atoms - 1)

            # Distribute probability mass
            projected = jnp.zeros_like(next_dist)
            # Lower
            projected = projected.at[jnp.arange(batch_size)[:, None], lo].add(
                next_dist * (hi.astype(jnp.float32) - b)
            )
            # Upper
            projected = projected.at[jnp.arange(batch_size)[:, None], hi].add(
                next_dist * (b - lo.astype(jnp.float32))
            )

            projected = jax.lax.stop_gradient(projected)

            # Cross-entropy loss
            element_loss = -jnp.sum(projected * chosen_log_probs, axis=-1)  # (B,)
            weighted_loss = jnp.mean(is_weights * element_loss)

            # TD errors for PER priority update
            chosen_probs = jax.nn.softmax(logits, axis=-1)[batch_idx, actions]
            q_current = jnp.sum(chosen_probs * support[None, :], axis=-1)
            q_target = jnp.sum(projected * support[None, :], axis=-1)
            td_errors = jnp.abs(q_current - q_target)

            return weighted_loss, {
                "loss": weighted_loss,
                "mean_q": jnp.mean(q_current),
                "min_q": jnp.min(q_current),
                "max_q": jnp.max(q_current),
                "td_errors": td_errors,
            }

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        td_errors = info.pop("td_errors")

        # Polyak update target network
        new_network = self._target_update(new_network)

        return self.replace(network=new_network, rng=new_rng), info, td_errors

    def _target_update(self, network):
        """Polyak update: target <- tau * online + (1-tau) * target."""
        tau = self.config.get("tau", 0.005)
        new_params = flax.core.unfreeze(network.params)
        new_params["modules_target_q_network"] = jax.tree_util.tree_map(
            lambda p, tp: tau * p + (1 - tau) * tp,
            new_params["modules_q_network"],
            new_params["modules_target_q_network"],
        )
        return network.replace(params=flax.core.freeze(new_params))

    def _get_support(self):
        """Get C51 atom support vector."""
        v_min = self.config.get("v_min", -10.0)
        v_max = self.config.get("v_max", 10.0)
        num_atoms = self.config.get("num_atoms", 51)
        return jnp.linspace(v_min, v_max, num_atoms)


def _label_param(path):
    """Label parameters for multi_transform optimizer."""
    key_str = "/".join(str(getattr(k, "key", "")) for k in path)
    if "modules_encoder" in key_str or "modules_target_q_network" in key_str:
        return "frozen"
    elif "modules_q_network" in key_str:
        return "q_network"
    return "frozen"
