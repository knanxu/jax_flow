"""ACFQL (Action-Chunking Flow Q-Learning) agent.

Offline-to-online RL with flow matching, based on qc project.
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp

from jax_flow.agents.train_state import ModuleDict, TrainState, nonpytree_field


class ACFQLAgent(flax.struct.PyTreeNode):
    """ACFQL agent for offline-to-online RL.

    Components:
    - actor_bc_flow: Multi-step flow policy (BC training)
    - actor_onestep_flow: One-step distilled policy (fast inference)
    - critic: Q-function ensemble
    - target_critic: Target Q-function (Polyak averaged)
    """

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    def _flatten_actions(self, batch_actions):
        """Flatten action sequences for chunking or select first step."""
        if self.config["action_chunking"]:
            return jnp.reshape(batch_actions, (batch_actions.shape[0], -1))
        else:
            return batch_actions[..., 0, :]

    def _effective_action_dim(self):
        """Get effective action dimension with optional chunking."""
        action_dim = self.config["action_dim"]
        if self.config["action_chunking"]:
            return action_dim * self.config["horizon_length"]
        return action_dim

    def _aggregate_q(self, qs):
        """Aggregate Q-ensemble values (min or mean)."""
        if self.config["q_agg"] == "min":
            return qs.min(axis=0)
        else:
            return qs.mean(axis=0)

    def critic_loss(self, batch, grad_params, rng):
        """Compute critic (Q-function) loss."""
        batch_actions = self._flatten_actions(batch["actions"])

        # Sample next actions from policy
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch["next_observations"], rng=sample_rng)

        # Compute target Q
        next_qs = self.network.select("target_critic")(
            batch["next_observations"], actions=next_actions
        )
        next_q = self._aggregate_q(next_qs)

        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_q

        # Compute current Q
        q = self.network.select("critic")(
            batch["observations"], actions=batch_actions, params=grad_params
        )

        critic_loss = jnp.mean(jnp.square(q - target_q))

        return critic_loss, {
            "critic_loss": critic_loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute actor loss (BC flow + distillation + Q)."""
        batch_actions = self._flatten_actions(batch["actions"])
        batch_size, action_dim = batch_actions.shape
        rng, x_rng, t_rng = jax.random.split(rng, 3)

        # ========== BC flow loss ==========
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch_actions
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select("actor_bc_flow")(
            batch["observations"], x_t, t, params=grad_params
        )
        bc_flow_loss = jnp.mean(jnp.square(pred - vel))

        # ========== Distillation + Q loss ==========
        if self.config["actor_type"] == "distill-ddpg":
            rng, noise_rng = jax.random.split(rng)
            noises = jax.random.normal(noise_rng, (batch_size, action_dim))

            # Target: multi-step flow actions (no grad)
            target_flow_actions = self.compute_flow_actions(
                batch["observations"], noises=noises
            )

            # Student: one-step prediction (with grad)
            actor_actions = self.network.select("actor_onestep_flow")(
                batch["observations"], noises, params=grad_params
            )

            distill_loss = jnp.mean(jnp.square(actor_actions - target_flow_actions))

            # Q loss
            actor_actions_clipped = jnp.clip(actor_actions, -1, 1)
            qs = self.network.select("critic")(
                batch["observations"], actions=actor_actions_clipped
            )
            q_loss = -jnp.mean(qs)
        else:
            distill_loss = jnp.zeros(())
            q_loss = jnp.zeros(())

        actor_loss = bc_flow_loss + self.config["alpha"] * distill_loss + q_loss

        return actor_loss, {
            "actor_loss": actor_loss,
            "bc_flow_loss": bc_flow_loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute total loss (critic + actor)."""
        info = {}
        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f"critic/{k}"] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f"actor/{k}"] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Polyak average update for target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            network.params[f"modules_{module_name}"],
            network.params[f"modules_target_{module_name}"],
        )
        # Return new network with updated target params (immutable)
        updated_params = dict(network.params)
        updated_params[f"modules_target_{module_name}"] = new_target_params
        return network.replace(params=updated_params)

    @staticmethod
    def _update(agent, batch):
        """Static update for use with jax.lax.scan."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = agent.target_update(new_network, "critic")
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        """Update agent with a batch of data."""
        return self._update(self, batch)

    @jax.jit
    def compute_flow_actions(self, observations, noises):
        """Compute actions via multi-step Euler integration of BC flow."""
        actions = noises
        flow_steps = self.config["flow_steps"]

        for i in range(flow_steps):
            t = jnp.full((*observations.shape[:-1], 1), i / flow_steps)
            vels = self.network.select("actor_bc_flow")(observations, actions, t)
            actions = actions + vels / flow_steps

        return jnp.clip(actions, -1, 1)

    @jax.jit
    def sample_actions(self, observations, rng=None):
        """Sample actions for environment interaction."""
        rng = rng if rng is not None else self.rng

        if self.config["actor_type"] == "distill-ddpg":
            action_dim = self._effective_action_dim()
            noises = jax.random.normal(
                rng, (*observations.shape[:-1], action_dim)
            )
            actions = self.network.select("actor_onestep_flow")(observations, noises)
            return jnp.clip(actions, -1, 1)

        elif self.config["actor_type"] == "best-of-n":
            action_dim = self._effective_action_dim()
            n_samples = self.config["actor_num_samples"]

            noises = jax.random.normal(
                rng, (*observations.shape[:-1], n_samples, action_dim)
            )
            obs_expanded = jnp.repeat(
                observations[..., None, :], n_samples, axis=-2
            )
            actions = self.compute_flow_actions(obs_expanded, noises)

            # Select best action by Q value
            q = self.network.select("critic")(obs_expanded, actions)
            q = self._aggregate_q(q)
            indices = jnp.argmax(q, axis=-1)

            bshape = indices.shape
            indices = indices.reshape(-1)
            bsize = len(indices)
            actions = jnp.reshape(actions, (-1, n_samples, action_dim))
            actions = actions[jnp.arange(bsize), indices, :]
            return actions.reshape(bshape + (action_dim,))

        else:
            raise ValueError(f"Unknown actor_type: {self.config['actor_type']}")

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new ACFQLAgent.

        Args:
            seed: Random seed.
            ex_observations: Example observations.
            ex_actions: Example actions.
            config: Configuration dict.

        Returns:
            New ACFQLAgent instance.
        """
        from jax_flow.core.utils import create_optimizer
        from jax_flow.networks.mlp import MLP

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        action_dim = ex_actions.shape[-1]
        if config["action_chunking"]:
            full_action_dim = action_dim * config["horizon_length"]
        else:
            full_action_dim = action_dim

        ex_times = ex_actions[..., :1]
        ex_full_actions = jnp.zeros((ex_observations.shape[0], full_action_dim))

        # ========== Define networks ==========

        # Critic (Q-function ensemble)
        from jax_flow.networks.value import Value

        critic_def = Value(
            hidden_dims=tuple(config.get("value_hidden_dims", (512, 512, 512, 512))),
            layer_norm=config.get("layer_norm", True),
            num_ensembles=config.get("num_qs", 2),
        )

        # Actor BC flow (multi-step)
        actor_bc_flow_def = MLP(
            action_dim=full_action_dim,
            hidden_dims=tuple(config.get("actor_hidden_dims", (512, 512, 512, 512))),
            activation="gelu",
            layer_norm=config.get("actor_layer_norm", False),
        )

        # Actor one-step flow (distilled)
        actor_onestep_flow_def = MLP(
            action_dim=full_action_dim,
            hidden_dims=tuple(config.get("actor_hidden_dims", (512, 512, 512, 512))),
            activation="gelu",
            layer_norm=config.get("actor_layer_norm", False),
        )

        # ========== Build ModuleDict ==========
        networks = {
            "actor_bc_flow": actor_bc_flow_def,
            "actor_onestep_flow": actor_onestep_flow_def,
            "critic": critic_def,
            "target_critic": copy.deepcopy(critic_def),
        }
        network_args = {
            "actor_bc_flow": (ex_observations, ex_full_actions, ex_times),
            "actor_onestep_flow": (ex_observations, ex_full_actions),
            "critic": (ex_observations, ex_full_actions),
            "target_critic": (ex_observations, ex_full_actions),
        }

        network_def = ModuleDict(networks)

        # Optimizer
        tx = create_optimizer(
            lr=config.get("lr", 3e-4),
            weight_decay=config.get("weight_decay", 0.0),
            schedule_type=config.get("schedule_type", "constant"),
            warmup_steps=config.get("warmup_steps", 0),
            total_steps=config.get("gradient_steps", 100000),
        )

        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=tx)

        # Copy critic params to target (immutable)
        updated_params = dict(network.params)
        updated_params["modules_target_critic"] = network.params["modules_critic"]
        network = network.replace(params=updated_params)

        # Store dimensions in config
        config["action_dim"] = action_dim
        config["ob_dims"] = ex_observations.shape

        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(**config),
        )
