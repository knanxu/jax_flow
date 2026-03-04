"""Value network (critic) for Q-learning."""

import flax.linen as nn
import jax.numpy as jnp

from jax_flow.core.utils import get_activation


class Value(nn.Module):
    """Value/critic network for Q-learning.

    Supports ensemble of Q-functions for better stability.
    """

    hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    layer_norm: bool = True
    num_ensembles: int = 2
    activation: str = "gelu"

    @nn.compact
    def __call__(self, observations, actions):
        """Compute Q-values.

        Args:
            observations: Observations. Shape: (batch, obs_dim)
            actions: Actions. Shape: (batch, action_dim)

        Returns:
            Q-values. Shape: (num_ensembles, batch) if ensemble, else (batch,)
        """
        # Concatenate observations and actions
        x = jnp.concatenate([observations, actions], axis=-1)

        # Activation function
        act_fn = get_activation(self.activation)

        # Ensemble via vmap
        if self.num_ensembles > 1:

            def single_q(x):
                for dim in self.hidden_dims:
                    x = nn.Dense(dim)(x)
                    if self.layer_norm:
                        x = nn.LayerNorm()(x)
                    x = act_fn(x)
                return nn.Dense(1)(x).squeeze(-1)

            # Vmap over ensemble dimension
            ensemble_q = nn.vmap(
                single_q,
                variable_axes={"params": 0},
                split_rngs={"params": True},
                in_axes=None,
                out_axes=0,
                axis_size=self.num_ensembles,
            )
            return ensemble_q(x)
        else:
            # Single Q-function
            for dim in self.hidden_dims:
                x = nn.Dense(dim)(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = act_fn(x)
            return nn.Dense(1)(x).squeeze(-1)
