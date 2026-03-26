"""Value network (critic) for Q-learning."""

import flax.linen as nn
import jax.numpy as jnp

from jax_flow.core.utils import get_activation


class _QHead(nn.Module):
    """Single Q-function MLP head."""

    hidden_dims: tuple[int, ...]
    layer_norm: bool
    activation: str

    @nn.compact
    def __call__(self, x):
        act_fn = get_activation(self.activation)
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = act_fn(x)
        return nn.Dense(1)(x).squeeze(-1)


class Value(nn.Module):
    """Value/critic network for Q-learning.

    Supports ensemble of Q-functions for better stability.
    Optionally embeds an independent encoder for raw observation input.
    """

    encoder: nn.Module | None = None
    hidden_dims: tuple[int, ...] = (512, 512, 512, 512)
    layer_norm: bool = True
    num_ensembles: int = 2
    activation: str = "gelu"

    @nn.compact
    def __call__(self, observations, actions=None, training=False):
        """Compute Q-values Q(s,a) or state-values V(s).

        Args:
            observations: Observations — raw obs if encoder is set, else pre-encoded features.
            actions: Actions. Shape: (batch, action_dim). If None, computes V(s).
            training: Training mode flag (passed to encoder if present).

        Returns:
            Values. Shape: (num_ensembles, batch) if ensemble, else (batch,)
        """
        if self.encoder is not None:
            obs_encoded = self.encoder(observations, training=training)
        else:
            obs_encoded = observations

        if actions is not None:
            x = jnp.concatenate([obs_encoded, actions], axis=-1)
        else:
            x = obs_encoded

        if self.num_ensembles > 1:
            EnsembleQ = nn.vmap(
                _QHead,
                variable_axes={"params": 0},
                split_rngs={"params": True},
                in_axes=None,
                out_axes=0,
                axis_size=self.num_ensembles,
            )
            return EnsembleQ(
                hidden_dims=self.hidden_dims,
                layer_norm=self.layer_norm,
                activation=self.activation,
            )(x)
        else:
            return _QHead(
                hidden_dims=self.hidden_dims,
                layer_norm=self.layer_norm,
                activation=self.activation,
            )(x)
