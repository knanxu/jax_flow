"""Dueling C51 network for Rainbow DQN speed tuning."""

import flax.linen as nn
import jax.numpy as jnp

from jax_flow.core.utils import get_activation


class DuelingC51Network(nn.Module):
    """Dueling architecture with C51 distributional Q-learning.

    Splits into value and advantage streams, each outputting atom logits.
    Q(s,a) atoms = V(s) atoms + A(s,a) atoms - mean_a(A(s,a)) atoms.
    Final Q(s,a) = sum(softmax(logits) * support).

    Attributes:
        num_actions: Number of discrete speed options.
        num_atoms: Number of atoms for C51 distribution.
        hidden_dims: Hidden layer dimensions for shared + stream layers.
        activation: Activation function name.
        layer_norm: Whether to use layer normalization.
    """

    num_actions: int = 11
    num_atoms: int = 51
    hidden_dims: tuple[int, ...] = (512, 512)
    activation: str = "relu"
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x):
        """Forward pass.

        Args:
            x: Encoded observations (batch, cond_dim).

        Returns:
            Atom logits (batch, num_actions, num_atoms).
        """
        act_fn = get_activation(self.activation)

        # Shared feature layers
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = act_fn(x)

        # Value stream: (batch, num_atoms)
        v = nn.Dense(self.hidden_dims[-1])(x)
        if self.layer_norm:
            v = nn.LayerNorm()(v)
        v = act_fn(v)
        v = nn.Dense(self.num_atoms)(v)  # (batch, num_atoms)

        # Advantage stream: (batch, num_actions * num_atoms)
        a = nn.Dense(self.hidden_dims[-1])(x)
        if self.layer_norm:
            a = nn.LayerNorm()(a)
        a = act_fn(a)
        a = nn.Dense(self.num_actions * self.num_atoms)(a)
        a = a.reshape(
            -1, self.num_actions, self.num_atoms
        )  # (batch, num_actions, num_atoms)

        # Dueling aggregation: Q = V + A - mean(A)
        v = v[:, None, :]  # (batch, 1, num_atoms)
        logits = v + a - jnp.mean(a, axis=1, keepdims=True)

        return logits  # (batch, num_actions, num_atoms)
