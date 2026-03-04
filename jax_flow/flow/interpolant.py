"""Interpolant for flow matching.

Defines the path from noise (x0) to data (x1).
"""

import jax.numpy as jnp


class Interpolant:
    """Stochastic interpolant with different interpolation types."""

    def __init__(self, interp_type="linear"):
        """Initialize interpolant.

        Args:
            interp_type: Type of interpolation ('linear' or 'trig').
        """
        self.interp_type = interp_type

        if interp_type == "linear":
            self.alpha = lambda t: 1.0 - t
            self.beta = lambda t: t
            self.alpha_dot = lambda t: -1.0
            self.beta_dot = lambda t: 1.0
        elif interp_type == "trig":
            self.alpha = lambda t: jnp.cos(t * jnp.pi / 2)
            self.beta = lambda t: jnp.sin(t * jnp.pi / 2)
            self.alpha_dot = lambda t: -0.5 * jnp.pi * jnp.sin(t * jnp.pi / 2)
            self.beta_dot = lambda t: 0.5 * jnp.pi * jnp.cos(t * jnp.pi / 2)
        else:
            raise ValueError(f"Unknown interpolant type: {interp_type}")

    def interpolate(self, t, x0, x1):
        """Compute interpolated point x_t = alpha(t) * x0 + beta(t) * x1.

        Args:
            t: Time in [0, 1]. Shape: (batch,) or (batch, 1)
            x0: Starting point (noise). Shape: (batch, ...)
            x1: Ending point (data). Shape: (batch, ...)

        Returns:
            Interpolated point x_t. Shape: (batch, ...)
        """
        # Ensure t has correct shape for broadcasting
        t = jnp.atleast_1d(t)
        if t.ndim == 1:
            # Add dimensions to match x0/x1
            for _ in range(x0.ndim - 1):
                t = t[..., None]

        return self.alpha(t) * x0 + self.beta(t) * x1

    def velocity(self, t, x0, x1):
        """Compute velocity dx_t/dt = alpha_dot(t) * x0 + beta_dot(t) * x1.

        Args:
            t: Time in [0, 1]. Shape: (batch,) or (batch, 1)
            x0: Starting point (noise). Shape: (batch, ...)
            x1: Ending point (data). Shape: (batch, ...)

        Returns:
            Velocity at time t. Shape: (batch, ...)
        """
        # Ensure t has correct shape for broadcasting
        t = jnp.atleast_1d(t)
        if t.ndim == 1:
            for _ in range(x0.ndim - 1):
                t = t[..., None]

        return self.alpha_dot(t) * x0 + self.beta_dot(t) * x1


def create_interpolant(interp_type="linear"):
    """Factory function to create interpolant.

    Args:
        interp_type: Type of interpolation ('linear' or 'trig').

    Returns:
        Interpolant instance.
    """
    return Interpolant(interp_type=interp_type)
