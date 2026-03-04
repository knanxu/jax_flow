"""Type definitions for JAX Flow."""

from typing import Any, NamedTuple

import flax
from jaxtyping import Array, Float, PRNGKeyArray

# Type aliases
PRNGKey = PRNGKeyArray
Params = flax.core.FrozenDict[str, Any]
InfoDict = dict[str, float]


class Batch(NamedTuple):
    """Batch of training data."""

    observations: Float[Array, "batch obs_steps obs_dim"]
    actions: Float[Array, "batch horizon act_dim"]
    rewards: Float[Array, "batch"] | None = None
    dones: Float[Array, "batch"] | None = None
    next_observations: Float[Array, "batch obs_steps obs_dim"] | None = None
    valid: Float[Array, "batch horizon"] | None = (
        None  # Validity mask for action chunking
    )


__all__ = [
    "Array",
    "Float",
    "PRNGKey",
    "Params",
    "InfoDict",
    "Batch",
]
