"""General utilities for JAX Flow."""

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn


def get_activation(name: str) -> Callable:
    """Get activation function by name."""
    activations = {
        "relu": nn.relu,
        "gelu": nn.gelu,
        "silu": nn.silu,
        "swish": nn.swish,
        "tanh": nn.tanh,
        "sigmoid": nn.sigmoid,
        "elu": nn.elu,
        "leaky_relu": functools.partial(nn.leaky_relu, negative_slope=0.2),
    }

    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")

    return activations[name]


def create_learning_rate_schedule(
    base_lr: float,
    schedule_type: str = "constant",
    warmup_steps: int = 0,
    total_steps: int = 100000,
    min_lr: float = 0.0,
) -> optax.Schedule:
    """Create learning rate schedule."""
    if schedule_type == "constant":
        schedule = optax.constant_schedule(base_lr)
    elif schedule_type == "linear":
        schedule = optax.linear_schedule(
            init_value=base_lr,
            end_value=min_lr,
            transition_steps=max(1, total_steps - warmup_steps),
        )
    elif schedule_type == "cosine":
        schedule = optax.cosine_decay_schedule(
            init_value=base_lr,
            decay_steps=max(1, total_steps - warmup_steps),
            alpha=min_lr / base_lr,
        )
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")

    if warmup_steps > 0:
        warmup_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=base_lr,
            transition_steps=warmup_steps,
        )
        schedule = optax.join_schedules(
            schedules=[warmup_schedule, schedule],
            boundaries=[warmup_steps],
        )

    return schedule


def at_least_ndim(x: jnp.ndarray, ndim: int) -> jnp.ndarray:
    """Ensure array has at least ndim dimensions."""
    while x.ndim < ndim:
        x = x[..., None]
    return x


def create_optimizer(
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    schedule_type: str = "constant",
    warmup_steps: int = 0,
    total_steps: int = 100000,
) -> optax.GradientTransformation:
    """Create optimizer with optional learning rate schedule.

    Args:
        lr: Base learning rate.
        weight_decay: Weight decay coefficient.
        schedule_type: Learning rate schedule type ('constant', 'linear', 'cosine').
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps.

    Returns:
        Optax optimizer.
    """
    # Create learning rate schedule
    if schedule_type != "constant" or warmup_steps > 0:
        lr_schedule = create_learning_rate_schedule(
            base_lr=lr,
            schedule_type=schedule_type,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
    else:
        lr_schedule = lr

    # Create optimizer
    if weight_decay > 0.0:
        return optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
    else:
        return optax.adam(learning_rate=lr_schedule)


def get_batch_size(obs):
    """Get batch size from observations (array or dict of arrays)."""
    if isinstance(obs, dict):
        return next(iter(obs.values())).shape[0]
    return obs.shape[0]


def make_param_labels(params):
    """Generate parameter label tree for optax.multi_transform.

    Labels 'modules_encoder' subtree as 'encoder', everything else as 'flow'.
    ModuleDict names submodules as 'modules_{key}' under Flax auto-naming.
    """
    return jax.tree_util.tree_map_with_path(
        lambda path, _: (
            "encoder"
            if any(
                str(getattr(k, "key", "")).startswith("modules_encoder") for k in path
            )
            else "flow"
        ),
        params,
    )


__all__ = [
    "get_activation",
    "create_learning_rate_schedule",
    "create_optimizer",
    "make_param_labels",
    "at_least_ndim",
    "get_batch_size",
]
