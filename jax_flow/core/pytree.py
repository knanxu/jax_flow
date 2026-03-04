"""PyTree utilities for JAX Flow."""

from typing import Any

import jax
import jax.numpy as jnp


def tree_norm(tree: Any, ord: int = 2) -> float:
    """Compute the norm of a PyTree.

    Args:
        tree: PyTree to compute norm of
        ord: Order of the norm (default: 2 for L2 norm)

    Returns:
        Norm of the tree
    """
    leaves = jax.tree_util.tree_leaves(tree)
    if ord == 2:
        return jnp.sqrt(sum(jnp.sum(x**2) for x in leaves))
    elif ord == 1:
        return sum(jnp.sum(jnp.abs(x)) for x in leaves)
    elif ord == float("inf"):
        return max(jnp.max(jnp.abs(x)) for x in leaves)
    else:
        raise ValueError(f"Unsupported norm order: {ord}")


def tree_size(tree: Any) -> int:
    """Count total number of parameters in a PyTree.

    Args:
        tree: PyTree to count parameters of

    Returns:
        Total number of parameters
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(tree))


def tree_stack(trees: list[Any]) -> Any:
    """Stack a list of PyTrees along a new axis.

    Args:
        trees: List of PyTrees with the same structure

    Returns:
        Stacked PyTree
    """
    return jax.tree_map(lambda *xs: jnp.stack(xs), *trees)


def tree_unstack(tree: Any) -> list[Any]:
    """Unstack a PyTree along the first axis.

    Args:
        tree: PyTree to unstack

    Returns:
        List of PyTrees
    """
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    n = leaves[0].shape[0]
    return [
        jax.tree_util.tree_unflatten(treedef, [leaf[i] for leaf in leaves])
        for i in range(n)
    ]


def tree_mean(trees: list[Any]) -> Any:
    """Compute the mean of a list of PyTrees.

    Args:
        trees: List of PyTrees with the same structure

    Returns:
        Mean PyTree
    """
    return jax.tree_map(lambda *xs: jnp.mean(jnp.stack(xs), axis=0), *trees)


def tree_zeros_like(tree: Any) -> Any:
    """Create a PyTree of zeros with the same structure.

    Args:
        tree: PyTree to match structure of

    Returns:
        PyTree of zeros
    """
    return jax.tree_map(jnp.zeros_like, tree)


def tree_ones_like(tree: Any) -> Any:
    """Create a PyTree of ones with the same structure.

    Args:
        tree: PyTree to match structure of

    Returns:
        PyTree of ones
    """
    return jax.tree_map(jnp.ones_like, tree)


__all__ = [
    "tree_norm",
    "tree_size",
    "tree_stack",
    "tree_unstack",
    "tree_mean",
    "tree_zeros_like",
    "tree_ones_like",
]
