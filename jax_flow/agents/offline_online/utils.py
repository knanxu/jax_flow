"""Shared utilities for offline-to-online RL agents (ACFQL, DQC)."""

import jax
import jax.numpy as jnp


def expectile_loss(pred, target, expectile=0.7):
    """IQL-style asymmetric L2 loss for expectile regression.

    When expectile > 0.5, upweights cases where target > pred,
    pushing the prediction toward the upper quantile of the target distribution.

    Args:
        pred: Predicted values. Shape: (batch,).
        target: Target values. Shape: (batch,).
        expectile: Asymmetry coefficient. 0.5 = symmetric MSE.

    Returns:
        Scalar loss.
    """
    diff = target - pred
    weight = jnp.where(diff > 0, expectile, 1.0 - expectile)
    return jnp.mean(weight * diff ** 2)


def reward_transform(rewards):
    """Transform robomimic rewards from {0, 1} to {-1, 0}.

    Penalty formulation: agent receives -1 at every non-success step,
    0 on success. This encourages the agent to reach the goal quickly.

    Args:
        rewards: Raw rewards. Shape: (batch, 1) or (batch,).

    Returns:
        Transformed rewards, same shape.
    """
    return rewards - 1.0


def flatten_action_chunk(actions):
    """Flatten action chunk for critic input.

    Args:
        actions: (batch, chunk_len, action_dim)

    Returns:
        (batch, chunk_len * action_dim)
    """
    return actions.reshape(actions.shape[0], -1)


def unflatten_action_chunk(flat_actions, chunk_len, action_dim):
    """Unflatten action chunk back to sequence form.

    Args:
        flat_actions: (batch, chunk_len * action_dim)
        chunk_len: Number of steps in the chunk.
        action_dim: Action dimension per step.

    Returns:
        (batch, chunk_len, action_dim)
    """
    return flat_actions.reshape(flat_actions.shape[0], chunk_len, action_dim)


def aggregate_q(qs, mode="min"):
    """Aggregate Q-ensemble values.

    Args:
        qs: Q-values from ensemble. Shape: (num_ensembles, batch).
        mode: 'min' or 'mean'.

    Returns:
        Aggregated Q-values. Shape: (batch,).
    """
    if mode == "min":
        return qs.min(axis=0)
    return qs.mean(axis=0)
