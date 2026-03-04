"""Custom TrainState for JAX Flow agents.

Based on qc project's design with apply_loss_fn method.
"""

import functools
from collections.abc import Callable
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

# Helper for non-pytree fields
nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    """A dictionary of modules for managing multiple networks.

    This allows sharing parameters between modules and provides a convenient
    way to access them via the select() method.

    Attributes:
        modules: Dictionary of Flax modules.
    """

    modules: dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with name=None and provide arguments for each
        module in kwargs. Otherwise, call with name=<module_name> and provide
        the arguments for that module.

        Args:
            *args: Positional arguments for the module.
            name: Name of the module to call. If None, initializes all modules.
            **kwargs: Keyword arguments for the module(s).

        Returns:
            If name is None: dict of outputs for each module.
            Otherwise: output of the specified module.
        """
        if name is None:
            # Initialization mode: call all modules
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f"When name is None, kwargs must contain arguments for each "
                    f"module. Got {kwargs.keys()} but expected {self.modules.keys()}"
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, dict):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, (list, tuple)):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        # Normal mode: call specific module
        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Custom train state for JAX Flow agents.

    Based on qc project's design. Manages model parameters, optimizer state,
    and training step in an immutable PyTreeNode.

    Attributes:
        step: Training step counter.
        apply_fn: Apply function of the model.
        model_def: Model definition (non-pytree).
        params: Model parameters.
        tx: Optax optimizer (non-pytree).
        opt_state: Optimizer state.
    """

    step: int
    apply_fn: Callable[..., Any] = nonpytree_field()
    model_def: nn.Module = nonpytree_field()
    params: Any
    tx: optax.GradientTransformation = nonpytree_field()
    opt_state: optax.OptState

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        """Create a new TrainState.

        Args:
            model_def: Flax module definition.
            params: Initial parameters.
            tx: Optax optimizer. If None, no optimizer is used.
            **kwargs: Additional fields to store in the state.

        Returns:
            New TrainState instance.
        """
        opt_state = tx.init(params) if tx is not None else None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, rngs=None, **kwargs):
        """Forward pass through the model.

        When params is not provided, uses stored parameters without flowing
        gradients. To flow gradients, explicitly pass the traced parameters.

        Args:
            *args: Positional arguments for the model.
            params: Parameters to use. If None, uses stored params (no grad flow).
            method: Method name to call in the model. If None, uses default apply.
            rngs: Optional dict of PRNG keys (e.g. {'dropout': key}) for stochastic layers.
            **kwargs: Keyword arguments for the model.

        Returns:
            Model output.
        """
        if params is None:
            params = self.params
        variables = {"params": params}

        method_fn = getattr(self.model_def, method) if method is not None else None

        extra_kwargs = {}
        if rngs is not None:
            extra_kwargs["rngs"] = rngs

        return self.apply_fn(variables, *args, method=method_fn, **extra_kwargs, **kwargs)

    def select(self, name):
        """Helper to select a module from ModuleDict.

        Args:
            name: Name of the module to select.

        Returns:
            Partial function that calls the selected module.
        """
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        """Apply gradients and return updated state.

        Args:
            grads: Gradients to apply.
            **kwargs: Additional fields to update in the state.

        Returns:
            New TrainState with updated parameters and optimizer state.
        """
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_loss_fn(self, loss_fn):
        """Compute gradients from loss function and apply them.

        This is a convenience method that combines gradient computation and
        parameter update in one step. It also computes gradient statistics.

        Args:
            loss_fn: Function that takes params and returns (loss, info_dict).

        Returns:
            Tuple of (new_state, info_dict) where info_dict includes gradient stats.
        """
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)

        # Compute gradient statistics (single tree traversal)
        grad_flat = jnp.concatenate(
            [jnp.reshape(x, -1) for x in jax.tree_util.tree_leaves(grads)], axis=0
        )
        final_grad_max = jnp.max(grad_flat)
        final_grad_min = jnp.min(grad_flat)
        final_grad_norm = jnp.linalg.norm(grad_flat)

        # Add gradient stats to info
        info.update(
            {
                "grad/max": final_grad_max,
                "grad/min": final_grad_min,
                "grad/norm": final_grad_norm,
            }
        )

        return self.apply_gradients(grads=grads), info
