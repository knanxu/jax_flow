"""Kitchen environment for jax_flow.

Requires: pip install jax_flow[kitchen]
"""


def __getattr__(name):
    if name == "make_kitchen_env":
        from jax_flow.envs.kitchen.kitchen_wrapper import make_kitchen_env

        return make_kitchen_env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["make_kitchen_env"]
