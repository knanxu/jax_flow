"""Environment utilities."""

from jax_flow.envs.residual_wrapper import ResidualEnvWrapper
from jax_flow.envs.robomimic_env import (
    ActionChunkingWrapper,
    FrameStackWrapper,
    RobomimicImageWrapper,
    RobomimicWrapper,
    make_robomimic_env,
)


def make_env(task_source, **kwargs):
    """Dispatch environment creation based on task_source.

    Args:
        task_source: 'robomimic', 'pusht', or 'kitchen'.
        **kwargs: Arguments passed to the environment factory.

    Returns:
        Wrapped environment.
    """
    if task_source == "pusht":
        from jax_flow.envs.pusht.pusht_wrapper import make_pusht_env
        return make_pusht_env(**kwargs)
    elif task_source == "kitchen":
        from jax_flow.envs.kitchen.kitchen_wrapper import make_kitchen_env
        return make_kitchen_env(**kwargs)
    else:
        return make_robomimic_env(**kwargs)


__all__ = [
    "RobomimicWrapper",
    "RobomimicImageWrapper",
    "FrameStackWrapper",
    "ActionChunkingWrapper",
    "ResidualEnvWrapper",
    "make_robomimic_env",
    "make_env",
]
