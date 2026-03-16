"""Push-T environment for jax_flow.

Requires: pip install jax_flow[pusht]
"""


def __getattr__(name):
    if name == "PushTEnv":
        from jax_flow.envs.pusht.pusht_env import PushTEnv

        return PushTEnv
    elif name == "PushTImageEnv":
        from jax_flow.envs.pusht.pusht_image_env import PushTImageEnv

        return PushTImageEnv
    elif name == "PushTKeypointsEnv":
        from jax_flow.envs.pusht.pusht_keypoints_env import PushTKeypointsEnv

        return PushTKeypointsEnv
    elif name == "make_pusht_env":
        from jax_flow.envs.pusht.pusht_wrapper import make_pusht_env

        return make_pusht_env
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PushTEnv",
    "PushTImageEnv",
    "PushTKeypointsEnv",
    "make_pusht_env",
]
