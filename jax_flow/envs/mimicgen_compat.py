"""MimicGen compatibility shim for robosuite 1.5.

MimicGen 1.0.1 was built for robosuite 1.4.x which had SingleArmEnv.
robosuite 1.5 removed that class and renamed mount_types -> base_types,
robot.controller -> robot.composite_controller + robot.part_controllers.

This module injects shims so that `import mimicgen` succeeds and all
MimicGen environments get registered into robosuite's registry.
"""

import sys
import types

_patched = False

# MimicGen-specific environments (not built into robosuite)
MIMICGEN_ENVS = {
    "Threading",
    "Threading_D0",
    "Threading_D1",
    "Threading_D2",
    "Coffee",
    "Coffee_D0",
    "Coffee_D1",
    "Coffee_D2",
    "CoffeePreparation",
    "CoffeePreparation_D0",
    "CoffeePreparation_D1",
    "StackThree",
    "StackThree_D0",
    "StackThree_D1",
    "ThreePieceAssembly",
    "ThreePieceAssembly_D0",
    "ThreePieceAssembly_D1",
    "ThreePieceAssembly_D2",
    "MugCleanup",
    "MugCleanup_D0",
    "MugCleanup_D1",
    "MugCleanup_O1",
    "MugCleanup_O2",
    "HammerCleanup",
    "HammerCleanup_D0",
    "HammerCleanup_D1",
    "Kitchen",
    "Kitchen_D0",
    "Kitchen_D1",
    # Variant names from dataset env_meta (before _D0/_D1 stripping)
    "Stack_D0",
    "Stack_D1",
    "NutAssembly_D0",
    "Square_D0",
    "Square_D1",
    "Square_D2",
    "PickPlace_D0",
}


def ensure_mimicgen_compat():
    """Inject robosuite 1.5 compatibility shims and import mimicgen.

    Safe to call multiple times — patches are applied only once.
    Must be called BEFORE any code that needs MimicGen environments.
    """
    global _patched
    if _patched:
        return
    _patched = True

    # --- Shim 1: Restore SingleArmEnv import path ---
    # MimicGen imports from robosuite.environments.manipulation.single_arm_env
    # which no longer exists in robosuite 1.5. We create a shim module that
    # provides SingleArmEnv as a subclass of ManipulationEnv with mount_types
    # -> base_types parameter remapping.
    shim_module_name = "robosuite.environments.manipulation.single_arm_env"
    if shim_module_name not in sys.modules:
        from robosuite.environments.manipulation.manipulation_env import (
            ManipulationEnv,
        )

        class SingleArmEnv(ManipulationEnv):
            """Backward-compat shim for robosuite < 1.5."""

            def __init__(self, **kwargs):
                if "mount_types" in kwargs:
                    kwargs["base_types"] = kwargs.pop("mount_types")
                super().__init__(**kwargs)

        shim = types.ModuleType(shim_module_name)
        shim.SingleArmEnv = SingleArmEnv
        sys.modules[shim_module_name] = shim

    # --- Shim 2: Backward-compat robot.controller + controller.eef_name ---
    # MimicGen's Coffee env accesses robot.controller.eef_name which was
    # removed in robosuite 1.5 (replaced by composite_controller / part_controllers).
    from robosuite.robots import FixedBaseRobot

    if not getattr(FixedBaseRobot, "_mimicgen_controller_patched", False):
        _orig_load_controller = FixedBaseRobot._load_controller

        def _patched_load_controller(self):
            _orig_load_controller(self)
            # Add backward-compat controller attribute
            if not hasattr(self, "controller") or self.controller is None:
                arm_ctrl = self.part_controllers.get("right")
                if arm_ctrl is not None and not hasattr(arm_ctrl, "eef_name"):
                    # Reconstruct the old eef_name (grip site name)
                    gripper_name = (
                        list(self.gripper.keys())[0] if self.gripper else "right"
                    )
                    grip_prefix = (
                        self.gripper[gripper_name].naming_prefix
                        if self.gripper
                        else ""
                    )
                    arm_ctrl.eef_name = f"{grip_prefix}grip_site"
                self.controller = arm_ctrl

        FixedBaseRobot._load_controller = _patched_load_controller
        FixedBaseRobot._mimicgen_controller_patched = True

    # --- Import mimicgen to trigger environment registration ---
    try:
        import mimicgen  # noqa: F401
    except ImportError:
        print(
            "Warning: mimicgen not installed. "
            "Install with: pip install -e /path/to/mimicgen"
        )
