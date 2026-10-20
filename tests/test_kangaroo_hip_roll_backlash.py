"""Checks for the Kangaroo hip roll backlash variant."""

import math

import mujoco
import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg

TASK = "Mjlab-Velocity-Flat-Pal-Kangaroo-Lower-Body"


def test_backlash_joints_added_unactuated():
  base = Scene(load_env_cfg(TASK).scene, "cpu").compile()
  model = Scene(load_env_cfg(f"{TASK}-Backlash").scene, "cpu").compile()
  assert model.njnt == base.njnt + 2
  assert model.nu == base.nu
  for side in ("left", "right"):
    jnt = model.joint(f"robot/leg_{side}_3_backlash_joint")
    assert model.jnt_bodyid[jnt.id] == model.body(f"robot/leg_{side}_3_link").id
    assert jnt.type == mujoco.mjtJoint.mjJNT_HINGE
    assert list(jnt.axis) == [1, 0, 0]
    assert math.isclose(jnt.range[1], math.radians(1.0))
