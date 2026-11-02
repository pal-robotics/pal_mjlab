"""The Kangaroo arm and wrist joints turn about the axes of PAL's URDF.

pal_sea_arm (humble-devel) urdf/arm/arm_properties/humanoid: 7dof_arm.urdf.xacro has arm_5 and arm_7 about +z
(and arm_6 about -z), 5dof_arm.urdf.xacro arm_5 about +z; the RH8D wrist roll is mirrored, -z on the left and +z
on the right. A sign flip turns the arm the other way for the same joint value: retargeted motions and policies
no longer match the robot (docs/supermimic/audit/kangaroo_7dof.md).
"""

import mujoco
import pytest
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  KANGAROO_GRIPPERS_XML,
  KANGAROO_HANDS_XML,
)

GRIPPERS = {
  **{f"arm_{s}_5_joint": (0, 0, 1) for s in ("left", "right")},
  **{f"arm_{s}_6_joint": (0, 0, -1) for s in ("left", "right")},
  **{f"arm_{s}_7_joint": (0, 0, 1) for s in ("left", "right")},
}
HANDS = {
  "arm_left_5_joint": (0, 0, 1),
  "arm_right_5_joint": (0, 0, 1),
  "hand_left_palm_roll_joint": (0, 0, -1),
  "hand_right_palm_roll_joint": (0, 0, 1),
}


@pytest.mark.parametrize(
  ("xml", "axes"), [(KANGAROO_GRIPPERS_XML, GRIPPERS), (KANGAROO_HANDS_XML, HANDS)]
)
def test_arm_axes_match_urdf(xml, axes):
  spec = mujoco.MjSpec.from_file(str(xml))
  for name, axis in axes.items():
    assert tuple(spec.joint(name).axis) == axis, name
