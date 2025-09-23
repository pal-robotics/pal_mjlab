"""Unitree Go1 constants."""

from pathlib import Path

import mujoco

from mjlab_kangaroo import mjlab_kangaroo_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.os import update_assets
from mjlab.utils.spec_editor import ActuatorCfg, CollisionCfg

##
# MJCF and assets.
##

KANG_XML: Path = (
  mjlab_kangaroo_SRC_PATH / "robots" / "pal_kangaroo" / "xmls" / "kangaroo.xml"
)
assert KANG_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, KANG_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(KANG_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##


KANG_LEG_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=[
    "leg_.*_1_joint",
    "leg_.*_2_joint",
    "leg_.*_3_joint",
    "leg_.*_length_joint",
    "leg_.*_4_joint",
    "leg_.*_5_joint",
    "leg_.*_femur_joint",
    "leg_.*_knee_joint",
  ],
  effort_limit_sim={
    "leg_.*_1_joint": 80,
    "leg_.*_2_joint": 230,
    "leg_.*_3_joint": 139,
    "leg_.*_length_joint": 1100,
    "leg_.*_4_joint": 140,
    "leg_.*_5_joint": 82,
    "leg_.*_femur_joint": 100,
    "leg_.*_knee_joint": 100,
  },
  velocity_limit_sim={
    "leg_.*_1_joint": 3.87,
    "leg_.*_2_joint": 3.87,
    "leg_.*_3_joint": 3.87,
    "leg_.*_length_joint": 10,
    "leg_.*_4_joint": 3.87,
    "leg_.*_5_joint": 3.87,
    "leg_.*_femur_joint": 3.87,
    "leg_.*_knee_joint": 3.87,
  },
  armature=0.01,
  stiffness={
    "leg_.*_1_joint": 40,
    "leg_.*_2_joint": 100,
    "leg_.*_3_joint": 100,
    "leg_.*_length_joint": 1100,
    "leg_.*_4_joint": 100,
    "leg_.*_5_joint": 40,
    "leg_.*_femur_joint": 0,
    "leg_.*_knee_joint": 0,
  },
  damping={
    "leg_.*_1_joint": 2,
    "leg_.*_2_joint": 5,
    "leg_.*_3_joint": 5,
    "leg_.*_length_joint": 150, #55,
    "leg_.*_4_joint": 5,
    "leg_.*_5_joint": 2,
    "leg_.*_femur_joint": 0,
    "leg_.*_knee_joint": 0,
  },
)
KANG_ARM_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=[
    "arm_.*_1_joint",
    "arm_.*_2_joint",
    "arm_.*_3_joint",
    "arm_.*_4_joint",
  ],
  armature=0.01,
  velocity_limit_sim={
    "arm_.*_1_joint": 1.95,
    "arm_.*_2_joint": 1.95,
    "arm_.*_3_joint": 2.35,
    "arm_.*_4_joint": 2.35,
  },
  effort_limit_sim=43.0,
  stiffness={
    "arm_.*_1_joint": 100,
    "arm_.*_2_joint": 100,
    "arm_.*_3_joint": 100,
    "arm_.*_4_joint": 100,
  },
  damping={
    "arm_.*_1_joint": 10,
    "arm_.*_2_joint": 10,
    "arm_.*_3_joint": 10,
    "arm_.*_4_joint": 10,
  },
)
KANG_TORSO_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=[
    "pelvis_1_joint",
    "pelvis_2_joint",
  ],
  effort_limit_sim=100.0,
  velocity_limit_sim=3.14,
  armature=0.01,
  stiffness={
    "pelvis_1_joint": 80,
    "pelvis_2_joint": 40,
  },
  damping={
    "pelvis_1_joint": 4,
    "pelvis_2_joint": 2,
  },
)
##
# Keyframes.
##


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.0),
  joint_pos={
    # legs
    "leg_.*_1_joint": 0.0,
    "leg_.*_2_joint": 0.05,
    "leg_.*_3_joint": 0.0,
    "leg_.*_length_joint": 0.6832,
    "leg_.*_4_joint": -0.05,
    "leg_.*_5_joint": 0.0,
    "leg_.*_femur_joint": 1.1172,
    "leg_.*_knee_joint": 2.2345,
    # arms
    # "arm_.*": 0, # need to do that as target is set to 0, possible workaround is to set the buffer to init position in the reset event though
    "arm_left_1_joint": 0.24,
    "arm_right_1_joint": -0.24,
    "arm_.*_2_joint": 1.32,
    "arm_left_3_joint": 1.57,
    "arm_right_3_joint": -1.57,
    "arm_.*_4_joint": 0.8,
    # torso
    "pelvis_1_joint": 0,
    "pelvis_2_joint": 0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[left][right]_foot_collision$"

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=[_foot_regex],
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions, excluding self collisions.
# Foot collisions are given custom condim, friction and solimp.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=[".*_collision"],
  condim={_foot_regex: 3},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

##
# Final config.
##

GO1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO1_HIP_ACTUATOR_CFG,
    GO1_KNEE_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)

GO1_ROBOT_CFG = EntityCfg(
  init_state=INIT_STATE,
  collisions=(FULL_COLLISION,),
  spec_fn=get_spec,
  articulation=GO1_ARTICULATION,
)

GO1_ACTION_SCALE: dict[str, float] = {}
for a in GO1_ARTICULATION.actuators:
  e = a.effort_limit
  s = a.stiffness
  names = a.joint_names_expr
  if not isinstance(e, dict):
    e = {n: e for n in names}
  if not isinstance(s, dict):
    s = {n: s for n in names}
  for n in names:
    if n in e and n in s and s[n]:
      GO1_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
