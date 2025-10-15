"""Unitree Go1 constants."""

from pathlib import Path

import mujoco

from mjlab_kangaroo import mjlab_kangaroo_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import ActuatorCfg, CollisionCfg

##
# MJCF and assets.
##

TALOS_XML: Path = (
  mjlab_kangaroo_SRC_PATH / "robots" / "pal_talos" / "xmls" / "talos.xml"
)
assert TALOS_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, TALOS_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(TALOS_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0
REDUCTION_RATIO = 100
factor = 0.01
factor_leg = 0.01
# ---- arm joints params

# joints inertia
ARM_1_MOTOR_INERTIA = 0.000207288
ARM_2_MOTOR_INERTIA = 0.000140493
ARM_34_MOTOR_INERTIA = 8.60398e-05
ARM_567_MOTOR_INERTIA = 1.e-05 # this one has not been properly identified
# joints armature (reflected inertia)
ARM_1_ARMATURE = factor * ARM_1_MOTOR_INERTIA * REDUCTION_RATIO ** 2
ARM_2_ARMATURE = factor * ARM_2_MOTOR_INERTIA * REDUCTION_RATIO ** 2
ARM_34_ARMATURE = factor * ARM_34_MOTOR_INERTIA * REDUCTION_RATIO ** 2
ARM_567_ARMATURE = factor * ARM_567_MOTOR_INERTIA * REDUCTION_RATIO ** 2
# joints effort limit
ARM_1_EFFORT_LIMIT = 100.0
ARM_2_EFFORT_LIMIT = 100.0
ARM_34_EFFORT_LIMIT = 70.0
ARM_567_EFFORT_LIMIT = 8.0
# joints stiffness
ARM_1_STIFFNESS = ARM_1_ARMATURE * NATURAL_FREQ ** 2
ARM_2_STIFFNESS = ARM_2_ARMATURE * NATURAL_FREQ ** 2
ARM_34_STIFFNESS = ARM_34_ARMATURE * NATURAL_FREQ ** 2
ARM_567_STIFFNESS = ARM_567_ARMATURE * NATURAL_FREQ ** 2
# joints damping
ARM_1_DAMPING = 2.0 * DAMPING_RATIO * ARM_1_ARMATURE * NATURAL_FREQ
ARM_2_DAMPING = 2.0 * DAMPING_RATIO * ARM_2_ARMATURE * NATURAL_FREQ
ARM_34_DAMPING = 2.0 * DAMPING_RATIO * ARM_34_ARMATURE * NATURAL_FREQ
ARM_567_DAMPING = 2.0 * DAMPING_RATIO * ARM_567_ARMATURE * NATURAL_FREQ

# ---- torso joints params

# joints inertia
TORSO_MOTOR_INERTIA = 0.000207288
# joints armature (reflected inertia)
TORSO_ARMATURE = factor * TORSO_MOTOR_INERTIA * REDUCTION_RATIO ** 2
# joints effort limit
TORSO_EFFORT_LIMIT = 200.0
# joints stiffness
TORSO_STIFFNESS = TORSO_ARMATURE * NATURAL_FREQ ** 2
# joints damping
TORSO_DAMPING = 2.0 * DAMPING_RATIO * TORSO_ARMATURE * NATURAL_FREQ

# ---- head joints params

# joints inertia
HEAD_MOTOR_INERTIA = 1.e-05 # this one has not been properly identified
# joints reduction ratio
HEAD_REDUCTION_RATIO = 144
# joints armature (reflected inertia)
HEAD_ARMATURE = HEAD_MOTOR_INERTIA * HEAD_REDUCTION_RATIO ** 2
# joints effort limit
HEAD_1_EFFORT_LIMIT = 8.0
HEAD_2_EFFORT_LIMIT = 4.0
# joints stiffness
HEAD_STIFFNESS = HEAD_ARMATURE * NATURAL_FREQ ** 2
# joints damping limit
HEAD_DAMPING = 2.0 * DAMPING_RATIO * HEAD_ARMATURE * NATURAL_FREQ

# ---- leg joints params

# joints inertia
LEG_16_MOTOR_INERTIA = 8.60398e-05
LEG_235_MOTOR_INERTIA = 0.000207288
LEG_4_MOTOR_INERTIA = 0.000195461
# joints reduction ratio
LEG_1_REDUCTION_RATIO = 150
LEG_26_REDUCTION_RATIO = 101
LEG_4_REDUCTION_RATIO = 144
# joints armature (reflected inertia)
LEG_1_ARMATURE = factor_leg * LEG_16_MOTOR_INERTIA * LEG_1_REDUCTION_RATIO ** 2
LEG_2_ARMATURE = factor_leg * LEG_235_MOTOR_INERTIA * LEG_26_REDUCTION_RATIO ** 2
LEG_35_ARMATURE = factor_leg * LEG_235_MOTOR_INERTIA * REDUCTION_RATIO ** 2
LEG_4_ARMATURE = factor_leg * LEG_4_MOTOR_INERTIA * LEG_4_REDUCTION_RATIO ** 2
LEG_6_ARMATURE = factor_leg * LEG_16_MOTOR_INERTIA * LEG_26_REDUCTION_RATIO ** 2
print(LEG_1_ARMATURE)
print(LEG_2_ARMATURE)
print(LEG_35_ARMATURE)
print(LEG_4_ARMATURE)
print(LEG_6_ARMATURE)
# joints effort limit
LEG_1_EFFORT_LIMIT = 100.0
LEG_2_EFFORT_LIMIT = 160.0
LEG_35_EFFORT_LIMIT = 160.0
LEG_4_EFFORT_LIMIT = 400.0
LEG_6_EFFORT_LIMIT = 100.0
# joints stiffness
LEG_1_STIFFNESS = LEG_1_ARMATURE * NATURAL_FREQ ** 2
LEG_2_STIFFNESS = LEG_2_ARMATURE * NATURAL_FREQ ** 2 
LEG_35_STIFFNESS = LEG_35_ARMATURE * NATURAL_FREQ ** 2 
LEG_4_STIFFNESS = LEG_4_ARMATURE * NATURAL_FREQ ** 2 
LEG_6_STIFFNESS = LEG_6_ARMATURE * NATURAL_FREQ ** 2 
# joints damping limit
LEG_1_DAMPING = 2.0 * DAMPING_RATIO * LEG_1_ARMATURE * NATURAL_FREQ
LEG_2_DAMPING = 2.0 * DAMPING_RATIO * LEG_2_ARMATURE * NATURAL_FREQ
LEG_35_DAMPING = 2.0 * DAMPING_RATIO * LEG_35_ARMATURE * NATURAL_FREQ
LEG_4_DAMPING = 2.0 * DAMPING_RATIO * LEG_4_ARMATURE * NATURAL_FREQ
LEG_6_DAMPING = 2.0 * DAMPING_RATIO * LEG_6_ARMATURE * NATURAL_FREQ

##
# Actuator config.
##

# arm actuators
ARM_1_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=["arm_.*_1_joint"],
  effort_limit=ARM_1_EFFORT_LIMIT,
  armature=ARM_1_ARMATURE,
  stiffness=ARM_1_STIFFNESS,
  damping=ARM_1_DAMPING,
)
ARM_2_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=["arm_.*_2_joint"],
  effort_limit=ARM_2_EFFORT_LIMIT,
  armature=ARM_2_ARMATURE,
  stiffness=ARM_2_STIFFNESS,
  damping=ARM_2_DAMPING,
)
ARM_34_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=["arm_.*_3_joint", "arm_.*_4_joint"],
  effort_limit=ARM_34_EFFORT_LIMIT,
  armature=ARM_34_ARMATURE,
  stiffness=ARM_34_STIFFNESS,
  damping=ARM_34_DAMPING,
)
ARM_567_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=["arm_.*_5_joint", "arm_.*_6_joint", "arm_.*_7_joint"],
  effort_limit=ARM_567_EFFORT_LIMIT,
  armature=ARM_567_ARMATURE,
  stiffness=ARM_567_STIFFNESS,
  damping=ARM_567_DAMPING,
)
# torso actuators
TORSO_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=["torso_.*_joint"],
  effort_limit=TORSO_EFFORT_LIMIT,
  armature=TORSO_ARMATURE,
  stiffness=TORSO_STIFFNESS,
  damping=TORSO_DAMPING,
)
# head actuators
HEAD_1_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=["head_1_joint"],
  effort_limit=HEAD_1_EFFORT_LIMIT,
  armature=HEAD_ARMATURE,
  stiffness=HEAD_STIFFNESS,
  damping=HEAD_DAMPING,
)
HEAD_2_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=["head_2_joint"],
  effort_limit=HEAD_2_EFFORT_LIMIT,
  armature=HEAD_ARMATURE,
  stiffness=HEAD_STIFFNESS,
  damping=HEAD_DAMPING,
)
# leg actuators
LEG_1_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=["leg_.*_1_joint"],
  effort_limit=LEG_1_EFFORT_LIMIT,
  armature=LEG_1_ARMATURE,
  stiffness=LEG_1_STIFFNESS,
  damping=LEG_1_DAMPING,
)
LEG_2_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=["leg_.*_2_joint"],
  effort_limit=LEG_2_EFFORT_LIMIT,
  armature=LEG_2_ARMATURE,
  stiffness=LEG_2_STIFFNESS,
  damping=LEG_2_DAMPING,
)
LEG_35_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=["leg_.*_3_joint", "leg_.*_5_joint"],
  effort_limit=LEG_35_EFFORT_LIMIT,
  armature=LEG_35_ARMATURE,
  stiffness=LEG_35_STIFFNESS,
  damping=LEG_35_DAMPING,
)
LEG_4_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=["leg_.*_4_joint"],
  effort_limit=LEG_4_EFFORT_LIMIT,
  armature=LEG_4_ARMATURE,
  stiffness=LEG_4_STIFFNESS,
  damping=LEG_4_DAMPING,
)
LEG_6_ACTUATOR_CFG = ActuatorCfg(
  joint_names_expr=["leg_.*_6_joint"],
  effort_limit=LEG_6_EFFORT_LIMIT,
  armature=LEG_6_ARMATURE,
  stiffness=LEG_6_STIFFNESS,
  damping=LEG_6_DAMPING,
)

##
# Keyframes.
##


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 1.0),
  joint_pos={
    # legs
    "leg_.*_1_joint": 0.0,
    "leg_.*_2_joint": 0.0,
    "leg_.*_3_joint": -0.4,
    "leg_.*_4_joint": 0.8,
    "leg_.*_5_joint": -0.4,
    "leg_.*_6_joint": 0.0,
    # arms
    "arm_left_1_joint":   0.3,
    "arm_right_1_joint": -0.3,
    "arm_left_2_joint":   0.4,
    "arm_right_2_joint": -0.4,
    "arm_left_3_joint": -0.5,
    "arm_right_3_joint": 0.5,
    "arm_.*_4_joint": -1.5,
    "arm_.*_5_joint": 0.0,
    "arm_.*_6_joint": 0.0,
    "arm_.*_7_joint": 0.0,
    # head
    "head_1_joint": 0.0,
    "head_2_joint": 0.0,
    # torso
    "torso_1_joint": 0.0,
    "torso_2_joint": 0.15,
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

TALOS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    ARM_1_ACTUATOR_CFG,
    ARM_2_ACTUATOR_CFG,
    ARM_34_ACTUATOR_CFG,
    ARM_567_ACTUATOR_CFG,
    LEG_1_ACTUATOR_CFG,
    LEG_2_ACTUATOR_CFG,
    LEG_35_ACTUATOR_CFG,
    LEG_4_ACTUATOR_CFG,
    LEG_6_ACTUATOR_CFG,
    HEAD_1_ACTUATOR_CFG,
    HEAD_2_ACTUATOR_CFG,
    TORSO_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)

TALOS_ROBOT_CFG = EntityCfg(
  init_state=INIT_STATE,
  collisions=(FULL_COLLISION,),
  spec_fn=get_spec,
  articulation=TALOS_ARTICULATION,
)

TALOS_ACTION_SCALE: dict[str, float] = {}

for a in TALOS_ARTICULATION.actuators:
    e = a.effort_limit
    s = a.stiffness
    names = a.joint_names_expr

    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}

    for n in names:
        if n in e and n in s and s[n]:
            TALOS_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

