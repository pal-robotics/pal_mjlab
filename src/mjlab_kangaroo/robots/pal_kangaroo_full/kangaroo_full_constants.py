"""PAL Robotics Kangaroo (full-model) constants."""

from pathlib import Path
import re

import mujoco

from mjlab_kangaroo import mjlab_kangaroo_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import ActuatorCfg, CollisionCfg

##
# MJCF and assets.
##

KANG_FULL_XML: Path = (
    mjlab_kangaroo_SRC_PATH
    / "robots"
    / "pal_kangaroo_full"
    / "xmls"
    / "kangaroo_full.xml"
)
assert KANG_FULL_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, KANG_FULL_XML.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(KANG_FULL_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Actuator config.
##

KANG_FULL_PASSIVE_JOINTS = [
    "baselink__.*_hip_z_motor",
    ".*_hip_z_yaw",
    ".*_hipyaw_yaw__hip_xy_bracket_l",
    ".*_hip_xy_bracket_l__hip_xy_motor_l",
    ".*_hip_xy_baselink__hip_xy_bracket_r",
    ".*_hip_xy_bracket_r__hip_xy_motor_r",
    ".*_hip_xy_pitch",
    ".*_hip_xy_roll",
    ".*_hip_xy_legholder__leg_length_femur",
    ".*_leg_length_slider__leg_length_bar3",
    ".*_leg_length_slider__leg_length_bar4",
    ".*_leg_length_femur__leg_length_triangle",
    ".*_leg_length_triangle__leg_length_bar2",
    ".*_ankle_xy_femur__ankle_xy_butterfly_l",
    ".*_ankle_xy_butterfly_l__ankle_xy_bar2_l",
    ".*_ankle_xy_femur__ankle_xy_butterfly_r",
    ".*_ankle_xy_butterfly_r__ankle_xy_bar2_r",
    ".*_knee",
    ".*_ankle_xy_pitch",
    ".*_ankle_xy_roll",
    ".*_leg_length_baselink__leg_length_bar1",
    ".*_hip_xy_legholder__ankle_xy_motor_l",
    ".*_hip_xy_legholder__ankle_xy_motor_r",
    ".*_hip_xy_legholder__ankle_xy_crank_l",
    ".*_ankle_xy_crank_l__ankle_xy_bar1_l",
    ".*_hip_xy_legholder__ankle_xy_crank_r",
    ".*_ankle_xy_crank_r__ankle_xy_bar1_r",
]

KANG_FULL_MEASURED_PASSIVE_JOINTS = None  # TODO

KANG_FULL_LINEAR_ACTUATORS = [
    ".*_hip_z_slider",
    ".*_hip_xy_slider_l",
    ".*_hip_xy_slider_r",
    ".*_ankle_xy_slider_l",
    ".*_ankle_xy_slider_r",
    ".*_leg_length_slider$",
]

KANG_FULL_REVOLUTE_ACTUATORS = [
    "arm_.*_1_joint",
    "arm_.*_2_joint",
    "arm_.*_3_joint",
    "arm_.*_4_joint",
    "pelvis_1_joint",
    "pelvis_2_joint",
]

KANG_FULL_LEGS_PASSIVE_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=KANG_FULL_PASSIVE_JOINTS,
    effort_limit=100.0,
    armature=0.01,
    stiffness=0.0,
    damping=0.0,
)

# TODO: check `stiffness`, `damping`, and `effort_limit`

KANG_FULL_HIP_Z_SLIDERS_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=[
        ".*_hip_z_slider",
    ],
    effort_limit=10000.0,
    armature=0.01,
    stiffness=1000000.0,
    damping=1000.0,
)

KANG_FULL_HIP_XY_SLIDERS_L_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=[
        ".*_hip_xy_slider_l",
    ],
    effort_limit=10000.0,
    armature=0.01,
    stiffness=1000000.0,
    damping=1000.0,
)

KANG_FULL_HIP_XY_SLIDERS_R_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=[
        ".*_hip_xy_slider_r",
    ],
    effort_limit=10000.0,
    armature=0.01,
    stiffness=1000000.0,
    damping=1000.0,
)

KANG_FULL_ANKLE_XY_SLIDERS_L_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=[
        ".*_ankle_xy_slider_l",
    ],
    effort_limit=10000.0,
    armature=0.01,
    stiffness=1000000.0,
    damping=1000.0,
)

KANG_FULL_ANKLE_XY_SLIDERS_R_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=[
        ".*_ankle_xy_slider_r",
    ],
    effort_limit=10000.0,
    armature=0.01,
    stiffness=1000000.0,
    damping=1000.0,
)

KANG_FULL_LEG_LENGTH_SLIDERS_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=[
        ".*_leg_length_slider$",
    ],
    effort_limit=10000.0,
    armature=0.01,
    stiffness=1000000.0,
    damping=1000.0,
)

KANG_FULL_ARMS_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=[
        "arm_.*_1_joint",
        "arm_.*_2_joint",
        "arm_.*_3_joint",
        "arm_.*_4_joint",
    ],
    armature=0.01,
    effort_limit=43.0,
    stiffness=100.0,
    damping=0.0,
)

KANG_FULL_PELVIS_1_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=[
        "pelvis_1_joint",
    ],
    effort_limit=100.0,
    armature=0.01,
    stiffness=500.0,
    damping=0.0,
)

KANG_FULL_PELVIS_2_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=[
        "pelvis_2_joint",
    ],
    effort_limit=100.0,
    armature=0.01,
    stiffness=500.0,
    damping=0.0,
)
##
# Keyframes.
##


INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 1.02),
    joint_pos={
        # arms
        "arm_left_1_joint": 0.24,
        "arm_right_1_joint": -0.24,
        "arm_.*_2_joint": 1.32,
        "arm_left_3_joint": 1.57,
        "arm_right_3_joint": -1.57,
        "arm_.*_4_joint": 0.8,
        # torso
        "pelvis_1_joint": 0.0,
        "pelvis_2_joint": 0.0,
        # legs active
        ".*_hip_z_slider": 0.0,
        ".*_hip_xy_slider_l": 0.0,
        ".*_hip_xy_slider_r": 0.0,
        ".*_ankle_xy_slider_l": 0.0,
        ".*_ankle_xy_slider_r": 0.0,
        ".*_leg_length_slider$": 0.0,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = ".*_foot_collision"

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

KANG_FULL_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        KANG_FULL_LEGS_PASSIVE_ACTUATOR_CFG,
        KANG_FULL_HIP_Z_SLIDERS_ACTUATOR_CFG,
        KANG_FULL_HIP_XY_SLIDERS_L_ACTUATOR_CFG,
        KANG_FULL_HIP_XY_SLIDERS_R_ACTUATOR_CFG,
        KANG_FULL_ANKLE_XY_SLIDERS_L_ACTUATOR_CFG,
        KANG_FULL_ANKLE_XY_SLIDERS_R_ACTUATOR_CFG,
        KANG_FULL_LEG_LENGTH_SLIDERS_ACTUATOR_CFG,
        KANG_FULL_ARMS_ACTUATOR_CFG,
        KANG_FULL_PELVIS_1_ACTUATOR_CFG,
        KANG_FULL_PELVIS_2_ACTUATOR_CFG,
    ),
    soft_joint_pos_limit_factor=0.9,
)

KANG_FULL_ROBOT_CFG = EntityCfg(
    init_state=INIT_STATE,
    collisions=(FEET_ONLY_COLLISION,),
    spec_fn=get_spec,
    articulation=KANG_FULL_ARTICULATION,
)


def match_list(n: str, targets: list[str]) -> bool:
    """Check name match on a list of target patterns."""
    for t in targets:
        if re.match(t, n):
            return True
    return False


import pdb

KANG_FULL_ACTION_SCALE: dict[str, float] = {}
for a in KANG_FULL_ARTICULATION.actuators:
    # e = a.effort_limit
    # s = a.stiffness
    names = a.joint_names_expr

    for n in names:
        if match_list(n, KANG_FULL_PASSIVE_JOINTS):
            continue
        elif match_list(n, KANG_FULL_REVOLUTE_ACTUATORS):
            KANG_FULL_ACTION_SCALE[n] = 0.5
        elif match_list(n, KANG_FULL_LINEAR_ACTUATORS):
            KANG_FULL_ACTION_SCALE[n] = 0.005
        else:
            continue
