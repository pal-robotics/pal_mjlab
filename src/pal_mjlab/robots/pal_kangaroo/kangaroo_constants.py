"""Pal Robotics` KANGAROO constants."""

from pathlib import Path

import mujoco

from pal_mjlab import PAL_MJLAB_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg
from mjlab.actuator import BuiltinPositionActuatorCfg

##
# MJCF and assets.
##

# There are multiple arm-wise variants of the KANGAROO robot. For clarity, we use the following naming:
# - kangaroo: simplified model with 4 DoF per arm and a fake forearm
# - kangaroo_hands: simplified model with 5 DoF per arm and a Seed Robotics hand
# - kangaroo_gripper: simplified model with 7 DoF per arm and a gripper
# - kangaroo_full: full model with 4 DoF per arm and a fake forearm

KANGAROO_PATH: Path = PAL_MJLAB_SRC_PATH / "robots" / "pal_kangaroo" / "xmls"
assert KANGAROO_PATH.exists()

KANGAROO_XML: Path = KANGAROO_PATH / "kangaroo.xml"
assert KANGAROO_XML.exists()

KANGAROO_HANDS_XML: Path = KANGAROO_PATH / "kangaroo_hands.xml"
assert KANGAROO_HANDS_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, KANGAROO_PATH / "assets", meshdir)
    return assets


def get_kangaroo_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(KANGAROO_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


def get_kangaroo_hands_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(KANGAROO_HANDS_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Actuator parameters calculs.
##

# params (BeyondMimic paper methodology)
NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0  # over-damped
factor = 0.05
# gear ratio
S_PLUS_GEAR_RATIO = 121
S_MINUS_GEAR_RATIO = 101
XS_GEAR_RATIO = 101
# joints inertia
S_PLUS_MOTOR_INERTIA = 1.728e-5
S_MINUS_MOTOR_INERTIA = 1.3e-5
XS_MOTOR_INERTIA = 1.3e-5
# joints armature (reflected inertia)
S_PLUS_ARMATURE = factor * S_PLUS_MOTOR_INERTIA * S_PLUS_GEAR_RATIO**2
S_MINUS_ARMATURE = factor * S_MINUS_MOTOR_INERTIA * S_MINUS_GEAR_RATIO**2
XS_ARMATURE = factor * XS_MOTOR_INERTIA * XS_GEAR_RATIO**2
# joints effort limit (mehhh...)
S_PLUS_EFFORT_LIMIT = 50
S_MINUS_EFFORT_LIMIT = 25
XS_EFFORT_LIMIT = 25
# joints stiffness
S_PLUS_STIFFNESS = round(S_PLUS_ARMATURE * NATURAL_FREQ**2, 3)
S_MINUS_STIFFNESS = round(S_MINUS_ARMATURE * NATURAL_FREQ**2, 3)
XS_STIFFNESS = round(XS_ARMATURE * NATURAL_FREQ**2, 3)
# joints damping
S_PLUS_DAMPING = round(2.0 * DAMPING_RATIO * S_PLUS_ARMATURE * NATURAL_FREQ, 3)
S_MINUS_DAMPING = round(2.0 * DAMPING_RATIO * S_MINUS_ARMATURE * NATURAL_FREQ, 3)
XS_DAMPING = round(2.0 * DAMPING_RATIO * XS_ARMATURE * NATURAL_FREQ, 3)


# leg stiffness
LEG_45_STIFFNESS = 30
LEG_12_STIFFNESS = 100  # pitch, yaw
LEG_3_STIFFNESS = 100  # roll
LEG_LENGTH_STIFFNESS = 1600
# leg damping
LEG_45_DAMPING = round(2.0 * DAMPING_RATIO * LEG_45_STIFFNESS / NATURAL_FREQ, 3)
LEG_12_DAMPING = round(2.0 * DAMPING_RATIO * LEG_12_STIFFNESS / NATURAL_FREQ, 3)
LEG_3_DAMPING = round(2.0 * DAMPING_RATIO * LEG_3_STIFFNESS / NATURAL_FREQ, 3)
LEG_LENGTH_DAMPING = round(2.0 * DAMPING_RATIO * LEG_LENGTH_STIFFNESS / NATURAL_FREQ, 3)


##
# Actuator config.
##

# LEGS ACTUATORS
KANGAROO_LEGS_1_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("leg_.*_1_joint",),
    effort_limit=80.0,
    armature=0.01,
    stiffness=LEG_12_STIFFNESS,
    damping=LEG_12_DAMPING,
)
KANGAROO_LEGS_2_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("leg_.*_2_joint",),
    effort_limit=230.0,
    armature=0.01,
    stiffness=LEG_12_STIFFNESS,
    damping=LEG_12_DAMPING,
)
KANGAROO_LEGS_3_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("leg_.*_3_joint",),
    effort_limit=139.0,
    armature=0.01,
    stiffness=LEG_3_STIFFNESS,
    damping=LEG_3_DAMPING,
)
KANGAROO_LEGS_4_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("leg_.*_4_joint",),
    effort_limit=140.0,
    armature=0.01,
    stiffness=LEG_45_STIFFNESS,
    damping=LEG_45_DAMPING,
)
KANGAROO_LEGS_5_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("leg_.*_5_joint",),
    effort_limit=82.0,
    armature=0.01,
    stiffness=LEG_45_STIFFNESS,
    damping=LEG_45_DAMPING,
)
KANGAROO_LEGS_LENGTH_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("leg_.*_length_joint",),
    effort_limit=1100.0,
    armature=0.01,
    stiffness=LEG_LENGTH_STIFFNESS,
    damping=LEG_LENGTH_DAMPING,
)
# ACTUATORS
KANGAROO_S_PLUS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(
        "arm_.*_1_joint",
        "arm_.*_2_joint",
        "pelvis_1_joint",
        "pelvis_2_joint",
    ),
    armature=S_PLUS_ARMATURE,
    effort_limit=S_PLUS_EFFORT_LIMIT,
    stiffness=S_PLUS_STIFFNESS,
    damping=S_PLUS_DAMPING,
)
KANGAROO_S_MINUS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(r"arm_.*_(?![1267]_joint)\d+_joint",),
    armature=S_MINUS_ARMATURE,
    effort_limit=S_MINUS_EFFORT_LIMIT,
    stiffness=S_MINUS_STIFFNESS,
    damping=S_MINUS_DAMPING,
)
KANGAROO_XS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(r"arm_.*_(?![12345]_joint)\d+_joint",),
    armature=XS_ARMATURE,
    effort_limit=XS_EFFORT_LIMIT,
    stiffness=XS_STIFFNESS,
    damping=XS_DAMPING,
)

# TODO: hands and gripper actuators cfg

##
# Keyframes.
##

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.9),
    joint_pos={
        # legs
        "leg_left_1_joint": -0.012,
        "leg_right_1_joint": 0.012,
        "leg_.*_2_joint": 0.054,
        "leg_left_3_joint": 0.04,
        "leg_right_3_joint": -0.04,
        "leg_.*_length_joint": 0.6,
        "leg_.*_4_joint": -0.053,
        "leg_.*_5_joint": 0.0,
        "leg_.*_femur_joint": 0.9,
        "leg_.*_knee_joint": 1.8,
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
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = ".*_foot.*_collision"

FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(_foot_regex,),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
)
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={_foot_regex: 3, ".*_collision": 1},
    priority={_foot_regex: 1},
    friction={_foot_regex: (0.6,)},
)

##
# Final config.
##

KANGAROO_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        # KANGAROO_LEGS_ACTUATOR_CFG,
        KANGAROO_LEGS_1_ACTUATOR_CFG,
        KANGAROO_LEGS_2_ACTUATOR_CFG,
        KANGAROO_LEGS_3_ACTUATOR_CFG,  # hips
        KANGAROO_LEGS_4_ACTUATOR_CFG,
        KANGAROO_LEGS_5_ACTUATOR_CFG,  # ankles
        KANGAROO_LEGS_LENGTH_ACTUATOR_CFG,
        KANGAROO_S_PLUS_ACTUATOR_CFG,
        KANGAROO_S_MINUS_ACTUATOR_CFG,
    ),
    soft_joint_pos_limit_factor=0.9,
)
KANGAROO_HANDS_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        # KANGAROO_LEGS_ACTUATOR_CFG,
        KANGAROO_LEGS_1_ACTUATOR_CFG,
        KANGAROO_LEGS_2_ACTUATOR_CFG,
        KANGAROO_LEGS_3_ACTUATOR_CFG,  # hips
        KANGAROO_LEGS_4_ACTUATOR_CFG,
        KANGAROO_LEGS_5_ACTUATOR_CFG,  # ankles
        KANGAROO_LEGS_LENGTH_ACTUATOR_CFG,
        KANGAROO_S_PLUS_ACTUATOR_CFG,
        KANGAROO_S_MINUS_ACTUATOR_CFG,
        # KANGAROO_XS_ACTUATOR_CFG,
        # KANGAROO_HANDS_ACTUATOR_CFG,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_kangaroo_robot_cfg() -> EntityCfg:
    """Get a fresh KANGAROO (4 DoF per arms) robot configuration instance."""
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_kangaroo_spec,
        articulation=KANGAROO_ARTICULATION,
    )


def get_kangaroo_hands_robot_cfg() -> EntityCfg:
    """Get a fresh KANGAROO with hands (5 DoF per arms) robot configuration instance."""
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FEET_ONLY_COLLISION,),
        spec_fn=get_kangaroo_hands_spec,
        articulation=KANGAROO_HANDS_ARTICULATION,
    )


KANGAROO_ACTION_SCALE: dict[str, float] = {}
KANGAROO_HANDS_ACTION_SCALE: dict[str, float] = {}
KANGAROO_ACTUATOR_NAMES: tuple = ()


def test_jn(name: str) -> bool:
    if name == "leg_left_knee_joint":
        return False
    if name == "leg_right_knee_joint":
        return False
    if name == "leg_left_femur_joint":
        return False
    if name == "leg_right_femur_joint":
        return False
    return True


for a in KANGAROO_ARTICULATION.actuators:
    e = a.effort_limit
    s = a.stiffness
    names = a.joint_names_expr

    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}

    for n in names:
        if n in e and n in s and s[n] and test_jn(n):
            KANGAROO_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
            KANGAROO_ACTUATOR_NAMES += (n,)

for a in KANGAROO_HANDS_ARTICULATION.actuators:
    e = a.effort_limit
    s = a.stiffness
    names = a.joint_names_expr

    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}

    for n in names:
        if n in e and n in s and s[n]:
            KANGAROO_HANDS_ACTION_SCALE[n] = 0.25 * e[n] / s[n]


# TODO: make an argument for the choice of the robot
if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(get_kangaroo_robot_cfg())

    viewer.launch(robot.spec.compile())
