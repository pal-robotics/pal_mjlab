"""PAL Robotics TIAGo PRO constants."""

from pathlib import Path

import mujoco

from pal_mjlab import PAL_MJLAB_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg
from mjlab.actuator import BuiltinPositionActuatorCfg

# Reuse same utilities as YAM for rotary->linear reflection
from mjlab.utils.actuator import ElectricActuator, reflect_rotary_to_linear

TIAGO_PRO_XML: Path = (
    PAL_MJLAB_SRC_PATH / "robots" / "pal_tiago_pro" / "xmls" / "tiago_pro.xml"
)
assert TIAGO_PRO_XML.exists()

##
# Actuator Parameters (BeyondMimic methodology)
##

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0
FACTOR = 0.05


def _calc_actuator_params(
    gear_ratio: float, motor_inertia: float, effort: float
) -> dict:
    """Calculate armature, stiffness, and damping for an actuator."""
    armature = FACTOR * motor_inertia * gear_ratio**2
    stiffness = round(armature * NATURAL_FREQ**2, 3)
    damping = round(2.0 * DAMPING_RATIO * armature * NATURAL_FREQ, 3)
    return {
        "armature": armature,
        "stiffness": stiffness,
        "damping": damping,
        "effort_limit": effort,
    }


# Motor parameters: (gear_ratio, motor_inertia, effort_limit)
S_PLUS = _calc_actuator_params(121, 1.728e-5, 50)
S_MINUS = _calc_actuator_params(101, 1.3e-5, 25)
XS = _calc_actuator_params(101, 1.3e-5, 25)

TORSO = {"armature": 0.1, "stiffness": 1500, "damping": 300, "effort_limit": 2200}

# -------------------------------------------------------------------
# YAM gripper constants (ported from your YAM constants)
# -------------------------------------------------------------------

# YAM DM4310 motor parameters used for gripper crank
ARMATURE_DM_4310 = 0.0018
DM_4310 = ElectricActuator(
    reflected_inertia=ARMATURE_DM_4310,
    velocity_limit=30.0,
    effort_limit=10.0,
)

# Crank transmission: 2.7 rad motor range -> 0.071 m linear stroke
GRIPPER_MOTOR_STROKE_CRANK = 2.7        # [rad]
GRIPPER_LINEAR_STROKE_CRANK = 0.071     # [m]
GRIPPER_TRANSMISSION_RATIO_CRANK = (
    GRIPPER_LINEAR_STROKE_CRANK / GRIPPER_MOTOR_STROKE_CRANK
)

(
    ARMATURE_DM_4310_LINEAR_CRANK,
    VELOCITY_LIMIT_DM_4310_LINEAR_CRANK,
    EFFORT_LIMIT_DM_4310_LINEAR_CRANK,
) = reflect_rotary_to_linear(
    armature_rotary=ARMATURE_DM_4310,
    velocity_limit_rotary=DM_4310.velocity_limit,
    effort_limit_rotary=DM_4310.effort_limit,
    transmission_ratio=GRIPPER_TRANSMISSION_RATIO_CRANK,
)

# PD gains for gripper (same as YAM)
NATURAL_FREQ_GRIPPER = 2 * 2.0 * 3.1415926535  # 2Hz
STIFFNESS_DM_4310_LINEAR_CRANK = ARMATURE_DM_4310_LINEAR_CRANK * NATURAL_FREQ_GRIPPER**2
DAMPING_DM_4310_LINEAR_CRANK = (
    2.0 * DAMPING_RATIO * ARMATURE_DM_4310_LINEAR_CRANK * NATURAL_FREQ_GRIPPER
)

# Sim-safe force limit (same as YAM)
EFFORT_LIMIT_DM_4310_LINEAR_CRANK_SAFE = EFFORT_LIMIT_DM_4310_LINEAR_CRANK * 0.1

YAM_RIGHT_GRIPPER = {
    "armature": ARMATURE_DM_4310_LINEAR_CRANK,
    "stiffness": STIFFNESS_DM_4310_LINEAR_CRANK,
    "damping": DAMPING_DM_4310_LINEAR_CRANK,
    "effort_limit": EFFORT_LIMIT_DM_4310_LINEAR_CRANK_SAFE,
}

##
# MJCF & Assets
##


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, TIAGO_PRO_XML.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(TIAGO_PRO_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


## --------------------------------------------------------
# Actuator configurations.
## --------------------------------------------------------

# Arms
TIAGO_PRO_S_PLUS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(r"arm_.*_(1|2)_joint",),
    **S_PLUS,
)
TIAGO_PRO_S_MINUS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(r"arm_.*_(?![1267]_joint)\d+_joint",),
    **S_MINUS,
)
TIAGO_PRO_XS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(r"arm_.*_(?![12345]_joint)\d+_joint",),
    **XS,
)

# Torso
TORSO_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("torso_lift_joint",),
    **TORSO,
)

# Right YAM gripper: actuate only left finger; right finger is coupled via equality
YAM_RIGHT_GRIPPER_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("yam_right_left_finger_joint",),
    **YAM_RIGHT_GRIPPER,
)

##
# Initial State
##

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "torso_lift_joint": 0.1,
        "arm_left_1_joint": 0.3578,
        "arm_right_1_joint": -0.3578,
        "arm_.*_2_joint": -1.8266,
        "arm_left_3_joint": 0.4698,
        "arm_right_3_joint": -0.4698,
        "arm_.*_4_joint": -2.3409,
        "arm_.*_6_joint": -1.2006,

        # YAM right gripper (set both for a clean starting pose)
        "yam_right_left_finger_joint": 0.0375 / 2,
        "yam_right_right_finger_joint": -0.0375 / 2,
    },
    joint_vel={".*": 0.0},
)

##
# Collision Configs
##

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*",),  # all geoms
    condim=3,
    priority=1,
    friction=(0.7,),
)

##
# Articulation Config
##

TIAGO_PRO_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        TIAGO_PRO_S_PLUS_ACTUATOR_CFG,
        TIAGO_PRO_S_MINUS_ACTUATOR_CFG,
        TIAGO_PRO_XS_ACTUATOR_CFG,
        TORSO_ACTUATOR_CFG,
        YAM_RIGHT_GRIPPER_ACTUATOR_CFG,
    ),
    soft_joint_pos_limit_factor=0.9,
)

TIAGO_PRO_ACTION_SCALE: dict[str, float] = {}
TIAGO_PRO_ACTUATOR_NAMES: tuple = ()


def get_tiago_pro_robot_cfg() -> EntityCfg:
    """Get a fresh TIAGo Pro robot configuration instance."""
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=TIAGO_PRO_ARTICULATION,
    )


for a in TIAGO_PRO_ARTICULATION.actuators:
    e = a.effort_limit
    s = a.stiffness
    names = a.joint_names_expr

    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}

    for n in names:
        if n in e and n in s and s[n]:
            TIAGO_PRO_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
            TIAGO_PRO_ACTUATOR_NAMES += (n,)


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_tiago_pro_robot_cfg())
    viewer.launch(robot.spec.compile())
