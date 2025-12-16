"""PAL Robotics TIAGo PRO constants."""

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

TIAGO_PRO_XML: Path = PAL_MJLAB_SRC_PATH / "robots" / "pal_tiago_pro" / "xmls" / "tiago_pro.xml"
assert TIAGO_PRO_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, TIAGO_PRO_XML.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(TIAGO_PRO_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec

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
S_MINUS_ARMATURE =factor * S_MINUS_MOTOR_INERTIA * S_MINUS_GEAR_RATIO**2
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

# --------------------------------------------------------
# TORSO JOINTS
# effort limit 
TORSO_EFFORT_LIMIT = 2200 # 40.0

# joint stiffness 
TORSO_STIFFNESS = 1500 # (NATURAL_FREQ**2) * ARM_1_ARMATURE

# joint damping
TORSO_DAMPING = 300 # 2.0 * DAMPING_RATIO * NATURAL_FREQ * ARM_1_ARMATURE

# --------------------------------------------------------
# GRIPPER JOINTS
# inertia
GRIPPER_MOTOR_INERTIA = 0.000207288
GRIPPER_REDUCTION_RATIO = 100

# armature
GRIPPER_ARMATURE = factor * GRIPPER_MOTOR_INERTIA * GRIPPER_REDUCTION_RATIO**2

# effort limit 
GRIPPER_EFFORT_LIMIT = 40.0

# joint stiffness 
GRIPPER_STIFFNESS = (NATURAL_FREQ**2) * GRIPPER_ARMATURE

# joint damping
GRIPPER_DAMPING = 2.0 * DAMPING_RATIO * NATURAL_FREQ * GRIPPER_ARMATURE

## --------------------------------------------------------
# Actuator configurations.
## --------------------------------------------------------

TIAGO_S_PLUS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(
        "arm_.*_1_joint",
        "arm_.*_2_joint",
    ),
    armature=S_PLUS_ARMATURE,
    effort_limit=S_PLUS_EFFORT_LIMIT,
    stiffness=S_PLUS_STIFFNESS,
    damping=S_PLUS_DAMPING,
)
TIAGO_S_MINUS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
        joint_names_expr=(
        "arm_.*_3_joint",
        "arm_.*_4_joint",
        "arm_.*_5_joint",
    ),
    armature=S_MINUS_ARMATURE,
    effort_limit=S_MINUS_EFFORT_LIMIT,
    stiffness=S_MINUS_STIFFNESS,
    damping=S_MINUS_DAMPING,
)
TIAGO_XS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(
        "arm_.*_6_joint",
        "arm_.*_7_joint",
    ),
    armature=XS_ARMATURE,
    effort_limit=XS_EFFORT_LIMIT,
    stiffness=XS_STIFFNESS,
    damping=XS_DAMPING,
)
TORSO_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("torso_lift_joint",),
    effort_limit=TORSO_EFFORT_LIMIT,
    armature=S_MINUS_ARMATURE,
    stiffness=TORSO_STIFFNESS,
    damping=TORSO_DAMPING,
)       
GRIPPER_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(
        "gripper_left_.*_joint",
        "gripper_right_.*_joint",),
    effort_limit=GRIPPER_EFFORT_LIMIT,
    armature=GRIPPER_ARMATURE,    
    stiffness=GRIPPER_STIFFNESS,
    damping=GRIPPER_DAMPING,
)


INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        # torso
        "torso_lift_joint": 0.1 ,  
        # arms
        "arm_left_1_joint": 0.3578,
        "arm_left_2_joint": -1.8266,
        "arm_left_3_joint": 0.4698,
        "arm_left_4_joint": -2.3409,
        "arm_left_5_joint": 0.0,
        "arm_left_6_joint": -1.2006,
        "arm_left_7_joint": 0.0,
        "arm_right_1_joint": -0.3576,
        "arm_right_2_joint": -1.8266,
        "arm_right_3_joint": -0.4698,
        "arm_right_4_joint": -2.3409,
        "arm_right_5_joint": 0.0,
        "arm_right_6_joint": -1.2006,
        "arm_right_7_joint": 0.0,
        # grippers 
        # "gripper_.*_joint": 0.0,
    },
    joint_vel={".*": 0.0},
)

## --------------------------------------------------------
# Collision configurations.
## --------------------------------------------------------

FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*",),  # all geoms
    condim=3,
    priority=1,
    friction=(0.7,),
)

## --------------------------------------------------------
# Final configurations.
## --------------------------------------------------------

TIAGO_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        TIAGO_S_PLUS_ACTUATOR_CFG,
        TIAGO_S_MINUS_ACTUATOR_CFG,
        TIAGO_XS_ACTUATOR_CFG,
        TORSO_ACTUATOR_CFG,
        # GRIPPER_ACTUATOR_CFG,
    ),
    soft_joint_pos_limit_factor=0.9,
)

TIAGO_ACTION_SCALE: dict[str, float] = {}

def get_tiago_robot_cfg() -> EntityCfg:
    """Get a fresh TIAGo Pro robot configuration instance."""
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=TIAGO_ARTICULATION,
    )

for a in TIAGO_ARTICULATION.actuators:
    e = a.effort_limit
    s = a.stiffness
    names = a.joint_names_expr

    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}

    for n in names:
        if n in e and n in s and s[n]:
            TIAGO_ACTION_SCALE[n] = 0.25 * e[n] / s[n]


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_tiago_robot_cfg())
    viewer.launch(robot.spec.compile())