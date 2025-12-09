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

# ARM JOINTS parameters
# inertia
ARM_1_MOTOR_INERTIA = 1.728e-5
ARM_2_MOTOR_INERTIA = 1.728e-5
ARM_3_MOTOR_INERTIA = 1.3e-5
ARM_4_MOTOR_INERTIA = 1.3e-5
ARM_5_MOTOR_INERTIA = 1.3e-5
ARM_6_MOTOR_INERTIA = 1.3e-5
ARM_7_MOTOR_INERTIA = 1.3e-5

ARM_12_REDUCTION_RATIO = 121
ARM_REDUCTION_RATIO = 101

# armature
ARM_1_ARMATURE = factor * ARM_1_MOTOR_INERTIA * ARM_12_REDUCTION_RATIO**2
ARM_2_ARMATURE = factor * ARM_2_MOTOR_INERTIA * ARM_12_REDUCTION_RATIO**2
ARM_3_ARMATURE = factor * ARM_3_MOTOR_INERTIA * ARM_REDUCTION_RATIO**2
ARM_4_ARMATURE = factor * ARM_4_MOTOR_INERTIA * ARM_REDUCTION_RATIO**2
ARM_5_ARMATURE = factor * ARM_5_MOTOR_INERTIA * ARM_REDUCTION_RATIO**2
ARM_6_ARMATURE = factor * ARM_6_MOTOR_INERTIA * ARM_REDUCTION_RATIO**2
ARM_7_ARMATURE = factor * ARM_7_MOTOR_INERTIA * ARM_REDUCTION_RATIO**2

# effort limit 
ARM_1_EFFORT_LIMIT = 46.0
ARM_2_EFFORT_LIMIT = 46.0
ARM_3_EFFORT_LIMIT = 46.0      
ARM_4_EFFORT_LIMIT = 46.0
ARM_5_EFFORT_LIMIT = 46.0
ARM_6_EFFORT_LIMIT = 46.0
ARM_7_EFFORT_LIMIT = 46.0

# joint stiffness 
ARM_1_JOINT_STIFFNESS = (NATURAL_FREQ**2) * ARM_1_ARMATURE
ARM_2_JOINT_STIFFNESS = (NATURAL_FREQ**2) * ARM_2_ARMATURE
ARM_3_JOINT_STIFFNESS = (NATURAL_FREQ**2) * ARM_3_ARMATURE
ARM_4_JOINT_STIFFNESS = (NATURAL_FREQ**2) * ARM_4_ARMATURE
ARM_5_JOINT_STIFFNESS = (NATURAL_FREQ**2) * ARM_5_ARMATURE
ARM_6_JOINT_STIFFNESS = (NATURAL_FREQ**2) * ARM_6_ARMATURE
ARM_7_JOINT_STIFFNESS = (NATURAL_FREQ**2) * ARM_7_ARMATURE  

# joint damping
ARM_1_JOINT_DAMPING = 2.0 * DAMPING_RATIO * NATURAL_FREQ * ARM_1_ARMATURE
ARM_2_JOINT_DAMPING = 2.0 * DAMPING_RATIO * NATURAL_FREQ * ARM_2_ARMATURE
ARM_3_JOINT_DAMPING = 2.0 * DAMPING_RATIO * NATURAL_FREQ * ARM_3_ARMATURE
ARM_4_JOINT_DAMPING = 2.0 * DAMPING_RATIO * NATURAL_FREQ * ARM_4_ARMATURE
ARM_5_JOINT_DAMPING = 2.0 * DAMPING_RATIO * NATURAL_FREQ * ARM_5_ARMATURE
ARM_6_JOINT_DAMPING = 2.0 * DAMPING_RATIO * NATURAL_FREQ * ARM_6_ARMATURE
ARM_7_JOINT_DAMPING = 2.0 * DAMPING_RATIO * NATURAL_FREQ * ARM_7_ARMATURE

# --------------------------------------------------------
# TORSO JOINTS
# inertia
TORSO_MOTOR_INERTIA = 0.01 # 0.000207288
TORSO_REDUCTION_RATIO = 100

# armature
TORSO_ARMATURE = factor * ARM_1_MOTOR_INERTIA * TORSO_REDUCTION_RATIO**2

# effort limit 
TORSO_EFFORT_LIMIT = 2200 # 40.0

# joint stiffness 
TORSO_STIFFNESS = 1500 # (NATURAL_FREQ**2) * ARM_1_ARMATURE

# joint damping
TORSO_DAMPING = 300 # 2.0 * DAMPING_RATIO * NATURAL_FREQ * ARM_1_ARMATURE

# --------------------------------------------------------
# HEAD JOINTS
# inertia
HEAD_MOTOR_INERTIA = 0.000207288
HEAD_REDUCTION_RATIO = 100

# armature
HEAD_ARMATURE = factor * HEAD_MOTOR_INERTIA * HEAD_REDUCTION_RATIO**2

# effort limit 
HEAD_1_EFFORT_LIMIT = 8.0
HEAD_2_EFFORT_LIMIT = 4.0

# joint stiffness 
HEAD_STIFFNESS = (NATURAL_FREQ**2) * HEAD_ARMATURE

# joint damping
HEAD_DAMPING = 2.0 * DAMPING_RATIO * NATURAL_FREQ * HEAD_ARMATURE

# --------------------------------------------------------
# WHEEL JOINTS
# inertia
WHEEL_MOTOR_INERTIA = 0.000207288
WHEEL_REDUCTION_RATIO = 100

# armature
WHEEL_ARMATURE = factor * WHEEL_MOTOR_INERTIA * WHEEL_REDUCTION_RATIO**2

# effort limit 
WHEEL_EFFORT_LIMIT = 40.0

# joint stiffness 
WHEEL_STIFFNESS = (NATURAL_FREQ**2) * WHEEL_ARMATURE

# joint damping
WHEEL_DAMPING = 2.0 * DAMPING_RATIO * NATURAL_FREQ * WHEEL_ARMATURE

# --------------------------------------------------------
# GRIPPER JOINTS
# inertia
GRIPPER_MOTOR_INERTIA = 0.000019
GRIPPER_REDUCTION_RATIO = 100.0 # 0.0013

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

ARM_1_L_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("arm_left_1_joint",),
    effort_limit=ARM_1_EFFORT_LIMIT,
    armature=ARM_1_ARMATURE,
    stiffness=50,
    damping=1.0,
)
ARM_1_R_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("arm_right_1_joint",),
    effort_limit=ARM_1_EFFORT_LIMIT,
    armature=ARM_1_ARMATURE,
    stiffness=50,
    damping=1.0,
)
ARM_2_L_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("arm_left_2_joint",),
    effort_limit=ARM_2_EFFORT_LIMIT,
    armature=ARM_2_ARMATURE,
    stiffness=45,
    damping=0.5,
)
ARM_2_R_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("arm_right_2_joint",),
    effort_limit=ARM_2_EFFORT_LIMIT,
    armature=ARM_2_ARMATURE,
    stiffness=45,
    damping=0.5,
)
ARM_3_L_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("arm_left_3_joint",),
    effort_limit=ARM_3_EFFORT_LIMIT,
    armature=ARM_3_ARMATURE,
    stiffness=70,
    damping=1.0,
)
ARM_3_R_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("arm_right_3_joint",),
    effort_limit=ARM_3_EFFORT_LIMIT,
    armature=ARM_3_ARMATURE,
    stiffness=70,
    damping=1.0,
)
ARM_4_R_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("arm_right_4_joint",),
    effort_limit=ARM_4_EFFORT_LIMIT,
    armature=ARM_4_ARMATURE,
    stiffness=35,
    damping=0.0,
)
ARM_4_L_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("arm_left_4_joint",),
    effort_limit=ARM_4_EFFORT_LIMIT,
    armature=ARM_4_ARMATURE,
    stiffness=35,
    damping=0.0,
)
ARM_5_L_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("arm_left_5_joint",),
    effort_limit=ARM_5_EFFORT_LIMIT,
    armature=ARM_5_ARMATURE,
    stiffness=15,
    damping=0,
)
ARM_5_R_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("arm_right_5_joint",),
    effort_limit=ARM_5_EFFORT_LIMIT,
    armature=ARM_5_ARMATURE,
    stiffness=15,
    damping=0,
)
ARM_6_L_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("arm_left_6_joint",),
    effort_limit=ARM_6_EFFORT_LIMIT,
    armature=ARM_6_ARMATURE,
    stiffness=15,
    damping=0.0,
)
ARM_6_R_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("arm_right_6_joint",),
    effort_limit=ARM_6_EFFORT_LIMIT,
    armature=ARM_6_ARMATURE,
    stiffness=15,
    damping=0.0,
)
ARM_7_R_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("arm_right_7_joint",),
    effort_limit=ARM_7_EFFORT_LIMIT,
    armature=ARM_7_ARMATURE,
    stiffness=10,
    damping=0.0,
)
ARM_7_L_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("arm_left_7_joint",),
    effort_limit=ARM_7_EFFORT_LIMIT,
    armature=ARM_7_ARMATURE,
    stiffness=10,
    damping=0.0,
)
TORSO_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("torso_lift_joint",),
    effort_limit=TORSO_EFFORT_LIMIT,
    armature=TORSO_ARMATURE,
    stiffness=TORSO_STIFFNESS,
    damping=TORSO_DAMPING,
)       
HEAD_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("head_.*_joint",),
    effort_limit=(HEAD_1_EFFORT_LIMIT, HEAD_2_EFFORT_LIMIT),
    armature=HEAD_ARMATURE,
    stiffness=HEAD_STIFFNESS,
    damping=HEAD_DAMPING,
)
WHEEL_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(        
        "wheel_front_right_joint",
        "wheel_front_left_joint",
        "wheel_rear_right_joint",
        "wheel_rear_left_joint",),
    effort_limit=WHEEL_EFFORT_LIMIT,
    armature=WHEEL_ARMATURE,    
    stiffness=WHEEL_STIFFNESS,
    damping=WHEEL_DAMPING,
)
GRIPPER_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(
        "gripper_left_outer_finger_left_joint",
        "gripper_right_outer_finger_left_joint",),
    effort_limit=GRIPPER_EFFORT_LIMIT,
    armature=GRIPPER_ARMATURE,    
    stiffness=60.0, #GRIPPER_STIFFNESS,
    damping= 4.0, #GRIPPER_DAMPING,
)


INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        # wheels
        "wheel_.*_joint": 0.0,
        # torso
        "torso_lift_joint": 0.08 ,  
        # head
        "head_1_joint": 0.0,
        "head_2_joint": 0.0,
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
        "gripper_.*_joint": 0.40,
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
        ARM_1_R_ACTUATOR_CFG,
        ARM_2_R_ACTUATOR_CFG,
        ARM_3_R_ACTUATOR_CFG,
        ARM_4_R_ACTUATOR_CFG,
        ARM_5_R_ACTUATOR_CFG,
        ARM_6_R_ACTUATOR_CFG,
        ARM_7_R_ACTUATOR_CFG,
        ARM_1_L_ACTUATOR_CFG,
        ARM_2_L_ACTUATOR_CFG,
        ARM_3_L_ACTUATOR_CFG,
        ARM_4_L_ACTUATOR_CFG,
        ARM_5_L_ACTUATOR_CFG,
        ARM_6_L_ACTUATOR_CFG,
        ARM_7_L_ACTUATOR_CFG,
        TORSO_ACTUATOR_CFG,
        # HEAD_ACTUATOR_CFG,
        # WHEEL_ACTUATOR_CFG,
        GRIPPER_ACTUATOR_CFG,
    ),
    soft_joint_pos_limit_factor=0.75,
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