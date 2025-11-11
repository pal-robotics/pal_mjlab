"""Unitree Go1 constants."""

from pathlib import Path

import mujoco

from pal_mjlab import PAL_MJLAB_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import ActuatorCfg, CollisionCfg

##
# MJCF and assets.
##

# There are multiple arm-wise variants of the KANGAROO robot. For clarity, we use the following naming:
# 
# - kangaroo: simplified model with 4 DoF per arm and a fake forearm
# - kangaroo_hands: simplified model with 7 DoF per arm and a Seed Robotics hand
# - kangaroo_gripper: simplified model with 5 DoF per arm and a gripper
# - kangaroo_full: full model with 4 DoF per arm and a fake forearm

KANGAROO_PATH: Path = (
    PAL_MJLAB_SRC_PATH / "robots" / "pal_kangaroo" / "xmls"
)
assert KANGAROO_PATH.exists()

KANGAROO_XML: Path = (
    KANGAROO_PATH / "kangaroo.xml"
)
assert KANGAROO_XML.exists()

KANGAROO_HANDS_XML: Path = (
    KANGAROO_PATH / "kangaroo_hands.xml"
)
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
# Actuator config.
##

# LEGS ACTUATORS 
# TODO: use this instead once https://github.com/mujocolab/mjlab/pull/290 is merged
# KANGAROO_LEGS_ACTUATOR_CFG = ActuatorCfg(
#     joint_names_expr=(
#         "leg_.*_1_joint", 
#         "leg_.*_2_joint", 
#         "leg_.*_3_joint", 
#         "leg_.*_length_joint", 
#         "leg_.*_4_joint", 
#         "leg_.*_5_joint",
#     ),
#     effort_limit={
#         "leg_.*_1_joint": 80.0, 
#         "leg_.*_2_joint": 230.0, 
#         "leg_.*_3_joint": 139.0, 
#         "leg_.*_length_joint": 1100.0, 
#         "leg_.*_4_joint": 140.0, 
#         "leg_.*_5_joint": 82.0,
#     },
#     stiffness={
#         "leg_.*_1_joint": 40.0, 
#         "leg_.*_2_joint": 100.0, 
#         "leg_.*_3_joint": 100.0, 
#         "leg_.*_length_joint": 1100.0, 
#         "leg_.*_4_joint": 100.0, 
#         "leg_.*_5_joint": 40.0,
#     },
#     damping={
#         "leg_.*_1_joint": 2.55, 
#         "leg_.*_2_joint": 6.35, 
#         "leg_.*_3_joint": 6.35, 
#         "leg_.*_length_joint": 70.0, 
#         "leg_.*_4_joint": 6.35, 
#         "leg_.*_5_joint": 2.55,
#     },
#     armature=0.01,
# )
KANGAROO_LEGS_1_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=("leg_.*_1_joint",),
    effort_limit=80.0,
    armature=0.01,
    stiffness=40.0,
    damping=2.55,
)
KANGAROO_LEGS_2_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=("leg_.*_2_joint",),
    effort_limit=230.0,
    armature=0.01,
    stiffness=100.0,
    damping=6.35,
)
KANGAROO_LEGS_3_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=("leg_.*_3_joint",),
    effort_limit=139.0,
    armature=0.01,
    stiffness=100.0,
    damping=6.35,
)
KANGAROO_LEGS_4_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=("leg_.*_4_joint",),
    effort_limit=140.0,
    armature=0.01,
    stiffness=100.0,
    damping=6.35,
)
KANGAROO_LEGS_5_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=("leg_.*_5_joint",),
    effort_limit=82.0,
    armature=0.01,
    stiffness=40.0,
    damping=2.55,
)
KANGAROO_LEGS_LENGTH_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=("leg_.*_length_joint",),
    effort_limit=1100.0,
    armature=0.01,
    stiffness=1100.0,
    damping=70.0,
)
# ARMS ACTUATORS
KANGAROO_ARMS_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=(
        "arm_.*",
    ),
    armature=0.01,
    effort_limit=43.0,
    stiffness=100.0,
    damping=10.0,
)
# PELVIS ACTUATORS
KANGAROO_PELVIS_1_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=("pelvis_1_joint",),
    effort_limit=100.0,
    armature=0.01,
    stiffness=80.0,
    damping=5.1,
)
KANGAROO_PELVIS_2_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=("pelvis_2_joint",),
    effort_limit=100.0,
    armature=0.01,
    stiffness=40.0,
    damping=2.55,
)

# TODO: hands and gripper actuators cfg

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

_foot_regex = ".*_foot_collision"

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
        KANGAROO_LEGS_1_ACTUATOR_CFG, KANGAROO_LEGS_2_ACTUATOR_CFG, KANGAROO_LEGS_3_ACTUATOR_CFG, # hips
        KANGAROO_LEGS_4_ACTUATOR_CFG, KANGAROO_LEGS_5_ACTUATOR_CFG, # ankles
        KANGAROO_LEGS_LENGTH_ACTUATOR_CFG,
        KANGAROO_ARMS_ACTUATOR_CFG,
        KANGAROO_PELVIS_1_ACTUATOR_CFG, KANGAROO_PELVIS_2_ACTUATOR_CFG,
    ),
    soft_joint_pos_limit_factor=0.9,
)
KANGAROO_HANDS_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        KANGAROO_LEGS_1_ACTUATOR_CFG, KANGAROO_LEGS_2_ACTUATOR_CFG, KANGAROO_LEGS_3_ACTUATOR_CFG, # hips
        KANGAROO_LEGS_4_ACTUATOR_CFG, KANGAROO_LEGS_5_ACTUATOR_CFG, # ankles
        KANGAROO_LEGS_LENGTH_ACTUATOR_CFG,
        KANGAROO_ARMS_ACTUATOR_CFG,
        KANGAROO_PELVIS_1_ACTUATOR_CFG, KANGAROO_PELVIS_2_ACTUATOR_CFG,
        #KANGAROO_HANDS_ACTUATOR_CFG,
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
    """Get a fresh KANGAROO with hands (7 DoF per arms) robot configuration instance."""
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_kangaroo_hands_spec,
        articulation=KANGAROO_HANDS_ARTICULATION,
    )

KANGAROO_ACTION_SCALE: dict[str, float] = {}
KANGAROO_HANDS_ACTION_SCALE: dict[str, float] = {}

for a in KANGAROO_ARTICULATION.actuators:
    e = a.effort_limit
    s = a.stiffness
    names = a.joint_names_expr

    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}

    for n in names:
        if n in e and n in s and s[n]:
            KANGAROO_ACTION_SCALE[n] = 0.25 * e[n] / s[n]

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
