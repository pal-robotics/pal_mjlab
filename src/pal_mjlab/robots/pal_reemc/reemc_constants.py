"""PAL Robotics REEM-C constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from pal_mjlab import PAL_MJLAB_SRC_PATH

##
# MJCF and assets.
##

REEMC_XML: Path = PAL_MJLAB_SRC_PATH / "robots" / "pal_reemc" / "xmls" / "reemc.xml"
assert REEMC_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, REEMC_XML.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(REEMC_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


# Arms joints params
# joints armature (reflected inertia)
ARM_1234_ARMATURE = 0.5
ARM_567_ARMATURE = 0.1
# joints effort limit
ARM_1234_EFFORT_LIMIT = 54.0
ARM_5_EFFORT_LIMIT = 3.5
ARM_67_EFFORT_LIMIT = 7.0
# joints stiffness
ARM_1234_STIFFNESS = 3000.0
ARM_567_STIFFNESS = 500.0
# joints damping
ARM_1234_DAMPING = 10.0
ARM_567_DAMPING = 5.0

# Head joints params
HEAD_12_ARMATURE = 0.1
# joints effort limit
HEAD_12_EFFORT_LIMIT = 4.0
# joints stiffness
HEAD_12_STIFFNESS = 300.0
# joints damping
HEAD_12_DAMPING = 0.1

# Torso joint params
TORSO_12_ARMATURE = 1.0
# joints effort limit
TORSO_12_EFFORT_LIMIT = 108.0
# joints stiffness
TORSO_12_STIFFNESS = 10000.0
# joints damping
TORSO_12_DAMPING = 10.0

# Leg joints params
LEG_123456_ARMATURE = 0.5
# joints effort limit
LEG_13_EFFORT_LIMIT = 127.0
LEG_26_EFFORT_LIMIT = 76.0
LEG_4_EFFORT_LIMIT = 284.0
LEG_5_EFFORT_LIMIT = 147.0
# joints stiffness
LEG_123456_STIFFNESS = 3000.0
# joints damping
LEG_123456_DAMPING = 10.0

##
# Actuator config.
##

# arm actuators
ARM_1234_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "arm_.*_1_joint",
        "arm_.*_2_joint",
        "arm_.*_3_joint",
        "arm_.*_4_joint",
    ),
    effort_limit=ARM_1234_EFFORT_LIMIT,
    armature=ARM_1234_ARMATURE,
    stiffness=ARM_1234_STIFFNESS,
    damping=ARM_1234_DAMPING,  
)

ARM_5_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "arm_.*_5_joint",
    ),
    effort_limit=ARM_5_EFFORT_LIMIT,
    armature=ARM_567_ARMATURE,
    stiffness=ARM_567_STIFFNESS,
    damping=ARM_567_DAMPING,
)
ARM_67_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "arm_.*_6_joint",
        "arm_.*_7_joint",
    ),
    effort_limit=ARM_67_EFFORT_LIMIT,
    armature=ARM_567_ARMATURE,
    stiffness=ARM_567_STIFFNESS,
    damping=ARM_567_DAMPING,
)
# torso actuators
TORSO_12_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=("torso_.*_joint",),
    effort_limit=TORSO_12_EFFORT_LIMIT,
    armature=TORSO_12_ARMATURE,
    stiffness=TORSO_12_STIFFNESS,
    damping=TORSO_12_DAMPING,
)
# head actuators
HEAD_12_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=("head_.*_joint",),
    effort_limit=HEAD_12_EFFORT_LIMIT,
    armature=HEAD_12_ARMATURE,
    stiffness=HEAD_12_STIFFNESS,
    damping=HEAD_12_DAMPING,
)
# leg actuators
LEG_13_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_1_joint", "leg_.*_3_joint"),
    effort_limit=LEG_13_EFFORT_LIMIT,
    armature=LEG_123456_ARMATURE,
    stiffness=LEG_123456_STIFFNESS,
    damping=LEG_123456_DAMPING,
)
LEG_26_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_2_joint", "leg_.*_6_joint"),
    effort_limit=LEG_26_EFFORT_LIMIT,
    armature=LEG_123456_ARMATURE,
    stiffness=LEG_123456_STIFFNESS,
    damping=LEG_123456_DAMPING,
)
LEG_4_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_4_joint",),
    effort_limit=LEG_4_EFFORT_LIMIT,
    armature=LEG_123456_ARMATURE,
    stiffness=LEG_123456_STIFFNESS,
    damping=LEG_123456_DAMPING,
)
LEG_5_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_5_joint",),
    effort_limit=LEG_5_EFFORT_LIMIT,
    armature=LEG_123456_ARMATURE,
    stiffness=LEG_123456_STIFFNESS,
    damping=LEG_123456_DAMPING,
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
        "leg_.*_4_joint": 0.641,
        "leg_.*_5_joint": -0.318,
        "leg_.*_6_joint": 0.0,
        # arms
        "arm_.*_1_joint": -0.44,
        "arm_.*_2_joint": 0.26,
        "arm_.*_3_joint": 0.0,
        "arm_.*_4_joint": 0.88,
        "arm_.*_5_joint": 0.0,
        "arm_.*_6_joint": 0.0,
        "arm_.*_7_joint": 0.0,
        # head
        "head_1_joint": 0.0,
        "head_2_joint": 0.0,
        # torso
        "torso_1_joint": 0.0,
        "torso_2_joint": 0.0,
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

REEMC_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        ARM_1234_ACTUATOR_CFG,
        ARM_5_ACTUATOR_CFG,
        ARM_67_ACTUATOR_CFG,
        TORSO_12_ACTUATOR_CFG,
        HEAD_12_ACTUATOR_CFG,
        LEG_13_ACTUATOR_CFG,
        LEG_26_ACTUATOR_CFG,
        LEG_4_ACTUATOR_CFG,
        LEG_5_ACTUATOR_CFG,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_reemc_robot_cfg() -> EntityCfg:
    """Get a fresh Talos robot configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=REEMC_ARTICULATION,
    )


REEMC_ACTION_SCALE: dict[str, float] = {}

for a in REEMC_ARTICULATION.actuators:
    e = a.effort_limit
    s = a.stiffness
    names = a.target_names_expr

    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}

    for n in names:
        if n in e and n in s and s[n]:
            REEMC_ACTION_SCALE[n] = 0.25 * e[n] / s[n]


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_reemc_robot_cfg())

    viewer.launch(robot.spec.compile())
