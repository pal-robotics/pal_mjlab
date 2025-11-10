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

KANGAROO_XML: Path = (
    PAL_MJLAB_SRC_PATH / "robots" / "pal_kangaroo" / "xmls" / "kangaroo.xml"
)
assert KANGAROO_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, KANGAROO_XML.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(KANGAROO_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Actuator config.
##


KANGAROO_LEGS_PASSIVE_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=(
        "leg_.*_femur_joint",
        "leg_.*_knee_joint",
    ),
    effort_limit=100.0,
    armature=0.01,
    stiffness=0.0,
    damping=0.0,
)


KANGAROO_LEGS_1_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=(
        "leg_.*_1_joint",
    ),
    effort_limit=80.0,
    armature=0.01,
    stiffness=40.0,
    damping=2.55,
)

KANGAROO_LEGS_2_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=(
        "leg_.*_2_joint",
    ),
    effort_limit=230.0,
    armature=0.01,
    stiffness=100.0,
    damping=6.35,
)
KANGAROO_LEGS_3_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=(
        "leg_.*_3_joint",
    ),
    effort_limit=139.0,
    armature=0.01,
    stiffness=100.0,
    damping=6.35,
)
KANGAROO_LEGS_4_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=(
        "leg_.*_4_joint",
    ),
    effort_limit=140.0,
    armature=0.01,
    stiffness=100.0,
    damping=6.35,
)
KANGAROO_LEGS_5_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=(
        "leg_.*_5_joint",
    ),
    effort_limit=82.0,
    armature=0.01,
    stiffness=40.0,
    damping=2.55,
)

KANGAROO_LEGS_LENGTH_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=(
        "leg_.*_length_joint",
    ),
    effort_limit=1100.0,
    armature=0.01,
    stiffness=1100.0,
    damping=70.0,
)
KANGAROO_ARMS_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=(
        "arm_.*_1_joint",
        "arm_.*_2_joint",
        "arm_.*_3_joint",
        "arm_.*_4_joint",
    ),
    armature=0.01,
    effort_limit=43.0,
    stiffness=100.0,
    damping=10.0,
)
KANGAROO_PELVIS_1_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=(
        "pelvis_1_joint",
    ),
    effort_limit=100.0,
    armature=0.01,
    stiffness=80.0,
    damping=5.1,
)

KANGAROO_PELVIS_2_ACTUATOR_CFG = ActuatorCfg(
    joint_names_expr=(
        "pelvis_2_joint",
    ),
    effort_limit=100.0,
    armature=0.01,
    stiffness=40.0,
    damping=2.55,
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

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=[_foot_regex),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
)

# This enables all collisions, excluding self collisions.
# Foot collisions are given custom condim, friction and solimp.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=[".*_collision"),
    condim={_foot_regex: 3, ".*_collision": 1},
    priority={_foot_regex: 1},
    friction={_foot_regex: (0.6,)},
)

##
# Final config.
##

KANGAROO_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        KANGAROO_LEGS_1_ACTUATOR_CFG,
        KANGAROO_LEGS_2_ACTUATOR_CFG,
        KANGAROO_LEGS_3_ACTUATOR_CFG,
        KANGAROO_LEGS_4_ACTUATOR_CFG,
        KANGAROO_LEGS_5_ACTUATOR_CFG,
        KANGAROO_LEGS_LENGTH_ACTUATOR_CFG,
        # KANGAROO_LEGS_PASSIVE_ACTUATOR_CFG,
        KANGAROO_ARMS_ACTUATOR_CFG,
        KANGAROO_PELVIS_1_ACTUATOR_CFG,
        KANGAROO_PELVIS_2_ACTUATOR_CFG,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_kangaroo_robot_cfg() -> EntityCfg:
    """Get a fresh KANGAROO robot configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=KANGAROO_ARTICULATION,
    )


KANGAROO_ACTION_SCALE: dict[str, float] = {}

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


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(get_kangaroo_robot_cfg())

    viewer.launch(robot.spec.compile())
