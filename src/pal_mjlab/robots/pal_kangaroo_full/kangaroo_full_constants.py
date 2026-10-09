"""PAL Robotics Kangaroo (full-model) constants."""

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

KANG_FULL_XML: Path = (
    PAL_MJLAB_SRC_PATH / "robots" / "pal_kangaroo_full" / "xmls" / "kangaroo_full.xml"
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

KANG_FULL_BENT_KNEES_JOINTS = {
    "baselink__left_hip_z_motor": 0.012612334452569485,
    "left_hip_z_slider": -0.0069928900338709354,
    "left_hip_z_yaw": 0.17689171433448792,
    "left_hipyaw_yaw__hip_xy_bracket_l": 0.0,
    "left_hip_xy_bracket_l__hip_xy_motor_l": 0.0,
    "left_hip_xy_slider_l": 0.0,
    "left_hip_xy_baselink__hip_xy_bracket_r": 0.0,
    "left_hip_xy_bracket_r__hip_xy_motor_r": 0.0,
    "left_hip_xy_slider_r": 0.0,
    "left_hip_xy_pitch": 0.0,
    "left_hip_xy_roll": -0.001043891767039895,
    "left_hip_xy_legholder__leg_length_femur": 0.4631879925727844,
    "left_leg_length_slider": 0.049737099558115005,
    "left_leg_length_slider__leg_length_bar3": 0.2861635684967041,
    "left_leg_length_slider__leg_length_bar4": 0.2861635386943817,
    "left_leg_length_femur__leg_length_triangle": -0.9329532384872437,
    "left_leg_length_triangle__leg_length_bar2": 0.9330295324325562,
    "left_ankle_xy_femur__ankle_xy_butterfly_l": -0.493692010641098,
    "left_ankle_xy_butterfly_l__ankle_xy_bar2_l": 0.43707865476608276,
    "left_ankle_xy_femur__ankle_xy_butterfly_r": -0.49363481998443604,
    "left_ankle_xy_butterfly_r__ankle_xy_bar2_r": 0.43713614344596863,
    "left_knee": -0.9305110573768616,
    "left_ankle_xy_pitch": 0.4340316653251648,
    "left_ankle_xy_roll": -4.455562157090753e-05,
    "left_leg_length_baselink__leg_length_bar1": -0.10036752372980118,
    "left_hip_xy_legholder__ankle_xy_motor_l": 0.0004405663057696074,
    "left_ankle_xy_slider_l": -0.0007190561154857278,
    "left_hip_xy_legholder__ankle_xy_motor_r": 0.00043989793630316854,
    "left_ankle_xy_slider_r": -0.0007202239357866347,
    "left_hip_xy_legholder__ankle_xy_crank_l": -0.028633885085582733,
    "left_ankle_xy_crank_l__ankle_xy_bar1_l": 0.49185872077941895,
    "left_hip_xy_legholder__ankle_xy_crank_r": -0.0285696592181921,
    "left_ankle_xy_crank_r__ankle_xy_bar1_r": 0.4917941689491272,
    "baselink__right_hip_z_motor": -0.012612465769052505,
    "right_hip_z_slider": -0.006992915645241737,
    "right_hip_z_yaw": -0.1768926978111267,
    "right_hipyaw_yaw__hip_xy_bracket_l": 0.0,
    "right_hip_xy_bracket_l__hip_xy_motor_l": 0.0,
    "right_hip_xy_slider_l": 0.0,
    "right_hip_xy_baselink__hip_xy_bracket_r": 0.0,
    "right_hip_xy_bracket_r__hip_xy_motor_r": 0.0,
    "right_hip_xy_slider_r": 0.0,
    "right_hip_xy_pitch": 0.0,
    "right_hip_xy_roll": 0.0010353424586355686,
    "right_hip_xy_legholder__leg_length_femur": 0.46318718791007996,
    "right_leg_length_slider": 0.049737025052309036,
    "right_leg_length_slider__leg_length_bar3": 0.2861640155315399,
    "right_leg_length_slider__leg_length_bar4": 0.28616413474082947,
    "right_leg_length_femur__leg_length_triangle": -0.9329521059989929,
    "right_leg_length_triangle__leg_length_bar2": 0.9330282211303711,
    "right_ankle_xy_femur__ankle_xy_butterfly_l": -0.4936346411705017,
    "right_ankle_xy_butterfly_l__ankle_xy_bar2_l": 0.4371365010738373,
    "right_ankle_xy_femur__ankle_xy_butterfly_r": -0.49369052052497864,
    "right_ankle_xy_butterfly_r__ankle_xy_bar2_r": 0.43708041310310364,
    "right_knee": -0.9305117130279541,
    "right_ankle_xy_pitch": 0.43403327465057373,
    "right_ankle_xy_roll": 0.0,
    "right_leg_length_baselink__leg_length_bar1": -0.10036752372980118,
    "right_hip_xy_legholder__ankle_xy_motor_l": 0.00043951268889941275,
    "right_ankle_xy_slider_l": -0.0007198702660389245,
    "right_hip_xy_legholder__ankle_xy_motor_r": 0.00044148124288767576,
    "right_ankle_xy_slider_r": -0.0007187846931628883,
    "right_hip_xy_legholder__ankle_xy_crank_l": -0.028571173548698425,
    "right_ankle_xy_crank_l__ankle_xy_bar1_l": 0.49179553985595703,
    "right_hip_xy_legholder__ankle_xy_crank_r": -0.02863118425011635,
    "right_ankle_xy_crank_r__ankle_xy_bar1_r": 0.49185508489608765,
}

# Passive
KANG_FULL_LEGS_PASSIVE_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=KANG_FULL_PASSIVE_JOINTS,
    effort_limit=100.0,
    armature=0.01,
    stiffness=0.0,
    damping=0.0,
)

KANG_FULL_HIP_Z_SLIDERS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=[
        ".*_hip_z_slider",
    ],
    effort_limit=3000.0,
    armature=0.01,
    stiffness=75000.0,
    damping=550.0,
)

KANG_FULL_HIP_XY_SLIDERS_L_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=[
        ".*_hip_xy_slider_l",
    ],
    effort_limit=3000.0,
    armature=0.01,
    stiffness=175000.0,
    damping=840.0,
)

KANG_FULL_HIP_XY_SLIDERS_R_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=[
        ".*_hip_xy_slider_r",
    ],
    effort_limit=3000.0,
    armature=0.01,
    stiffness=175000.0,
    damping=840.0,
)

KANG_FULL_ANKLE_XY_SLIDERS_L_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=[
        ".*_ankle_xy_slider_l",
    ],
    effort_limit=3000.0,
    armature=0.01,
    stiffness=200000.0,
    damping=890.0,
)

KANG_FULL_ANKLE_XY_SLIDERS_R_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=[
        ".*_ankle_xy_slider_r",
    ],
    effort_limit=3000.0,
    armature=0.01,
    stiffness=200000.0,
    damping=890.0,
)

KANG_FULL_LEG_LENGTH_SLIDERS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=[
        ".*_leg_length_slider$",
    ],
    effort_limit=5000.0,
    armature=0.01,
    stiffness=250000.0,
    damping=1000.0,
)

KANG_FULL_ARMS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=[
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

KANG_FULL_PELVIS_1_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=[
        "pelvis_1_joint",
    ],
    effort_limit=100.0,
    armature=0.01,
    stiffness=500.0,
    damping=0.0,
)

KANG_FULL_PELVIS_2_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    target_names_expr=[
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
        ".*_hip_z_slider": 0.0,
        ".*_hip_xy_slider_l": 0.0,
        ".*_hip_xy_slider_r": 0.0,
        ".*_ankle_xy_slider_l": 0.0,
        ".*_ankle_xy_slider_r": 0.0,
        ".*_leg_length_slider$": 0.0,
    },
    joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.85),
    joint_pos=KANG_FULL_BENT_KNEES_JOINTS,
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
        KANG_FULL_PELVIS_2_ACTUATOR_CFG
    ),
    soft_joint_pos_limit_factor=0.99,
)


_EXCLUDED_JOINTS = { }


_ROBOT_CONFIGS = {
    "kangaroo_full": (get_spec, KANG_FULL_ARTICULATION, FULL_COLLISION),
}


def _make_robot_cfg(variant: str) -> EntityCfg:
    return EntityCfg(
        init_state=KNEES_BENT_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=KANG_FULL_ARTICULATION,
    )


def get_kangaroo_full_robot_cfg() -> EntityCfg:
    return _make_robot_cfg("kangaroo_full")


def _build_action_scales(
    articulation: EntityArticulationInfoCfg, exclude: set = frozenset()
) -> tuple[dict, tuple]:
    """Build action scale dict and actuator names from articulation config."""
    scales, names = {}, []
    for a in articulation.actuators:
        e = (
            a.effort_limit
            if isinstance(a.effort_limit, dict)
            else {n: a.effort_limit for n in a.target_names_expr}
        )
        s = (
            a.stiffness
            if isinstance(a.stiffness, dict)
            else {n: a.stiffness for n in a.target_names_expr}
        )
        for n in a.target_names_expr:
            if n in e and n in s and s[n] and n not in exclude:
                scales[n] = 0.25 * e[n] / s[n]
                names.append(n)
    return scales, tuple(names)

##
# Final config.
##

KANG_FULL_ACTION_SCALE, KANG_FULL_ACTUATOR_NAMES = _build_action_scales(
    KANG_FULL_ARTICULATION, _EXCLUDED_JOINTS)


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_kangaroo_full_robot_cfg())
    viewer.launch(robot.spec.compile())
