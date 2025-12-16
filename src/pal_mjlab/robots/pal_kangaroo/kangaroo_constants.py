"""Pal Robotics KANGAROO constants."""

from pathlib import Path
import mujoco

from pal_mjlab import PAL_MJLAB_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg
from mjlab.actuator import BuiltinPositionActuatorCfg

# There are multiple variants of the KANGAROO robot. For simplicity, we only implemented the following:
# - kangaroo: simplified model with 4 DoF per arm and a fake forearm
# - kangaroo_hands: simplified model with 5 DoF per arm and a Seed Robotics hand
# - kangaroo_gripper: simplified model with 7 DoF per arm and a gripper
# - kangaroo_full: full model with 4 DoF per arm and a fake forearm

KANGAROO_PATH = PAL_MJLAB_SRC_PATH / "robots" / "pal_kangaroo" / "xmls"
KANGAROO_XML = KANGAROO_PATH / "kangaroo.xml"
KANGAROO_HANDS_XML = KANGAROO_PATH / "kangaroo_hands.xml"
KANGAROO_GRIPPERS_XML = KANGAROO_PATH / "kangaroo_grippers.xml"

for p in [KANGAROO_PATH, KANGAROO_XML, KANGAROO_HANDS_XML, KANGAROO_GRIPPERS_XML]:
    assert p.exists(), f"Missing: {p}"

##
# Actuator Parameters (BeyondMimic methodology)
##

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0
FACTOR = 0.05


def _calc_actuator_params(gear_ratio: float, motor_inertia: float, effort: float) -> dict:
    """Calculate armature, stiffness, and damping for an actuator."""
    armature = FACTOR * motor_inertia * gear_ratio**2
    stiffness = round(armature * NATURAL_FREQ**2, 3)
    damping = round(2.0 * DAMPING_RATIO * armature * NATURAL_FREQ, 3)
    return {"armature": armature, "stiffness": stiffness, "damping": damping, "effort_limit": effort}


def _calc_leg_params(stiffness: float, effort: float) -> dict:
    """Calculate leg actuator parameters."""
    damping = round(2.0 * DAMPING_RATIO * stiffness / NATURAL_FREQ, 3)
    return {"armature": 0.01, "stiffness": stiffness, "damping": damping, "effort_limit": effort}


# Motor parameters: (gear_ratio, motor_inertia, effort_limit)
S_PLUS = _calc_actuator_params(121, 1.728e-5, 50)
S_MINUS = _calc_actuator_params(101, 1.3e-5, 25)
XS = _calc_actuator_params(101, 1.3e-5, 25)

##
# MJCF & Assets
##

def get_assets(meshdir: str) -> dict[str, bytes]:
    assets = {}
    update_assets(assets, KANGAROO_PATH / "assets", meshdir)
    return assets


def _load_spec(xml_path: Path) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(xml_path))
    spec.assets = get_assets(spec.meshdir)
    return spec


get_kangaroo_spec = lambda: _load_spec(KANGAROO_XML)
get_kangaroo_hands_spec = lambda: _load_spec(KANGAROO_HANDS_XML)
get_kangaroo_grippers_spec = lambda: _load_spec(KANGAROO_GRIPPERS_XML)

##
# Actuator Configs
##

# Legs
KANGAROO_LEG_ACTUATORS = (
    BuiltinPositionActuatorCfg(joint_names_expr=("leg_.*_1_joint",), **_calc_leg_params(100, 80)),
    BuiltinPositionActuatorCfg(joint_names_expr=("leg_.*_2_joint",), **_calc_leg_params(100, 230)),
    BuiltinPositionActuatorCfg(joint_names_expr=("leg_.*_3_joint",), **_calc_leg_params(100, 139)),
    BuiltinPositionActuatorCfg(joint_names_expr=("leg_.*_4_joint",), **_calc_leg_params(30, 140)),
    BuiltinPositionActuatorCfg(joint_names_expr=("leg_.*_5_joint",), **_calc_leg_params(30, 82)),
    BuiltinPositionActuatorCfg(joint_names_expr=("leg_.*_length_joint",), **_calc_leg_params(1600, 1100)),
)

# Arms & Torso
KANGAROO_S_PLUS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=("arm_.*_1_joint", "arm_.*_2_joint", "pelvis_1_joint", "pelvis_2_joint"),
    **S_PLUS,
)
KANGAROO_S_MINUS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(r"arm_.*_(?![1267]_joint)\d+_joint",),
    **S_MINUS,
)
KANGAROO_XS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
    joint_names_expr=(r"arm_.*_(?![12345]_joint)\d+_joint",),
    **XS,
)

COMMON_ACTUATORS = KANGAROO_LEG_ACTUATORS + (KANGAROO_S_PLUS_ACTUATOR_CFG, KANGAROO_S_MINUS_ACTUATOR_CFG)

##
# Initial State
##

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.9),
    joint_pos={
        "leg_left_1_joint": -0.012, "leg_right_1_joint": 0.012,
        "leg_.*_2_joint": 0.054,
        "leg_left_3_joint": 0.04, "leg_right_3_joint": -0.04,
        "leg_.*_length_joint": 0.6,
        "leg_.*_4_joint": -0.053, "leg_.*_5_joint": 0.0,
        "leg_.*_femur_joint": 0.9, "leg_.*_knee_joint": 1.8,
        "arm_left_1_joint": 0.24, "arm_right_1_joint": -0.24,
        "arm_.*_2_joint": 1.32,
        "arm_left_3_joint": 1.57, "arm_right_3_joint": -1.57,
        "arm_.*_4_joint": 0.8,
        "pelvis_1_joint": 0.0, "pelvis_2_joint": 0.0,
    },
    joint_vel={".*": 0.0},
)

##
# Collision Configs
##

_FOOT_REGEX = ".*_foot.*_collision"

FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(_FOOT_REGEX,), contype=0, conaffinity=1, condim=3, priority=1, friction=(0.6,)
)
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={_FOOT_REGEX: 3, ".*_collision": 1},
    priority={_FOOT_REGEX: 1},
    friction={_FOOT_REGEX: (0.6,)},
)

##
# Articulation Configs
##

KANGAROO_ARTICULATION = EntityArticulationInfoCfg(actuators=COMMON_ACTUATORS, soft_joint_pos_limit_factor=0.9)
KANGAROO_HANDS_ARTICULATION = EntityArticulationInfoCfg(actuators=COMMON_ACTUATORS, soft_joint_pos_limit_factor=0.9)
KANGAROO_GRIPPERS_ARTICULATION = EntityArticulationInfoCfg(
    actuators=COMMON_ACTUATORS + (KANGAROO_XS_ACTUATOR_CFG,), soft_joint_pos_limit_factor=0.9
)


_ROBOT_CONFIGS = {
    "kangaroo": (get_kangaroo_spec, KANGAROO_ARTICULATION, FULL_COLLISION),
    "hands": (get_kangaroo_hands_spec, KANGAROO_HANDS_ARTICULATION, FEET_ONLY_COLLISION),
    "grippers": (get_kangaroo_grippers_spec, KANGAROO_GRIPPERS_ARTICULATION, FEET_ONLY_COLLISION),
}


def _make_robot_cfg(variant: str) -> EntityCfg:
    spec_fn, articulation, collision = _ROBOT_CONFIGS[variant]
    return EntityCfg(init_state=INIT_STATE, collisions=(collision,), spec_fn=spec_fn, articulation=articulation)


get_kangaroo_robot_cfg = lambda: _make_robot_cfg("kangaroo")
get_kangaroo_hands_robot_cfg = lambda: _make_robot_cfg("hands")
get_kangaroo_grippers_robot_cfg = lambda: _make_robot_cfg("grippers")


_EXCLUDED_JOINTS = {"leg_left_knee_joint", "leg_right_knee_joint", "leg_left_femur_joint", "leg_right_femur_joint"}


def _build_action_scales(articulation: EntityArticulationInfoCfg, exclude: set = frozenset()) -> tuple[dict, tuple]:
    """Build action scale dict and actuator names from articulation config."""
    scales, names = {}, []
    for a in articulation.actuators:
        e = a.effort_limit if isinstance(a.effort_limit, dict) else {n: a.effort_limit for n in a.joint_names_expr}
        s = a.stiffness if isinstance(a.stiffness, dict) else {n: a.stiffness for n in a.joint_names_expr}
        for n in a.joint_names_expr:
            if n in e and n in s and s[n] and n not in exclude:
                scales[n] = 0.25 * e[n] / s[n]
                names.append(n)
    return scales, tuple(names)


KANGAROO_ACTION_SCALE, KANGAROO_ACTUATOR_NAMES = _build_action_scales(KANGAROO_ARTICULATION, _EXCLUDED_JOINTS)
KANGAROO_HANDS_ACTION_SCALE, KANGAROO_HANDS_ACTUATOR_NAMES = _build_action_scales(KANGAROO_HANDS_ARTICULATION)
KANGAROO_GRIPPERS_ACTION_SCALE, KANGAROO_GRIPPERS_ACTUATOR_NAMES = _build_action_scales(KANGAROO_GRIPPERS_ARTICULATION)

if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_kangaroo_robot_cfg())
    viewer.launch(robot.spec.compile())