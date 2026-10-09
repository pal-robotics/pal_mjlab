"""PAL Robotics Kangaroo (full-model) constants."""

from pathlib import Path

import mujoco
import torch
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from pal_mjlab import PAL_MJLAB_SRC_PATH

##
# MJCF and assets.
##

_LEG_ACTUATORS_EFFORT_LIMITS = [5000.0, 2000.0]  # [LEG LENGTH, OTHERS]

KANG_FULL_XML: Path = (
  PAL_MJLAB_SRC_PATH / "robots" / "pal_kangaroo_full" / "xmls" / "kangaroo_full.xml"
)
assert KANG_FULL_XML.exists()


HIP_XY_CONVEX_HULL_POINTS = torch.tensor(
  [
    [-0.742, 0.035],
    [-0.742, -0.094],
    [-0.742, -0.167],
    [-0.707, -0.243],
    [-0.655, -0.349],
    [-0.61, -0.411],
    [-0.344, -0.413],
    [-0.061, -0.41],
    [0.307, -0.404],
    [0.486, -0.4],
    [0.55, -0.354],
    [0.638, -0.282],
    [0.709, -0.186],
    [0.72, -0.081],
    [0.722, 0.054],
    [0.708, 0.18],
    [0.641, 0.301],
    [0.531, 0.389],
    [0.448, 0.45],
    [0.171, 0.453],
    [-0.164, 0.455],
    [-0.434, 0.461],
    [-0.605, 0.467],
    [-0.659, 0.404],
    [-0.713, 0.309],
    [-0.742, 0.222],
    [-0.742, 0.133],
  ]
)

ANKLE_XY_CONVEX_HULL_POINTS = torch.tensor(
  [
    [0.707, 0.005],
    [0.648, 0.112],
    [0.576, 0.23],
    [0.484, 0.38],
    [0.443, 0.439],
    [0.266, 0.443],
    [0.008, 0.441],
    [-0.293, 0.45],
    [-0.46, 0.448],
    [-0.505, 0.379],
    [-0.594, 0.244],
    [-0.686, 0.098],
    [-0.744, 0.001],
    [-0.688, -0.099],
    [-0.604, -0.231],
    [-0.499, -0.394],
    [-0.445, -0.472],
    [-0.254, -0.469],
    [0.005, -0.462],
    [0.232, -0.456],
    [0.429, -0.46],
    [0.475, -0.382],
    [0.583, -0.207],
    [0.665, -0.071],
  ]
)


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


def _calc_leg_params(stiffness: float, effort: float) -> dict:
  """Calculate leg actuator parameters."""
  damping = round(2.0 * DAMPING_RATIO * stiffness / NATURAL_FREQ, 3)
  return {
    "armature": 0.01,
    "stiffness": stiffness,
    "damping": damping,
    "effort_limit": effort,
  }


# Motor parameters: (gear_ratio, motor_inertia, effort_limit)
S_PLUS = _calc_actuator_params(121, 1.728e-5, 50)


def _load_spec(xml_path: Path) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(xml_path))
  return spec


def get_kangaroo_spec() -> mujoco.MjSpec:
  return _load_spec(KANG_FULL_XML)


##
# Actuator config.
##

KANG_FULL_HIP_Z_SLIDERS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_z_slider",),
  **_calc_leg_params(16000.0, _LEG_ACTUATORS_EFFORT_LIMITS[1]),
)

KANG_FULL_HIP_XY_SLIDERS_L_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_xy_slider_l",),
  **_calc_leg_params(16000.0, _LEG_ACTUATORS_EFFORT_LIMITS[1]),
)

KANG_FULL_HIP_XY_SLIDERS_R_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_xy_slider_r",),
  **_calc_leg_params(16000.0, _LEG_ACTUATORS_EFFORT_LIMITS[1]),
)

KANG_FULL_ANKLE_XY_SLIDERS_L_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_xy_slider_l",),
  **_calc_leg_params(16000.0, _LEG_ACTUATORS_EFFORT_LIMITS[1]),
)

KANG_FULL_ANKLE_XY_SLIDERS_R_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_xy_slider_r",),
  **_calc_leg_params(16000.0, _LEG_ACTUATORS_EFFORT_LIMITS[1]),
)

KANG_FULL_LEG_LENGTH_SLIDERS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_leg_length_slider$",),
  **_calc_leg_params(200000.0, _LEG_ACTUATORS_EFFORT_LIMITS[0]),
)

KANG_FULL_ARMS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "arm_.*_1_joint",
    "arm_.*_2_joint",
    "arm_.*_3_joint",
    "arm_.*_4_joint",
  ),
  **S_PLUS,
)

KANG_FULL_PELVIS_1_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("pelvis_1_joint",),
  **S_PLUS,
)

KANG_FULL_PELVIS_2_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("pelvis_2_joint",),
  **S_PLUS,
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
    "arm_left_1_joint": 0.24,
    "arm_right_1_joint": -0.24,
    "arm_.*_2_joint": 1.32,
    "arm_left_3_joint": 1.57,
    "arm_right_3_joint": -1.57,
    "arm_.*_4_joint": 0.8,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_FOOT_REGEX = ".*_foot_collision"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_FOOT_REGEX: 3, ".*_collision": 1},
  priority={_FOOT_REGEX: 1},
  friction={_FOOT_REGEX: (0.6,)},
)


KANG_FULL_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
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
  soft_joint_pos_limit_factor=0.99,
)


def get_kangaroo_full_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_kangaroo_spec,
    articulation=KANG_FULL_ARTICULATION,
  )


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
  KANG_FULL_ARTICULATION
)


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_kangaroo_full_robot_cfg())
  viewer.launch(robot.spec.compile())
