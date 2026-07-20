"""PAL Robotics TIAGo PRO constants."""

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from pal_mjlab import PAL_MJLAB_SRC_PATH

##
# MJCF and assets.
##

TIAGO_PRO_XML: Path = (
  PAL_MJLAB_SRC_PATH / "robots" / "pal_tiago_pro" / "xmls" / "tiago_pro.xml"
)
assert TIAGO_PRO_XML.exists()


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(TIAGO_PRO_XML))
  return spec


##
# Actuator Parameters (BeyondMimic methodology)
##

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0
FACTOR = 1.0


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
GRIPPER = _calc_actuator_params(101, 1.3e-5, 8)
TORSO = {"armature": 0.1, "stiffness": 1500.0, "damping": 300.0, "effort_limit": 2200.0}


## --------------------------------------------------------
# Actuator configurations.
## --------------------------------------------------------

# Arms
TIAGO_PRO_S_PLUS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(r"arm_right_(1|2)_joint",),
  **S_PLUS,
)
TIAGO_PRO_S_MINUS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(r"arm_right_(?![1267]_joint)\d+_joint",),
  **S_MINUS,
)
TIAGO_PRO_XS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(r"arm_right_(?![12345]_joint)\d+_joint",),
  **XS,
)
# Torso
TORSO_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("torso_lift_joint",),
  **TORSO,
)
# Gripper
TIAGO_PRO_GRIPPER_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=("gripper_right_finger_joint",),
  **GRIPPER,
)

##
# Initial State
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(-0.25, 0.0, 0.0),
  joint_pos={
    # "torso_lift_joint": 0.1,
    "arm_right_1_joint": -3.14,
    "arm_right_2_joint": -1.7,
    "arm_right_3_joint": 0.71,
    "arm_right_4_joint": -1.14,
    "arm_right_5_joint": 2.8,
    "arm_right_6_joint": 1.0,  # -1
    "arm_right_7_joint": 0.0,
    "gripper_right_finger_joint": 0.045,
    "gripper_right_inner_finger_left_joint": -0.379495,
    "gripper_right_fingertip_left_joint": 0.417219,
    "gripper_right_outer_finger_left_joint": -0.353050,
    "gripper_right_finger_right_joint": 0.009982,
    "gripper_right_inner_finger_right_joint": -0.371652,
    "gripper_right_fingertip_right_joint": 0.406154,
    "gripper_right_outer_finger_right_joint": -0.349496,
    # "arm_left_1_joint": 0.36,
    # "arm_left_2_joint": -1.83,
    # "arm_left_3_joint": 0.47,
    # "arm_left_4_joint": -2.35,
    # "arm_left_5_joint": 0.0,
    # "arm_left_6_joint": -1.20,
    # "arm_left_7_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision Configs
##

# Collision enabled for all geoms except:
#   - Left arm:  col_left_*, col_upper_left_arm, col_lower_left_arm, col_ee_left
#   - Base:      col_base
#   - Torso/Head:col_torso_head
#   - Right arm joints 3, 5, 7: col_upper_right_arm, col_lower_right_arm, col_arm_right_7
# (right arm joint 1 has no collision geom in the XML)
# Non-matched geoms are disabled automatically (disable_other_geoms=True by default).
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(
    r"^(?!"
    r"col_base$|col_torso_head$"
    r"|col_upper_left_arm$|col_lower_left_arm$|col_ee_left$|col_left_"
    r"|col_upper_right_arm$|col_lower_right_arm$|col_arm_right_7$"
    r").*",
  ),
  condim=3,
  priority=1,
  friction=(0.7,),
)

# Only overrides friction for right fingertips; does NOT disable other geoms
# (disable_other_geoms=False) to avoid conflicting with FULL_COLLISION.
FINGERTIP_COLLISION = CollisionCfg(
  geom_names_expr=(r".*right.*fingertip.*",),
  condim=3,
  priority=1,
  friction=(1.0,),
  disable_other_geoms=False,
)

##
# Articulation Config
##

TIAGO_PRO_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    TIAGO_PRO_S_PLUS_ACTUATOR_CFG,
    TIAGO_PRO_S_MINUS_ACTUATOR_CFG,
    TIAGO_PRO_XS_ACTUATOR_CFG,
    TIAGO_PRO_GRIPPER_ACTUATOR_CFG,
  ),
  soft_joint_pos_limit_factor=0.9,
)

TIAGO_PRO_ACTION_SCALE: dict[str, float] = {}
TIAGO_PRO_ACTUATOR_NAMES: tuple = ()


def get_tiago_pro_robot_cfg() -> EntityCfg:
  """Get a fresh TIAGo Pro robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION, FINGERTIP_COLLISION),
    spec_fn=get_spec,
    articulation=TIAGO_PRO_ARTICULATION,
  )


for a in TIAGO_PRO_ARTICULATION.actuators:
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr

  if not isinstance(e, dict):
    e = {n: e for n in names}
  if not isinstance(s, dict):
    s = {n: s for n in names}

  for n in names:
    if n in e and n in s and s[n]:
      TIAGO_PRO_ACTION_SCALE[n] = 0.05 * e[n] / s[n]  # 0.05
      TIAGO_PRO_ACTUATOR_NAMES += (n,)

# Override gripper scale: tuned to 0.01 (formula-derived ~0.001528 is too small for control)
TIAGO_PRO_ACTION_SCALE["gripper_right_finger_joint"] = 0.01  # 0.01


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_tiago_pro_robot_cfg())
  viewer.launch(robot.spec.compile())
