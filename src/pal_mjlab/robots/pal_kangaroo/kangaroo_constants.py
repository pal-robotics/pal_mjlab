"""Pal Robotics KANGAROO constants."""

import math
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from pal_mjlab import PAL_MJLAB_SRC_PATH

# There are multiple variants of the KANGAROO robot. For simplicity, we only implemented the following:
# - kangaroo: simplified model with 4 DoF per arm and a fake forearm
# - kangaroo_hands: simplified model with 5 DoF per arm and a Seed Robotics hand
# - kangaroo_gripper: simplified model with 7 DoF per arm and a gripper
# - kangaroo_full: full model with 4 DoF per arm and a fake forearm

REGEX_LEG_LENGTH_JOINTS_ONLY = r"leg_.*_length_joint"
REGEX_ALL_ACTUATED_JOINTS = r"^(?!leg_.*_femur_joint$|leg_.*_knee_joint$).*$"
REGEX_FEMUR_AND_KNEE_LINKS = (
  r"^(leg_left_femur_link|leg_right_femur_link|leg_left_knee_link|leg_right_knee_link)$"
)
REGEX_JOINTS_WITHOUT_ENVELOPE = r"^(?!leg_.*_[2-5]_joint$).*$"

KANGAROO_PATH = PAL_MJLAB_SRC_PATH / "robots" / "pal_kangaroo" / "xmls"
KANGAROO_XML = KANGAROO_PATH / "kangaroo.xml"
KANGAROO_LOWER_BODY_XML = KANGAROO_PATH / "kangaroo_lower_body.xml"
KANGAROO_HANDS_XML = KANGAROO_PATH / "kangaroo_hands.xml"
KANGAROO_GRIPPERS_XML = KANGAROO_PATH / "kangaroo_grippers.xml"

for p in [
  KANGAROO_PATH,
  KANGAROO_XML,
  KANGAROO_LOWER_BODY_XML,
  KANGAROO_HANDS_XML,
  KANGAROO_GRIPPERS_XML,
]:
  assert p.exists(), f"Missing: {p}"

##
# Actuator Parameters (BeyondMimic methodology)
##

NATURAL_FREQ = 3.5 * 2.0 * math.pi  # 3.5Hz
DAMPING_RATIO = 2.0

##
# Joint limits described as affine functions and planes
##

KANGAROO_TIBIA_PITCH_OFFSET = math.radians(70.0)

KANGAROO_ANKLE_ENVELOPE_LIMITS = {
  # b0j5, dj4ndj5, j4n_min
  "foot_tibia_collision": (-1.095692, 0.730921, -0.998403),
  # a0, dadj4, dadj5, min, max
  "actuator": (0.000963, -0.031860, -0.018999, -0.022294, 0.024684),
  # b0l, dj4dl, dj4dj5
  "crank_bar_collision": (0.207172, 1.378701, 0.543604),
}

KANGAROO_HIP_ENVELOPE_LIMITS = {
  # a0, dadj2, dadj3, min, max
  "actuator": (-0.000940, 0.054716, 0.033618, -0.04, 0.04),
}


def _calc_actuator_params(
  gear_ratio: float, motor_inertia: float, effort: float
) -> dict:
  """Calculate armature, stiffness, and damping for an actuator."""
  armature = motor_inertia * gear_ratio**2
  stiffness = round(armature * NATURAL_FREQ**2, 3)
  damping = round(2.0 * DAMPING_RATIO * armature * NATURAL_FREQ, 3)
  return {
    "armature": armature,
    "stiffness": stiffness,
    "damping": damping,
    "effort_limit": effort,
  }


def _calc_leg_params(
  effort: float,
  armature: float,
  frictionloss: float,
  viscous_damping: float,
) -> dict:
  """Calculate leg actuator parameters."""
  stiffness = round(armature * NATURAL_FREQ**2, 3)
  damping = round(2.0 * DAMPING_RATIO * armature * NATURAL_FREQ, 3)
  return {
    "armature": armature,
    "stiffness": stiffness,
    "damping": damping,
    "effort_limit": effort,
    "frictionloss": frictionloss,
    "viscous_damping": viscous_damping,
  }


# Motor parameters: (gear_ratio, motor_inertia, effort_limit)
S_PLUS = _calc_actuator_params(121, 3.3e-05, 50)  # joint armature = 0.483
S_MINUS = _calc_actuator_params(101, 2.99e-05, 25)  # joint armature = 0.305
XS = _calc_actuator_params(101, 1.99e-05, 25)  # joint armature = 0.203

##
# MJCF & Assets
##


def _load_spec(xml_path: Path) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(xml_path))
  return spec


def get_kangaroo_spec() -> mujoco.MjSpec:
  return _load_spec(KANGAROO_XML)


def get_kangaroo_lower_body_spec() -> mujoco.MjSpec:
  return _load_spec(KANGAROO_LOWER_BODY_XML)


def get_kangaroo_hands_spec() -> mujoco.MjSpec:
  return _load_spec(KANGAROO_HANDS_XML)


def get_kangaroo_grippers_spec() -> mujoco.MjSpec:
  return _load_spec(KANGAROO_GRIPPERS_XML)


##
# Joint envelopes: the limits above as A x <= b
##


def _normalize_envelope(
  A: list[list[float]], b: list[float]
) -> tuple[list[list[float]], list[float]]:
  """Divide every row of ``A x <= b`` by the length of its first two columns.

  A x - b is then the signed distance [rad] to every edge in the plane of the first
  two joints. The remaining joints only shift the edges, so they are left out of the
  norm.
  """
  norms = [math.hypot(row[0], row[1]) for row in A]
  A = [[value / n for value in row] for row, n in zip(A, norms, strict=True)]
  b = [bound / n for bound, n in zip(b, norms, strict=True)]
  return A, b


def get_kangaroo_hip_envelope(
  shrink_joint: float = 0.0,
  shrink_actuator: float = 0.0,
) -> tuple[list[list[float]], list[float]]:
  """Feasible joint space of hip xy as ``A x <= b`` in x = (j2, j3).

  The space is mostly bounded by the joint 3 range and the actuator strokes;
  the actuators are modelled as affine lines, symmetric along both joints. Joint
  limits come from the model, the actuators are described by
  ``KANGAROO_HIP_ENVELOPE_LIMITS``:

    a0: stroke of both actuators at j2 = j3 = 0 [m].
    dadj2, dadj3: rates of change of the actuator with respect to joints 2 and
      3, representing linearized reprojections of the actuator limits onto joint
      space [m/rad].
    min, max: actuator stroke limits [m].

  ``shrink_joint`` [rad] and ``shrink_actuator`` [m] are the margins of the joint
  and actuator limits. A x - b is the signed distance [rad] in the (j2, j3) plane.

  For a visual explanation of the parameters see
    https://github.com/pal-robotics/pal_mjlab/pull/133
  """
  # Both legs of all variants share these ranges.
  spec = get_kangaroo_spec()
  j2_min, j2_max = spec.joint("leg_left_2_joint").range.tolist()
  j3_min, j3_max = spec.joint("leg_left_3_joint").range.tolist()

  a0, dadj2, dadj3, act_min, act_max = KANGAROO_HIP_ENVELOPE_LIMITS["actuator"]
  act_min, act_max = act_min + shrink_actuator, act_max - shrink_actuator

  # fmt: off
  A = [
    # j2       j3
    [-1.0,     0.0   ],  # 0 j2 >= j2_min
    [ 1.0,     0.0   ],  # 1 j2 <= j2_max
    [ 0.0,     1.0   ],  # 2 j3 <= j3_max
    [ 0.0,    -1.0   ],  # 3 j3 >= j3_min
    [ dadj2,   dadj3 ],  # 4 a2 <= act_max
    [-dadj2,  -dadj3 ],  # 5 a2 >= act_min
    [ dadj2,  -dadj3 ],  # 6 a3 <= act_max
    [-dadj2,   dadj3 ],  # 7 a3 >= act_min
  ]
  b = [
    -(j2_min + shrink_joint),  # 0
    j2_max - shrink_joint,     # 1
    j3_max - shrink_joint,     # 2
    -(j3_min + shrink_joint),  # 3
    act_max - a0,              # 4
    a0 - act_min,              # 5
    act_max - a0,              # 6
    a0 - act_min,              # 7
  ]
  # fmt: on

  return _normalize_envelope(A, b)


def get_kangaroo_ankle_envelope(
  shrink_joint: float = 0.0,
  shrink_actuator: float = 0.0,
  shrink_crank: float = 0.0,
) -> tuple[list[list[float]], list[float]]:
  """Feasible joint space of the ankle as ``A x <= b`` in x = (j4, j5, L, femur, knee).

  The space is bounded by the joint 5 range, the foot-tibia and crank-bar
  collisions and the actuator strokes. The collisions and actuators are modelled
  as affine lines, symmetric in joint 5; the collisions are written as one row per
  sign of joint 5. The foot-tibia limits act on the foot angle relative to the tibia
  instead of the hip, which is linear in x: j4n = j4 - femur + knee - 70 deg; A holds
  its joint coefficients and b its constant offset. Joint limits come from the model,
  the other limits are described by ``KANGAROO_ANKLE_ENVELOPE_LIMITS``:

    foot_tibia_collision:
      j4n_min: lowest foot angle relative to the tibia [rad].
      b0j5: j4n where the affine lines approximating the foot-tibia collision
        meet, at j5 = 0 [rad].
      dj4ndj5: rate of change of j4n with respect to |j5|, the slope of the
        foot-tibia collision [rad/rad].
    actuator:
      a0: stroke of both actuators at j4 = j5 = 0 [m].
      dadj4, dadj5: rates of change of the actuator with respect to joints 4 and
        5 [m/rad].
      min, max: actuator stroke limits [m].
    crank_bar_collision:
      b0l: value of j4 at the crank-bar collision when L = 0, j5 = 0 [rad].
      dj4dl, dj4dj5: rates of change of j4 with respect to leg length [rad/m] and
        |j5| [rad/rad], approximating the reprojected crank-bar collision with a
        plane.

  ``shrink_joint`` [rad], ``shrink_actuator`` [m] and ``shrink_crank`` [rad] are the
  margins of the joint and foot-tibia, the actuator and the crank-bar limits. A x - b
  is the signed distance [rad] in the (j4, j5) plane.

  For a visual explanation of the parameters see
    https://github.com/pal-robotics/pal_mjlab/pull/133
  """
  # Both legs of all variants share this range.
  j5_min, j5_max = get_kangaroo_spec().joint("leg_left_5_joint").range.tolist()

  limits = KANGAROO_ANKLE_ENVELOPE_LIMITS
  ft_b0j5, ft_dj4ndj5, j4n_min = limits["foot_tibia_collision"]
  act_a0, act_dadj4, act_dadj5, act_min, act_max = limits["actuator"]
  cb_b0l, cb_dj4dl, cb_dj4dj5 = limits["crank_bar_collision"]

  # The collision lines are lines in joint space, so their margin in rad is a
  # distance there: b - margin * |(a_j4, a_j5)| before normalizing.
  margin_ft = shrink_joint * math.hypot(1.0, ft_dj4ndj5)
  margin_cb = shrink_crank * math.hypot(1.0, cb_dj4dj5)
  act_min, act_max = act_min + shrink_actuator, act_max - shrink_actuator

  # Rows 0, 3 and 4 limit j4n = j4 - femur + knee - KANGAROO_TIBIA_PITCH_OFFSET.

  # fmt: off
  A = [
    # j4           j5            L          femur  knee
    [-1.0,         0.0,          0.0,        1.0,  -1.0],  #  0 j4n >= j4n_min
    [ 0.0,         1.0,          0.0,        0.0,   0.0],  #  1 j5 <= j5_max
    [ 0.0,        -1.0,          0.0,        0.0,   0.0],  #  2 j5 >= j5_min
    [-1.0,         ft_dj4ndj5,   0.0,        1.0,  -1.0],  #  3 j4n >= ft_b0j5 + ft_dj4ndj5 j5
    [-1.0,        -ft_dj4ndj5,   0.0,        1.0,  -1.0],  #  4 j4n >= ft_b0j5 - ft_dj4ndj5 j5
    [ act_dadj4,   act_dadj5,    0.0,        0.0,   0.0],  #  5 a4 <= act_max
    [-act_dadj4,  -act_dadj5,    0.0,        0.0,   0.0],  #  6 a4 >= act_min
    [ act_dadj4,  -act_dadj5,    0.0,        0.0,   0.0],  #  7 a5 <= act_max
    [-act_dadj4,   act_dadj5,    0.0,        0.0,   0.0],  #  8 a5 >= act_min
    [ 1.0,         cb_dj4dj5,   -cb_dj4dl,   0.0,   0.0],  #  9 j4 <= cb_b0l + cb_dj4dl L - cb_dj4dj5 j5
    [ 1.0,        -cb_dj4dj5,   -cb_dj4dl,   0.0,   0.0],  # 10 j4 <= cb_b0l + cb_dj4dl L + cb_dj4dj5 j5
  ]
  b = [
    -(j4n_min + KANGAROO_TIBIA_PITCH_OFFSET + shrink_joint),  #  0
    j5_max - shrink_joint,                                    #  1
    -(j5_min + shrink_joint),                                 #  2
    -(ft_b0j5 + KANGAROO_TIBIA_PITCH_OFFSET + margin_ft),     #  3
    -(ft_b0j5 + KANGAROO_TIBIA_PITCH_OFFSET + margin_ft),     #  4
    act_max - act_a0,                                         #  5
    act_a0 - act_min,                                         #  6
    act_max - act_a0,                                         #  7
    act_a0 - act_min,                                         #  8
    cb_b0l - margin_cb,                                       #  9
    cb_b0l - margin_cb,                                       # 10
  ]
  # fmt: on

  return _normalize_envelope(A, b)


##
# Actuator Configs
##

# Legs
KANGAROO_LEG_ACTUATORS = (
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_1_joint",),
    **_calc_leg_params(80.0, 0.25, 0.9, 1.65),
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_2_joint",),
    **_calc_leg_params(230.0, 0.5, 4.0, 6.49),
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_3_joint",),
    **_calc_leg_params(139.0, 0.2, 2.34, 2.705),
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_4_joint",),
    **_calc_leg_params(140.0, 0.1, 1.25, 2.8),
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_5_joint",),
    **_calc_leg_params(82.0, 0.05, 0.8, 1.0),
  ),
  # The leg length joint uses an armature calculated as
  # actuator inertia x velocity ratio. As the velocity ratio
  # is not constant, its average value was used. To compensate this
  # +- 0.5 kg randomization is added later to the armature
  BuiltinPositionActuatorCfg(
    target_names_expr=("leg_.*_length_joint",),
    **_calc_leg_params(1100.0, 4.1, None, None),
  ),
)

# Arms & Torso
KANGAROO_PELVIS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "pelvis_1_joint",
    "pelvis_2_joint",
  ),
  **S_PLUS,
)
KANGAROO_S_PLUS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "arm_.*_1_joint",
    "arm_.*_2_joint",
    "pelvis_1_joint",
    "pelvis_2_joint",
  ),
  **S_PLUS,
)
KANGAROO_S_MINUS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(r"arm_.*_(?![1267]_joint)\d+_joint",),
  **S_MINUS,
)
KANGAROO_XS_ACTUATOR_CFG = BuiltinPositionActuatorCfg(
  target_names_expr=(r"arm_.*_(?![12345]_joint)\d+_joint",),
  **XS,
)

COMMON_ACTUATORS = KANGAROO_LEG_ACTUATORS + (
  KANGAROO_S_PLUS_ACTUATOR_CFG,
  KANGAROO_S_MINUS_ACTUATOR_CFG,
)

KANGAROO_LOWER_BODY_ACTUATORS = KANGAROO_LEG_ACTUATORS + (KANGAROO_PELVIS_ACTUATOR_CFG,)

##
# Initial State
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.9),
  joint_pos={
    "leg_left_1_joint": -0.012,
    "leg_right_1_joint": 0.012,
    "leg_.*_2_joint": 0.054,
    "leg_left_3_joint": 0.04,
    "leg_right_3_joint": -0.04,
    "leg_.*_length_joint": 0.6,
    "leg_.*_4_joint": -0.053,
    "leg_.*_5_joint": 0.0,
    "leg_.*_femur_joint": 0.9,
    "leg_.*_knee_joint": 1.8,
    "arm_left_1_joint": 0.24,
    "arm_right_1_joint": -0.24,
    "arm_.*_2_joint": 1.32,
    "arm_left_3_joint": 1.57,
    "arm_right_3_joint": -1.57,
    "arm_.*_4_joint": 0.8,
    "pelvis_1_joint": 0.0,
    "pelvis_2_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision Configs
##

_FOOT_REGEX = ".*_foot.*_collision"

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_FOOT_REGEX,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={_FOOT_REGEX: 3, ".*_collision": 1},
  priority={_FOOT_REGEX: 1, ".*_collision": 0},
  friction={_FOOT_REGEX: (0.6,)},
)

##
# Articulation Configs
##

KANGAROO_ARTICULATION = EntityArticulationInfoCfg(
  actuators=COMMON_ACTUATORS, soft_joint_pos_limit_factor=0.9
)
KANGAROO_LOWER_BODY_ARTICULATION = EntityArticulationInfoCfg(
  actuators=KANGAROO_LOWER_BODY_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)
KANGAROO_HANDS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=COMMON_ACTUATORS, soft_joint_pos_limit_factor=0.9
)
KANGAROO_GRIPPERS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=COMMON_ACTUATORS + (KANGAROO_XS_ACTUATOR_CFG,),
  soft_joint_pos_limit_factor=0.9,
)


_ROBOT_CONFIGS = {
  "kangaroo": (get_kangaroo_spec, KANGAROO_ARTICULATION, FULL_COLLISION),
  "lower_body": (
    get_kangaroo_lower_body_spec,
    KANGAROO_LOWER_BODY_ARTICULATION,
    FULL_COLLISION,
  ),
  "hands": (
    get_kangaroo_hands_spec,
    KANGAROO_HANDS_ARTICULATION,
    FULL_COLLISION,
  ),
  "grippers": (
    get_kangaroo_grippers_spec,
    KANGAROO_GRIPPERS_ARTICULATION,
    FULL_COLLISION,
  ),
}


def _make_robot_cfg(variant: str) -> EntityCfg:
  spec_fn, articulation, collision = _ROBOT_CONFIGS[variant]
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(collision,),
    spec_fn=spec_fn,
    articulation=articulation,
  )


def get_kangaroo_robot_cfg() -> EntityCfg:
  return _make_robot_cfg("kangaroo")


def get_kangaroo_lower_body_robot_cfg() -> EntityCfg:
  lower_body_cfg = _make_robot_cfg("lower_body")
  # This is needed for the new feet
  lower_body_cfg.init_state.pos = (0.0, 0.0, 0.945)
  return lower_body_cfg


def get_kangaroo_hands_robot_cfg() -> EntityCfg:
  return _make_robot_cfg("hands")


def get_kangaroo_grippers_robot_cfg() -> EntityCfg:
  return _make_robot_cfg("grippers")


_EXCLUDED_JOINTS = {
  "leg_left_knee_joint",
  "leg_right_knee_joint",
  "leg_left_femur_joint",
  "leg_right_femur_joint",
}


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
    # Torque-normalized action scale: at rest, a unit action shifts the PD
    # target by 0.25 * effort / Kp, producing ~25% of the joint's effort limit
    # (saturation at |a| ~ 4). This gives every joint the same torque authority
    # per unit action regardless of its stiffness. The 0.25 is a heuristic
    # (not derived), inherited from the legged_gym default action_scale
    # (Rudin et al., CoRL 2021, arXiv:2109.11978) and made torque-relative in
    # BeyondMimic (Liao et al., 2025, arXiv:2508.08241;
    # github.com/HybridRobotics/whole_body_tracking), which mjlab's G1 config
    # follows.
    for n in a.target_names_expr:
      if n in e and n in s and s[n] and n not in exclude:
        scales[n] = 0.25 * e[n] / s[n]
        names.append(n)
  return scales, tuple(names)


KANGAROO_ACTION_SCALE, KANGAROO_ACTUATOR_NAMES = _build_action_scales(
  KANGAROO_ARTICULATION, _EXCLUDED_JOINTS
)
KANGAROO_LOWER_BODY_ACTION_SCALE, KANGAROO_LOWER_BODY_ACTUATOR_NAMES = (
  _build_action_scales(KANGAROO_LOWER_BODY_ARTICULATION, _EXCLUDED_JOINTS)
)
KANGAROO_HANDS_ACTION_SCALE, KANGAROO_HANDS_ACTUATOR_NAMES = _build_action_scales(
  KANGAROO_HANDS_ARTICULATION
)
KANGAROO_GRIPPERS_ACTION_SCALE, KANGAROO_GRIPPERS_ACTUATOR_NAMES = _build_action_scales(
  KANGAROO_GRIPPERS_ARTICULATION
)

if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_kangaroo_robot_cfg())
  viewer.launch(robot.spec.compile())
