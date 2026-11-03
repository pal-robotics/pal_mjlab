"""PAL Kangaroo SuperMimic configuration: 7-DoF arms with grippers, and 5-DoF arms with Seed
Robotics hands (``hands``: the hand's 2-DoF wrist in the place of arm joints 6-7, fingers fixed).

Robot-specific choices only; every paper setting lives in ``supermimic_env_cfg.py``.
Kangaroo facts that shape the mapping:
  - legs are a hip-ankle slide (``leg_*_length_joint``) with a passive four-bar knee;
  - the IMU sits on the torso (``pelvis_2_link``), after the two waist joints;
  - the passive four-bar joints are not measured, so proprioception covers the 28 actuated
    joints (both variants).
"""

import math
from dataclasses import replace

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pal_mjlab.robots import (
  REGEX_ALL_ACTUATED_JOINTS,
  get_kangaroo_grippers_robot_cfg,
  get_kangaroo_hands_robot_cfg,
)
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  DAMPING_RATIO,
  _build_action_scales,
)
from pal_mjlab.tasks.supermimic.mdp import MotionLibraryCommandCfg
from pal_mjlab.tasks.supermimic.supermimic_env_cfg import (
  make_supermimic_env_cfg,
  make_velocity_env_cfg,
)

# Default motion libraries, 2.5 h of the BONES-SEED retarget of each robot, relative to the working directory
# (artifacts/ is ignored by git; link or copy the retarget there). The CLI overrides them:
# --env.commands.motion.library-path <folder>
GRIPPERS_LIBRARY = "artifacts/kangaroo_bones_seed/dataset/libraries/dev"
HANDS_LIBRARY = "artifacts/kangaroo_bones_seed/dataset_5dof/libraries/dev"
# The velocity task's: every walk/jog/run clip of the 5-DoF retarget (D56).
HANDS_LOCOMOTION_LIBRARY = (
  "artifacts/kangaroo_bones_seed/dataset_5dof/libraries/locomotion"
)

# Arm and waist actuators of both robots: the natural frequency of their gains (None: kangaroo_constants' 3.5 Hz,
# the robot's) and the one their action scales are computed at (None: the gains'). The robot's gains with the 1 Hz
# action scales (D55; D27 had 1 Hz gains). The legs keep theirs.
ARM_WAIST_GAIN_FREQ_HZ: float | None = None
ARM_WAIST_ACTION_SCALE_FREQ_HZ: float | None = 1.0

# The paper's 14 tracked bodies (BeyondMimic's G1 set), mapped to Kangaroo (D25). The first
# one is the root/anchor.
#   pelvis             -> base_link
#   torso_link         -> pelvis_2_link     (after the two waist joints)
#   hip_roll_link      -> leg_*_3_link
#   knee_link          -> leg_*_knee_link   (passive four-bar knee: the serial leg is a
#                                            hip-ankle slide, leg_*_4/5_link both sit at the ankle)
#   ankle_roll_link    -> leg_*_5_link      (foot)
#   shoulder_roll_link -> arm_*_2_link
#   elbow_link         -> arm_*_4_link
#   wrist_yaw_link     -> arm_*_7_link      (last arm link; arm_*_tip_link is an end-effector frame
#                                            at the same point, defined differently in GMR)
TRACKED_BODIES = (
  "base_link",
  "pelvis_2_link",
  "leg_left_3_link",
  "leg_left_knee_link",
  "leg_left_5_link",
  "leg_right_3_link",
  "leg_right_knee_link",
  "leg_right_5_link",
  "arm_left_2_link",
  "arm_left_4_link",
  "arm_left_7_link",
  "arm_right_2_link",
  "arm_right_4_link",
  "arm_right_7_link",
)
# With hands the wrist_yaw_link slot is the middle of the hand (hand_*_center_link, the frame
# the 5-DoF retarget matches to the middle of the G1 hand). It moves with the Seed wrist as
# arm_*_7_link does with the 7-DoF wrist.
HANDS_TRACKED_BODIES = tuple(
  {
    "arm_left_7_link": "hand_left_center_link",
    "arm_right_7_link": "hand_right_center_link",
  }.get(b, b)
  for b in TRACKED_BODIES
)

# Per variant: robot, tracked bodies, default motion library.
VARIANTS = {
  "grippers": (get_kangaroo_grippers_robot_cfg, TRACKED_BODIES, GRIPPERS_LIBRARY),
  "hands": (get_kangaroo_hands_robot_cfg, HANDS_TRACKED_BODIES, HANDS_LIBRARY),
}

FOOT_GEOMS = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(0, 12, 2)
)
TORSO_BODY = "pelvis_2_link"


def kangaroo_robot_cfg(
  arm_waist_natural_freq_hz: float | None = None, variant: str = "grippers"
) -> EntityCfg:
  """Kangaroo with grippers or hands; gains from ``kangaroo_constants`` (3.5 Hz everywhere).

  ``arm_waist_natural_freq_hz`` re-derives the arm and waist gains at another natural
  frequency with the same rule and rounding (D27); the legs keep theirs.
  """
  cfg = VARIANTS[variant][0]()
  if arm_waist_natural_freq_hz is None:
    return cfg
  w = 2.0 * math.pi * arm_waist_natural_freq_hz
  assert cfg.articulation is not None
  actuators = tuple(
    a
    if all(name.startswith("leg_") for name in a.target_names_expr)
    else replace(
      a,
      stiffness=round(a.armature * w**2, 3),
      damping=round(2.0 * DAMPING_RATIO * a.armature * w, 3),
    )
    for a in cfg.articulation.actuators
  )
  cfg.articulation = replace(cfg.articulation, actuators=actuators)
  return cfg


def action_scales(robot: EntityCfg) -> dict[str, float]:
  """BeyondMimic's action scale 0.25 * effort / Kp per actuator group (``kangaroo_constants``)."""
  assert robot.articulation is not None
  return _build_action_scales(robot.articulation)[0]


def pal_kangaroo_grippers_supermimic_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Kangaroo SuperMimic tracking env; ``play`` is the paper's evaluation setting."""
  return kangaroo_supermimic_env_cfg("grippers", play)


def pal_kangaroo_hands_supermimic_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The same with 5-DoF arms and Seed Robotics hands, on the 5-DoF retarget."""
  return kangaroo_supermimic_env_cfg("hands", play)


def pal_kangaroo_hands_velocity_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """The velocity interface on the Hands robot (D56), post-trained on a frozen Hands policy.

  The actuators are pinned to the frozen policy's (run B, ``hands-tight-full-24h``: 1 Hz arm and
  waist gains, their own action scales), not to the module defaults. The motions are the
  walk/jog/run library. The fall height is the paper's 0.35 m scaled from the G1's root height
  (0.79 m) to Kangaroo's (0.92 m).
  """
  cfg = kangaroo_supermimic_env_cfg(
    "hands", play, gain_freq_hz=1.0, action_scale_freq_hz=None
  )
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionLibraryCommandCfg)
  motion_cmd.library_path = HANDS_LOCOMOTION_LIBRARY
  return make_velocity_env_cfg(cfg, torso=TORSO_BODY, min_root_height=0.4)


def kangaroo_supermimic_env_cfg(
  variant: str,
  play: bool = False,
  gain_freq_hz: float | None = ARM_WAIST_GAIN_FREQ_HZ,
  action_scale_freq_hz: float | None = ARM_WAIST_ACTION_SCALE_FREQ_HZ,
) -> ManagerBasedRlEnvCfg:
  """The arm and waist gains at ``gain_freq_hz`` and their action scales at ``action_scale_freq_hz``
  (``ARM_WAIST_*``): e.g. gains None with scales 1.0 gives the 1 Hz action scales (exploration and
  action-rate cost per radian) on the 3.5 Hz servos."""
  _, tracked_bodies, library = VARIANTS[variant]
  cfg = make_supermimic_env_cfg()

  robot = kangaroo_robot_cfg(gain_freq_hz, variant)
  cfg.scene.entities = {"robot": robot}
  cfg.viewer.body_name = "base_link"

  scales = action_scales(
    robot
    if action_scale_freq_hz is None
    else kangaroo_robot_cfg(action_scale_freq_hz, variant)
  )
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = scales
  joint_pos_action.actuator_names = tuple(scales)

  assert cfg.commands is not None
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionLibraryCommandCfg)
  motion_cmd.library_path = library
  motion_cmd.body_names = tracked_bodies
  motion_cmd.goal_joint_names = (REGEX_ALL_ACTUATED_JOINTS,)

  # Measured joints: the actuated ones (proprioception, calibration offsets, limits).
  actuated = SceneEntityCfg("robot", joint_names=(REGEX_ALL_ACTUATED_JOINTS,))
  proprio = cfg.observations["proprio"].terms
  proprio["joint_pos"].params["asset_cfg"] = actuated
  proprio["joint_vel"].params["asset_cfg"] = actuated
  proprio["imu_ang_vel"].params["sensor_name"] = "robot/imu_ang_vel"
  proprio["imu_projected_gravity"].params["sensor_name"] = "robot/imu_quat"
  cfg.rewards["joint_limit"].params["asset_cfg"] = actuated

  context = cfg.observations["critic"].terms["sim_context"].params
  context["friction_asset_cfg"] = SceneEntityCfg("robot", geom_names=FOOT_GEOMS)
  context["bias_asset_cfg"] = actuated
  context["com_asset_cfg"] = SceneEntityCfg("robot", body_names=(TORSO_BODY,))
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
  cfg.events["encoder_bias"].params["asset_cfg"] = actuated
  cfg.events["torso_com"].params["asset_cfg"].body_names = (TORSO_BODY,)

  if play:
    # Evaluation setting of the paper (F.1): every clip from its first frame, foot
    # friction 1.0, no randomization, no observation noise, no reset noise.
    cfg.episode_length_s = int(1e9)
    cfg.observations["proprio"].enable_corruption = False
    motion_cmd.sampling_mode = "start"
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    cfg.events.pop("encoder_bias")
    cfg.events.pop("torso_com")
    friction = cfg.events["foot_friction"]
    friction.mode = "startup"
    friction.params["ranges"] = (1.0, 1.0)

  return cfg
