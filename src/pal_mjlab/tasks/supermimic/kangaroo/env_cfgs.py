"""PAL Kangaroo (7-DoF arms with grippers) SuperMimic configuration.

Robot-specific choices only; every paper setting lives in ``supermimic_env_cfg.py``.
Kangaroo facts that shape the mapping:
  - a leg is a hip-ankle slide (``leg_*_length_joint``) with a passive knee linkage closed
    on it (``leg_*_femur_joint``, ``leg_*_knee_joint``). Motions are retargeted to the leg
    length only, so the motion library holds the 28 actuated joints and the linkage follows
    from the leg length (``kangaroo_passive_leg_joints``);
  - the IMU sits on the torso (``pelvis_2_link``), after the two waist joints.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pal_mjlab.robots import (
  KANGAROO_GRIPPERS_ACTION_SCALE,
  REGEX_ALL_ACTUATED_JOINTS,
  get_kangaroo_grippers_robot_cfg,
  kangaroo_passive_leg_joints,
)
from pal_mjlab.tasks.supermimic.mdp import MotionLibraryCommandCfg
from pal_mjlab.tasks.supermimic.supermimic_env_cfg import make_supermimic_env_cfg

# Default motion library, relative to the working directory (artifacts/ is ignored by git).
# Training overrides it: --env.commands.motion.library-path <folder>.
LIBRARY_PATH = "artifacts/kangaroo_grippers_library"

# Natural frequency [Hz] of the gains the arm and waist action scales are computed for. The
# robot's gains are at 3.5 Hz, where these scales are 12 times smaller: too little
# exploration per unit of action noise, and too high an action-rate cost per radian.
ARM_WAIST_ACTION_SCALE_FREQ_HZ = 1.0

# The paper's tracked bodies (BeyondMimic's G1 set) mapped to Kangaroo; the first one is the
# root/anchor. The G1 knee has no counterpart: the leg is a hip-ankle slide.
#   pelvis             -> base_link
#   torso_link         -> pelvis_2_link  (after the two waist joints)
#   hip_roll_link      -> leg_*_3_link
#   ankle_roll_link    -> leg_*_5_link   (foot)
#   shoulder_roll_link -> arm_*_2_link
#   elbow_link         -> arm_*_4_link
#   wrist_yaw_link     -> arm_*_7_link
TRACKED_BODIES = (
  "base_link",
  "pelvis_2_link",
  "leg_left_3_link",
  "leg_left_5_link",
  "leg_right_3_link",
  "leg_right_5_link",
  "arm_left_2_link",
  "arm_left_4_link",
  "arm_left_7_link",
  "arm_right_2_link",
  "arm_right_4_link",
  "arm_right_7_link",
)

FOOT_GEOMS = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(0, 12, 2)
)
TORSO_BODY = "pelvis_2_link"


def _action_scales() -> dict[str, float]:
  """BeyondMimic's action scale, 0.25 * effort / Kp, per actuator: the legs' for the robot's
  gains, the arms' and waist's for gains of ``ARM_WAIST_ACTION_SCALE_FREQ_HZ``."""
  articulation = get_kangaroo_grippers_robot_cfg().articulation
  assert articulation is not None
  scales = dict(KANGAROO_GRIPPERS_ACTION_SCALE)
  w = 2.0 * math.pi * ARM_WAIST_ACTION_SCALE_FREQ_HZ
  for actuator in articulation.actuators:
    for name in actuator.target_names_expr:
      if not name.startswith("leg_"):
        scales[name] = 0.25 * actuator.effort_limit / (actuator.armature * w**2)
  return scales


def pal_kangaroo_grippers_supermimic_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create the PAL Kangaroo SuperMimic tracking configuration."""
  cfg = make_supermimic_env_cfg()

  cfg.scene.entities = {"robot": get_kangaroo_grippers_robot_cfg()}
  cfg.viewer.body_name = "base_link"

  scales = _action_scales()
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = scales
  joint_pos_action.actuator_names = tuple(scales)

  assert cfg.commands is not None
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionLibraryCommandCfg)
  motion_cmd.library_path = LIBRARY_PATH
  motion_cmd.body_names = TRACKED_BODIES
  motion_cmd.passive_joint_state = kangaroo_passive_leg_joints

  # The actuated joints are the library's joints, and the ones measured.
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

  # Apply play mode overrides: the paper's evaluation setting (Appendix F.1). Every clip
  # from its first frame, foot friction 1.0, no randomization, no observation or reset noise.
  if play:
    cfg.auto_reset = True
    cfg.episode_length_s = int(1e9)
    cfg.observations["proprio"].enable_corruption = False
    motion_cmd.sampling_mode = "start"
    motion_cmd.compile_kinematics = False  # not worth its wait for a few envs
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    cfg.events.pop("encoder_bias")
    cfg.events.pop("torso_com")
    friction = cfg.events["foot_friction"]
    friction.mode = "startup"
    friction.params["ranges"] = (1.0, 1.0)

  return cfg
