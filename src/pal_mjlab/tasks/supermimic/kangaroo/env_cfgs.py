"""PAL Kangaroo (7-DoF arms with grippers) SuperMimic configuration.

Robot-specific choices only; every paper setting lives in ``supermimic_env_cfg.py``.
Kangaroo facts that shape the mapping:
  - legs are a hip-ankle slide (``leg_*_length_joint``) with a passive four-bar knee;
  - the IMU sits on the torso (``pelvis_2_link``), after the two waist joints;
  - the passive four-bar joints are not measured, so proprioception covers the 28
    actuated joints.
"""

import math
from dataclasses import replace

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pal_mjlab.robots import REGEX_ALL_ACTUATED_JOINTS, get_kangaroo_grippers_robot_cfg
from pal_mjlab.tasks.supermimic.mdp import MotionLibraryCommandCfg
from pal_mjlab.tasks.supermimic.supermimic_env_cfg import make_supermimic_env_cfg

# BONES-SEED retarget packed by scripts/bones_to_library.py (``dev`` = 2.5 h subset).
DEFAULT_LIBRARY = "~/Projects/supermimic/kangaroo_bones_seed/data/library/dev"

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

FOOT_GEOMS = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(0, 12, 2)
)
TORSO_BODY = "pelvis_2_link"

# BeyondMimic's actuator rule (paper Appendix A, "low PD gains"): one natural frequency
# for every joint, Kp = armature * w^2, Kd = 2 * zeta * armature * w, and an action scale
# of 0.25 * effort / Kp. The frequency is PAL's identified value for Kangaroo (3.5 Hz on
# main, #121); the armatures, friction and effort limits are PAL's identified ones (D26).
NATURAL_FREQ_HZ = 3.5
DAMPING_RATIO = 2.0


def kangaroo_robot_cfg() -> EntityCfg:
  """Kangaroo with grippers, gains from the BeyondMimic rule at one natural frequency."""
  cfg = get_kangaroo_grippers_robot_cfg()
  w = 2.0 * math.pi * NATURAL_FREQ_HZ
  assert cfg.articulation is not None
  actuators = tuple(
    replace(
      a, stiffness=a.armature * w**2, damping=2.0 * DAMPING_RATIO * a.armature * w
    )
    for a in cfg.articulation.actuators
  )
  cfg.articulation = replace(cfg.articulation, actuators=actuators)
  return cfg


def action_scales(robot: EntityCfg) -> dict[str, float]:
  """0.25 * effort / Kp per actuator group (BeyondMimic's torque-relative scale)."""
  assert robot.articulation is not None
  return {
    name: 0.25 * a.effort_limit / a.stiffness
    for a in robot.articulation.actuators
    for name in a.target_names_expr
  }


def pal_kangaroo_grippers_supermimic_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Kangaroo SuperMimic tracking env; ``play`` is the paper's evaluation setting."""
  cfg = make_supermimic_env_cfg()

  robot = kangaroo_robot_cfg()
  cfg.scene.entities = {"robot": robot}
  cfg.viewer.body_name = "base_link"

  scales = action_scales(robot)
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = scales
  joint_pos_action.actuator_names = tuple(scales)

  assert cfg.commands is not None
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionLibraryCommandCfg)
  motion_cmd.library_path = DEFAULT_LIBRARY
  motion_cmd.body_names = TRACKED_BODIES
  motion_cmd.goal_joint_names = (REGEX_ALL_ACTUATED_JOINTS,)

  # Measured joints: the 28 actuated ones (proprioception, calibration offsets, limits).
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
