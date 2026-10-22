"""PAL Kangaroo (7-DoF arms with grippers) SuperMimic configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pal_mjlab.robots import (
  ANKLE_XY_CONVEX_HULL_POINTS,
  HIP_XY_CONVEX_HULL_POINTS,
  KANGAROO_GRIPPERS_ACTION_SCALE,
  KANGAROO_GRIPPERS_ACTUATOR_NAMES,
  REGEX_ALL_ACTUATED_JOINTS,
  get_kangaroo_grippers_robot_cfg,
)
from pal_mjlab.tasks.supermimic.mdp import MotionLibraryCommandCfg
from pal_mjlab.tasks.supermimic.supermimic_env_cfg import make_supermimic_env_cfg
from pal_mjlab.tasks.velocity.mdp import joint_limits_convex_hull

# BONES-SEED retarget packed by scripts/bones_to_library.py (``dev`` = 2.5 h subset).
DEFAULT_LIBRARY = "~/Projects/supermimic/kangaroo_bones_seed/data/library/dev"

# Tracked bodies (as in the WBC task); the first one is the anchor.
TRACKED_BODIES = (
  "base_link",
  "pelvis_2_link",
  "leg_left_3_link",
  "leg_left_4_link",
  "leg_left_5_link",
  "leg_right_3_link",
  "leg_right_4_link",
  "leg_right_5_link",
  "arm_left_2_link",
  "arm_left_4_link",
  "arm_left_tip_link",
  "arm_right_2_link",
  "arm_right_4_link",
  "arm_right_tip_link",
)

FOOT_GEOMS = tuple(
  f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(0, 12, 2)
)
TORSO_BODY = "pelvis_2_link"


def pal_kangaroo_grippers_supermimic_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Kangaroo SuperMimic tracking env; ``play`` is the paper's evaluation setting."""
  cfg = make_supermimic_env_cfg()

  cfg.scene.entities = {"robot": get_kangaroo_grippers_robot_cfg()}
  cfg.viewer.body_name = "base_link"

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_GRIPPERS_ACTION_SCALE
  joint_pos_action.actuator_names = KANGAROO_GRIPPERS_ACTUATOR_NAMES

  assert cfg.commands is not None
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionLibraryCommandCfg)
  motion_cmd.library_path = DEFAULT_LIBRARY
  motion_cmd.body_names = TRACKED_BODIES
  motion_cmd.goal_joint_names = (REGEX_ALL_ACTUATED_JOINTS,)

  # Proprioception covers the 28 actuated joints; the passive four-bar joints are not
  # measured.
  actuated = SceneEntityCfg("robot", joint_names=(REGEX_ALL_ACTUATED_JOINTS,))
  proprio = cfg.observations["proprio"].terms
  proprio["joint_pos"].params["asset_cfg"] = actuated
  proprio["joint_vel"].params["asset_cfg"] = actuated
  proprio["imu_ang_vel"].params["sensor_name"] = "robot/imu_ang_vel"
  proprio["imu_projected_gravity"].params["sensor_name"] = "robot/imu_quat"

  context = cfg.observations["critic"].terms["sim_context"].params
  context["friction_asset_cfg"] = SceneEntityCfg("robot", geom_names=FOOT_GEOMS)
  context["com_asset_cfg"] = SceneEntityCfg("robot", body_names=(TORSO_BODY,))

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
  cfg.events["torso_com"].params["asset_cfg"].body_names = (TORSO_BODY,)

  cfg.rewards["joint_limit"].params["asset_cfg"] = actuated
  # Kangaroo addition (D12): hip and ankle XY are coupled through linear actuators, so
  # their feasible set is a convex hull, not a box.
  for name, joints, hull in (
    ("hip", ("leg_{}_2_joint", "leg_{}_3_joint"), HIP_XY_CONVEX_HULL_POINTS),
    ("ankle", ("leg_{}_4_joint", "leg_{}_5_joint"), ANKLE_XY_CONVEX_HULL_POINTS),
  ):
    cfg.rewards[f"joint_limit_hull_{name}"] = RewardTermCfg(
      func=joint_limits_convex_hull,
      weight=-10.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
        "metrics_suffix": name,
        "joint_names_group": [
          [j.format(side) for j in joints] for side in ("left", "right")
        ],
        "margin": 0.02,
        "hull_points": hull,
      },
    )

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
