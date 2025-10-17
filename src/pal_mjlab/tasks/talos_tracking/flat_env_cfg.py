from dataclasses import dataclass, replace

from pal_mjlab.robots import TALOS_ACTION_SCALE, TALOS_ROBOT_CFG
from mjlab.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from mjlab.utils.spec_config import ContactSensorCfg


@dataclass
class TalosFlatEnvCfg(TrackingEnvCfg):
  def __post_init__(self):
    self_collision_sensor = ContactSensorCfg(
      name="self_collision",
      subtree1="base_link",
      subtree2="base_link",
      data=("found",),
      reduce="netforce",
      num=10,  # Report up to 10 contacts.
    )
    talos_cfg = replace(TALOS_ROBOT_CFG, sensors=(self_collision_sensor,))

    self.scene.entities = {"robot": talos_cfg}
    self.actions.joint_pos.scale = TALOS_ACTION_SCALE

    self.commands.motion.anchor_body_name = "base_link"
    self.commands.motion.body_names = [
      "base_link",
      "torso_2_link",
      "leg_left_3_link",
      "leg_left_4_link",
      "leg_left_6_link",
      "leg_right_3_link",
      "leg_right_4_link",
      "leg_right_6_link",
      "arm_left_3_link",
      "arm_left_4_link",
      "arm_left_7_link",
      "arm_right_3_link",
      "arm_right_4_link",
      "arm_right_7_link",
    ]

    self.events.foot_friction.params["asset_cfg"].geom_names = [
      r"^(left|right)_foot_collision$"
    ]
    self.events.base_com.params["asset_cfg"].body_names = "base_link"

    self.terminations.ee_body_pos.params["body_names"] = [
      "leg_left_6_link",
      "leg_right_6_link",
      "arm_left_7_link",
      "arm_right_7_link",
    ]

    self.viewer.body_name = "base_link"


@dataclass
class TalosFlatNoStateEstimationEnvCfg(TalosFlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.observations.policy.motion_anchor_pos_b = None
    self.observations.policy.base_lin_vel = None


@dataclass
class TalosFlatEnvCfg_PLAY(TalosFlatEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.observations.policy.enable_corruption = False
    self.events.push_robot = None

    # Disable RSI randomization.
    self.commands.motion.pose_range = {}
    self.commands.motion.velocity_range = {}

    # Disable adaptive sampling to play through motion from start to finish.
    self.commands.motion.disable_adaptive_sampling = True

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)


@dataclass
class TalosFlatNoStateEstimationEnvCfg_PLAY(TalosFlatNoStateEstimationEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.observations.policy.enable_corruption = False
    self.events.push_robot = None

    # Disable RSI randomization.
    self.commands.motion.pose_range = {}
    self.commands.motion.velocity_range = {}

    # Disable adaptive sampling to play through motion from start to finish.
    self.commands.motion.disable_adaptive_sampling = True

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)