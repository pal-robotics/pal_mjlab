from dataclasses import dataclass, field
from typing import Any

from mjlab.entity import EntityCfg

from .tiago_pro_constants import get_tiago_pro_robot_cfg


@dataclass
class TiagoProRobot:
  entity_cfg: EntityCfg = field(default_factory=get_tiago_pro_robot_cfg)
  arm_joint_pattern: str = "arm_right_.*_joint"
  gripper_joint_pattern: str = "gripper_right_finger_joint"
  ee_site: str = "gripper_right_grasping_site"
  fingertip_geom_pattern: str = "col_right_fingertip_.*"
  fingertip_site_pattern: str = "gripper_right_fingertip_.*_site"
  collision_link_pattern: str = "(arm_right|gripper_right)_.*_link"
  arm_collision_link_pattern: str = "arm_right_.*_link"
  gripper_collision_link_pattern: str = "gripper_right_.*_link"
  viewer_body: str = "base_footprint"
  camera_name: str = "head_realsense_camera"
  wrist_camera_name: str = "wrist_realsense_camera"
  head_camera_name: str = "head_realsense_camera"

  def arm_action_cfg(self) -> Any:
    from mjlab.envs.mdp.actions import DifferentialIKActionCfg

    return DifferentialIKActionCfg(
      entity_name="robot",
      actuator_names=(self.arm_joint_pattern,),
      frame_name=self.ee_site,
      frame_type="site",
      delta_pos_scale=0.005,  # Max displacement of 0.01m per step (0.5m/s max velocity)
      delta_ori_scale=0.005,  # Max rotation of 0.01 rad per step (0.5 rad/s max angular velocity)
    )

  def gripper_action_cfg(self) -> Any:
    from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg

    return RelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(self.gripper_joint_pattern,),
      scale=0.01,
    )
