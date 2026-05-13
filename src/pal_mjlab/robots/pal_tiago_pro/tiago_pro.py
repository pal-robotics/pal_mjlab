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
  collision_link_pattern: str = "(arm_right|gripper_right)_.*_link"
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
    )

  def gripper_action_cfg(self) -> Any:
    from mjlab.envs.mdp.actions import JointPositionActionCfg

    return JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(self.gripper_joint_pattern,),
      scale=0.035,
      offset=0.035,
      use_default_offset=False,
    )
