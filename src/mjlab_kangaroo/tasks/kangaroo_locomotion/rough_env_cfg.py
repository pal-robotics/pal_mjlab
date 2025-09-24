from dataclasses import dataclass, replace

from mjlab_kangaroo.robots.pal_kangaroo.kangaroo_constants import (
  KANG_ACTION_SCALE,
  KANG_ROBOT_CFG,
)
from mjlab.tasks.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)

@dataclass
class KangRoughEnvCfg(LocomotionVelocityEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    self.scene.entities = {"robot": replace(KANG_ROBOT_CFG)}
    # self.actions.joint_pos.actuator_names=[r"^(pelvis|arm|leg)_.*(1|2|3|length|4|5)_joint$"]
    # print(self.actions.joint_pos.actuator_names)
    self.actions.joint_pos.scale = KANG_ACTION_SCALE

    self.events.foot_friction.params["asset_cfg"].geom_names = [
      r"^(left|right)_foot_collision$"
    ]

    self.rewards.pose_l2.params["std"] = {
      # r"^leg_(left|right)_(?:knee|femur|length)_joint$": 6.0,
      r".*leg_(left|right)_(2|length)_joint.*": 6.0,
      r".*leg_(left|right)_(1|3|4|5)_joint.*": 3.0,
      r".*(pelvis_(1|2)_joint|arm_(left|right)_(1|4)_joint).*": 1.0,
      r".*leg_(left|right)_(femur|knee)_joint.*": 4.0,
      r".*arm_(left|right)_(2|3)_joint.*": 0.3,
    }


    self.rewards.power.weight = -0.001

    # self.rewards.ang_vel_xy_l2 = None
    # self.rewards.action_rate_l2 = None
    # self.rewards.power = None
    # self.rewards.dof_pos_limits = None
    # self.rewards.pose_l2 = None


    self.viewer.body_name = "pelvis_2_link"


@dataclass
class KangRoughEnvCfg_PLAY(KangRoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.border_width = 10.0
