from dataclasses import dataclass, replace, field

from bdx_r_mjlab.robots.bdx_r.bdx_r_constants import (
  BDX_R_ACTION_SCALE,
  BDX_R_ROBOT_CFG,
)
from bdx_r_mjlab.tasks.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)
from mjlab.utils.spec_config import ContactSensorCfg

from mjlab.managers.manager_term_config import TerminationTermCfg as DoneTerm
from mjlab.managers.manager_term_config import term

from bdx_r_mjlab.tasks.velocity import mdp

@dataclass
class TerminationCfg:
  time_out: DoneTerm = term(DoneTerm, func=mdp.time_out, time_out=True)
  base_contact: DoneTerm = term(
    DoneTerm,
    func=mdp.illegal_contacts, 
  )
  
@dataclass
class BdxRRoughEnvCfg(LocomotionVelocityEnvCfg):
  terminations: TerminationCfg = field(default_factory=TerminationCfg)
  def __post_init__(self):
    super().__post_init__()

    contact_sensors = [
    #   ContactSensorCfg(
    #     name=f"{side}_foot_ground_contact",
    #     body1=f"{side}_Foot",
    #     body2="terrain",
    #     num=1,
    #     data=("found",),
    #     reduce="netforce",
    #   )
    #   for side in ["Left", "Right"]
    # ]
    # contact_sensors += [
      ContactSensorCfg(
        name="base_link_contact",
        body1="base_link",
        body2="terrain",
        num=1,
        data=("found",),
        reduce="netforce",
      )
    ]
    bdx_r_cfg = replace(BDX_R_ROBOT_CFG, sensors=tuple(contact_sensors))
    self.scene.entities = {"robot": bdx_r_cfg}

    # sensor_names = ["Left_foot_ground_contact", "Right_foot_ground_contact"]
    geom_names = ["left_foot_collision", "right_foot_collision"]

    self.events.foot_friction.params["asset_cfg"].geom_names = geom_names

    self.actions.joint_pos.scale = BDX_R_ACTION_SCALE

    # self.rewards.air_time.params["sensor_names"] = sensor_names
    self.rewards.pose.params["std"] = {
      r".*_Hip_Yaw": 0.15,
      r".*_Hip_Roll": 0.15,
      r".*_Hip_Pitch": 0.3,
      r".*_Knee": 0.35,
      r".*_Ankle": 0.1,
    }

    self.viewer.body_name = "base_link"
    self.commands.twist.viz.z_offset = 0.75

    self.curriculum.command_vel = None

    self.events.push_robot.params["velocity_range"] = {
      "x": (-0.3, 0.3),
      "y": (-0.3, 0.3),
    }
    self.terminations.base_contact.params["sensor_names"] = ["base_link_contact"]

    self.rewards.air_time = None


@dataclass
class BdxRRoughEnvCfg_PLAY(BdxRRoughEnvCfg):
  def __post_init__(self):
    super().__post_init__()

    # Effectively infinite episode length.
    self.episode_length_s = int(1e9)

    if self.scene.terrain is not None:
      if self.scene.terrain.terrain_generator is not None:
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.border_width = 10.0