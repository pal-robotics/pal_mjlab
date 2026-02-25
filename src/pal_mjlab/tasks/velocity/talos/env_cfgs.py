"""PAL Robotics Talos velocity tracking environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from pal_mjlab.robots import TALOS_ACTION_SCALE, get_talos_robot_cfg


def pal_talos_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics Talos rough terrain velocity tracking configuration."""
  cfg = make_velocity_env_cfg()

  cfg.scene.entities = {"robot": get_talos_robot_cfg()}

  site_names = ("left_foot", "right_foot")
  geom_names = ("left_foot_collision", "right_foot_collision")

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(leg_left_6_link|leg_right_6_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  body_ground_cfg = ContactSensorCfg(
    name="body_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r"^(leg_left_4_link|leg_right_4_link|torso_2_link|arm_left_7_link|arm_right_7_link|arm_left_5_link|arm_right_5_link|)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, body_ground_cfg)

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = TALOS_ACTION_SCALE

  cfg.viewer.body_name = "torso_2_link"

  assert cfg.commands is not None
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  cfg.observations["actor"].terms["height_scan"] = None
  cfg.observations["critic"].terms["height_scan"] = None
  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_2_link",)

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r"leg_.*_3_.*": 0.3,  # pitch
    r"leg_.*_2_.*": 0.15,  # roll
    r"leg_.*_1_.*": 0.15,
    r"leg_.*_4_.*": 0.35,  # knee
    r"leg_.*_5_.*": 0.25,
    r"leg_.*_6_.*": 0.1,
    # Waist.
    r".*torso_2.*": 0.1,  # pitch
    r".*torso_1.*": 0.2,  # yaw
    r".*head.*": 0.1,
    # Arms.
    r"arm_.*_1_.*": 0.15,  # yaw
    r"arm_.*_2_.*": 0.15,  # roll
    r"arm_.*_3_.*": 0.1,  # yaw
    r"arm_.*_4_.*": 0.15,  # elbow
    r"arm_.*_5_.*": 0.1,  # elbow
    r"arm_.*_6_.*": 0.1,  # wrist
    r"arm_.*_7_.*": 0.2,  # wrist
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r"leg_.*_3_.*": 0.5,  # pitch
    r"leg_.*_2_.*": 0.2,  # roll
    r"leg_.*_1_.*": 0.2,
    r"leg_.*_4_.*": 0.6,
    r"leg_.*_5_.*": 0.35,
    r"leg_.*_6_.*": 0.15,
    # Waist.
    r".*torso_2.*": 0.2,  # pitch
    r".*torso_1.*": 0.3,  # yaw
    r".*head.*": 0.1,
    # Arms.
    r"arm_.*_1_.*": 0.2,  # yaw
    r"arm_.*_2_.*": 0.2,  # roll
    r"arm_.*_3_.*": 0.1,  # yaw
    r"arm_.*_4_.*": 0.35,  # elbow
    r"arm_.*_5_.*": 0.1,  # elbow
    r"arm_.*_6_.*": 0.1,  # wrist
    r"arm_.*_7_.*": 0.2,  # wrist
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_2_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_2_link",)

  for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.0

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )

  cfg.terminations["illegal_contacts"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "body_ground_contact"},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def pal_talos_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Talos flat terrain velocity configuration."""
  cfg = pal_talos_rough_env_cfg(play=play)

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  if play:
    commands = cfg.commands
    assert commands is not None
    twist_cmd = commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg
