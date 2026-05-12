"""PAL Robotics KANGAROO velocity tracking environment configurations."""

from dataclasses import replace
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import MetricsTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
  RayCastSensorCfg,
  PinholeCameraPatternCfg
)
from mjlab.terrains.config import STAIRS_TERRAINS_CFG
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from pal_mjlab.tasks.velocity.mdp import UniformVelocityCommandWithProgressTracking, UniformVelocityCommandWithProgressTrackingCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.config import pyramid_stairs, pyramid_stairs_inv, flat, random_spread_boxes
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg

from mjlab.utils.noise import UniformNoiseCfg as Unoise
import math
from mjlab.sensor import RayCastSensorCfg

from pal_mjlab.robots import (
  ANKLE_XY_CONVEX_HULL_POINTS,
  HIP_XY_CONVEX_HULL_POINTS,
  KANGAROO_ACTION_SCALE,
  KANGAROO_ACTUATOR_NAMES,
  KANGAROO_GRIPPERS_ACTION_SCALE,
  KANGAROO_GRIPPERS_ACTUATOR_NAMES,
  KANGAROO_HANDS_ACTION_SCALE,
  KANGAROO_HANDS_ACTUATOR_NAMES,
  REGEX_ALL_ACTUATED_JOINTS,
  REGEX_FEMUR_AND_KNEE_LINKS,
  REGEX_KNEE_LINKS,
  REGEX_ILLEGAL_GEOMS,
  REGEX_LEG_LENGTH_JOINTS_ONLY,
  get_kangaroo_grippers_robot_cfg,
  get_kangaroo_hands_robot_cfg,
  get_kangaroo_robot_cfg,
)
from pal_mjlab.tasks.velocity import mdp


def pal_kangaroo_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": get_kangaroo_robot_cfg()}
  
  # nconmax is the max number of contacts that will be generated at runtime
  # due to https://github.com/google-deepmind/mujoco_warp/blob/c62864ed2bf816c0a724d4cbf153921188f78eae/mujoco_warp/_src/io.py#L649-L660
  # for collision-rich envs, it is recommended to be manually set through experimentation
  cfg.sim.nconmax = 50
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.mujoco.timestep = 0.002
  cfg.decimation = 10

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision"
    for side in ("left", "right")
    for i in [0, 2, 4, 6, 8, 10]
  )
  actuated_joints = REGEX_ALL_ACTUATED_JOINTS  # Exclude femur and knee joints.

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(leg_left_5_link|leg_right_5_link)$",  # subtree so foot link is included
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
      pattern=REGEX_FEMUR_AND_KNEE_LINKS,
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
    history_length=4 # To also penalize collisions within policy steps
  )

  # Remove the default terrain scan sensor
  # cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "terrain_scan")

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    body_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_ACTION_SCALE
  joint_pos_action.actuator_names = KANGAROO_ACTUATOR_NAMES

  cfg.viewer.body_name = "pelvis_2_link"

  assert cfg.commands is not None
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  # Wire foot height scan to per-foot sites.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot") for s in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  # -- Observations

  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["actor"].terms["base_lin_vel"]
  del cfg.observations["actor"].terms["projected_gravity"]
  del cfg.observations["critic"].terms["projected_gravity"]

  cfg.observations["actor"].terms["imu_projected_gravity"] = ObservationTermCfg(
    func=mdp.imu_projected_gravity,
    params={"sensor_name": "robot/imu_quat"},
    noise=Unoise(n_min=-0.035, n_max=0.035),
  )
  cfg.observations["actor"].terms["base_lin_acc"] = ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_lin_acc"},
    noise=Unoise(n_min=-0.075, n_max=0.075),
  )
  cfg.observations["critic"].terms["imu_projected_gravity"] = ObservationTermCfg(
    func=mdp.imu_projected_gravity,
    params={"sensor_name": "robot/imu_quat"},
  )
  cfg.observations["critic"].terms["base_lin_acc"] = ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_lin_acc"},
  )

  ### Disabling the use of history length as we haven't seen much improvements with it
  ### Moreover, our best policy #62 doesn't use any history length
  # cfg.observations["actor"].history_length = 5  # Keep last 5 frames
  # cfg.observations["critic"].history_length = 5  # Keep last 5 frames

  # -- Events

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis_2_link",)

  # Domain Randomization for joint friction
  cfg.events["joint_friction"] = EventTermCfg(
    mode="startup",
    func=dr.dof_frictionloss,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),  # Set per-robot.
      "operation": "add",
      "ranges": (-0.008, 0.008),
      "shared_random": False,
    },
  )
  cfg.events["encoder_bias"].params["asset_cfg"].joint_names = [
    r"^(?!leg_.*_length_.*$).*"
  ]
  cfg.events["leg_length_encoder_bias"] = EventTermCfg(
    mode="startup",
    func=dr.encoder_bias,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(REGEX_LEG_LENGTH_JOINTS_ONLY)),
      "bias_range": (-0.005, 0.005),
    },
  )

  # -- Rewards

  cfg.rewards["pose"].params["asset_cfg"].joint_names = (actuated_joints,)
  cfg.rewards["pose"].params["std_standing"] = {actuated_joints: 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    # Lower body.
    r"leg_.*_1_.*": 0.15,
    r"leg_.*_2_.*": 0.3,  # pitch
    r"leg_.*_3_.*": 0.15,
    r"leg_.*_length_.*": 0.1,  # length
    r"leg_.*_4_.*": 0.25,
    r"leg_.*_5_.*": 0.1,
    # Waist.
    r"pelvis_1.*": 0.08,
    r"pelvis_2.*": 0.2,
    # Arms.
    r"arm_.*_1_.*": 0.2,  # pitch
    r"arm_.*_4_.*": 0.2,  # elbow
    r"arm_.*_(?![14]_joint)\d+_joint": 0.1,
  }
  cfg.rewards["pose"].params["std_running"] = {
    # Lower body.
    r"leg_.*_1_.*": 0.2,
    r"leg_.*_2_.*": 0.5,
    r"leg_.*_3_.*": 0.2,
    r"leg_.*_length_.*": 0.15,
    r"leg_.*_4_.*": 0.35,
    r"leg_.*_5_.*": 0.15,
    # Waist.
    r"pelvis_1.*": 0.08,
    r"pelvis_2.*": 0.3,
    # Arms.
    r"arm_.*_1_.*": 0.4,
    r"arm_.*_4_.*": 0.35,
    r"arm_.*_(?![14]_joint)\d+_joint": 0.15,
  }
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("pelvis_2_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis_2_link",)
  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02
  cfg.rewards["air_time"].weight = 0.25
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": self_collision_cfg.name},
  )

  # The hull points should correspond to the respective joints defined in the joint_names_group order
  # leg_*_2_joint corresponds to Hip Pitch and leg_*_3_joint corresponds to Hip roll
  cfg.rewards["convex_hull_joint_limits_hip"] = RewardTermCfg(
    func=mdp.joint_limits_convex_hull,
    weight=-10.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
      "metrics_suffix": "hipXY",
      "joint_names_group": [
        [r"leg_left_2_joint", r"leg_left_3_joint"],
        [r"leg_right_2_joint", r"leg_right_3_joint"],
      ],
      "margin": 0.02,
      "hull_points": HIP_XY_CONVEX_HULL_POINTS,
    },
  )

  cfg.rewards["convex_hull_joint_limits_ankle"] = RewardTermCfg(
    func=mdp.joint_limits_convex_hull,
    weight=-10.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
      "margin": 0.02,
      "metrics_suffix": "ankleXY",
      "joint_names_group": [
        [r"leg_left_4_joint", r"leg_left_5_joint"],
        [r"leg_right_4_joint", r"leg_right_5_joint"],
      ],
      "hull_points": ANKLE_XY_CONVEX_HULL_POINTS,
    },
  )
  cfg.rewards["joint_vel_limits"] = RewardTermCfg(
    func=mdp.joint_vel_limits,
    weight=-10.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(REGEX_LEG_LENGTH_JOINTS_ONLY,)),
      "velocity_limits": {REGEX_LEG_LENGTH_JOINTS_ONLY: (-1.6, 1.6)},
    },
  )

  ## Metrics
  cfg.metrics = {
    "joint_vel_mag": MetricsTermCfg(
      func=mdp.joint_velocity_magnitude,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_acc_mag": MetricsTermCfg(
      func=mdp.joint_accelerations_magnitude,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_torque_mag": MetricsTermCfg(
      func=mdp.joint_torques_magnitude,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "action_rate_l2": MetricsTermCfg(func=mdp.action_rate_l2, params={}),
    "action_acc_l2": MetricsTermCfg(func=mdp.action_acc_l2, params={}),
    "max_feet_delta_vel_along_gravity": MetricsTermCfg(
      func=mdp.max_feet_delta_velocity_along_gravity,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=site_names)},
    ),
  }

  # # All except leg length joints
  # cfg.rewards["joint_accel"] = RewardTermCfg(
  #     func=mdp.joint_acc_l2,
  #     weight=-1.0e-8,
  #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
  # )

  # cfg.curriculum["joint_accel"] = CurriculumTermCfg(
  #   func=mdp.reward_curriculum,
  #   params={"reward_name": "joint_accel",
  #           "stages": [
  #               {"step": 0, "weight": 0.0 },
  #               {"step": 2000 * 24, "weight": -1.0e-8},
  #               {"step": 8000 * 24, "weight": -1.0e-7},
  #               {"step": 15000 * 24, "weight": -1.0e-6},
  #           ],
  #   },
  # )

  # -- Terminations

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
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def pal_kangaroo_hands_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO with hands (5 DoF per arms) rough terrain velocity configuration."""
  cfg = pal_kangaroo_rough_env_cfg(play=play)

  cfg.scene.entities = {"robot": get_kangaroo_hands_robot_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_HANDS_ACTION_SCALE
  joint_pos_action.actuator_names = KANGAROO_HANDS_ACTUATOR_NAMES

  return cfg


def pal_kangaroo_grippers_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO with grippers (7 DoF per arms) rough terrain velocity configuration."""
  cfg = pal_kangaroo_rough_env_cfg(play=play)

  cfg.scene.entities = {"robot": get_kangaroo_grippers_robot_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_GRIPPERS_ACTION_SCALE
  joint_pos_action.actuator_names = KANGAROO_GRIPPERS_ACTUATOR_NAMES

  return cfg


def pal_kangaroo_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO flat terrain velocity configuration."""
  cfg = pal_kangaroo_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["critic"].terms["height_scan"] # Actor already removed

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Disable terrain curriculum.
  assert cfg.curriculum is not None
  assert "terrain_levels" in cfg.curriculum
  del cfg.curriculum["terrain_levels"]

  if play:
    # Disable command curriculum.
    assert "command_vel" in cfg.curriculum
    del cfg.curriculum["command_vel"]

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def pal_kangaroo_hands_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO with hands (5 DoF per arms) flat terrain velocity configuration."""
  cfg = pal_kangaroo_flat_env_cfg(play=play)

  cfg.scene.entities = {"robot": get_kangaroo_hands_robot_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_HANDS_ACTION_SCALE
  joint_pos_action.actuator_names = KANGAROO_HANDS_ACTUATOR_NAMES

  return cfg


def pal_kangaroo_grippers_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO with grippers (7 DoF per arms) flat terrain velocity configuration."""
  cfg = pal_kangaroo_flat_env_cfg(play=play)

  cfg.scene.entities = {"robot": get_kangaroo_grippers_robot_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_GRIPPERS_ACTION_SCALE
  joint_pos_action.actuator_names = KANGAROO_GRIPPERS_ACTUATOR_NAMES

  return cfg


def pal_kangaroo_stairs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO stairs terrain velocity configuration."""
  cfg = pal_kangaroo_rough_env_cfg(play=play)

  ### SENSORS

  # Configure the terrain scan with the pelvis
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame = ObjRef(type="body", name="pelvis_2_link", entity="robot")

  # # More permissive illegal contact sensor
  # # preventing the envs for terminating early due to tibia collisions
  # new_body_ground_contact = ContactSensorCfg(
  #   name="body_ground_contact",
  #   primary=ContactMatch(
  #     mode="geom",
  #     pattern=REGEX_ILLEGAL_GEOMS,
  #     entity="robot",
  #   ),
  #   secondary=ContactMatch(mode="body", pattern="terrain"),
  #   fields=("found",),
  #   reduce="none",
  #   num_slots=1,
  # )

  # # The info of this sensor can be used to penalize contacts by time
  # knee_ground_cfg = ContactSensorCfg(
  #   name="knee_ground_contact",
  #   primary=ContactMatch(
  #     mode="geom",
  #     pattern=r"^leg_(left|right)_knee_collision$",
  #     entity="robot",
  #   ),
  #   secondary=ContactMatch(mode="body", pattern="terrain"),
  #   fields=("found",),
  #   reduce="none",
  #   num_slots=1,
  #   track_air_time=True,   # required for current_contact_time
  # )

  # cfg.scene.sensors = tuple(
  #   new_body_ground_contact if s.name == "body_ground_contact" else s
  #   for s in (cfg.scene.sensors or ())
  # ) + (knee_ground_cfg,)

  ### OBSERVATIONS

  # OBSERVATION LAG (sensor lag)
  cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
  cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 2
  cfg.observations["actor"].terms["base_lin_acc"].delay_min_lag = 0
  cfg.observations["actor"].terms["base_lin_acc"].delay_max_lag = 1
  cfg.observations["actor"].terms["imu_projected_gravity"].delay_min_lag = 0
  cfg.observations["actor"].terms["imu_projected_gravity"].delay_max_lag = 2
  cfg.observations["actor"].terms["joint_pos"].delay_min_lag = 0
  cfg.observations["actor"].terms["joint_pos"].delay_max_lag = 1
  cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 0
  cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1

  # OBSERVATION history (to better infer dynamics)

  # Only applied to the most important observations
  cfg.observations["actor"].terms["base_ang_vel"].history_length = 3
  cfg.observations["critic"].terms["base_ang_vel"].history_length = 3

  cfg.observations["actor"].terms["base_lin_acc"].history_length = 3
  cfg.observations["critic"].terms["base_lin_acc"].history_length = 3

  cfg.observations["actor"].terms["imu_projected_gravity"].history_length = 3
  cfg.observations["critic"].terms["imu_projected_gravity"].history_length = 3

  cfg.observations["actor"].terms["joint_pos"].history_length = 3
  cfg.observations["critic"].terms["joint_pos"].history_length = 3

  cfg.observations["actor"].terms["joint_vel"].history_length = 3
  cfg.observations["critic"].terms["joint_vel"].history_length = 3

  # Reduce the joint vel obs noise
  cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.15, n_max=0.15)
  del cfg.observations["critic"].terms["joint_vel"].noise

  # Give some evolution info
  cfg.observations["critic"].terms["height_scan"].history_length = 3

  ### COMMANDS

  # Start with a small forward-only cmd range
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  
  twist_cmd.resampling_time_range=(6.0, 9.0) # More constant commands
  twist_cmd.heading_command = True
  twist_cmd.rel_standing_envs = 0.1
  twist_cmd.rel_forward_envs = 0.0 # This is only straight, the random sampling takes care of this
  twist_cmd.rel_heading_envs = 1.0
  twist_cmd.ranges.lin_vel_x = (-0.3, -0.2) # Only forward and backward
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (-0.3, 0.3) # When heading = true, it is the clip value
  twist_cmd.ranges.heading = (0.0, 0.0) # Lock heading

  ### REWARDS

  # Experimentally, this reward just makes the robot learn upright faster
  # cfg.rewards["is_terminated"] = RewardTermCfg(
  #   func=mdp.is_terminated,
  #   weight=-100.0
  # )

  # cfg.rewards["knee_ground_contact"] = RewardTermCfg(
  #   func=mdp.knee_ground_contact_time,
  #   weight=-2.0,
  #   params={
  #     "sensor_name": knee_ground_cfg.name,
  #     "saturation_time": 0.5, 
  #   },
  # )

  # cfg.rewards["torso_height"] = RewardTermCfg(
  #   func=mdp.torso_height,
  #   weight=-3.0,
  #   params={
  #     "z_des": 0.85,
  #     "std": 0.1,
  #   },
  # )

  # Reward policies that lasted more before illegal contact
  # cfg.rewards["illegal_contact_termination_penalty"] = RewardTermCfg(
  #   func=mdp.illegal_contact_termination_penalty,
  #   weight=-50.0,
  #   params={"term_name": "illegal_contacts"},
  # )

  # With ``terrain_sensor_names``, penalizes tilt relative to the terrain surface normal.
  cfg.rewards["upright"].params["terrain_sensor_names"] = ("terrain_scan",)

  # Extra clearance to go over steps
  cfg.rewards["foot_clearance"].params["target_height"] = 0.2
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.2
  cfg.rewards["air_time"].weight = 0.5 # More important
  cfg.rewards["air_time"].params["threshold_min"] = 0.1
  cfg.rewards["air_time"].params["threshold_max"] = 1.0
  cfg.rewards["air_time"].params["command_threshold"] = 0.1

  # Use deocupled ang vel z-xy reward / penalties
  cfg.rewards["track_angular_velocity"].func = mdp.ang_vel_z_exp
  cfg.rewards["body_ang_vel"].func = mdp.ang_vel_xy_l2

  # More principled std values (Rudin et al)
  cfg.rewards["track_angular_velocity"].weight = 2.0
  cfg.rewards["track_linear_velocity"].weight = 2.0
  cfg.rewards["track_linear_velocity"].params["std"] = 0.25
  cfg.rewards["track_angular_velocity"].params["std"] = 0.25
  cfg.rewards["upright"].params["std"] = 0.25 # Stricter
  # cfg.rewards["body_ang_vel"].weight = -0.1 # Stricter

  ### EVENTS

  # PUSHING

  # Milder, low velocity adapted pushes
  # cfg.events["push_robot"] = EventTermCfg(
  #   func=mdp.push_by_setting_velocity,
  #   mode="interval",
  #   interval_range_s=(4.0, 6.0),
  #   params={
  #     "velocity_range": {
  #       "x": (-0.3, 0.3),
  #       "y": (-0.3, 0.3),
  #       "z": (0.0, 0.0), # Doesn't make a lot of sense
  #       "roll": (-0.2, 0.2),
  #       "pitch": (-0.2, 0.2),
  #       "yaw": (-0.25, 0.25),
  #     }
  #   }
  # )

  # Safer spawning close to the center (to avoid directly spawning unbalanced most of the time)
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.05, 0.05),
    "y": (-0.05, 0.05),
    "z": (0.01, 0.05),
    "yaw": (-0.1, 0.1),
  }

  ### CURRICULUM

  # TERRAIN
  
  # Use a better progess ratio based promotion function
  cfg.curriculum["terrain_levels"].func = mdp.terrain_levels_vel
  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    num_rows=12,
    num_cols=15,
    border_width=20.0,
    curriculum=True,
    sub_terrains={
      "flat": flat(proportion=0.1),
      "pebbles": random_spread_boxes(
        proportion=0.1,
        num_boxes=350,
        box_width_range=(0.02, 0.05),
        box_length_range=(0.02, 0.05),
        box_height_range=(0.02, 0.05),
        platform_width=0.5,
        border_width=0.2,
      ),
      "random_obstacles": random_spread_boxes(
        proportion=0.1,
        num_boxes=50,
        box_width_range=(0.2, 0.6),
        box_length_range=(0.2, 0.6),
        box_height_range=(0.02, 0.05),
        platform_width=0.5,
        border_width=0.2,
      ),
      "easy_stairs_50": pyramid_stairs_inv(
        proportion=0.1,
        step_height_range=(0.02, 0.05),
        step_width=0.5,
        platform_width=0.5,
        border_width=0.2,
      ),
      "easy_stairs_40": pyramid_stairs_inv(
        proportion=0.1,
        step_height_range=(0.02, 0.05),
        step_width=0.4,
        platform_width=0.5,
        border_width=0.2,
      ),
      "easy_stairs_30": pyramid_stairs_inv(
        proportion=0.1,
        step_height_range=(0.02, 0.05),
        step_width=0.3,
        platform_width=0.5,
        border_width=0.2,
      ),
      "mid_stairs_50": pyramid_stairs_inv(
        proportion=0.1,
        step_height_range=(0.05, 0.1),
        step_width=0.5,
        platform_width=0.5,
        border_width=0.2,
      ),
      "mid_stairs_40": pyramid_stairs_inv(
        proportion=0.1,
        step_height_range=(0.05, 0.1),
        step_width=0.4,
        platform_width=0.5,
        border_width=0.2,
      ),
      "mid_stairs_30": pyramid_stairs_inv(
        proportion=0.1,
        step_height_range=(0.05, 0.1),
        step_width=0.3,
        platform_width=0.5,
        border_width=0.2,
      ),
      "hard_stairs_50": pyramid_stairs_inv(
        proportion=0.1,
        step_height_range=(0.1, 0.15),
        step_width=0.5,
        platform_width=0.5,
        border_width=0.2,
      ),
      "hard_stairs_40": pyramid_stairs_inv(
        proportion=0.1,
        step_height_range=(0.1, 0.15),
        step_width=0.4,
        platform_width=0.5,
        border_width=0.2,
      ),
      "hard_stairs_30": pyramid_stairs_inv(
        proportion=0.1,
        step_height_range=(0.1, 0.15),
        step_width=0.3,
        platform_width=0.5,
        border_width=0.2,
      ),
      "super_hard_stairs_50": pyramid_stairs_inv(
        proportion=0.1,
        step_height_range=(0.15, 0.2),
        step_width=0.5,
        platform_width=0.5,
        border_width=0.2,
      ),
      "super_hard_stairs_40": pyramid_stairs_inv(
        proportion=0.1,
        step_height_range=(0.15, 0.2),
        step_width=0.4,
        platform_width=0.5,
        border_width=0.2,
      ),
      "super_hard_stairs_30": pyramid_stairs_inv(
        proportion=0.1,
        step_height_range=(0.15, 0.2),
        step_width=0.3,
        platform_width=0.5,
        border_width=0.2,
      ),
    },
  )

  # VELOCITY COMMAND

  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {"step": 0, "lin_vel_x": (-0.3, -0.2)},
        {"step": 5000 * 24, "lin_vel_x": (-0.4, -0.2)},
        {"step": 10000 * 24, "lin_vel_x": (-0.5, -0.2)},
        {"step": 20000 * 24, "lin_vel_x": (-0.6, -0.2)},
      ],
    },
  )

  # REWARDS PARAMS

  # Adapt std so a motionless robot at the maximum commanded velocity earns the same fraction of the tracking reward as in Rudin et al. 2022 (legged_gym default: ~0.018 — i.e., effectively zero).
  # The principled rule, derived from Rudin's σ_lin/cmd_max = 0.5: σ = cmd_max / 2 at each stage.
  # cfg.curriculum["track_linear_velocity_params"] = CurriculumTermCfg(
  #   func=mdp.reward_params,
  #   params={
  #     "reward_name": "track_linear_velocity",
  #     "param_stages": [
  #       {"step": 0, "params": {"std": 0.15}},
  #       {"step": 5000 * 24, "params": {"std": 0.2}},
  #       {"step": 10000 * 24, "params": {"std": 0.25}},
  #       {"step": 20000 * 24, "params": {"std": 0.3}},
  #     ],
  #   },
  # )

  # No need for a track curriculum as the target is always 0
  # It would only narrow ned
  # cfg.curriculum["track_angular_velocity_params"] = CurriculumTermCfg(
  #   func=mdp.reward_params,
  #   params={
  #     "reward_name": "track_angular_velocity",
  #     "param_stages": [
  #       {"step": 0, "params": {"std": math.sqrt(0.25)}},
  #       {"step": 5000 * 24, "params": {"std": math.sqrt(0.2)}},
  #       {"step": 10000 * 24, "params": {"std": math.sqrt(0.15)}},
  #       {"step": 20000 * 24, "params": {"std": math.sqrt(0.1)}},
  #     ],
  #   },
  # )

  ### PLAY
  if play:
    # Disable command curriculum.
    assert "command_vel" in cfg.curriculum
    del cfg.curriculum["command_vel"]

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.6, -0.2)
    twist_cmd.ranges.heading = (0.1, 0.1) # Lock heading

    # if cfg.scene.terrain is not None:
    #   if cfg.scene.terrain.terrain_generator is not None:
    #     cfg.scene.terrain.terrain_generator.num_cols = 5
    #     cfg.scene.terrain.terrain_generator.num_rows = 5
    #     cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg