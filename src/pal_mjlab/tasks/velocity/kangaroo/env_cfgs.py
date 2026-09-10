"""PAL Robotics KANGAROO velocity tracking environment configurations."""

import math

import mjlab.terrains as terrain_gen
import mujoco as _mujoco
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import CurriculumTermCfg, MetricsTermCfg
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
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pal_mjlab.robots import (
  ANKLE_XY_CONVEX_HULL_POINTS,
  HIP_XY_CONVEX_HULL_POINTS,
  KANGAROO_ACTION_SCALE,
  KANGAROO_ACTUATOR_NAMES,
  KANGAROO_GRIPPERS_ACTION_SCALE,
  KANGAROO_GRIPPERS_ACTUATOR_NAMES,
  KANGAROO_HANDS_ACTION_SCALE,
  KANGAROO_HANDS_ACTUATOR_NAMES,
  KANGAROO_LOWER_BODY_ACTION_SCALE,
  KANGAROO_LOWER_BODY_ACTUATOR_NAMES,
  REGEX_ALL_ACTUATED_JOINTS,
  REGEX_FEMUR_AND_KNEE_LINKS,
  REGEX_LEG_LENGTH_JOINTS_ONLY,
  get_kangaroo_grippers_robot_cfg,
  get_kangaroo_hands_robot_cfg,
  get_kangaroo_lower_body_robot_cfg,
  get_kangaroo_robot_cfg,
)
from pal_mjlab.tasks.velocity import mdp


def pal_kangaroo_baseline_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": get_kangaroo_robot_cfg()}
  cfg.sim.nconmax = None
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
  )

  # Remove the default terrain scan sensor
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "terrain_scan")

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

  cfg.observations["actor"].terms["height_scan"] = None
  cfg.observations["critic"].terms["height_scan"] = None
  cfg.observations["actor"].terms["base_lin_vel"] = None
  cfg.observations["actor"].terms["projected_gravity"] = None
  cfg.observations["actor"].terms["imu_projected_gravity"] = ObservationTermCfg(
    func=mdp.imu_projected_gravity,
    params={"sensor_name": "robot/imu_quat"},
    noise=Unoise(n_min=-0.05, n_max=0.05),
  )
  cfg.observations["actor"].terms["base_lin_acc"] = ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_lin_acc"},
    noise=Unoise(n_min=-0.5, n_max=0.5),
  )
  cfg.observations["critic"].terms["imu_projected_gravity"] = ObservationTermCfg(
    func=mdp.imu_projected_gravity,
    params={"sensor_name": "robot/imu_quat"},
  )
  cfg.observations["critic"].terms["base_lin_acc"] = ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_lin_acc"},
  )
  cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.5, n_max=0.5)

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
      "asset_cfg": SceneEntityCfg("robot", joint_names=[REGEX_LEG_LENGTH_JOINTS_ONLY]),
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
  cfg.rewards["upright"].weight = 1.25
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


def _soften_terrain_contacts(spec: _mujoco.MjSpec) -> None:
  """Soften terrain box geom contacts to reduce edge-contact NaN instability.

  Box terrains place adjacent geoms at different heights. The hard edges where
  heights change cause contact normal instability when feet land on them, which
  can produce impulsive NaN forces in the MuJoCo solver.

  Doubling the solref time constant (0.02 → 0.04 s) makes contact springs
  2× softer — enough to damp the instability without noticeably changing the
  macro-level walking physics. Applied to all geoms in the "terrain" body,
  which contains every box generated by TerrainGenerator.
  """
  body = spec.body("terrain")
  count = 0
  for geom in body.geoms:
    geom.solref = [0.04, 1.0]  # 2× softer time constant (default: 0.02)
    geom.solimp = [0.85, 0.95, 0.001, 0.5, 2.0]  # slightly softer impedance
    count += 1
  print(f"[rough terrain] spec_fn: softened {count} terrain geoms (solref=0.04)")


def _adapt_rough_terrain(terrain_gen_cfg: TerrainGeneratorCfg):
  # normal pyramid
  hf_pyramid = terrain_gen_cfg.sub_terrains["hf_pyramid_slope"]
  assert isinstance(hf_pyramid, terrain_gen.HfPyramidSlopedTerrainCfg)
  hf_pyramid.slope_range = (0.1, 0.4)
  hf_pyramid.vertical_scale = 0.001

  # inverted pyramid
  hf_pyramid_inv = terrain_gen_cfg.sub_terrains["hf_pyramid_slope_inv"]
  assert isinstance(hf_pyramid_inv, terrain_gen.HfPyramidSlopedTerrainCfg)
  hf_pyramid_inv.slope_range = (0.1, 0.4)
  hf_pyramid_inv.vertical_scale = 0.001

  # wave terrain
  hf_wave = terrain_gen_cfg.sub_terrains["wave_terrain"]
  assert isinstance(hf_wave, terrain_gen.HfWaveTerrainCfg)
  hf_wave.amplitude_range = (0.05, 0.2)


def pal_kangaroo_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics custom rough terrain velocity configuration."""

  ### GENERAL CONFIGURATION

  cfg = pal_kangaroo_baseline_env_cfg(play=play)

  # nconmax is the max number of contacts at runtime
  cfg.sim.njmax = 300
  cfg.sim.nconmax = 300
  cfg.sim.mujoco.iterations = 30
  cfg.sim.mujoco.ls_iterations = 50
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64

  # softer terrains
  cfg.scene.spec_fn = _soften_terrain_contacts

  ### OBSERVATIONS

  # Most actors in SOTA do not see base lin acc
  del cfg.observations["actor"].terms["base_lin_acc"]

  # Observation noise configuration (edit these values as needed)
  # cfg.observations["actor"].terms["base_ang_vel"].noise = Unoise(
  #   n_min=-0.03, n_max=0.03
  # )  # was 0.2
  # cfg.observations["actor"].terms["imu_projected_gravity"].noise = Unoise(
  #   n_min=-0.01, n_max=0.01
  # )  # was 0.15
  # cfg.observations["actor"].terms["joint_pos"].noise = Unoise(
  #   n_min=-0.001, n_max=0.001
  # )  # was 0.05
  # cfg.observations["actor"].terms["joint_vel"].noise = Unoise(
  #   n_min=-0.25, n_max=0.25
  # )  # was 2.0

  ### COMMANDS

  # Delete the speed curriculum altogether, the initial task is enough
  del cfg.curriculum["command_vel"]

  # Low vel task, only different is that with this command there's a turn in place bucket
  cfg.commands["twist"] = mdp.UniformVelocityCommandWithTurningBucketCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    rel_standing_envs=0.02,
    rel_forward_envs=0.2,
    rel_turn_in_place_envs=0.2,
    debug_vis=True,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-0.5, 0.5),
      lin_vel_y=(-0.4, 0.4),
      ang_vel_z=(-0.5, 0.5),
    ),
  )

  ### REWARDS

  # More human-like air time and stronger, specially important with obstacles
  cfg.rewards["air_time"].weight = 2.0
  cfg.rewards["air_time"].params["threshold_min"] = 0.2
  cfg.rewards["air_time"].params["threshold_max"] = 0.45
  cfg.rewards["air_time"].params["command_threshold"] = 0.01

  # Safer torso stance
  cfg.rewards["upright"].weight = 2.0
  cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

  # Tighter linear tracking std
  cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(
    0.1
  )  # duck std, suitable for low velocity

  # Only the leg length to try to penalize stomping
  # cfg.rewards["leg_length_acc"] = RewardTermCfg(
  #   func=mdp.joint_acc_l2,
  #   weight=-1e-8,
  #   params={
  #     "asset_cfg": SceneEntityCfg("robot", joint_names=(REGEX_LEG_LENGTH_JOINTS_ONLY,))
  #   },
  # )

  # Any non-zero command will now generate movement, needs to be penalized
  cfg.rewards["soft_landing"].params["command_threshold"] = 0.01

  ### TERRAIN

  # The default terrain slightly adapted to kangaroo capabilities
  assert cfg.scene.terrain is not None
  terrain_generator = cfg.scene.terrain.terrain_generator
  assert isinstance(terrain_generator, TerrainGeneratorCfg)
  _adapt_rough_terrain(terrain_generator)

  ### CURRICULUM

  # TODO: review fairness of the curriculum (https://github.com/mujocolab/mjlab/issues/934)
  cfg.curriculum["terrain_levels"].func = mdp.terrain_levels_vel

  # action_rate weight ramp: gentle smoothing while the gait bootstraps, then
  # tighten to -1.0 by iter 1500.
  cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
    func=mdp.reward_curriculum,
    params={
      "reward_name": "action_rate_l2",
      "stages": [
        {"step": 0, "weight": -0.1},
        {"step": 500 * 24, "weight": -0.2},
        {"step": 750 * 24, "weight": -0.4},
        {"step": 1000 * 24, "weight": -0.6},
        {"step": 1250 * 24, "weight": -0.8},
        {"step": 1500 * 24, "weight": -1.0},
      ],
    },
  )

  # Gradually increase standing env fraction after walking is established
  cfg.curriculum["standing_envs"] = CurriculumTermCfg(
    func=mdp.standing_envs_curriculum,
    params={
      "command_name": "twist",
      "standing_stages": [
        {"step": 0, "rel_standing_envs": 0.02},
        {"step": 500 * 24, "rel_standing_envs": 0.05},
        {"step": 750 * 24, "rel_standing_envs": 0.1},
        {"step": 1000 * 24, "rel_standing_envs": 0.15},
        {"step": 1500 * 24, "rel_standing_envs": 0.2},
        {"step": 2000 * 24, "rel_standing_envs": 0.25},
      ],
    },
  )

  # cfg.curriculum["leg_length_acc_weight"] = CurriculumTermCfg(
  #   func=mdp.reward_curriculum,
  #   params={
  #     "reward_name": "leg_length_acc",
  #     "stages": [
  #       {"step": 0, "weight": -1.0e-8},
  #       {"step": 500 * 24, "weight": -1.0e-7},
  #       {"step": 1000 * 24, "weight": -1.0e-6},
  #       {"step": 2000 * 24, "weight": -1.0e-5},
  #     ],
  #   },
  # )

  # cfg.curriculum["soft_landing_weight"] = CurriculumTermCfg(
  #   func=mdp.reward_curriculum,
  #   params={
  #     "reward_name": "soft_landing",
  #     "stages": [
  #       {"step": 0, "weight": -1.0e-5},
  #       {"step": 500 * 24, "weight": -1.0e-4},
  #       {"step": 1000 * 24, "weight": -1.0e-3},
  #       {"step": 2000 * 24, "weight": -1.0e-2},
  #     ],
  #   },
  # )

  # if play:
  #   twist_cmd = cfg.commands["twist"]
  #   assert isinstance(twist_cmd, mdp.UniformVelocityCommandWithTurningBucketCfg)
  #   twist_cmd.rel_turn_in_place_envs = 0.0
  #   twist_cmd.rel_standing_envs = 0.0
  #   twist_cmd.rel_forward_envs = 1.0

  #   del cfg.curriculum["action_rate_weight"]
  #   del cfg.curriculum["standing_envs"]
  #   del cfg.curriculum["soft_landing_weight"]

  return cfg


def pal_kangaroo_lower_body_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO with lower_body (Legs + Pelvis) rough terrain velocity configuration."""
  cfg = pal_kangaroo_rough_env_cfg(play=play)

  for pose_type in ("std_walking", "std_running"):
    del cfg.rewards["pose"].params[pose_type][r"arm_.*_1_.*"]
    del cfg.rewards["pose"].params[pose_type][r"arm_.*_4_.*"]
    del cfg.rewards["pose"].params[pose_type][r"arm_.*_(?![14]_joint)\d+_joint"]

  cfg.scene.entities = {"robot": get_kangaroo_lower_body_robot_cfg()}

  # Prevents feet instability
  cfg.rewards["action_rate_l2"].weight = -0.2

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_LOWER_BODY_ACTION_SCALE
  joint_pos_action.actuator_names = KANGAROO_LOWER_BODY_ACTUATOR_NAMES

  return cfg


def pal_kangaroo_hands_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO with hands (5 DoF per arms) rough terrain velocity configuration."""
  cfg = pal_kangaroo_baseline_env_cfg(play=play)

  cfg.scene.entities = {"robot": get_kangaroo_hands_robot_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_HANDS_ACTION_SCALE
  joint_pos_action.actuator_names = KANGAROO_HANDS_ACTUATOR_NAMES

  return cfg


def pal_kangaroo_grippers_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO with grippers (7 DoF per arms) rough terrain velocity configuration."""
  cfg = pal_kangaroo_baseline_env_cfg(play=play)

  cfg.scene.entities = {"robot": get_kangaroo_grippers_robot_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_GRIPPERS_ACTION_SCALE
  joint_pos_action.actuator_names = KANGAROO_GRIPPERS_ACTUATOR_NAMES

  return cfg


def pal_kangaroo_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO flat terrain velocity configuration."""
  cfg = pal_kangaroo_baseline_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

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


def pal_kangaroo_lower_body_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO with lower_body (Legs + Pelvis) flat terrain velocity configuration."""
  cfg = pal_kangaroo_flat_env_cfg(play=play)

  for pose_type in ("std_walking", "std_running"):
    del cfg.rewards["pose"].params[pose_type][r"arm_.*_1_.*"]
    del cfg.rewards["pose"].params[pose_type][r"arm_.*_4_.*"]
    del cfg.rewards["pose"].params[pose_type][r"arm_.*_(?![14]_joint)\d+_joint"]

  cfg.scene.entities = {"robot": get_kangaroo_lower_body_robot_cfg()}

  cfg.rewards["action_rate_l2"].weight = -0.2

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_LOWER_BODY_ACTION_SCALE
  joint_pos_action.actuator_names = KANGAROO_LOWER_BODY_ACTUATOR_NAMES

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
