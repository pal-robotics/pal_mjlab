"""PAL Robotics KANGAROO velocity tracking environment configurations."""

import math
import re
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import MetricsTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg, ObservationGroupCfg
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
from pal_mjlab.tasks.velocity.mdp.observations import _JOINT_NAMES_FOR_DP, _JOINT_NAMES_FOR_DP_2

from pal_mjlab import PAL_MJLAB_SRC_PATH

def pal_kangaroo_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
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

  scales_dict = KANGAROO_ACTION_SCALE  # regex -> scale

  ordered_scales = {}
  ordered_names = []

  for name in _JOINT_NAMES_FOR_DP_2:
      matched = False

      for pattern, scale in scales_dict.items():
          if re.fullmatch(pattern, name):
              ordered_scales[name] = scale
              ordered_names.append(name)
              matched = True
              break

      if not matched:
          raise ValueError(f"No scale found for joint: {name}")

  joint_pos_action.scale = ordered_scales
  joint_pos_action.actuator_names = tuple(ordered_names)

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
    noise=Unoise(n_min=-0.1, n_max=0.1),
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

  cfg.observations["actor"].terms["joint_vel"].noise = Unoise(n_min=-0.5, n_max=0.5)

  ### Disabling the use of history length as we haven't seen much improvements with it
  ### Moreover, our best policy #62 doesn't use any history length
  # cfg.observations["actor"].history_length = 5  # Keep last 5 frames
  # cfg.observations["critic"].history_length = 5  # Keep last 5 frames


  dp_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_dp,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"asset_cfg" : SceneEntityCfg("robot", joint_names=_JOINT_NAMES_FOR_DP)},
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_dp,
      noise=Unoise(n_min=-0.5, n_max=0.5),
      params={"asset_cfg" : SceneEntityCfg("robot", joint_names=_JOINT_NAMES_FOR_DP)},
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "base_lin_acc" : ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_acc"},
        noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "Action": ObservationTermCfg(func=mdp.last_action),
  }

  cfg.observations["dynamics_prior"] = ObservationGroupCfg(
      terms=dp_terms,
      concatenate_terms=True,
      enable_corruption=False,
    )

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

  cfg.rewards["dynamics_prior_accuracy"] = RewardTermCfg(
    func=mdp.dynamics_prior_accuracy,
    weight=0.5,
    params={
      "path": str(PAL_MJLAB_SRC_PATH / "dp_models" / "Good_Complete.pt"),
      "tokens": 10,
      "std" : math.sqrt(0.25),
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
