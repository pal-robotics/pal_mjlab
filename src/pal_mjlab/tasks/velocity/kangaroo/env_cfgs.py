"""PAL Robotics KANGAROO velocity tracking environment configurations."""

import math
from dataclasses import replace

import mjlab.terrains as terrain_gen
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
  GridPatternCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.utils.spec_config import CollisionCfg

from pal_mjlab.actuators import BacklashPdActuatorCfg
from pal_mjlab.robots import (
  ANKLE_XY_CONVEX_HULL_POINTS,
  HIP_XY_CONVEX_HULL_POINTS,
  KANGAROO_ACTION_SCALE,
  KANGAROO_ACTUATOR_NAMES,
  KANGAROO_GRIPPERS_ACTION_SCALE,
  KANGAROO_GRIPPERS_ACTUATOR_NAMES,
  KANGAROO_HANDS_ACTION_SCALE,
  KANGAROO_HANDS_ACTUATOR_NAMES,
  KANGAROO_HIP_ROLL_PARAMS,
  KANGAROO_LOWER_BODY_ACTION_SCALE,
  KANGAROO_LOWER_BODY_ACTUATOR_NAMES,
  REGEX_ALL_ACTUATED_JOINTS,
  REGEX_FEMUR_AND_KNEE_LINKS,
  REGEX_HIP_ROLL_JOINTS,
  REGEX_LEG_LENGTH_JOINTS_ONLY,
  get_kangaroo_grippers_robot_cfg,
  get_kangaroo_hands_robot_cfg,
  get_kangaroo_lower_body_robot_cfg,
  get_kangaroo_robot_cfg,
)
from pal_mjlab.tasks.velocity import mdp
from pal_mjlab.terrains import BoxStraightStairsTerrainCfg

# Hip roll backlash measured on the real robot: ~2 cm of free lateral foot motion
# with the leg in the air in position control. With the ~0.74-0.80 m hip-roll-to-sole
# lever arm that is ~0.025-0.027 rad of total play; the range goes up to ~3 cm at the
# foot for robustness (and down to 0 so a tighter joint is also covered).
HIP_ROLL_BACKLASH_RANGE = (0.0, 0.04)
# Minimum surface distance (m) the policy is asked to keep between both feet.
FEET_SAFETY_MARGIN = 0.04


def _add_hip_roll_backlash(
  cfg: ManagerBasedRlEnvCfg,
  backlash_range: tuple[float, float] = HIP_ROLL_BACKLASH_RANGE,
) -> None:
  """Replace the hip roll actuators by an explicit PD with backlash (dead zone).

  Must be called after the robot entity is set, since it edits its articulation.
  """
  robot_cfg = cfg.scene.entities["robot"]
  assert robot_cfg.articulation is not None
  actuators = []
  replaced = 0
  for act in robot_cfg.articulation.actuators:
    if act.target_names_expr == (REGEX_HIP_ROLL_JOINTS,):
      act = BacklashPdActuatorCfg(
        target_names_expr=act.target_names_expr,
        backlash=backlash_range[1],
        **KANGAROO_HIP_ROLL_PARAMS,
      )
      replaced += 1
    actuators.append(act)
  assert replaced == 1, "Expected exactly one hip roll actuator config."
  # Copy, the articulation config is shared between robot variants.
  robot_cfg.articulation = replace(robot_cfg.articulation, actuators=tuple(actuators))

  # Resampled every episode, independently per leg.
  cfg.events["hip_roll_backlash"] = EventTermCfg(
    mode="reset",
    func=mdp.randomize_backlash,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "backlash_range": backlash_range,
    },
  )


def _add_feet_safety_margin(
  cfg: ManagerBasedRlEnvCfg, margin: float = FEET_SAFETY_MARGIN
) -> None:
  """Ask the policy to keep a safety gap between the real foot geometries.

  The oversized foot "avoidance" capsules used to collide physically, acting as
  bumpers that push the feet apart in sim but do not exist on the real robot. They
  are made non-colliding and replaced by a graded distance penalty on the real
  sole + ankle capsules. Real foot contacts are still penalized by
  "self_collisions".

  Must be called after the robot entity is set, since it edits its collisions.
  """
  robot_cfg = cfg.scene.entities["robot"]
  robot_cfg.collisions = tuple(robot_cfg.collisions) + (
    CollisionCfg(
      geom_names_expr=(r".*_foot_avoidance_collision_\d+$",),
      contype=0,
      conaffinity=0,
      condim=1,
      priority=0,
      disable_other_geoms=False,
    ),
  )

  cfg.rewards["feet_proximity"] = RewardTermCfg(
    func=mdp.feet_proximity,
    weight=-2.0,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "left_geom_names": (
        r"^left_foot\d+_collision$",
        r"^leg_left_ankle_(bar_)?collision$",
      ),
      "right_geom_names": (
        r"^right_foot\d+_collision$",
        r"^leg_right_ankle_(bar_)?collision$",
      ),
      "margin": margin,
    },
  )


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
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "operation": "add",
      "ranges": {
        r"leg_.*_1_joint": (-0.1, 0.1),
        r"leg_.*_2_joint": (-0.25, 0.25),
        r"leg_.*_3_joint": (-0.2, 0.2),
        r"leg_.*_4_joint": (-0.15, 0.15),
        r"leg_.*_5_joint": (-0.1, 0.1),
        r"^(?!(?:leg_.*_1_joint|leg_.*_2_joint|leg_.*_3_joint|leg_.*_4_joint|leg_.*_5_joint)$).*": (
          -0.008,
          0.008,
        ),
      },
      "shared_random": False,
    },
  )
  # Domain Randomization for joint damping
  cfg.events["joint_damping"] = EventTermCfg(
    mode="startup",
    func=dr.dof_damping,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "operation": "add",
      "ranges": {
        r"leg_.*_1_joint": (-0.2, 0.2),
        r"leg_.*_2_joint": (-0.2, 0.2),
        r"leg_.*_3_joint": (-0.1, 0.1),
        r"leg_.*_4_joint": (-0.1, 0.1),
        r"leg_.*_5_joint": (-0.1, 0.1),
      },
      "shared_random": False,
    },
  )
  # Domain Randomization for joint armature
  cfg.events["joint_armature"] = EventTermCfg(
    mode="startup",
    func=dr.dof_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "operation": "add",
      "ranges": {
        r"leg_.*_1_joint": (-0.1, 0.1),
        r"leg_.*_2_joint": (-0.25, 0.25),
        r"leg_.*_3_joint": (-0.075, 0.075),
        r"leg_.*_4_joint": (-0.05, 0.05),
        r"leg_.*_5_joint": (-0.005, 0.005),
        r"leg_.*_length_joint": (-0.5, 0.5),
      },
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
  cfg.rewards["upright"].weight = 2.0
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


def pal_kangaroo_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics custom rough terrain velocity configuration."""

  ### GENERAL CONFIGURATION

  cfg = pal_kangaroo_baseline_env_cfg(play=play)

  # nconmax is the max number of contacts that will be generated at runtime
  # due to https://github.com/google-deepmind/mujoco_warp/blob/c62864ed2bf816c0a724d4cbf153921188f78eae/mujoco_warp/_src/io.py#L649-L660
  # for collision-rich envs, it is recommended to be manually set through experimentation
  cfg.sim.nconmax = 200
  # To avoid memory issues when allocating EPA buffer
  cfg.sim.mujoco.ccd_iterations = 100

  ### SENSORS

  # Idealized 1.0x1.0 m elevation map around the robot
  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="pelvis_2_link", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.0, 1.0), resolution=0.05),
    max_distance=2.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=True,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (terrain_scan,)

  ### OBSERVATIONS

  # The default scan was deleted, we add a custom one, only to the critic
  cfg.observations["critic"].terms["height_scan"] = ObservationTermCfg(
    func=mdp.height_scan,
    params={"sensor_name": "terrain_scan"},
    scale=1 / terrain_scan.max_distance,
  )

  ### COMMANDS

  # Easy starting velocity
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, mdp.UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-0.5, 0.5)
  twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  # Delete the speed curriculum altogether
  # del cfg.curriculum["command_vel"]
  command_vel_curriculum = cfg.curriculum["command_vel"]
  assert isinstance(command_vel_curriculum, CurriculumTermCfg)
  command_vel_curriculum.params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (-0.5, 0.5),
      "lin_vel_y": (-0.5, 0.5),
      "ang_vel_z": (-1.0, 1.0),
    },
    {"step": 5000 * 24, "lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0)},
    {"step": 10000 * 24, "lin_vel_x": (-2.0, 2.0)},
    {"step": 15000 * 24, "lin_vel_x": (-2.0, 3.0)},
    {"step": 20000 * 24, "lin_vel_x": (-2.0, 4.0)},
    {"step": 25000 * 24, "lin_vel_x": (-2.0, 5.0)},
  ]

  ### REWARDS

  # More human-like air time and stronger, specially important with obstacles
  cfg.rewards["air_time"].weight = 1.0
  cfg.rewards["air_time"].params["threshold_min"] = 0.1
  cfg.rewards["air_time"].params["threshold_max"] = 0.4
  cfg.rewards["air_time"].params["command_threshold"] = 0.1

  # Reduce upright std, to have torso stance more stable
  cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

  # Tightened from default so the reward stays discriminative at low command speeds
  # instead of flattening into a dead-zone.
  cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)

  # Human-like arm swing: shoulder pitch in counter-phase with the legs, amplitude
  # proportional to the forward command (0.2 rad per m/s, capped at 0.5 rad).
  # Signs give "forward" motion: +leg_*_2 moves the foot back, +arm_left_1 moves the
  # left hand back and +arm_right_1 moves the right hand forward.
  cfg.rewards["arm_swing"] = RewardTermCfg(
    func=mdp.arm_swing,
    weight=0.5,
    params={
      "command_name": "twist",
      "hip_joint_names": ("leg_left_2_joint", "leg_right_2_joint"),
      "shoulder_joint_names": ("arm_left_1_joint", "arm_right_1_joint"),
      "hip_flexion_signs": (-1.0, -1.0),
      "shoulder_flexion_signs": (-1.0, 1.0),
      "std": 0.15,
      "amplitude_per_mps": 0.2,
      "max_amplitude": 0.5,
      "hip_flexion_ref": 0.25,
      # Elbows (+arm_*_4 bends both): flex on the forward swing, open on the back
      # swing, and bend more with speed (0.1 rad per m/s, capped at 0.4 rad).
      "elbow_joint_names": ("arm_left_4_joint", "arm_right_4_joint"),
      "elbow_flexion_signs": (1.0, 1.0),
      "elbow_swing_ratio": 0.5,
      "elbow_bend_per_mps": 0.1,
      "max_elbow_bend": 0.4,
      # Torso yaw (pelvis_2, +q brings the left shoulder forward) counter-rotates
      # against the hips: 0.05 rad per m/s, capped at 0.15 rad. Torso roll
      # (pelvis_1) is left to the upright and pose rewards to keep the head stable.
      "torso_yaw_joint_name": "pelvis_2_joint",
      "torso_yaw_sign": -1.0,
      "torso_yaw_per_mps": 0.05,
      "max_torso_yaw": 0.15,
      "torso_yaw_std": 0.05,
      "command_threshold": 0.1,
    },
  )
  # Let the shoulder pitch and elbow leave the default pose enough to swing.
  cfg.rewards["pose"].params["std_walking"][r"arm_.*_1_.*"] = 0.35
  cfg.rewards["pose"].params["std_running"][r"arm_.*_1_.*"] = 0.6
  cfg.rewards["pose"].params["std_walking"][r"arm_.*_4_.*"] = 0.3
  cfg.rewards["pose"].params["std_running"][r"arm_.*_4_.*"] = 0.55

  # Sim2real: hip roll backlash + safety gap between the feet (feet collided on
  # the real robot at ~3.5 m/s).
  _add_hip_roll_backlash(cfg)
  _add_feet_safety_margin(cfg)

  ### EVENTS

  # Safer spawning close to the center (to avoid directly spawning unbalanced most of the time)
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.2, 0.2),
    "y": (-0.2, 0.2),
    "z": (0.01, 0.05),
    "yaw": (-3.14, 3.14),
  }

  ### CURRICULUM

  # Terrain

  # TODO: review fairness of the curriculum (https://github.com/mujocolab/mjlab/issues/934)
  cfg.curriculum["terrain_levels"].func = mdp.terrain_levels_vel

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
    size=(4.0, 4.0),
    num_rows=12,
    border_width=20.0,
    curriculum=True,  # one column per sub-terrain; num_cols is ignored
    add_lights=False,
    sub_terrains={
      "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.2),
      # Fine gravel-like ground: many tiny boxes scattered over a floor.
      "pebbles": terrain_gen.BoxRandomSpreadTerrainCfg(
        proportion=0.2,
        num_boxes=350,
        box_width_range=(0.02, 0.05),
        box_length_range=(0.02, 0.05),
        box_height_range=(0.02, 0.05),
        box_yaw_range=(0.0, 360.0),
        add_floor=True,
        platform_width=0.6,
        border_width=0.0,
      ),
      # Sparse medium obstacles to step over / around.
      "random_boxes": terrain_gen.BoxRandomSpreadTerrainCfg(
        proportion=0.2,
        num_boxes=30,
        box_width_range=(0.2, 0.6),
        box_length_range=(0.2, 0.6),
        box_height_range=(0.02, 0.12),
        box_yaw_range=(0.0, 360.0),
        add_floor=True,
        platform_width=0.6,
        border_width=0.0,
      ),
      # Fine uneven ground: small cells, low relief.
      "random_grid_fine": terrain_gen.BoxRandomGridTerrainCfg(
        proportion=0.1,
        grid_width=0.25,
        grid_height_range=(0.01, 0.04),
        platform_width=0.6,
        border_width=0.25,
        merge_similar_heights=True,
        height_merge_threshold=0.01,
        max_merge_distance=3,
      ),
      # Coarse uneven ground: foot-sized cells, larger steps between them.
      "random_grid_coarse": terrain_gen.BoxRandomGridTerrainCfg(
        proportion=0.1,
        grid_width=0.5,
        grid_height_range=(0.01, 0.07),
        platform_width=0.6,
        border_width=0.25,
        merge_similar_heights=True,
        height_merge_threshold=0.02,
        max_merge_distance=3,
      ),
      # Discrete obstacles
      "hf_discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
        proportion=0.1,
        obstacle_width_range=(0.3, 0.8),
        obstacle_height_range=(0.01, 0.06),
        num_obstacles=100,
        platform_width=0.5,
        border_width=0.1,
        base_thickness_ratio=0.5,
      ),
      # Natural ondulation fractal noise
      "perlin_noise": terrain_gen.HfPerlinNoiseTerrainCfg(
        proportion=0.1,
        height_range=(0.02, 0.2),
        resolution=0.10,
        scale=5.0,
      ),
      # Basic stairs, one descending + one ascending.
      # Pyramid: spawn on top platform, walk down. Inverted: spawn at bottom, walk up.
      # "stairs_down": terrain_gen.BoxPyramidStairsTerrainCfg(
      #   proportion=0.1,
      #   step_height_range=(0.02, 0.1),
      #   step_width=0.35,
      #   platform_width=0.6,
      #   border_width=0.1,
      # ),
      # "stairs_up": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      #   proportion=0.1,
      #   step_height_range=(0.02, 0.1),
      #   step_width=0.35,
      #   platform_width=0.6,
      #   border_width=0.1,
      # ),
    },
  )

  # PLAY
  if play:
    # Disable terrain curriculum.
    assert cfg.curriculum is not None
    assert "terrain_levels" in cfg.curriculum
    del cfg.curriculum["terrain_levels"]

  return cfg


def pal_kangaroo_lower_body_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO with lower_body (Legs + Pelvis) rough terrain velocity configuration."""
  cfg = pal_kangaroo_rough_env_cfg(play=play)

  for pose_type in ("std_walking", "std_running"):
    del cfg.rewards["pose"].params[pose_type][r"arm_.*_1_.*"]
    del cfg.rewards["pose"].params[pose_type][r"arm_.*_4_.*"]
    del cfg.rewards["pose"].params[pose_type][r"arm_.*_(?![14]_joint)\d+_joint"]
  del cfg.rewards["arm_swing"]  # No arms.

  cfg.scene.entities = {"robot": get_kangaroo_lower_body_robot_cfg()}
  # The robot entity was replaced: re-apply the robot-level sim2real edits.
  _add_hip_roll_backlash(cfg)
  _add_feet_safety_margin(cfg)

  # Maintain base linear acceleration for the kangaroo lower body policies
  # TODO: Train and test a policy without the base linear acceleration term
  cfg.observations["actor"].terms["base_lin_acc"] = ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_lin_acc"},
    noise=Unoise(n_min=-1.0, n_max=1.0),
  )

  # Prevents feet instability
  cfg.rewards["action_rate_l2"].weight = -0.2

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_LOWER_BODY_ACTION_SCALE
  joint_pos_action.actuator_names = KANGAROO_LOWER_BODY_ACTUATOR_NAMES

  return cfg


def pal_kangaroo_stairs_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO rough terrain with straight stairs velocity configuration."""
  cfg = pal_kangaroo_rough_env_cfg(play=play)

  ### SENSORS

  # The tibia brushing stair noses is expected (the actor is blind and feels risers
  # with the shins): only the femur stays an illegal contact, tibia contacts are
  # penalized instead (see "shin_contacts" reward).
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "body_ground_contact":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = r"^(leg_left_femur_link|leg_right_femur_link)$"

  shin_ground_cfg = ContactSensorCfg(
    name="shin_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r"^(leg_left_knee_link|leg_right_knee_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=cfg.decimation,  # Every physics substep of the policy step
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (shin_ground_cfg,)

  ### COMMANDS

  # More forward-only commands (x > 0, y = yaw = 0) for pure stair traversal
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, mdp.UniformVelocityCommandCfg)
  twist_cmd.rel_forward_envs = 0.55  # 50 % of envs (after the standing envs override)
  twist_cmd.heading_command = False
  twist_cmd.rel_heading_envs = 0.0
  del twist_cmd.ranges.heading

  ### REWARDS

  # Higher swing apex so 20 cm risers aren't penalized
  # (foot_clearance intentionally untouched: increasing it goes unstable)
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.25

  # Mild penalty for tibia-terrain contact (no longer a termination).
  # With force history the cost is the number of substeps (0..decimation) with a
  # shin contact above force_threshold, i.e. proportional to contact duration.
  cfg.rewards["shin_contacts"] = RewardTermCfg(
    func=mdp.self_collision_cost,  # generic: counts contacts of the given sensor
    weight=(-1.0 / cfg.decimation),  # -1.0 per fully-in-contact policy step
    params={"sensor_name": shin_ground_cfg.name, "force_threshold": 10.0},
  )

  ### EVENTS

  # On straight staircases spawn facing the stairs: at the bottom facing the up
  # flight (+x), on the landing facing either down flight (+x or -x).
  cfg.events["reset_base"].func = mdp.reset_root_state_terrain_yaw
  cfg.events["reset_base"].params["terrain_yaw"] = {
    "straight_stairs_up": (0.0,),
    "straight_stairs_down": (0.0, math.pi),
  }
  cfg.events["reset_base"].params["yaw_offset_range"] = (-0.3, 0.3)

  ### CURRICULUM

  # Terrain: the rough sub-terrains are reused with lower proportions, the rest
  # goes to stairs.
  assert cfg.scene.terrain is not None
  terrain_generator = cfg.scene.terrain.terrain_generator
  assert terrain_generator is not None
  rough = terrain_generator.sub_terrains

  # Stairs, risers up to 20 cm, three tread widths, both directions.
  stairs_down = replace(rough["stairs_down"], step_height_range=(0.02, 0.2))
  stairs_up = replace(rough["stairs_up"], step_height_range=(0.02, 0.2))

  terrain_generator.sub_terrains = {
    "flat": replace(rough["flat"], proportion=0.03),
    "pebbles": replace(rough["pebbles"], proportion=0.05),
    "random_boxes": replace(rough["random_boxes"], proportion=0.08),
    "random_grid_fine": replace(rough["random_grid_fine"], proportion=0.05),
    "random_grid_coarse": replace(rough["random_grid_coarse"], proportion=0.05),
    "hf_discrete_obstacles": replace(rough["hf_discrete_obstacles"], proportion=0.05),
    "perlin_noise": replace(rough["perlin_noise"], proportion=0.05),
    "stairs_down_narrow": replace(stairs_down, proportion=0.06, step_width=0.28),
    "stairs_up_narrow": replace(stairs_up, proportion=0.06, step_width=0.28),
    "stairs_down": replace(stairs_down, proportion=0.07, step_width=0.35),
    "stairs_up": replace(stairs_up, proportion=0.07, step_width=0.35),
    "stairs_down_wide": replace(stairs_down, proportion=0.06, step_width=0.45),
    "stairs_up_wide": replace(stairs_up, proportion=0.06, step_width=0.45),
    # Irregular stairs: each riser sampled from the range, scaled by (0.5 + 0.5 * difficulty).
    "random_stairs": terrain_gen.BoxRandomStairsTerrainCfg(
      proportion=0.07,
      step_height_range=(0.04, 0.2),
      step_width=0.35,
      platform_width=0.6,
      border_width=0.1,
    ),
    # Open-riser stairs (thin slabs), tread narrows from 0.45 to 0.30 with difficulty.
    "open_stairs_up": terrain_gen.BoxOpenStairsTerrainCfg(
      proportion=0.05,
      step_height_range=(0.02, 0.2),
      step_width_range=(0.3, 0.45),
      platform_width=0.6,
      border_width=0.1,
      inverted=True,
    ),
    # Straight limited-width staircases with open sides (real stairs without
    # railings): falling off the sides teaches caution and some fall recovery.
    "straight_stairs_down": BoxStraightStairsTerrainCfg(
      proportion=0.04,
      step_height_range=(0.02, 0.2),
      spawn="top",
    ),
    "straight_stairs_up": BoxStraightStairsTerrainCfg(
      proportion=0.10,
      step_height_range=(0.02, 0.2),
      spawn="bottom",
    ),
  }

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
  cfg.rewards["upright"].weight = 1.5

  # Adjust noise parameters for the lower body kangaroo
  # These parameters worked for the microstrain IMU
  cfg.observations["actor"].terms["imu_projected_gravity"].noise = Unoise(
    n_min=-0.025, n_max=0.025
  )
  # Maintain base linear acceleration for the kangaroo lower body policies
  # TODO: Train and test a policy without the base linear acceleration term
  cfg.observations["actor"].terms["base_lin_acc"] = ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_lin_acc"},
    noise=Unoise(n_min=-1.0, n_max=1.0),
  )

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
