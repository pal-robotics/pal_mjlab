"""PAL Robotics KANGAROO stand-up environment configuration."""

import copy
import math

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
from mjlab.sensor import ContactMatch, ContactSensorCfg

from pal_mjlab.robots import (
  ANKLE_XY_CONVEX_HULL_POINTS,
  FALL_COLLISION_MESH_SUFFIX,
  FALL_STANDING_HEIGHT,
  FALL_TERRAIN_COLLISION,
  HIP_XY_CONVEX_HULL_POINTS,
  NATURAL_FREQ,
  REGEX_ALL_ACTUATED_JOINTS,
  REGEX_LEG_LENGTH_JOINTS_ONLY,
  get_kangaroo_fall_robot_cfg,
)
from pal_mjlab.tasks.stand_up import mdp
from pal_mjlab.tasks.velocity.kangaroo.env_cfgs import pal_kangaroo_flat_env_cfg

# Every body with collision geometry, and the ones that carry the soles.
REGEX_GROUND_BODIES = (
  r"^(base_link|pelvis_[12]_link|arm_(left|right)_[1-4]_link"
  r"|leg_(left|right)_([1-4]|femur|knee|foot|ankle)_link)$"
)
REGEX_FOOT_BODIES = r"leg_(left|right)_foot_link"

# Actuated joints other than the leg length: the ones that can be offset
# without breaking the knee linkage.
REGEX_ROTARY_JOINTS = r"^(?!leg_.*_(length|femur|knee)_joint$).*_joint$"

TORSO_BODY = "pelvis_2_link"

# Base height [m]: about 0.15 lying or sitting, 0.45 in the deepest squat,
# 0.9 in the default pose.
LYING_HEIGHT = 0.2
STAND_HEIGHT = 0.8

# The arm action scale is the one of actuators with this natural frequency
# [Hz] instead of 3.5: 12.25 times larger. The gains are unchanged.
ARM_ACTION_SCALE_NATURAL_FREQ_HZ = 1.0

# Upward force [robot weights] on the torso of a robot on the ground at the
# start of training, and the step it is lowered by.
ASSIST_LEVEL = 1.0
ASSIST_STEP = 0.1

# Margin [rad] inside the hip and ankle hulls.
HULL_MARGIN = 0.05

# Rewards that penalize violent motion and the factor each weight is raised
# by at most during training. The ones that measure what exceeds a limit go
# furthest.
CARE_REWARDS = {
  "joint_speed": 50.0,
  "ground_force": 50.0,
  "base_acceleration": 50.0,
  "actuator_overload": 10.0,
  "actuator_effort": 10.0,
  "base_speed": 10.0,
  "self_collisions": 10.0,
  "action_rate_l2": 10.0,
  "action_acc_l2": 10.0,
}


def kangaroo_start_states(num_states: int) -> mdp.StartStateCfg:
  return mdp.StartStateCfg(
    num_states=num_states,
    joint_hulls=(
      ((r"leg_.*_2_joint", r"leg_.*_3_joint"), HIP_XY_CONVEX_HULL_POINTS),
      ((r"leg_.*_4_joint", r"leg_.*_5_joint"), ANKLE_XY_CONVEX_HULL_POINTS),
    ),
    leg_length_joint_regex=REGEX_LEG_LENGTH_JOINTS_ONLY,
    # With longer legs, or the ankles pitched less, the robot does not keep
    # its balance on its own.
    crouch_leg_length_range=(0.17, 0.5),
    crouch_joint_offset={
      r"leg_.*_[1235]_joint": (-0.1, 0.1),
      r"leg_.*_4_joint": (-0.6, -0.15),
      r"pelvis_.*_joint": (-0.15, 0.15),
      r"arm_.*_joint": (-0.6, 0.6),
    },
  )


def pal_kangaroo_stand_up_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO stand-up configuration."""
  cfg = pal_kangaroo_flat_env_cfg(play=play)

  # Mesh collisions: every link can end up on the ground.
  cfg.scene.entities = {"robot": get_kangaroo_fall_robot_cfg()}
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.collisions = (FALL_TERRAIN_COLLISION,)
  cfg.sim.nconmax = 96
  cfg.sim.njmax = 400
  cfg.sim.contact_sensor_maxmatch = 64

  # Long enough that most of an episode is spent standing. Play keeps the
  # velocity task's endless episodes.
  if not play:
    cfg.episode_length_s = 20.0
  cfg.viewer.body_name = TORSO_BODY

  # -- Actions

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  assert isinstance(joint_pos_action.scale, dict)
  arm_scale_factor = (
    NATURAL_FREQ / (2.0 * math.pi * ARM_ACTION_SCALE_NATURAL_FREQ_HZ)
  ) ** 2
  joint_pos_action.scale = {
    joint: scale * (arm_scale_factor if joint.startswith("arm_") else 1.0)
    for joint, scale in joint_pos_action.scale.items()
  }

  # -- Sensors

  ground_sensor = ContactSensorCfg(
    name="body_ground_force",
    primary=ContactMatch(mode="body", pattern=REGEX_GROUND_BODIES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=cfg.decimation,
  )
  self_collision_sensor = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=cfg.decimation,
  )
  cfg.scene.sensors = tuple(
    s
    for s in (cfg.scene.sensors or ())
    if s.name not in ("body_ground_contact", "self_collision", "foot_height_scan")
  ) + (ground_sensor, self_collision_sensor)

  contact_cfg = mdp.GroundContactCfg(
    sensor_name=ground_sensor.name, foot_body_regex=REGEX_FOOT_BODIES
  )
  stand_cfg = mdp.StandCfg(
    contact_cfg=contact_cfg,
    torso_body=TORSO_BODY,
    target_height=FALL_STANDING_HEIGHT,
    min_height=LYING_HEIGHT,
    stand_height=STAND_HEIGHT,
  )
  actuated_cfg = SceneEntityCfg("robot", joint_names=(REGEX_ALL_ACTUATED_JOINTS,))

  # -- Commands

  cfg.commands = {}

  # -- Observations

  actor_terms = cfg.observations["actor"].terms
  critic_terms = cfg.observations["critic"].terms
  actor_terms.pop("command", None)
  for name in ("command", "foot_height", "foot_air_time"):
    critic_terms.pop(name, None)
  # Before the first step: the action that holds the start pose.
  for terms in (actor_terms, critic_terms):
    terms["actions"] = ObservationTermCfg(func=mdp.last_action_from_start)
  critic_terms["body_ground_load"] = ObservationTermCfg(
    func=mdp.body_ground_load, params={"contact_cfg": contact_cfg}
  )
  critic_terms["base_height"] = ObservationTermCfg(
    func=mdp.base_height_above_ground, params={"stand_cfg": stand_cfg}
  )
  critic_terms["torso_tilt_cos"] = ObservationTermCfg(
    func=mdp.torso_tilt_cos, params={"stand_cfg": stand_cfg}
  )

  # -- Events

  for name in ("reset_base", "reset_robot_joints", "push_robot", "randomize_terrain"):
    cfg.events.pop(name, None)
  cfg.events["body_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", geom_names=(rf".*{FALL_COLLISION_MESH_SUFFIX}$",)
      ),
      "operation": "abs",
      "ranges": (0.3, 1.0),
      "shared_random": True,
    },
  )
  # After the domain randomization terms. Few states in play, to start sooner.
  start_cfg = kangaroo_start_states(num_states=16 if play else 8192)
  cfg.events["generate_start_states"] = EventTermCfg(
    func=mdp.generate_start_states,
    mode="startup",
    params={"start_cfg": start_cfg},
  )
  cfg.events["reset_to_start_state"] = EventTermCfg(
    func=mdp.reset_to_start_state,
    mode="reset",
    params={
      "start_cfg": start_cfg,
      "standing_fraction": 0.3,
      "asset_cfg": SceneEntityCfg("robot", joint_names=(REGEX_ROTARY_JOINTS,)),
    },
  )
  cfg.events["lift_assist"] = EventTermCfg(
    func=mdp.lift_assist,
    mode="step",
    params={"stand_cfg": stand_cfg, "level": 0.0},
  )
  if not play:
    cfg.events["push_robot"] = EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(3.0, 6.0),
      params={
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "roll": (-0.3, 0.3),
          "pitch": (-0.3, 0.3),
          "yaw": (-0.3, 0.3),
        },
      },
    )

  # -- Rewards

  walking_rewards = cfg.rewards
  cfg.rewards = {
    # Progress.
    "torso_upright": RewardTermCfg(
      func=mdp.torso_upright, weight=1.0, params={"stand_cfg": stand_cfg}
    ),
    "feet_load": RewardTermCfg(
      func=mdp.feet_load, weight=1.5, params={"stand_cfg": stand_cfg}
    ),
    "base_height": RewardTermCfg(
      func=mdp.base_height_progress, weight=3.0, params={"stand_cfg": stand_cfg}
    ),
    "standing": RewardTermCfg(
      func=mdp.standing, weight=2.0, params={"stand_cfg": stand_cfg}
    ),
    "standing_posture": RewardTermCfg(
      func=mdp.standing_posture,
      weight=2.0,
      params={
        "stand_cfg": stand_cfg,
        "std": {
          REGEX_LEG_LENGTH_JOINTS_ONLY: 0.08,
          r"leg_.*_[1-5]_joint": 0.15,
          r"pelvis_.*_joint": 0.15,
          r"arm_.*_joint": 0.4,
        },
        "asset_cfg": actuated_cfg,
      },
    ),
    "standing_still": RewardTermCfg(
      func=mdp.standing_still,
      weight=1.0,
      params={"stand_cfg": stand_cfg, "lin_speed_std": 0.3, "ang_speed_std": 0.6},
    ),
    # Care: initial weights, raised by the curriculum (see CARE_REWARDS).
    "action_rate_l2": RewardTermCfg(func=mdp.episode_action_rate_l2, weight=-0.01),
    "action_acc_l2": RewardTermCfg(func=mdp.episode_action_acc_l2, weight=-0.0025),
    "joint_speed": RewardTermCfg(
      func=mdp.joint_speed_excess,
      weight=-0.5,
      params={
        # [m/s] for the leg length, [rad/s] for the others.
        "speed_limits": {REGEX_LEG_LENGTH_JOINTS_ONLY: 0.5, REGEX_ROTARY_JOINTS: 2.0},
        "asset_cfg": actuated_cfg,
      },
    ),
    "actuator_effort": RewardTermCfg(func=mdp.actuator_effort_l2, weight=-0.2),
    "actuator_overload": RewardTermCfg(
      func=mdp.actuator_overload, weight=-0.2, params={"threshold": 0.8}
    ),
    "ground_force": RewardTermCfg(
      func=mdp.ground_force_excess,
      weight=-5.0,
      params={
        "contact_cfg": contact_cfg,
        "body_threshold": 1.2,
        "feet_threshold": 1.3,
      },
    ),
    "base_speed": RewardTermCfg(
      func=mdp.base_speed_l2, weight=-0.15, params={"ang_scale": 0.3}
    ),
    "base_acceleration": RewardTermCfg(
      func=mdp.base_acceleration_excess,
      weight=-2.5,
      params={"sensor_name": "robot/imu_lin_acc", "threshold": 3.0},
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-0.05,
      params={"sensor_name": self_collision_sensor.name, "force_threshold": 200.0},
    ),
    # Limits, as in the velocity task.
    "dof_pos_limits": walking_rewards["dof_pos_limits"],
    "joint_vel_limits": walking_rewards["joint_vel_limits"],
  }
  # The hip and ankle hulls: the real transmission has no solution outside
  # them. Linear in the distance, so the pull back does not fade near the
  # edge, with a margin for the tracking error of the real robot. The
  # targets lead the positions out, so they are penalized too.
  for joint in ("hip", "ankle"):
    hull = walking_rewards[f"convex_hull_joint_limits_{joint}"]
    for source, weight in (("position", -20.0), ("target", -5.0)):
      term = copy.deepcopy(hull)
      term.weight = weight
      term.params["margin"] = HULL_MARGIN
      term.params["source"] = source
      term.params["squared"] = False
      term.params["metrics_suffix"] += f"_{source}"
      cfg.rewards[f"{joint}_hull_{source}"] = term

  # -- Curriculum

  cfg.curriculum = {}
  if not play:
    cfg.curriculum["stand_up"] = CurriculumTermCfg(
      func=mdp.stand_up_curriculum,
      params={
        "care_rewards": CARE_REWARDS,
        "stand_cfg": stand_cfg,
        "assist_event_name": "lift_assist",
        "assist_level": ASSIST_LEVEL,
        "assist_step": ASSIST_STEP,
        "care_step_factor": 1.2,
        "raise_above": 0.95,
        "lower_below": 0.9,
        "update_interval": 25 * 24,  # 25 training iterations.
      },
    )

  # -- Terminations

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }
  if not play:
    # Falling once up costs every reward the rest of the episode would bring.
    # Play lets the robot get up again, as the deployment would.
    cfg.terminations["fell_after_standing"] = TerminationTermCfg(
      func=mdp.fell_after_standing, params={"stand_cfg": stand_cfg}
    )

  # -- Metrics

  cfg.metrics = {
    "stood_up": MetricsTermCfg(
      func=mdp.standing_metric, params={"stand_cfg": stand_cfg}, reduce="last"
    ),
    "standing_time_fraction": MetricsTermCfg(
      func=mdp.standing_metric, params={"stand_cfg": stand_cfg}, reduce="mean"
    ),
    "peak_body_ground_load": MetricsTermCfg(
      func=mdp.peak_body_ground_load,
      params={"contact_cfg": contact_cfg},
      reduce="max",
    ),
    "peak_joint_speed": MetricsTermCfg(
      func=mdp.peak_joint_speed,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(REGEX_ROTARY_JOINTS,))},
      reduce="max",
    ),
  }

  return cfg
