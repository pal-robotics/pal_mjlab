"""PAL Robotics Kangaroo Flat terrain tracking configuration."""

import dataclasses

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.tasks.tracking.mdp import MotionCommandCfg
# from mjlab.managers.curriculum_manager import CurriculumTermCfg
# from pal_mjlab.tasks.tracking.mdp.commands import PalMotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from pal_mjlab.tasks.tracking import mdp as tracking_mdp

from pal_mjlab.robots import (
  ANKLE_XY_CONVEX_HULL_POINTS,
  HIP_XY_CONVEX_HULL_POINTS,
  KANGAROO_ACTION_SCALE,
  KANGAROO_ACTUATOR_NAMES,
  REGEX_FEMUR_AND_KNEE_LINKS,
  get_kangaroo_robot_cfg,
)
from pal_mjlab.tasks.velocity import mdp


def pal_kangaroo_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics Talos flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_kangaroo_robot_cfg()}
  cfg.sim.mujoco.timestep = 0.002
  cfg.decimation = 10

  geom_names = tuple(
    f"{side}_foot{i}_collision"
    for side in ("left", "right")
    for i in [0, 2, 4, 6, 8, 10]
  )

  body_geoms = (
    # # Femur
    "leg_left_femur_collision",
    "leg_right_femur_collision",
    # Knee
    "leg_left_knee_collision",
    "leg_left_knee_bar_collision",
    "leg_right_knee_collision",
    "leg_right_knee_bar_collision",
    # Arms (4 only — arm 3 has no collision geom)
    "arm_left_4_collision",
    "arm_right_4_collision",
    # Pelvis
    "pelvis_2_collision",
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  feet_ground_contact_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(leg_left_5_link|leg_right_5_link)$",
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
  cfg.scene.sensors = (self_collision_cfg, feet_ground_contact_cfg, body_ground_cfg)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_ACTION_SCALE
  joint_pos_action.actuator_names = KANGAROO_ACTUATOR_NAMES

  assert cfg.commands is not None
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  # _base_motion_cmd = cfg.commands["motion"]
  # motion_cmd = PalMotionCommandCfg(
  #   **{f.name: getattr(_base_motion_cmd, f.name) for f in dataclasses.fields(_base_motion_cmd)}
  # )
  # cfg.commands["motion"] = motion_cmd
  motion_cmd.anchor_body_name = "base_link"
  motion_cmd.body_names = (
    "base_link",
    "pelvis_2_link",
    "leg_left_3_link",
    "leg_left_4_link",
    "leg_left_5_link",
    "leg_right_3_link",
    "leg_right_4_link",
    "leg_right_5_link",
    "arm_left_2_link",
    "arm_left_3_link",
    "arm_left_tip_link",
    "arm_right_2_link",
    "arm_right_3_link",
    "arm_right_tip_link",
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

  # cfg.rewards["joint_acc"] = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)

  # cfg.rewards["foot_contact"] = RewardTermCfg(
  #   func=tracking_mdp.motion_foot_contact,
  #   weight=0.5,
  #   params={
  #     "command_name": "motion",
  #     "sensor_name": "feet_ground_contact",
  #     "foot_body_names": ("leg_left_5_link", "leg_right_5_link"),
  #     "height_threshold": 0.03,
  #   },
  # )

  # cfg.rewards["motion_global_root_pos"].weight = 1.0

  # cfg.rewards["action_rate_l2"].weight = 1.e-2

  # cfg.curriculum["motion_trajectory_fraction"] = CurriculumTermCfg(
  #   func=tracking_mdp.motion_trajectory_fraction,
  #   params={
  #     "command_name": "motion",
  #     "fraction_stages": [
  #       {"step": 0, "fraction": 0.25},
  #       {"step": 10000 * 24, "fraction": 0.50},
  #       {"step": 15000 * 24, "fraction": 1.0},
  #     ],
  #   },
  # )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["body_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=body_geoms),  # Set per-robot.
      "operation": "abs",
      "ranges": (0.3, 2.0),
      "shared_random": False,  # All body geoms share the same friction.
    },
  )

  cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis_2_link",)

  cfg.terminations["illegal_contacts"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "body_ground_contact"},
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "leg_left_5_link",
    "leg_right_5_link",
    "arm_left_5_link",
    "arm_right_5_link",
  )
 # cfg.terminations["ee_body_pos"].params["threshold"] = 0.5

  cfg.viewer.body_name = "base_link"

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
  cfg.observations["critic"].terms["foot_air_time"] = ObservationTermCfg(
    func=mdp.foot_air_time,
    params={"sensor_name": "feet_ground_contact"},
  )

  cfg.rewards["motion_global_root_lin_vel_z"] = RewardTermCfg(
    func=tracking_mdp.motion_global_anchor_velocity_z_error_exp,
    weight=1.0,
    params={
      "command_name": "motion",
      "std": 1.0,
    },
  )

  cfg.rewards["foot_air"] = RewardTermCfg(
    func=tracking_mdp.feet_air_time,
    weight=10.0,
    params={
      "sensor_name": "feet_ground_contact",
      "threshold": 0.1,
    },
  )

  # cfg.observations["actor"].history_length = 3  # Keep last 3 frames
  # cfg.observations["critic"].history_length = 3  # Keep last 3 frames

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      # I added motion_anchor_ori_b but might not be necessary,
      # and i wonder if i should add lin acc when state
      # estimation is false
      if k not in ["motion_anchor_pos_b", "motion_anchor_ori_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
