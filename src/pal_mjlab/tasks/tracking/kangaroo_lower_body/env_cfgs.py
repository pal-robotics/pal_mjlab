"""PAL Robotics Kangaroo Lower Body Flat terrain tracking configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pal_mjlab.robots import (
  ANKLE_XY_CONVEX_HULL_POINTS,
  HIP_XY_CONVEX_HULL_POINTS,
  KANGAROO_LOWER_BODY_ACTION_SCALE,
  KANGAROO_LOWER_BODY_ACTUATOR_NAMES,
  REGEX_FEMUR_AND_KNEE_LINKS,
  get_kangaroo_lower_body_robot_cfg,
)
from pal_mjlab.tasks.tracking import mdp as tracking_mdp
from pal_mjlab.tasks.velocity import mdp


def pal_kangaroo_lower_body_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics Kangaroo lower body flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_kangaroo_lower_body_robot_cfg()}
  cfg.sim.mujoco.timestep = 0.002
  cfg.decimation = 10
  cfg.sim.nconmax = 64
  cfg.sim.njmax = 300

  geom_names = tuple(
    f"{side}_foot{i}_collision"
    for side in ("left", "right")
    for i in [0, 2, 4, 6, 8, 10]
  )

  body_geoms = (
    # Femur
    "leg_left_femur_collision",
    "leg_right_femur_collision",
    # Knee
    "leg_left_knee_collision",
    "leg_left_knee_bar_collision",
    "leg_right_knee_collision",
    "leg_right_knee_bar_collision",
    # Pelvis
    "pelvis_2_collision",
  )

  ## Sensors
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

  ## Actions
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_LOWER_BODY_ACTION_SCALE
  joint_pos_action.actuator_names = KANGAROO_LOWER_BODY_ACTUATOR_NAMES

  ## Commands
  assert cfg.commands is not None
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
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
  )

  ## Rewards
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

  # Loosen tracking precision so the policy receives useful gradient from the
  # start rather than near-zero signal when the initial error is large.
  cfg.rewards["motion_body_pos"].params["std"] = 0.7
  cfg.rewards["motion_body_ori"].params["std"] = 0.7
  cfg.rewards["motion_global_root_pos"].params["std"] = 0.7
  cfg.rewards["motion_global_root_ori"].params["std"] = 0.7

  # Reduce action-rate penalty so it does not crush exploration early on.
  cfg.rewards["action_rate_l2"].weight = -1e-2

  cfg.rewards["motion_global_root_lin_vel_z"] = RewardTermCfg(
    func=tracking_mdp.motion_global_anchor_velocity_z_error_exp,
    weight=1.0,
    params={
      "command_name": "motion",
      "std": 1.0,
    },
  )

  # Keep foot-air encouragement but at a weight comparable to tracking terms
  # so the policy does not learn to hop randomly while ignoring the motion ref.
  cfg.rewards["foot_air"] = RewardTermCfg(
    func=tracking_mdp.all_feet_air_time,
    weight=2.0,
    params={
      "sensor_name": "feet_ground_contact",
      "threshold": 0.1,
    },
  )

  ## Events (Domain Randomization)
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["foot_friction"].params["shared_random"] = (
    True  # All foot geoms share friction
  )
  cfg.events["body_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=body_geoms),
      "operation": "abs",
      "ranges": (0.3, 2.0),
      "shared_random": False,
    },
  )

  cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis_2_link",)

  ## Terminations
  cfg.terminations["illegal_contacts"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": "body_ground_contact"},
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "leg_left_5_link",
    "leg_right_5_link",
  )

  ## Viewer
  cfg.viewer.body_name = "base_link"

  ## Observations
  cfg.observations["actor"].terms["imu_projected_gravity"] = ObservationTermCfg(
    func=mdp.imu_projected_gravity,
    params={"sensor_name": "robot/imu_quat"},
    noise=Unoise(n_min=-0.02, n_max=0.02),
  )
  cfg.observations["actor"].terms["base_lin_acc"] = ObservationTermCfg(
    func=mdp.builtin_sensor,
    params={"sensor_name": "robot/imu_lin_acc"},
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
  cfg.observations["critic"].terms["foot_air_time"] = ObservationTermCfg(
    func=mdp.foot_air_time,
    params={"sensor_name": "feet_ground_contact"},
  )

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "motion_anchor_ori_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  ## Play mode overrides
  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.sampling_mode = "start"

  return cfg
