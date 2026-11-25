"""PAL Robotics Kangaroo Flat terrain tracking configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pal_mjlab.robots import (
  KANGAROO_ACTION_SCALE,
  KANGAROO_ACTUATOR_NAMES,
  KANGAROO_HANDS_ACTION_SCALE,
  KANGAROO_HANDS_ACTUATOR_NAMES,
  REGEX_JOINTS_WITHOUT_ENVELOPE,
  get_kangaroo_ankle_envelope,
  get_kangaroo_hands_robot_cfg,
  get_kangaroo_hip_envelope,
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
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_ACTION_SCALE
  joint_pos_action.actuator_names = KANGAROO_ACTUATOR_NAMES

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
    "arm_left_2_link",
    "arm_left_3_link",
    "arm_left_tip_link",
    "arm_right_2_link",
    "arm_right_3_link",
    "arm_right_tip_link",
  )
  # The joint position range was changed from +/- 0.1 to +/- 0.05 since the passive joints
  # were also being affected, sometimes causing the knee bar to penetrate the foot.
  motion_cmd.joint_position_range = (-0.05, 0.05)

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

  # Hip joints 2 / 3 and ankle joints 4 / 5 are limited by the envelope rewards below.
  cfg.rewards["joint_limit"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=(REGEX_JOINTS_WITHOUT_ENVELOPE,)
  )
  # Margins in the units of their limits: joint [rad], actuator [m].
  hip_A, hip_b = get_kangaroo_hip_envelope(shrink_joint=0.02, shrink_actuator=0.0015)
  cfg.rewards["joint_envelope_hip"] = RewardTermCfg(
    func=mdp.joint_envelope_limits,
    weight=-10.0,
    params={
      # One pattern per role [joint 2, joint 3], each matching both legs.
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(r"leg_.*_2_joint", r"leg_.*_3_joint"),
        preserve_order=True,
      ),
      "A": hip_A,
      "b": hip_b,
      "metrics_name": "hip",
    },
  )

  # Margins in the units of their limits: joint and crank-bar collision [rad],
  # actuator [m].
  ankle_A, ankle_b = get_kangaroo_ankle_envelope(
    shrink_joint=0.02, shrink_actuator=0.0008, shrink_crank=0.02
  )
  cfg.rewards["joint_envelope_ankle"] = RewardTermCfg(
    func=mdp.joint_envelope_limits,
    weight=-10.0,
    params={
      # One pattern per role [joint 4, joint 5, leg length, femur, knee], each
      # matching both legs.
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          r"leg_.*_4_joint",
          r"leg_.*_5_joint",
          r"leg_.*_length_joint",
          r"leg_.*_femur_joint",
          r"leg_.*_knee_joint",
        ),
        preserve_order=True,
      ),
      "A": ankle_A,
      "b": ankle_b,
      "metrics_name": "ankle",
    },
  )

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

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "leg_left_5_link",
    "leg_right_5_link",
    "arm_left_5_link",
    "arm_right_5_link",
  )

  cfg.viewer.body_name = "base_link"

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


def pal_kangaroo_hands_flat_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics KANGAROO with hands (5 DoF per arms) rough terrain velocity configuration."""
  cfg = pal_kangaroo_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation, play=play
  )

  cfg.scene.entities = {"robot": get_kangaroo_hands_robot_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_HANDS_ACTION_SCALE
  joint_pos_action.actuator_names = KANGAROO_HANDS_ACTUATOR_NAMES

  return cfg
