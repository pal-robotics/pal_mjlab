"""PAL Robotics Kangaroo Flat terrain tracking configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from pal_mjlab.tasks.WBC.mdp import OnlineMotionCommandCfg
from pal_mjlab.tasks.WBC import mdp
from pal_mjlab.tasks.WBC.wbc_env_cfg import make_wbc_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pal_mjlab.tasks.velocity.mdp import joint_torques_magnitude, raw_action_metrics

from pal_mjlab.robots import (
  ANKLE_XY_CONVEX_HULL_POINTS,
  HIP_XY_CONVEX_HULL_POINTS,
  KANGAROO_ACTION_SCALE,
  KANGAROO_ACTUATOR_NAMES,
  KANGAROO_GRIPPERS_ACTION_SCALE,
  KANGAROO_GRIPPERS_ACTUATOR_NAMES,
  get_kangaroo_robot_cfg,
  get_kangaroo_grippers_robot_cfg,
)

def pal_kangaroo_flat_wbc_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create PAL Robotics Talos flat terrain tracking configuration."""
  cfg = make_wbc_env_cfg()

  cfg.scene.entities = {"robot": get_kangaroo_robot_cfg()}
  cfg.sim.mujoco.timestep = 0.002
  cfg.decimation = 10

  end_effector_body_names = ("arm_right_tip_link","arm_left_tip_link")

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
  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(leg_left_5_link|leg_right_5_link)$",  # subtree so foot link is included
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (self_collision_cfg, feet_ground_cfg)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_ACTION_SCALE
  joint_pos_action.actuator_names = KANGAROO_ACTUATOR_NAMES

  assert cfg.commands is not None
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, OnlineMotionCommandCfg)
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
  motion_cmd.joint_position_range = (-0.05, 0.05)
  motion_cmd.tracked_body_names = end_effector_body_names
  motion_cmd.robot_type = "Kangaroo"

  # Setup end effector cartesian tracking instead of joint position/velocity tracking
  # cfg.observations["actor"].terms["ref_joint_pos"] = None
  # cfg.observations["actor"].terms["ref_joint_vel"] = None

  # cfg.observations["actor"].terms["ref_body_pos"] = ObservationTermCfg(
  #   func=mdp.ref_body_pos_b, 
  #   params={
  #     "command_name": "motion", 
  #     "body_names": end_effector_body_names,
  #   },
  #   noise=Unoise(n_min=-0.05, n_max=0.05),
  # )
  # cfg.observations["actor"].terms["ref_body_ori"] = ObservationTermCfg(
  #   func=mdp.ref_body_ori_b, 
  #   params={
  #     "command_name": "motion", 
  #     "body_names": end_effector_body_names,
  #   },
  #   noise=Unoise(n_min=-0.02, n_max=0.02),
  # )


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

  # # ZMP must stay inside the (shrunken) support polygon for clips in which the
  # # reference never lifts a foot. The outer foot capsules span the foot sole.
  # cfg.rewards["zmp_support_polygon"] = RewardTermCfg(
  #   func=mdp.zmp_support_polygon_penalty,
  #   weight=0.0,  # Disabled: superseded by dcm_support_center + unrequested_foot_lift.
  #   params={
  #     "command_name": "motion",
  #     "foot_body_names": ("leg_left_5_link", "leg_right_5_link"),
  #     "foot_geom_names": (
  #       "left_foot0_collision",
  #       "left_foot10_collision",
  #       "right_foot0_collision",
  #       "right_foot10_collision",
  #     ),
  #     "margin": 0.04,
  #     "lift_threshold": 0.03,
  #     "contact_height": 0.01,
  #   },
  # )

  # Stepping is allowed but not free: while the reference foot is planted, a
  # robot foot in the air (an unrequested step) or sliding on the ground costs
  # reward. Small disturbances should then be absorbed in double support, while
  # a step is still worth it when a push would otherwise make the robot fall.
  # Lift cost per footstep is about |weight| * swing_time (~0.35 s).
  ref_contact_params = {
    "command_name": "motion",
    "sensor_name": "feet_ground_contact",
    "foot_body_names": ("leg_left_5_link", "leg_right_5_link"),
    "lift_threshold": 0.03,
    "vel_threshold": 0.1,
  }
  cfg.rewards["unrequested_foot_lift"] = RewardTermCfg(
    func=mdp.ref_contact_foot_lift_penalty,
    weight=-5.0,
    params=dict(ref_contact_params),
  )
  cfg.rewards["unrequested_foot_slip"] = RewardTermCfg(
    func=mdp.ref_contact_foot_slip_penalty,
    weight=-1.0,
    params=dict(ref_contact_params),
  )

  # Ramp the foot penalties in: early on the policy can't stand yet, so they
  # only add unavoidable negative reward that pushes towards ending episodes.
  # Steps are env steps (iterations * num_steps_per_env = 24).
  cfg.curriculum["unrequested_foot_lift_curr"] = CurriculumTermCfg(
    func=mdp.reward_curriculum,
    params={
      "reward_name": "unrequested_foot_lift",
      "stages": [
        {"step": 0, "weight": 0.0},
        {"step": 1_000 * 24, "weight": -2.5},
        {"step": 3_000 * 24, "weight": -5.0},
      ],
    },
  )
  cfg.curriculum["unrequested_foot_slip_curr"] = CurriculumTermCfg(
    func=mdp.reward_curriculum,
    params={
      "reward_name": "unrequested_foot_slip",
      "stages": [
        {"step": 0, "weight": 0.0},
        {"step": 1_000 * 24, "weight": -0.5},
        {"step": 3_000 * 24, "weight": -1.0},
      ],
    },
  )

  # Balance task: while the reference is in double support, keep the DCM at the
  # centre of the robot's feet. Combined with the lift penalty: absorb
  # disturbances in double support, step only when the DCM can't be recovered.
  cfg.rewards["dcm_support_center"] = RewardTermCfg(
    func=mdp.dcm_support_center_exp,
    weight=1.0,
    params={
      **ref_contact_params,
      "left_foot_geom_names": tuple(n for n in geom_names if n.startswith("left_")),
      "right_foot_geom_names": tuple(n for n in geom_names if n.startswith("right_")),
      "std_x": 0.06,
      "std_y": 0.08,
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

  # Domain Randomization for joint friction
  cfg.events["joint_friction"] = EventTermCfg(
    mode="startup",
    func=dr.dof_frictionloss,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),  # Set per-robot.
      "operation": "add",
      "ranges": {
        r"leg_.*_1_joint" : (-0.1,0.1),
        r"leg_.*_2_joint" : (-0.25,0.25),
        r"leg_.*_3_joint" : (-0.2,0.2),
        r"leg_.*_4_joint" : (-0.15,0.15),
        r"leg_.*_5_joint" : (-0.1,0.1),
        r"^(?!(?:leg_.*_1_joint|leg_.*_2_joint|leg_.*_3_joint|leg_.*_4_joint|leg_.*_5_joint)$).*" : (-0.008, 0.008)
      },
      "shared_random": False,
    },
  )
  # Domain Randomization for joint damping
  cfg.events["joint_damping"] = EventTermCfg(
    mode="startup",
    func=dr.dof_damping,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),  # Set per-robot.
      "operation": "add",
      "ranges": {
        r"leg_.*_1_joint" : (-0.2,0.2),
        r"leg_.*_2_joint" : (-0.2,0.2),
        r"leg_.*_3_joint" : (-0.1,0.1),
        r"leg_.*_4_joint" : (-0.1,0.1),
        r"leg_.*_5_joint" : (-0.1,0.1),
      },
      "shared_random": False,
    },
  )
  # Domain Randomization for joint armature
  cfg.events["joint_armature"] = EventTermCfg(
    mode="startup",
    func=dr.dof_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),  # Set per-robot.
      "operation": "add",
      "ranges": {
        r"leg_.*_1_joint" : (-0.1,0.1),
        r"leg_.*_2_joint" : (-0.25,0.25),
        r"leg_.*_3_joint" : (-0.075,0.075),
        r"leg_.*_4_joint" : (-0.05,0.05),
        r"leg_.*_5_joint" : (-0.005,0.005),
      },
      "shared_random": False,
    },
  )

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "leg_left_5_link",
    "leg_right_5_link",
    "arm_left_5_link",
    "arm_right_5_link",
  )

  cfg.viewer.body_name = "base_link"

  ## Metrics
  cfg.metrics = {
    "arm_1_joint_torque_mag": MetricsTermCfg(
      func=joint_torques_magnitude,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"arm_left_1_joint",))},
    ),
    "arm_2_joint_torque_mag": MetricsTermCfg(
      func=joint_torques_magnitude,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"arm_left_2_joint",))},
    ),
    "arm_3_joint_torque_mag": MetricsTermCfg(
      func=joint_torques_magnitude,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"arm_left_3_joint",))},
    ),
    "arm_4_joint_torque_mag": MetricsTermCfg(
      func=joint_torques_magnitude,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"arm_left_4_joint",))},
    ),
  }

  cfg.metrics.update(
    raw_action_metrics([f"arm_left_{i}_joint" for i in range(1, 5)])
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    # Training-only term; its __init__ sweeps the whole motion library with
    # batch_size == num_envs, which is prohibitively slow with few play envs.
    cfg.rewards.pop("zmp_support_polygon", None)
    cfg.rewards.pop("unrequested_foot_lift", None)
    cfg.rewards.pop("unrequested_foot_slip", None)
    cfg.rewards.pop("dcm_support_center", None)
    cfg.curriculum.pop("unrequested_foot_lift_curr", None)
    cfg.curriculum.pop("unrequested_foot_slip_curr", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"
    motion_cmd.rel_fixed_command = 0.1

  return cfg

def pal_kangaroo_grippers_flat_wbc_env_cfg(
  play: bool = False
) -> ManagerBasedRlEnvCfg :
  """Create PAL Robotics KANGAROO with grippers (7 DoF per arms) flat terrain WBC configuration."""
  cfg = pal_kangaroo_flat_wbc_env_cfg(play=play)

  cfg.scene.entities = {"robot": get_kangaroo_grippers_robot_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_GRIPPERS_ACTION_SCALE
  joint_pos_action.actuator_names = KANGAROO_GRIPPERS_ACTUATOR_NAMES

  end_effector_body_names = ("arm_right_tip_link","arm_left_tip_link")

  assert cfg.commands is not None
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, OnlineMotionCommandCfg)
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
    "arm_left_4_link",
    "arm_left_tip_link",
    "arm_right_2_link",
    "arm_right_4_link",
    "arm_right_tip_link",
  )
  motion_cmd.tracked_body_names = end_effector_body_names
  motion_cmd.robot_type = "Kangaroo_grippers"

  # cfg.observations["actor"].terms["ref_body_pos"] = ObservationTermCfg(
  #   func=mdp.ref_body_pos_b, 
  #   params={
  #     "command_name": "motion", 
  #     "body_names": end_effector_body_names,
  #   },
  #   noise=Unoise(n_min=-0.02, n_max=0.02),
  # )
  # cfg.observations["actor"].terms["ref_body_ori"] = ObservationTermCfg(
  #   func=mdp.ref_body_ori_b, 
  #   params={
  #     "command_name": "motion", 
  #     "body_names": end_effector_body_names,
  #   },
  #   noise=Unoise(n_min=-0.01, n_max=0.01),
  # )

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "leg_left_5_link",
    "leg_right_5_link",
    "arm_left_7_link",
    "arm_right_7_link",
  )

  cfg.metrics["arm_5_joint_torque_mag"] = MetricsTermCfg(
    func=joint_torques_magnitude,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"arm_left_5_joint",))},
  )

  cfg.metrics["arm_6_joint_torque_mag"] = MetricsTermCfg(
    func=joint_torques_magnitude,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"arm_left_6_joint",))},
  )

  cfg.metrics["arm_7_joint_torque_mag"] = MetricsTermCfg(
    func=joint_torques_magnitude,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"arm_left_7_joint",))},
  )

  cfg.metrics.update(
    raw_action_metrics([f"arm_left_{i}_joint" for i in range(5, 8)])
  )

  return cfg
