from __future__ import annotations

from typing import Literal

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp import rewards as mjlab_rewards
from mjlab.envs.mdp import terminations as mdp_term
from mjlab.envs.mdp.dr import geom as dr_geom
from mjlab.managers import (
  CurriculumTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg, ContactMatch, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pal_mjlab.robots.pal_tiago_pro.tiago_pro import TiagoProRobot
from pal_mjlab.tasks.manipulation import mdp as manipulation_mdp_pal

EPISODE_LENGTH = 10


def lift_env_cfg(
  play: bool = False,
  robot_cfg=TiagoProRobot,
  cam_source: Literal["head", "wrist"] = "head",
) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg()
  robot = robot_cfg()

  cfg.sim.mujoco.timestep = 0.005 #0.005
  cfg.sim.mujoco.iterations = 20
  cfg.sim.mujoco.jacobian = "sparse"
  cfg.sim.nconmax = 50
  cfg.sim.njmax = 80
  cfg.decimation = 4   #4
  cfg.episode_length_s = EPISODE_LENGTH
  cfg.viewer.lookat = (0.4, 0.0, 0.55)
  cfg.viewer.distance = 1.7
  cfg.viewer.azimuth = 190.0
  cfg.viewer.elevation = 15.0
  cfg.sim.nan_guard.enabled = True
  cfg.sim.nan_guard.output_dir = "/tmp/mjlab/nan_dumps"
  cfg.observations["actor"].nan_policy = "sanitize"
  cfg.observations["critic"].nan_policy = "sanitize"

  cfg.scene.entities = {
    "robot": robot.entity_cfg,
    "table": EntityCfg(spec_fn=manipulation_mdp_pal.get_table_spec),
    "box": EntityCfg(
      spec_fn=manipulation_mdp_pal.get_box_spec,
      init_state=EntityCfg.InitialStateCfg(pos=(100.0, 0.0, 0.5)),
    ),
  }

  from mjlab.envs.mdp.actions import RelativeJointPositionActionCfg

  from pal_mjlab.robots import TIAGO_PRO_ACTION_SCALE

  cfg.actions.pop("ee_ik", None)
  cfg.actions["joint_pos"] = RelativeJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(robot.arm_joint_pattern,),
    scale={k: v for k, v in TIAGO_PRO_ACTION_SCALE.items() if "gripper" not in k},
  )
  cfg.actions["gripper"] = robot.gripper_action_cfg()

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    ContactSensorCfg(
      name="box_table_contact",
      primary=ContactMatch(mode="subtree", pattern="box_object", entity="box"),
      secondary=ContactMatch(mode="subtree", pattern="table", entity="table"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
    ContactSensorCfg(
      name="robot_table_contact",
      primary=ContactMatch(
        mode="body", pattern=robot.collision_link_pattern, entity="robot"
      ),
      secondary=ContactMatch(mode="subtree", pattern="table", entity="table"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
    ContactSensorCfg(
      name="box_fingertip_contact",
      primary=ContactMatch(
        mode="geom", pattern=robot.fingertip_geom_pattern, entity="robot"
      ),
      secondary=ContactMatch(mode="subtree", pattern="box_object", entity="box"),
      fields=("found", "pos", "dist"),
      reduce="none",
      num_slots=1,
    ),
    ContactSensorCfg(
      name="self_collision",
      primary=ContactMatch(
        mode="body", pattern=robot.collision_link_pattern, entity="robot"
      ),
      secondary=ContactMatch(mode="body", pattern="head_.*", entity="robot"),
      fields=("found",),
      reduce="none",
      num_slots=1,
    ),
  )

  cfg.commands["lift_height"] = manipulation_mdp_pal.LiftingCommandCfg(
    entity_name="box",
    object_half_height=manipulation_mdp_pal.BOX_HALF_Z,
    table_height=manipulation_mdp_pal.TABLE_HEIGHT,
    contact_sensor_name="box_table_contact",
    resampling_time_range=(EPISODE_LENGTH, EPISODE_LENGTH),
    debug_vis=True,
    success_threshold=0.05,
    target_position_range=manipulation_mdp_pal.LiftingCommandCfg.TargetPositionRangeCfg(
      x=(0.67, 0.77),
      y=(-0.766, -0.666),
      z=(0.55, 0.65),
    ),
    object_pose_range=manipulation_mdp_pal.LiftingCommandCfg.ObjectPoseRangeCfg(
      x=(-0.275, 0.275),
      y=(-0.275, 0.275),
      yaw=(-0.785, 0.785),   #yaw=(-0.785, 0.785),
    ),
  )

  # ====================================================================
  # OBSERVATIONS
  # ====================================================================
  # Ensure actor and critic observation configs are fully independent.
  import copy
  cfg.observations["actor"] = copy.deepcopy(cfg.observations["actor"])
  cfg.observations["critic"] = copy.deepcopy(cfg.observations["critic"])

  # 1. Base Observations Configuration
  for group in ["actor", "critic"]:
    terms = cfg.observations[group].terms
    
    # Configure robot joint assets
    terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=(robot.arm_joint_pattern,)
    )
    terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=(robot.arm_joint_pattern,)
    )
    
    # Remove unused base task terms
    terms.pop("ee_to_cube", None)
    terms.pop("cube_to_goal", None)
    
    # Add object states (default to ground-truth versions)
    terms["object_position"] = ObservationTermCfg(
      func=manipulation_mdp_pal.object_position_in_robot_root_frame,
      params={"command_name": "lift_height"},
    )
    # terms["object_width"] = ObservationTermCfg(
    #   func=manipulation_mdp_pal.object_width,
    #   params={"command_name": "lift_height"},
    # )
    terms["object_yaw"] = ObservationTermCfg(
      func=manipulation_mdp_pal.object_yaw_in_robot_root_frame,
      params={"command_name": "lift_height"},
    )

    # Add target/robot states
    terms["target_object_position"] = ObservationTermCfg(
      func=manipulation_mdp_pal.target_position_in_robot_base_frame,
      params={"command_name": "lift_height"},
    )
    terms["gripper_pos"] = ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(robot.gripper_joint_pattern,))
      },
    )
    terms["ee_position"] = ObservationTermCfg(
      func=manipulation_mdp_pal.ee_position_in_robot_base_frame,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=(robot.ee_site,))},
    )

  # 2. Noise & Dropout Configuration
  # Ensure all critic observations are completely clean (no noise).
  for name in cfg.observations["critic"].terms:
    cfg.observations["critic"].terms[name].noise = None

  if not play:
    # During training: apply observation noise to the actor.
    actor_terms = cfg.observations["actor"].terms
    actor_noise_configs = {
      "object_position": Unoise(n_min=-0.01, n_max=0.01),
      "object_yaw": Unoise(n_min=-0.05, n_max=0.05),
      "joint_pos": Unoise(n_min=-0.02, n_max=0.02),
      "joint_vel": Unoise(n_min=-0.05, n_max=0.05),
      # "target_object_position": Unoise(n_min=-0.01, n_max=0.01),
      "ee_position": Unoise(n_min=-0.01, n_max=0.01),
      "gripper_pos": Unoise(n_min=-0.003, n_max=0.003),
    }
    for name, noise_cfg in actor_noise_configs.items():
      if name in actor_terms:
        actor_terms[name].noise = noise_cfg

  else:
    # During evaluation/play: clear all noise for the actor observations.
    for name in cfg.observations["actor"].terms:
      cfg.observations["actor"].terms[name].noise = None

  #### REWARDS
  cfg.rewards.clear()
  _grasp_cfg = SceneEntityCfg("robot", site_names=(robot.ee_site,))
  cfg.rewards["reaching_object"] = RewardTermCfg(
    func=manipulation_mdp_pal.nan_safe(manipulation_mdp_pal.object_ee_distance_adaptive),
    weight=3.0,
    params={
      "std": 0.15,
      "min_reaching_reward": 0.0,
      "command_name": "lift_height",
      "asset_cfg": _grasp_cfg,
      "deactivate_on_contact": False,
      "sensor_name": "box_fingertip_contact",
      "site_names": [robot.fingertip_site_pattern],
    },
  )
  cfg.rewards["gripper_open_during_approach"] = RewardTermCfg(
    func=manipulation_mdp_pal.nan_safe(
      manipulation_mdp_pal.gripper_open_during_approach_reward
    ),
    weight=1.0,
    params={
      "command_name": "lift_height",
      "asset_cfg": _grasp_cfg,
      "std": 0.02,
      "max_open": 0.07,
    },
  )
  # cfg.rewards["lifting_object"] = RewardTermCfg(
  #   func=manipulation_mdp_pal.nan_safe(manipulation_mdp_pal.object_is_lifted_adaptive),
  #   weight=1.0,
  #   params={
  #     "command_name": "lift_height",
  #     "sensor_name": "box_fingertip_contact",
  #     "site_names": [robot.fingertip_site_pattern],
  #   },
  # )
  cfg.rewards["object_goal_tracking"] = RewardTermCfg(
    func=manipulation_mdp_pal.nan_safe(manipulation_mdp_pal.object_goal_distance_adaptive),
    weight=5.0,
    params={
      "command_name": "lift_height",
      "std": 0.3,
      "sensor_name": "box_fingertip_contact",
      "site_names": [robot.fingertip_site_pattern],
      "coordinate_weights": (1.0, 1.0, 3.0),
    },
  )

  cfg.rewards["arm_table_contact_penalty"] = RewardTermCfg(
    func=manipulation_mdp_pal.contact_penalty,
    weight=-1.0,
    params={"sensor_names": ["robot_table_contact"]},
  )

  cfg.rewards["object_table_sliding_penalty"] = RewardTermCfg(
    func=manipulation_mdp_pal.nan_safe(manipulation_mdp_pal.object_table_sliding_penalty_adaptive),
    weight=-5.0,
    params={"command_name": "lift_height"},
  )

  # cfg.rewards["object_contact_both_fingers"] = RewardTermCfg(
  #   func=manipulation_mdp_pal.nan_safe(manipulation_mdp_pal.object_contact_both_fingers_adaptive),
  #   weight=0.5,
  #   params={
  #     "sensor_name": "box_fingertip_contact",
  #     "site_names": [robot.fingertip_site_pattern],
  #     "command_name": "lift_height",
  #   },
  # )

  cfg.rewards["fingertip_cube_alignment"] = RewardTermCfg(
    func=manipulation_mdp_pal.nan_safe(
      manipulation_mdp_pal.fingertip_cube_alignment_reward_adaptive
    ),
    weight=-5.0,  # Note: Use a negative weight (e.g. -1.5) if as_penalty=True
    params={
      "command_name": "lift_height",
      "asset_cfg": _grasp_cfg,
      "std": 0.3,
      "as_penalty": True,
      "sensor_name": "box_fingertip_contact",
      "site_names": [robot.fingertip_site_pattern],
    },
  )

  cfg.rewards["release_cube"] = RewardTermCfg(
    func=manipulation_mdp_pal.nan_safe(manipulation_mdp_pal.release_cube_reward),
    weight=10.0,
    params={
      "command_name": "lift_height",
      "max_open": 0.08,
    },
  )

  cfg.rewards["object_falling"] = RewardTermCfg(
    func=manipulation_mdp_pal.nan_safe(manipulation_mdp_pal.object_falling_reward),
    weight=10.0,
    params={
      "command_name": "lift_height",
    },
  )

  cfg.rewards["success_reward"] = RewardTermCfg(
    func=manipulation_mdp_pal.nan_safe(manipulation_mdp_pal.task_success_reward),
    weight=1000.0,
    params={
      "command_name": "lift_height",
    },
  )

  cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=manipulation_mdp_pal.action_rate_l2,
    weight=-0.1,
    params={"action_indices": list(range(8))},
  )

  cfg.rewards["arm_right_1_joint_limit_penalty"] = RewardTermCfg(
    func=manipulation_mdp_pal.arm_right_1_joint_limit_penalty,
    weight=-0.5,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "threshold": -0.35,
    },
  )

  cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    func=mjlab_rewards.joint_torques_l2,
    weight=-5e-4,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(robot.arm_joint_pattern,))
    },
  )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": "self_collision"},
  )


  ### CURRICULUMS
  cfg.curriculum.clear()


  ##### DOMAIN RANDOMIZATION ON THE GRIPPER
  for friction_type in ("slide", "spin", "roll"):
    cfg.events.pop(f"fingertip_friction_{friction_type}", None)

  cfg.events["reset_robot_joints"] = EventTermCfg(
    func=mdp.reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (-0.1, 0.1),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=("^(?!torso_lift_joint|gripper_right).*$",)
      ),
    },
  )

  # cfg.events["reset_torso_joint"] = EventTermCfg(
  #   func=mdp.reset_joints_by_offset,
  #   mode="reset",
  #   params={
  #     "position_range": (0.0, 0.15),  # Randomizes torso height uniformly from 0.0m to 0.30m
  #     "velocity_range": (0.0, 0.0),
  #     "asset_cfg": SceneEntityCfg("robot", joint_names=("torso_lift_joint",)),
  #   },
  # )


  cfg.events["reset_gripper_joints"] = EventTermCfg(
    func=mdp.reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (-0.01, 0.01),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=("gripper_right.*",)),
    },
  )

  cfg.events["reset_robot_base"] = EventTermCfg(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.1, 0.1)},
      "velocity_range": {},
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  cfg.events["randomize_table_height"] = EventTermCfg(
    func=manipulation_mdp_pal.randomize_table_height,
    mode="reset",
    params={
      "table_asset_cfg": SceneEntityCfg(
        "table",
        body_names=("table",),
        geom_names=("table_geom",),
      ),
      "height_range": (-0.1, 0.1),
    },
  )

  cfg.events["reset_table"] = EventTermCfg(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {},
      "velocity_range": {},
      "asset_cfg": SceneEntityCfg("table"),
    },
  )

  _box_geom_cfg = SceneEntityCfg("box", geom_names=("box_geom",))
  cfg.events["randomize_box_size_x"] = EventTermCfg(
    func=dr_geom.geom_size,
    mode="reset",
    params={
      "ranges": {0: (0.01, 0.025)},
      "asset_cfg": _box_geom_cfg,
      "operation": "abs",
    },
  )
  cfg.events["randomize_box_size_y"] = EventTermCfg(
    func=dr_geom.geom_size,
    mode="reset",
    params={
      "ranges": {1: (0.01, 0.025)},
      "asset_cfg": _box_geom_cfg,
      "operation": "abs",
    },
  )
  cfg.events["randomize_box_size_z"] = EventTermCfg(
    func=dr_geom.geom_size,
    mode="reset",
    params={
      "ranges": {2: (0.02, 0.04)},
      "asset_cfg": _box_geom_cfg,
      "operation": "abs",
    },
  )

  _box_body_cfg = SceneEntityCfg("box", body_names=("box_object",))
  cfg.events["randomize_box_mass"] = EventTermCfg(
    func=dr.pseudo_inertia,
    mode="reset",
    params={
      "alpha_range": (0.0, 1.1513),
      "asset_cfg": _box_body_cfg,
    },
  )

  # Sim2Real: encoder calibration drift (zero-point offset per joint)
  cfg.events["encoder_bias"] = EventTermCfg(
    mode="startup",
    func=dr.encoder_bias,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(robot.arm_joint_pattern,)),
      "bias_range": (-0.01, 0.01),
    },
  )

  # Sim2Real: actuator friction variability across robot units
  cfg.events["joint_friction"] = EventTermCfg(
    mode="startup",
    func=dr.dof_frictionloss,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(robot.arm_joint_pattern,)),
      "operation": "add",
      "ranges": (-0.005, 0.005),
      "shared_random": False,
    },
  )

  #### TERMINATIONS
  cfg.terminations["nan_term"] = TerminationTermCfg(func=mdp_term.nan_detection)

  if not play:
    cfg.terminations["top_surface_penetration"] = TerminationTermCfg(
      func=manipulation_mdp_pal.top_surface_penetration_term,
      params={"command_name": "lift_height", "threshold": 0.0005},
    )
    cfg.terminations["object_released_on_floor"] = TerminationTermCfg(
      func=manipulation_mdp_pal.object_released_on_floor_term,
      params={"command_name": "lift_height"},
      time_out=True,
    )

  for s in cfg.scene.sensors:
    if isinstance(s, ContactSensorCfg) and s.name == "ee_ground_collision":
      s.primary = ContactMatch(
        mode="body", pattern=robot.arm_collision_link_pattern, entity="robot"
      )
      s.secondary = ContactMatch(mode="subtree", pattern="table", entity="table")
      break

  cfg.viewer.body_name = robot.viewer_body

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}

  cfg.observations["actor"].terms["object_both__contact_fingers"] = ObservationTermCfg(
    func=manipulation_mdp_pal.object_both__contact_fingers,
    params={
      "sensor_name": "box_fingertip_contact",
      "site_names": [robot.fingertip_site_pattern],
      "false_negative_rate": 0.0 if play else 0.05,
    },
  )
  cfg.observations["critic"].terms["object_both__contact_fingers"] = ObservationTermCfg(
    func=manipulation_mdp_pal.object_both__contact_fingers,
    params={
      "sensor_name": "box_fingertip_contact",
      "site_names": [robot.fingertip_site_pattern],
      "false_negative_rate": 0.0,
    },
  )

  return cfg
