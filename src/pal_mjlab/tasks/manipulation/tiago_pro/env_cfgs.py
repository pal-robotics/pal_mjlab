from __future__ import annotations

from typing import Literal

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import terminations as mdp_term
from mjlab.managers import (
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

  cfg.sim.mujoco.timestep = 0.002
  cfg.sim.mujoco.iterations = 20
  cfg.sim.mujoco.jacobian = "sparse"
  cfg.sim.nconmax = 500
  cfg.sim.njmax = 500
  cfg.decimation = 10
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

  cfg.actions.pop("joint_pos", None)
  cfg.actions["ee_ik"] = robot.arm_action_cfg()
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
      name="gripper_table_contact",
      primary=ContactMatch(
        mode="body", pattern=robot.gripper_collision_link_pattern, entity="robot"
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
  )

  cfg.commands["lift_height"] = manipulation_mdp_pal.LiftingCommandCfg(
    entity_name="box",
    object_half_height=manipulation_mdp_pal.BOX_HALF_Z,
    table_height=manipulation_mdp_pal.TABLE_HEIGHT,
    contact_sensor_name="box_table_contact",
    resampling_time_range=(EPISODE_LENGTH, EPISODE_LENGTH),
    debug_vis=True,
    target_position_range=manipulation_mdp_pal.LiftingCommandCfg.TargetPositionRangeCfg(
      x=(0.4, 0.6),
      y=(-0.25, 0.25),
      z=(0.65, 0.85),
    ),
    object_pose_range=manipulation_mdp_pal.LiftingCommandCfg.ObjectPoseRangeCfg(
      x=(0.4, 0.6),
      y=(-0.1, 0.1),
      yaw=(-3.1415926535, 3.1415926535),
    ),
  )

  for group in ["actor", "critic"]:
    terms = cfg.observations[group].terms
    terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=(robot.arm_joint_pattern,)
    )
    terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=(robot.arm_joint_pattern,)
    )
    terms.pop("ee_to_cube", None)
    terms.pop("cube_to_goal", None)
    terms["object_position"] = ObservationTermCfg(
      func=manipulation_mdp_pal.object_position_in_robot_root_frame,
      params={"command_name": "lift_height"},
    )
    terms["object_orientation"] = ObservationTermCfg(
      func=manipulation_mdp_pal.object_orientation_in_robot_root_frame,
      params={"command_name": "lift_height"},
    )
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

  cfg.observations["critic"].terms["finger_contact"] = ObservationTermCfg(
    func=manipulation_mdp_pal.site_contact_found,
    params={
      "sensor_name": "box_fingertip_contact",
      "site_names": [robot.fingertip_site_pattern],
    },
  )

  for name in ("object_position", "object_orientation", "target_object_position"):
    cfg.observations["actor"].terms[name].noise = Unoise(n_min=-0.01, n_max=0.01)

  #### REWARDS
  cfg.rewards.clear()
  _grasp_cfg = SceneEntityCfg("robot", site_names=(robot.ee_site,))
  cfg.rewards["reaching_object"] = RewardTermCfg(
    func=manipulation_mdp_pal.nan_safe(manipulation_mdp_pal.object_ee_distance),
    weight=5.0,
    params={
      "std": 0.3,
      "ee_vel_std": 0.3,
      "min_reaching_reward": 0.0,
      "command_name": "lift_height",
      "asset_cfg": _grasp_cfg,
    },
  )
  # cfg.rewards["lifting_object"] = RewardTermCfg(
  #   func=manipulation_mdp_pal.nan_safe(manipulation_mdp_pal.object_is_lifted),
  #   weight=1.0,
  #   params={
  #     "command_name": "lift_height",
  #     "sensor_name": "box_fingertip_contact",
  #     "site_names": [robot.fingertip_site_pattern],
  #   },
  # )
  cfg.rewards["object_goal_tracking"] = RewardTermCfg(
    func=manipulation_mdp_pal.nan_safe(manipulation_mdp_pal.object_goal_distance),
    weight=5.0,
    params={
      "command_name": "lift_height",
      "std": 0.3,
      "sensor_name": "box_fingertip_contact",
      "site_names": [robot.fingertip_site_pattern],
    },
  )
  # cfg.rewards["object_goal_tracking_fine_grained"] = RewardTermCfg(
  #   func=manipulation_mdp_pal.nan_safe(manipulation_mdp_pal.object_goal_distance),
  #   weight=10.0,
  #   params={
  #     "command_name": "lift_height",
  #     "std": 0.05,
  #     "sensor_name": "box_fingertip_contact",
  #     "site_names": [robot.fingertip_site_pattern],
  #   },
  # )
  cfg.rewards["arm_table_contact_penalty"] = RewardTermCfg(
    func=manipulation_mdp_pal.contact_penalty,
    weight=-0.5,
    params={"sensor_names": ["gripper_table_contact"]},
  )

  cfg.rewards["object_contact_both_fingers"] = RewardTermCfg(
    func=manipulation_mdp_pal.nan_safe(manipulation_mdp_pal.site_contact_both_fingers),
    weight=3.0,
    params={
      "sensor_name": "box_fingertip_contact",
      "site_names": [robot.fingertip_site_pattern],
    },
  )
  # cfg.rewards["action_rate_l2"] = RewardTermCfg(
  #   func=manipulation_mdp_pal.action_rate_l2,
  #   weight=-1.5,
  #   params={"action_indices": list(range(6))},
  # )
  # cfg.rewards["ee_vel_penalty"] = RewardTermCfg(
  #   func=manipulation_mdp_pal.nan_safe(manipulation_mdp_pal.ee_vel_penalty),
  #   weight=-1.0,
  #   params={
  #     "threshold": 0.06,
  #     "scale": 50.0,
  #     "max_penalty": 10.0,
  #     "asset_cfg": _grasp_cfg,
  #   },
  # )
  # cfg.rewards["ee_ground_collision_termination_penalty"] = RewardTermCfg(
  #   func=manipulation_mdp.illegal_contact,
  #   weight=-10.0,
  #   params={"sensor_name": "ee_ground_collision", "force_threshold": 1.0},
  # )

  ### CURRICULUMS
  cfg.curriculum.clear()

  # cfg.curriculum["reaching_object_std"] = CurriculumTermCfg(
  #   func=mdp.reward_curriculum,
  #   params={
  #     "reward_name": "reaching_object",
  #     "stages": [
  #       {"step": 0, "params": {"std": 0.15}},
  #       {"step": 1500 * 24, "params": {"std": 0.10}},
  #     ],
  #   },
  # )
  # cfg.curriculum["lifting_object_weight"] = CurriculumTermCfg(
  #   func=mdp.reward_curriculum,
  #   params={
  #     "reward_name": "lifting_object",
  #     "stages": [
  #       {"step": 1000 * 24, "weight": 5.0},
  #     ],
  #   },
  # )
  # cfg.curriculum["object_goal_curriculum"] = CurriculumTermCfg(
  #   func=mdp.reward_curriculum,
  #   params={
  #     "reward_name": "object_goal_tracking",
  #     "stages": [
  #       {"step": 4000 * 24, "weight": 25.0},
  #     ],
  #   },
  # )

  ##### DOMAIN RANDOMIZATION ON THE GRIPPER
  for friction_type in ("slide", "spin", "roll"):
    cfg.events.pop(f"fingertip_friction_{friction_type}", None)

  cfg.events["reset_robot_joints"] = EventTermCfg(
    func=mdp.reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (0.0, 0.0),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )

  cfg.events["reset_table"] = EventTermCfg(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {"z": (-0.05, 0.05)},
      "velocity_range": {},
      "asset_cfg": SceneEntityCfg("table"),
    },
  )

  #### TERMINATIONS
  cfg.terminations["nan_term"] = TerminationTermCfg(func=mdp_term.nan_detection)

  # cfg.terminations["object_dropped"] = TerminationTermCfg(
  #   func=mdp_term.root_height_below_minimum,
  #   params={
  #     "minimum_height": manipulation_mdp_pal.TABLE_HEIGHT - 0.1,
  #     "asset_cfg": SceneEntityCfg("box"),
  #   },
  # )

  cfg.terminations["ee_ground_collision"] = TerminationTermCfg(
    func=manipulation_mdp.illegal_contact,
    params={"sensor_name": "ee_ground_collision", "force_threshold": 1.0},
  )

  # cfg.terminations["arm_contact_while_lifting"] = TerminationTermCfg(
  #   func=manipulation_mdp_pal.arm_contact_while_lifting_term,
  #   params={
  #     "sensor_names": ["ee_ground_collision", "gripper_table_contact"],
  #     "command_name": "lift_height",
  #     "sensor_name": "box_fingertip_contact",
  #     "site_names": [robot.fingertip_site_pattern],
  #     "asset_cfg": _grasp_cfg,
  #   },
  # )

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

  return cfg


def lift_vision_env_cfg(
  cam_type: Literal["rgb", "depth", "rgbd"],
  cam_source: Literal["head", "wrist"] = "head",
  play: bool = False,
  robot_cfg=TiagoProRobot,
) -> ManagerBasedRlEnvCfg:
  cfg = lift_env_cfg(play=play, robot_cfg=robot_cfg, cam_source=cam_source)
  robot = robot_cfg()

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    CameraSensorCfg(
      name=f"{cam_source}_realsense_camera",
      height=128,
      width=128,
      data_types=("rgb", "depth"),
      camera_name=f"robot/{robot.head_camera_name if cam_source == 'head' else robot.wrist_camera_name}",
    ),
  )

  cfg.viewer.camera = f"robot/{robot.head_camera_name if cam_source == 'head' else robot.wrist_camera_name}"
  obs_sensor_name = f"{cam_source}_realsense_camera"

  terms = {}
  if cam_type == "rgbd":
    terms[f"{cam_source}_camera_rgbd"] = ObservationTermCfg(
      func=manipulation_mdp_pal.camera_rgbd, params={"sensor_name": obs_sensor_name}
    )
  elif cam_type == "rgb":
    terms[f"{cam_source}_camera_rgb"] = ObservationTermCfg(
      func=manipulation_mdp.camera_rgb, params={"sensor_name": obs_sensor_name}
    )
  elif cam_type == "depth":
    terms[f"{cam_source}_camera_depth"] = ObservationTermCfg(
      func=manipulation_mdp.camera_depth,
      params={"sensor_name": obs_sensor_name, "cutoff_distance": 1.5},
    )

  cfg.observations["camera"] = ObservationGroupCfg(
    terms=terms,
    enable_corruption=False,
    concatenate_terms=True,
    nan_policy="sanitize",
  )

  for name in ("object_position", "object_orientation", "target_object_position"):
    cfg.observations["actor"].terms.pop(name)

  cfg.observations["actor"].terms["goal_position"] = ObservationTermCfg(
    func=manipulation_mdp_pal.target_position_in_robot_base_frame,
    params={"command_name": "lift_height"},
  )

  return cfg
