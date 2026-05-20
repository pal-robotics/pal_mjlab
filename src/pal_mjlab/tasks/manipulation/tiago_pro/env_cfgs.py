from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Literal

import mujoco
import torch
from mjlab.entity import Entity, EntityCfg
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers import (
  CurriculumTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg, ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.velocity import mdp
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pal_mjlab.tasks.manipulation.mdp.contact_sensor import (
  site_contact_both_fingers,
  site_contact_found,
)

from pal_mjlab.robots.pal_tiago_pro.tiago_pro import TiagoProRobot

_TABLE_HEIGHT = 0.5
_TABLE_HALF_X = 0.35
_TABLE_HALF_Y = 0.35
_BOX_HALF_SIZE = 0.025

EPISODE_LENGTH = 10


def nan_safe(fn):
  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    return torch.nan_to_num(fn(*args, **kwargs), nan=0.0, posinf=0.0, neginf=0.0)

  return wrapper


def get_table_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(
    name="table",
    pos=(_TABLE_HALF_X + 0.15, 0.0, _TABLE_HEIGHT / 2),
  )
  body.add_geom(
    name="table_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(_TABLE_HALF_X, _TABLE_HALF_Y, _TABLE_HEIGHT / 2),
    rgba=(0.1, 0.1, 0.1, 1.0),
    solref=(0.001, 1),
    solimp=(0.95, 0.99, 0.001, 0.5, 2), 
  )
  return spec


def get_box_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="box_object")
  body.add_freejoint()
  body.add_geom(
    name="box_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(_BOX_HALF_SIZE, _BOX_HALF_SIZE, 1.5*_BOX_HALF_SIZE),
    rgba=(0.8, 0.2, 0.2, 1.0),
    mass=0.1,
    solref=(0.001, 1),
    solimp=(0.95, 0.99, 0.001, 0.5, 2), 
  )
  return spec


class LiftingCommand(CommandTerm):
  cfg: LiftingCommandCfg

  def __init__(self, cfg: LiftingCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.object: Entity = env.scene[cfg.entity_name]
    self.contact_sensor: ContactSensor = env.scene[cfg.contact_sensor_name]
    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)
    self.metrics["object_height"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  @property
  def object_pos_w(self) -> torch.Tensor:
    return self.object.data.root_link_pos_w

  @property
  def object_quat_w(self) -> torch.Tensor:
    return self.object.data.root_link_quat_w

  @property
  def object_bottom_z(self) -> torch.Tensor:
    return self.object_pos_w[:, 2] - self.cfg.object_half_height

  @property
  def object_on_table(self) -> torch.Tensor:
    return self.contact_sensor.data.found.any(dim=-1) > 0

  def _update_metrics(self) -> None:
    position_error = torch.norm(self.target_pos - self.object_pos_w, dim=-1)
    at_goal = (position_error < self.cfg.success_threshold).float()
    self.episode_success = torch.maximum(self.episode_success, at_goal)
    self.metrics["object_height"] = self.object_bottom_z
    self.metrics["position_error"] = position_error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success

  def compute_success(self) -> torch.Tensor:
    return self.metrics["position_error"] < self.cfg.success_threshold

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.episode_success[env_ids] = 0.0

    r = self.cfg.target_position_range
    lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
    upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
    target_pos = sample_uniform(lower, upper, (n, 3), device=self.device)
    self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]

    r = self.cfg.object_pose_range
    lower = torch.tensor([r.x[0], r.y[0], 0.0], device=self.device)
    upper = torch.tensor([r.x[1], r.y[1], 0.0], device=self.device)
    pos = sample_uniform(lower, upper, (n, 3), device=self.device)
    pos[:, 2] = self.cfg.table_height + self.cfg.object_half_height
    pos = pos + self._env.scene.env_origins[env_ids]

    yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
    zeros = torch.zeros(n, device=self.device)
    quat = quat_from_euler_xyz(zeros, zeros, yaw)
    pose = torch.cat([pos, quat], dim=-1)

    self.object.write_root_link_pose_to_sim(pose, env_ids=env_ids)
    self.object.write_root_link_velocity_to_sim(
      torch.zeros(n, 6, device=self.device), env_ids=env_ids
    )

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer) -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    for batch in env_indices:
      visualizer.add_sphere(
        center=self.target_pos[batch].cpu().numpy(),
        radius=0.03,
        color=self.cfg.viz.target_color,
        label=f"target_position_{batch}",
      )


@dataclass(kw_only=True)
class LiftingCommandCfg(CommandTermCfg):
  entity_name: str
  object_half_height: float
  table_height: float
  contact_sensor_name: str
  success_threshold: float = 0.05

  @dataclass
  class TargetPositionRangeCfg:
    x: tuple[float, float] = (0.3, 0.5)
    y: tuple[float, float] = (-0.2, 0.2)
    z: tuple[float, float] = (0.2, 0.4)

  target_position_range: TargetPositionRangeCfg = field(
    default_factory=TargetPositionRangeCfg
  )

  @dataclass
  class ObjectPoseRangeCfg:
    x: tuple[float, float] = (0.4, 0.6)
    y: tuple[float, float] = (-0.1, 0.1)
    yaw: tuple[float, float] = (math.pi, math.pi)

  object_pose_range: ObjectPoseRangeCfg = field(default_factory=ObjectPoseRangeCfg)

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> LiftingCommand:
    return LiftingCommand(self, env)


def object_position_in_robot_root_frame(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command: LiftingCommand = env.command_manager.get_term(command_name)
  return quat_apply(
    quat_inv(robot.data.root_link_quat_w),
    command.object_pos_w - robot.data.root_link_pos_w,
  )


def object_orientation_in_robot_root_frame(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command: LiftingCommand = env.command_manager.get_term(command_name)
  return quat_mul(quat_inv(robot.data.root_link_quat_w), command.object_quat_w)


def object_ee_distance(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command: LiftingCommand = env.command_manager.get_term(command_name)
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  distance = torch.norm(ee_pos_w - command.object_pos_w, dim=-1)
  
  # if (distance < 0.03).any():
  #     print("\033[96mFLAG: EE is within 3cm of the object center!\033[0m")
      
  return torch.clamp(1.0 - torch.tanh(distance / std), max=0.8)


def object_is_lifted(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  site_names: list[str],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  min_weight: float = 0.5,
  max_weight: float = 5.0,
  lift_threshold: float = 0.05,
) -> torch.Tensor:
  command: LiftingCommand = env.command_manager.get_term(command_name)
  
  # Check if both fingers are close (this uses our new distance logic)
  fingers_close = site_contact_both_fingers(env, sensor_name, site_names, asset_cfg=asset_cfg).bool()
  
  # Calculate elevation of the bottom of the object above the table
  elevation = command.object_bottom_z - command.cfg.table_height
  elevation = torch.clamp(elevation, min=0.0, max=lift_threshold)
  
  # Exponential scaling: weight = min_weight * (ratio ** (elevation / lift_threshold))
  ratio = max_weight / min_weight
  scale = min_weight * (ratio ** (elevation / lift_threshold))
  
  is_lifted = (~command.object_on_table & fingers_close).float()
  return is_lifted * scale


def object_goal_distance(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  sensor_name: str,
  site_names: list[str],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  command: LiftingCommand = env.command_manager.get_term(command_name)
  contact_both = site_contact_both_fingers(env, sensor_name, site_names, asset_cfg=asset_cfg).bool()
  
  distance = torch.norm(command.target_pos - command.object_pos_w, dim=-1)
  return (~command.object_on_table & contact_both) * (1.0 - torch.tanh(distance / std))


def contact_penalty(env: ManagerBasedRlEnv, sensor_names: list[str]) -> torch.Tensor:
  contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for name in sensor_names:
    sensor: ContactSensor = env.scene[name]
    contact |= sensor.data.found.any(dim=-1)
  return contact.float()


def arm_contact_while_lifting_term(
  env: ManagerBasedRlEnv,
  sensor_names: list[str],
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for name in sensor_names:
    sensor: ContactSensor = env.scene[name]
    contact |= sensor.data.found.any(dim=-1)
  lifted = object_is_lifted(env, command_name, asset_cfg).bool()
  return contact & lifted



def object_contact_both_fingers(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  return sensor.data.found.all(dim=-1).float()


def contact_sensor_found(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  return sensor.data.found.float()


def action_rate_l2(
  env: ManagerBasedRlEnv, action_indices: list[int] | None = None
) -> torch.Tensor:
  """Penalize the rate of change of the actions using L2 squared kernel."""
  if action_indices is None:
    action_diff = env.action_manager.action - env.action_manager.prev_action
  else:
    action_diff = (
      env.action_manager.action[:, action_indices]
      - env.action_manager.prev_action[:, action_indices]
    )
  return torch.sum(torch.square(action_diff), dim=1)


def camera_rgbd(env: ManagerBasedRlEnv, sensor_name: str, cutoff_distance: float = 1.0) -> torch.Tensor:
  rgb = manipulation_mdp.camera_rgb(env, sensor_name)
  depth = manipulation_mdp.camera_depth(env, sensor_name, cutoff_distance=cutoff_distance)
  return torch.cat([rgb, depth], dim=1)


def target_position_in_robot_base_frame(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:  
  robot: Entity = env.scene[asset_cfg.name]
  command: LiftingCommand = env.command_manager.get_term(command_name)
  return quat_apply(
    quat_inv(robot.data.root_link_quat_w),
    command.target_pos - robot.data.root_link_pos_w,
  )


def ee_position_in_robot_base_frame(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  return quat_apply(
    quat_inv(robot.data.root_link_quat_w),
    ee_pos_w - robot.data.root_link_pos_w,
  )


def ee_vel_penalty(
  env: ManagerBasedRlEnv,
  threshold: float = 0.06,
  scale: float = 50.0,
  max_penalty: float = 10.0,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  ee_lin_vel_w = robot.data.site_lin_vel_w[:, asset_cfg.site_ids].squeeze(1)
  ee_vel_norm = torch.linalg.norm(ee_lin_vel_w, dim=-1)
  
  # Differentiable exponential penalty for exceeding threshold
  excess_vel = torch.clamp(ee_vel_norm - threshold, min=0.0)
  penalty = torch.exp(scale * excess_vel) - 1.0
  return torch.clamp(penalty, max=max_penalty)


def lift_env_cfg(
  play: bool = False,
  robot_cfg=TiagoProRobot,
  cam_source: Literal["head", "wrist"] = "head",
) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg()
  robot = robot_cfg()

  cfg.sim.mujoco.timestep = 0.002
  cfg.sim.mujoco.iterations = 20
  # cfg.sim.mujoco.ls_iterations = 20
  # cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.mujoco.jacobian = "sparse"
  cfg.sim.nconmax = 500
  cfg.sim.njmax = 500
  cfg.decimation = 10
  cfg.episode_length_s = EPISODE_LENGTH
  cfg.viewer.lookat = (0.4, 0.0, 0.55)
  cfg.viewer.distance = 1.7
  cfg.viewer.azimuth = 190.0
  cfg.viewer.elevation = 15.0
  #cfg.viewer.camera = f"{cam_source}_realsense_camera"
  cfg.sim.nan_guard.enabled = True
  cfg.sim.nan_guard.output_dir = "/tmp/mjlab/nan_dumps"
  cfg.observations["actor"].nan_policy = "sanitize"
  cfg.observations["critic"].nan_policy = "sanitize"

  cfg.scene.entities = {
    "robot": robot.entity_cfg,
    "table": EntityCfg(spec_fn=get_table_spec),
    "box": EntityCfg(
      spec_fn=get_box_spec,
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
      primary=ContactMatch(mode="body", pattern=robot.gripper_collision_link_pattern, entity="robot"),
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

  cfg.commands["lift_height"] = LiftingCommandCfg(
    entity_name="box",
    object_half_height=1.5 * _BOX_HALF_SIZE,
    table_height=_TABLE_HEIGHT,
    contact_sensor_name="box_table_contact",
    resampling_time_range=(EPISODE_LENGTH, EPISODE_LENGTH),
    debug_vis=True,
    target_position_range=LiftingCommandCfg.TargetPositionRangeCfg(
      x=(0.4, 0.6),
      y=(-0.25, 0.25),
      z=(0.65, 0.85),
    ),
    object_pose_range=LiftingCommandCfg.ObjectPoseRangeCfg(
      x=(0.4, 0.6),
      y=(-0.1, 0.1),
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
      func=object_position_in_robot_root_frame,
      params={"command_name": "lift_height"},
    )
    terms["object_orientation"] = ObservationTermCfg(
      func=object_orientation_in_robot_root_frame,
      params={"command_name": "lift_height"},
    )
    terms["target_object_position"] = ObservationTermCfg(
      func=target_position_in_robot_base_frame,
      params={"command_name": "lift_height"},
    )
    terms["gripper_pos"] = ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(robot.gripper_joint_pattern,))
      },
    )
    terms["ee_position"] = ObservationTermCfg(
      func=ee_position_in_robot_base_frame,
      params={"asset_cfg": SceneEntityCfg("robot", site_names=(robot.ee_site,))},
    )

  cfg.observations["critic"].terms["finger_contact"] = ObservationTermCfg(
    func=site_contact_found,
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
    func=nan_safe(object_ee_distance),
    weight=5.0,
    params={"std": 0.15, "command_name": "lift_height", "asset_cfg": _grasp_cfg},
  )
  # cfg.rewards["lifting_object"] = RewardTermCfg(
  #   func=nan_safe(object_is_lifted),
  #   weight=1.0,
  #   params={
  #     "command_name": "lift_height", 
  #     "sensor_name": "box_fingertip_contact",
  #     "site_names": [robot.fingertip_site_pattern],
  #   },
  # )
  cfg.rewards["object_goal_tracking"] = RewardTermCfg(
    func=nan_safe(object_goal_distance),
    weight=5.0,
    params={
      "command_name": "lift_height", 
      "std": 0.3, 
      "sensor_name": "box_fingertip_contact",
      "site_names": [robot.fingertip_site_pattern],
    },
  )
  # cfg.rewards["object_goal_tracking_fine_grained"] = RewardTermCfg(
  #   func=nan_safe(object_goal_distance),
  #   weight=10.0,
  #   params={
  #     "command_name": "lift_height", 
  #     "std": 0.05, 
  #     "sensor_name": "box_fingertip_contact",
  #     "site_names": [robot.fingertip_site_pattern],
  #   },
  # )
  cfg.rewards["arm_table_contact_penalty"] = RewardTermCfg(
    func=contact_penalty,
    weight=-0.5,
    params={"sensor_names": ["gripper_table_contact"]},
  )


  cfg.rewards["object_contact_both_fingers"] = RewardTermCfg(
    func=nan_safe(site_contact_both_fingers),
    weight=3.0, 
    params={
      "sensor_name": "box_fingertip_contact",
      "site_names": [robot.fingertip_site_pattern],
    },
  )
  cfg.rewards["action_rate_l2"] = RewardTermCfg(
    func=action_rate_l2,
    weight=-1.5,
    params={"action_indices": list(range(6))},
  )
  # cfg.rewards["ee_vel_penalty"] = RewardTermCfg(
  #   func=nan_safe(ee_vel_penalty),
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
  #       {"step": 1500 * 24, "params": {"std": 0.10}}, # Dopo 1000 iterazioni (1000 * num_steps_per_env)
  #       # {"step": 3000 * 24, "params": {"std": 0.075}}, # Dopo 3000 iterazioni
  #     ],
  #   },
  # )
  # cfg.curriculum["lifting_object_weight"] = CurriculumTermCfg(
  #   func=mdp.reward_curriculum,
  #   params={
  #     "reward_name": "lifting_object",
  #     "stages": [
  #       {"step": 1000 * 24, "weight": 5.0}, # Enable after 800 iterations
  #     ],
  #   },
  # )

  # cfg.curriculum["object_goal_curriculum"] = CurriculumTermCfg(
  #   func=mdp.reward_curriculum,
  #   params={
  #     "reward_name": "object_goal_tracking",
  #     "stages": [
  #       {"step": 4000 * 24, "weight": 25.0}, # Enable after 800 iterations
  #     ],
  #   },
  # )  


##### DOMAIN RANDOMIZATION ON THE GRIPPER
  # Explicitly remove default fingertip friction randomizations to ensure they are inactive
  for friction_type in ("slide", "spin", "roll"):
    cfg.events.pop(f"fingertip_friction_{friction_type}", None)

  # for friction_type in ("slide", "spin", "roll"):
  #   cfg.events[f"fingertip_friction_{friction_type}"].params[
  #     "asset_cfg"
  #   ].geom_names = robot.fingertip_geom_pattern

  # cfg.events["reset_robot_base"] = EventTermCfg(
  #   func=mdp.reset_root_state_uniform,
  #   mode="reset",
  #   params={
  #     "pose_range": {},
  #     "velocity_range": {},
  #     "asset_cfg": SceneEntityCfg("robot"),
  #   },
  # )

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
      "pose_range": {},
      "velocity_range": {},
      "asset_cfg": SceneEntityCfg("table"),
    },
  )

  from mjlab.envs.mdp import terminations as mdp_term


#### TERMINATIONS
  # cfg.terminations.pop("ee_ground_collision", None)
  cfg.terminations["nan_term"] = TerminationTermCfg(func=mdp_term.nan_detection)

  # cfg.terminations["object_dropped"] = TerminationTermCfg(
  #   func=mdp_term.root_height_below_minimum,
  #   params={
  #     "minimum_height": _TABLE_HEIGHT - 0.1,
  #     "asset_cfg": SceneEntityCfg("box"),
  #   },
  # )

  cfg.terminations["ee_ground_collision"] = TerminationTermCfg(
    func=manipulation_mdp.illegal_contact,
    params={"sensor_name": "ee_ground_collision", "force_threshold": 1.0},
  )

  # cfg.terminations["arm_contact_while_lifting"] = TerminationTermCfg(
  #   func=arm_contact_while_lifting_term,
  #   params={
  #     "sensor_names": ["ee_ground_collision", "gripper_table_contact"],
  #     "command_name": "lift_height",
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
    # cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)

  return cfg


def lift_vision_env_cfg(
  cam_type: Literal["rgb", "depth", "rgbd"],
  cam_source: Literal["head", "wrist"] = "head",
  play: bool = False,
  robot_cfg=TiagoProRobot,
) -> ManagerBasedRlEnvCfg:
  cfg = lift_env_cfg(play=play, robot_cfg=robot_cfg, cam_source=cam_source)
  robot = robot_cfg()

  # Add camera sensor only for vision task
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    CameraSensorCfg(
      name=f"{cam_source}_realsense_camera",
      height=128,
      width=128,
      data_types=("rgb", "depth"),
      camera_name=f"robot/{robot.head_camera_name if cam_source == 'head' else robot.wrist_camera_name}",
    ),
  )

  # Start the viewer from the robot's camera
  cfg.viewer.camera = f"robot/{robot.head_camera_name if cam_source == 'head' else robot.wrist_camera_name}"

  # Choose the sensor name based on the source
  obs_sensor_name = f"{cam_source}_realsense_camera"

  terms = {}
  if cam_type == "rgbd":
    terms[f"{cam_source}_camera_rgbd"] = ObservationTermCfg(
      func=camera_rgbd, params={"sensor_name": obs_sensor_name}
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
    func=target_position_in_robot_base_frame,
    params={"command_name": "lift_height"},
  )

  return cfg