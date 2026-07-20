from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz, sample_uniform

TABLE_HEIGHT = 0.45
TABLE_HALF_X = 0.35
TABLE_HALF_Y = 0.35

BOX_HALF_X = 0.025
BOX_HALF_Y = 0.025
BOX_HALF_Z = 0.025
BOX_HALF_SIZE = 0.025  # Keep for compatibility


def get_table_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(
    name="table",
    pos=(TABLE_HALF_X + 0.15, 0.0, TABLE_HEIGHT / 2),
  )
  body.add_geom(
    name="table_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(TABLE_HALF_X, TABLE_HALF_Y, TABLE_HEIGHT / 2),
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
    size=(BOX_HALF_X, BOX_HALF_Y, BOX_HALF_Z),
    rgba=(0.8, 0.2, 0.2, 1.0),
    mass=0.01,
    solref=(-5000, -200),
    solimp=(0.99, 0.995, 0.001, 0.5, 2),
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
    self.reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.at_goal_time = torch.zeros(self.num_envs, device=self.device)
    self.reached_time = torch.zeros(
      self.num_envs, device=self.device
    )  # seconds elapsed since reached became True
    self.grasped_distance = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )
    self.prev_object_pos_w = self.object_pos_w.clone()

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  @property
  def object_pos_w(self) -> torch.Tensor:
    return self.object.data.root_link_pos_w

  @property
  def table_surface_z(self) -> torch.Tensor:
    table: Entity = self._env.scene["table"]
    # Read the table geom's world-frame center Z directly from physics data.
    # This is correct regardless of table height randomization, mocap wrapping, or
    # env_origin offsets. geom_xpos is updated by mujoco_warp's kinematics.
    table_geom_id = table.indexing.geom_ids[0]
    table_geom_center_z = self._env.sim.data.geom_xpos[:, table_geom_id, 2]
    table_geom_half_z = self._env.sim.model.geom_size[:, table_geom_id, 2]
    return table_geom_center_z + table_geom_half_z

  @property
  def object_quat_w(self) -> torch.Tensor:
    return self.object.data.root_link_quat_w

  @property
  def object_bottom_z(self) -> torch.Tensor:
    box_half_height = self._env.sim.model.geom_size[
      :, self.object.indexing.geom_ids[0], 2
    ]
    return self.object_pos_w[:, 2] - box_half_height

  @property
  def object_on_table(self) -> torch.Tensor:
    return self.contact_sensor.data.found.any(dim=-1) > 0

  def _update_metrics(self) -> None:
    position_error = torch.norm(self.target_pos - self.object_pos_w, dim=-1)
    at_goal = position_error < self.cfg.success_threshold

    # Increment at_goal_time if at goal, else reset to 0.0
    self.at_goal_time = torch.where(
      at_goal,
      self.at_goal_time + self._env.step_dt,
      torch.zeros_like(self.at_goal_time),
    )

    # reached becomes True if at_goal_time >= 0.5 seconds
    newly_reached = ~self.reached & (self.at_goal_time >= 0.1)
    self.reached = self.reached | newly_reached

    # Increment reached_time for all envs that are already (or just became) reached
    self.reached_time = torch.where(
      self.reached,
      self.reached_time + self._env.step_dt,
      self.reached_time,
    )

    # Track grasped distance and contact status
    from pal_mjlab.tasks.manipulation.mdp.contact_sensor import (
      site_contact_both_fingers,
    )

    contact_both = site_contact_both_fingers(
      self._env,
      sensor_name=self.cfg.fingertip_contact_sensor_name,
      site_names=[self.cfg.fingertip_site_pattern],
    ).bool()

    step_disp = torch.norm(self.object_pos_w - self.prev_object_pos_w, dim=-1)
    self.grasped_distance += torch.where(
      contact_both, step_disp, torch.zeros_like(step_disp)
    )
    self.prev_object_pos_w = self.object_pos_w.clone()

    # Success condition: target reached + dropped back to floor + released gripper (no contact).
    # We use a slightly permissive threshold (0.15 m instead of the termination's 0.1 m) because
    # `object_released_on_floor` fires when the object crosses 0.1 m during a MuJoCo *sub-step*,
    # but the step-level position sampled here is only updated after all sub-steps complete —
    # so the reported Z can be marginally above 0.1 m (e.g. after a small bounce).
    on_floor = self.object_pos_w[:, 2] < 0.15
    success = self.reached & on_floor & ~contact_both
    self.episode_success = torch.maximum(self.episode_success, success.float())

  def compute_success(self) -> torch.Tensor:
    return self.episode_success.bool()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.episode_success[env_ids] = 0.0
    self.reached[env_ids] = False
    self.at_goal_time[env_ids] = 0.0
    self.reached_time[env_ids] = 0.0
    self.grasped_distance[env_ids] = 0.0
    if hasattr(self, "frozen_rewards"):
      for key in self.frozen_rewards:
        self.frozen_rewards[key][env_ids] = 0.0
    table_surface_z = self.table_surface_z[env_ids]

    r = self.cfg.target_position_range
    lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
    upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
    target_pos = sample_uniform(lower, upper, (n, 3), device=self.device)

    # Target position is centered in base_footprint frame
    robot = self._env.scene["robot"]
    robot_pos = robot.data.root_link_pos_w[env_ids]
    robot_quat = robot.data.root_link_quat_w[env_ids]
    self.target_pos[env_ids] = quat_apply(robot_quat, target_pos) + robot_pos

    table: Entity = self._env.scene["table"]
    table_geom_id = table.indexing.geom_ids[0]

    # Table center in world coordinates (shape: (n, 2))
    table_center_x_y = self._env.sim.data.geom_xpos[env_ids, table_geom_id, 0:2]

    r = self.cfg.object_pose_range
    lower = torch.tensor([r.x[0], r.y[0], 0.0], device=self.device)
    upper = torch.tensor([r.x[1], r.y[1], 0.0], device=self.device)
    pos = sample_uniform(lower, upper, (n, 3), device=self.device)

    # Spawn box position: X & Y relative to table center, Z exactly on the actual table surface.
    pos_x_y = pos[:, 0:2] + table_center_x_y
    box_half_height = self._env.sim.model.geom_size[
      env_ids, self.object.indexing.geom_ids[0], 2
    ]
    pos_z = table_surface_z + box_half_height
    pos = torch.cat([pos_x_y, pos_z.unsqueeze(1)], dim=-1)

    self.prev_object_pos_w[env_ids] = pos

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
    robot = self._env.scene["robot"]
    ee_idx = None
    for target_name in ["gripper_right_grasping_site", "ee_site", "grasping_site"]:
      if target_name in robot.site_names:
        ee_idx = robot.site_names.index(target_name)
        break
    if ee_idx is None:
      for idx, name in enumerate(robot.site_names):
        if "grasping_site" in name or "ee" in name:
          ee_idx = idx
          break

    for batch in env_indices:
      visualizer.add_sphere(
        center=self.target_pos[batch].cpu().numpy(),
        radius=0.03,
        color=self.cfg.viz.target_color,
        label=f"target_position_{batch}",
      )
      if ee_idx is not None:
        ee_pos = robot.data.site_pos_w[batch, ee_idx].cpu().numpy()
        visualizer.add_sphere(
          center=ee_pos,
          radius=0.01,
          color=(1.0, 0.0, 0.0, 1.0),
          label=f"ee_position_{batch}",
        )


@dataclass(kw_only=True)
class LiftingCommandCfg(CommandTermCfg):
  entity_name: str
  object_half_height: float
  table_height: float
  contact_sensor_name: str
  success_threshold: float = 0.05
  fingertip_contact_sensor_name: str = "box_fingertip_contact"
  fingertip_site_pattern: str = "gripper_right_fingertip_.*_site"

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
    x: tuple[float, float] = (-0.2, 0.2)
    y: tuple[float, float] = (-0.2, 0.2)
    yaw: tuple[float, float] = (math.pi, math.pi)

  object_pose_range: ObjectPoseRangeCfg = field(default_factory=ObjectPoseRangeCfg)

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> LiftingCommand:
    return LiftingCommand(self, env)
