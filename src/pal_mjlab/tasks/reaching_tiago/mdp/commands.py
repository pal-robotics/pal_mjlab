from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
  quat_from_euler_xyz,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class LiftingCommand(CommandTerm):
  cfg: LiftingCommandCfg

  def __init__(self, cfg: LiftingCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.object: Entity = env.scene[cfg.asset_name]

    self.target_height = torch.zeros(self.num_envs, device=self.device)
    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)

    self.metrics["object_height"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    # Always return full 3D target position.
    return self.target_pos

  def _update_metrics(self) -> None:
    object_pos_w = self.object.data.root_link_pos_w
    object_height = object_pos_w[:, 2]

    position_error = torch.norm(self.target_pos - object_pos_w, dim=-1)
    height_error = torch.abs(self.target_height - object_height)

    self.metrics["object_height"] = object_height
    self.metrics["height_error"] = height_error
    self.metrics["position_error"] = position_error
    self.metrics["success_rate"] = (position_error < self.cfg.success_threshold).float()

  def compute_success(self) -> torch.Tensor:
    position_error = self.metrics["position_error"]
    return position_error < self.cfg.success_threshold

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)

    if self.cfg.difficulty == "fixed":
      target_pos = torch.tensor(
        [0.4, 0.0, 0.3], device=self.device, dtype=torch.float32
      ).expand(n, 3)
      # Add env_origins to make it absolute in world frame.
      self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]
    else:  # dynamic
      x = sample_uniform(
        self.cfg.target_position_range.x[0],
        self.cfg.target_position_range.x[1],
        (n,),
        device=self.device,
      )
      y = sample_uniform(
        self.cfg.target_position_range.y[0],
        self.cfg.target_position_range.y[1],
        (n,),
        device=self.device,
      )
      z = sample_uniform(
        self.cfg.target_position_range.z[0],
        self.cfg.target_position_range.z[1],
        (n,),
        device=self.device,
      )
      target_pos = torch.stack([x, y, z], dim=-1) + self._env.scene.env_origins[env_ids]
      self.target_pos[env_ids] = target_pos

    self.target_height[env_ids] = self.target_pos[env_ids, 2]

    # Reset object to new position.
    if self.cfg.object_pose_range is not None:
      x = sample_uniform(
        self.cfg.object_pose_range.x[0],
        self.cfg.object_pose_range.x[1],
        (n,),
        device=self.device,
      )
      y = sample_uniform(
        self.cfg.object_pose_range.y[0],
        self.cfg.object_pose_range.y[1],
        (n,),
        device=self.device,
      )
      z = sample_uniform(
        self.cfg.object_pose_range.z[0],
        self.cfg.object_pose_range.z[1],
        (n,),
        device=self.device,
      )
      pos = torch.stack([x, y, z], dim=-1) + self._env.scene.env_origins[env_ids]

      # Sample orientation (yaw only, keep upright).
      yaw = sample_uniform(
        self.cfg.object_pose_range.yaw[0],
        self.cfg.object_pose_range.yaw[1],
        (n,),
        device=self.device,
      )
      quat = quat_from_euler_xyz(
        torch.zeros(n, device=self.device),  # roll
        torch.zeros(n, device=self.device),  # pitch
        yaw,
      )
      pose = torch.cat([pos, quat], dim=-1)

      velocity = torch.zeros(n, 6, device=self.device)

      self.object.write_root_link_pose_to_sim(pose, env_ids=env_ids)
      self.object.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    batch = visualizer.env_idx
    if batch >= self.num_envs:
      return

    if self.cfg.viz.show_target_height:
      target_pos = self.target_pos[batch].cpu().numpy()
      visualizer.add_sphere(
        center=target_pos,
        radius=0.03,
        color=self.cfg.viz.target_color,
        label="target_position",
      )


@dataclass(kw_only=True)
class LiftingCommandCfg(CommandTermCfg):
  asset_name: str
  class_type: type[CommandTerm] = LiftingCommand
  success_threshold: float = 0.05
  difficulty: Literal["fixed", "dynamic"] = "fixed"

  @dataclass
  class TargetPositionRangeCfg:
    """Configuration for target position sampling in dynamic mode."""

    x: tuple[float, float] = (0.4, 0.8)
    y: tuple[float, float] = (-0.2, 0.5)
    z: tuple[float, float] = (0.2, 1.0)

  # Only used in dynamic mode.
  target_position_range: TargetPositionRangeCfg = field(
    default_factory=TargetPositionRangeCfg
  )

  @dataclass
  class ObjectPoseRangeCfg:
    """Configuration for object pose sampling when resampling commands."""

    x: tuple[float, float] = (0.4, 0.8)
    y: tuple[float, float] = (-0.2, 0.5)
    z: tuple[float, float] = (0.2, 1.0)
    yaw: tuple[float, float] = (-math.pi, math.pi)

  object_pose_range: ObjectPoseRangeCfg | None = field(
    default_factory=ObjectPoseRangeCfg
  )

  @dataclass
  class VizCfg:
    show_target_height: bool = True
    target_color: tuple[float, float, float, float] = (1.0, 0.5, 0.0, 0.3)

  viz: VizCfg = field(default_factory=VizCfg)