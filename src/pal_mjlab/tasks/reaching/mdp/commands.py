from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_conjugate,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_mul,
  quat_unique,
  sample_uniform,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


class UniformPoseCommand(CommandTerm):
  cfg: UniformPoseCommandCfg

  def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    # Extract the robot and site index for which the command is generated
    self.robot: Entity = env.scene[cfg.entity_name]
    self.site_idx = self.robot.site_names.index(cfg.site_name)

    # Create buffers
    # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
    self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
    self.pose_command_b[:, 3] = 1.0  # Initialize quaternion w component
    self.pose_command_w = torch.zeros_like(self.pose_command_b)

    # -- metrics
    self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

  # def __str__(self) -> str:
  #    msg = "UniformPoseCommand:\n"
  #    msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
  #    msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
  #    return msg

  @property
  def command(self) -> torch.Tensor:
    """The desired pose command. Shape is (num_envs, 7).

    The first three elements correspond to the position, followed by the quaternion
    orientation in (w, x, y, z).
    """
    return self.pose_command_b

  def _update_metrics(self):
    """Update tracking error metrics."""
    # Transform command from base frame to simulation world frame
    # Get robot root state
    root_pos_w = self.robot.data.site_pos_w[:, 0]  # Root site position
    root_quat_w = self.robot.data.site_quat_w[:, 0]  # Root site quaternion

    # Transform position: p_w = p_root + R_root * p_b
    pos_rotated = quat_mul(
      quat_mul(
        root_quat_w,
        torch.cat(
          [
            torch.zeros(self.num_envs, 1, device=self.device),
            self.pose_command_b[:, :3],
          ],
          dim=1,
        ),
      ),
      quat_conjugate(root_quat_w),
    )[:, 1:]  # Extract xyz from quaternion product
    self.pose_command_w[:, :3] = root_pos_w + pos_rotated

    # Transform orientation: q_w = q_root * q_b
    self.pose_command_w[:, 3:] = quat_mul(root_quat_w, self.pose_command_b[:, 3:])

    # Get current site pose
    current_site_pos_w = self.robot.data.site_pos_w[:, self.site_idx]
    current_site_quat_w = self.robot.data.site_quat_w[:, self.site_idx]

    # Compute position error
    pos_error = current_site_pos_w - self.pose_command_w[:, :3]
    self.metrics["position_error"] = torch.norm(pos_error, dim=-1)

    # Compute orientation error
    self.metrics["orientation_error"] = quat_error_magnitude(
      self.pose_command_w[:, 3:], current_site_quat_w
    )

  def _resample_command(self, env_ids: torch.Tensor):
    """Resample pose commands for specified environments.

    Args:
        env_ids: Environment indices to resample commands for.
    """
    # Sample new pose targets
    # -- position
    self.pose_command_b[env_ids, 0] = sample_uniform(
      self.cfg.ranges.pos_x[0],
      self.cfg.ranges.pos_x[1],
      (len(env_ids),),
      device=self.device,
    )
    self.pose_command_b[env_ids, 1] = sample_uniform(
      self.cfg.ranges.pos_y[0],
      self.cfg.ranges.pos_y[1],
      (len(env_ids),),
      device=self.device,
    )
    self.pose_command_b[env_ids, 2] = sample_uniform(
      self.cfg.ranges.pos_z[0],
      self.cfg.ranges.pos_z[1],
      (len(env_ids),),
      device=self.device,
    )

    # -- orientation (sample euler angles and convert to quaternion)
    roll = sample_uniform(
      self.cfg.ranges.roll[0],
      self.cfg.ranges.roll[1],
      (len(env_ids),),
      device=self.device,
    )
    pitch = sample_uniform(
      self.cfg.ranges.pitch[0],
      self.cfg.ranges.pitch[1],
      (len(env_ids),),
      device=self.device,
    )
    yaw = sample_uniform(
      self.cfg.ranges.yaw[0],
      self.cfg.ranges.yaw[1],
      (len(env_ids),),
      device=self.device,
    )

    quat = quat_from_euler_xyz(roll, pitch, yaw)

    # Make sure the quaternion has real part as positive
    self.pose_command_b[env_ids, 3:] = (
      quat_unique(quat) if self.cfg.make_quat_unique else quat
    )

  def _update_command(self):
    """Update command - no action needed for static pose commands."""
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Visualize goal and current poses using debug visualizer."""
    # if not self.robot.is_initialized:
    #    return

    env_idx = visualizer.env_idx

    # Visualize goal pose
    goal_pos = self.pose_command_w[env_idx, :3].cpu().numpy()
    # goal_quat = self.pose_command_w[env_idx, 3:].cpu().numpy()
    goal_rotm = (
      matrix_from_quat(self.pose_command_w[env_idx, 3:].unsqueeze(0))
      .squeeze(0)
      .cpu()
      .numpy()
    )

    visualizer.add_frame(
      position=goal_pos,
      rotation_matrix=goal_rotm,
      scale=self.cfg.viz.goal_frame_scale,
      label="goal_pose",
      axis_colors=self.cfg.viz.goal_frame_colors,
    )

    # Visualize current site pose
    current_pos = self.robot.data.site_pos_w[env_idx, self.site_idx].cpu().numpy()
    current_quat = self.robot.data.site_quat_w[env_idx, self.site_idx]
    current_rotm = matrix_from_quat(current_quat.unsqueeze(0)).squeeze(0).cpu().numpy()

    visualizer.add_frame(
      position=current_pos,
      rotation_matrix=current_rotm,
      scale=self.cfg.viz.current_frame_scale,
      label="current_pose",
      axis_colors=self.cfg.viz.current_frame_colors,
    )


@dataclass(kw_only=True)
class PoseRanges:
  """Ranges for sampling pose commands."""

  pos_x: tuple[float, float] = (-0.5, 0.5)
  pos_y: tuple[float, float] = (-0.5, 0.5)
  pos_z: tuple[float, float] = (-0.5, 1.0)
  roll: tuple[float, float] = (-np.pi, np.pi)
  pitch: tuple[float, float] = (-np.pi, np.pi)
  yaw: tuple[float, float] = (-np.pi, np.pi)


@dataclass(kw_only=True)
class UniformPoseCommandCfg(CommandTermCfg):
  """Configuration for uniform pose command generator."""

  class_type: type[CommandTerm] = UniformPoseCommand

  entity_name: str
  """Name of the robot asset in the scene."""

  site_name: str
  """Name of the site to track."""

  ranges: PoseRanges = field(default_factory=PoseRanges)
  """Ranges for sampling pose commands."""

  make_quat_unique: bool = True
  """Whether to make quaternions unique (positive real part)."""

  @dataclass
  class VizCfg:
    """Visualization configuration."""

    goal_frame_scale: float = 0.1
    current_frame_scale: float = 0.15
    goal_frame_colors: tuple[tuple[float, float, float], ...] = (
      (1.0, 0.5, 0.5),
      (0.5, 1.0, 0.5),
      (0.5, 0.5, 1.0),
    )
    current_frame_colors: tuple[tuple[float, float, float], ...] = (
      (1.0, 0.0, 0.0),
      (0.0, 1.0, 0.0),
      (0.0, 0.0, 1.0),
    )

  viz: VizCfg = field(default_factory=VizCfg)
  """Visualization configuration."""

  def build(self, env: ManagerBasedRlEnv) -> UniformPoseCommand:
    return UniformPoseCommand(self, env)
