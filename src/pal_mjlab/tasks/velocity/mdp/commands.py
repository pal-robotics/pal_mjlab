from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import math
import numpy as np
import torch
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

from mjlab.managers.command_manager import (
  CommandTerm,
  CommandTermCfg
)

from mjlab.utils.lab_api.math import matrix_from_quat, quat_mul, quat_from_matrix

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


def _quat_from_rpy(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor):
  """Intrinsic ZYX (yaw, pitch, roll) -> quaternion (w, x, y, z)."""
  cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
  cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
  cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
  return torch.stack(
    (
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
    ),
    dim=-1,
  )

def push_xy_outside_radius(xy: torch.Tensor, min_radius: float) -> torch.Tensor:
  norm = xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  return xy * (min_radius / norm).clamp_min(1.0)


def validate_base_position(base_position, min_radius: float) -> None:
  if math.hypot(base_position[0], base_position[1]) <= min_radius:
    raise ValueError(f"base_position={base_position} is inside the robot (xy-norm <= {min_radius}).")

class DualBandVelocityCommand(UniformVelocityCommand):
  """Uniform velocity command with configurable low-velocity sampling bands."""

  cfg: DualBandVelocityCommandCfg  # type: ignore[assignment]

  def __init__(self, cfg: DualBandVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._minimum_magnitudes = (
      cfg.min_lin_vel_x,
      cfg.min_lin_vel_y,
      cfg.min_ang_vel_z,
    )
    self._command_ranges = (
      cfg.ranges.lin_vel_x,
      cfg.ranges.lin_vel_y,
      cfg.ranges.ang_vel_z,
    )

  @staticmethod
  def _sample_uniform_interval(
    values: torch.Tensor, interval: tuple[float, float]
  ) -> torch.Tensor:
    lower, upper = interval
    return lower + torch.rand_like(values) * (upper - lower)

  def _sample_inside_low_velocity_band(self, command: torch.Tensor) -> torch.Tensor:
    """Sample configured nonzero components inside their low-velocity band."""
    for axis, (minimum, (lower, upper)) in enumerate(
      zip(self._minimum_magnitudes, self._command_ranges, strict=True)
    ):
      if minimum == 0.0 or (lower == 0.0 and upper == 0.0):
        continue

      component = command[:, axis]
      nonzero = component != 0.0
      inside_band = (max(lower, -minimum), min(upper, minimum))
      component[nonzero] = self._sample_uniform_interval(
        component[nonzero], inside_band
      )

    return command

  def _sample_outside_low_velocity_band(self, command: torch.Tensor) -> torch.Tensor:
    """Sample configured nonzero components outside their low-velocity band."""
    for axis, (minimum, (lower, upper)) in enumerate(
      zip(self._minimum_magnitudes, self._command_ranges, strict=True)
    ):
      if minimum == 0.0 or (lower == 0.0 and upper == 0.0):
        continue

      component = command[:, axis]
      nonzero = component != 0.0

      below_band = None
      if lower < -minimum:
        below_band = (lower, min(upper, -minimum))

      above_band = None
      if upper > minimum:
        above_band = (max(lower, minimum), upper)

      if below_band is None:
        assert above_band is not None
        component[nonzero] = self._sample_uniform_interval(
          component[nonzero], above_band
        )
      elif above_band is None:
        component[nonzero] = self._sample_uniform_interval(
          component[nonzero], below_band
        )
      else:
        below_band_lower, below_band_upper = below_band
        above_band_lower, above_band_upper = above_band
        below_band_width = below_band_upper - below_band_lower
        above_band_width = above_band_upper - above_band_lower
        # Choose each interval by its width to keep the union uniformly sampled.
        probability_below_band = below_band_width / (
          below_band_width + above_band_width
        )

        choose_below_band = (
          torch.rand_like(component) < probability_below_band
        ) & nonzero
        choose_above_band = (~choose_below_band) & nonzero
        component[choose_below_band] = self._sample_uniform_interval(
          component[choose_below_band], below_band
        )
        component[choose_above_band] = self._sample_uniform_interval(
          component[choose_above_band], above_band
        )

    return command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)

    straight_mask = (
      torch.rand(len(env_ids), device=self.device) < self.cfg.rel_straight_envs
    )
    straight_ids = env_ids[straight_mask]
    self.vel_command_b[straight_ids, 1:3] = 0.0
    self.is_heading_env[straight_ids] = False
    self.is_world_env[straight_ids] = False

    moving_ids = env_ids[~self.is_standing_env[env_ids]]
    inside_low_mask = (
      torch.rand(len(moving_ids), device=self.device) < self.cfg.rel_inside_low
    )
    inside_low_ids = moving_ids[inside_low_mask]
    outside_low_ids = moving_ids[~inside_low_mask]
    if len(inside_low_ids) > 0:
      self.vel_command_b[inside_low_ids] = self._sample_inside_low_velocity_band(
        self.vel_command_b[inside_low_ids]
      )
    if len(outside_low_ids) > 0:
      self.vel_command_b[outside_low_ids] = self._sample_outside_low_velocity_band(
        self.vel_command_b[outside_low_ids]
      )

    world_ids = moving_ids[self.is_world_env[moving_ids]]
    self.vel_command_w[world_ids] = self.vel_command_b[world_ids]


@dataclass(kw_only=True)
class DualBandVelocityCommandCfg(UniformVelocityCommandCfg):
  """Per-axis low-velocity sampling bands; zero remains a valid command."""

  rel_straight_envs: float = 0.0
  """Fraction of resampled commands constrained to signed x-only motion."""

  rel_inside_low: float = 0.0
  """Fraction of moving commands sampled inside every configured low-velocity band."""

  min_lin_vel_x: float = 0.0
  """Magnitude boundary of the x low-velocity band; zero disables the band."""

  min_lin_vel_y: float = 0.0
  """Magnitude boundary of the y low-velocity band; zero disables the band."""

  min_ang_vel_z: float = 0.0
  """Magnitude boundary of the yaw low-velocity band; zero disables the band."""

  def __post_init__(self) -> None:
    super().__post_init__()
    velocity_bands = {
      "min_lin_vel_x": (self.min_lin_vel_x, self.ranges.lin_vel_x),
      "min_lin_vel_y": (self.min_lin_vel_y, self.ranges.lin_vel_y),
      "min_ang_vel_z": (self.min_ang_vel_z, self.ranges.ang_vel_z),
    }
    if not 0.0 <= self.rel_inside_low <= 1.0:
      raise ValueError(
        f"rel_inside_low must be between 0 and 1; got {self.rel_inside_low}."
      )

    for name, (minimum, (lower, upper)) in velocity_bands.items():
      if minimum < 0.0:
        raise ValueError(f"{name} must be non-negative; got {minimum}.")
      if minimum == 0.0 or (lower == 0.0 and upper == 0.0):
        continue

      maximum_range_magnitude = max(abs(lower), abs(upper))
      if minimum > maximum_range_magnitude:
        raise ValueError(
          f"{name}={minimum} exceeds the maximum magnitude "
          f"{maximum_range_magnitude} available in range ({lower}, {upper})."
        )

      inside_band_lower = max(lower, -minimum)
      inside_band_upper = min(upper, minimum)
      has_inside_band = inside_band_lower < inside_band_upper
      has_below_band = lower < -minimum
      has_above_band = upper > minimum

      if self.rel_inside_low > 0.0 and not has_inside_band:
        raise ValueError(
          f"{name}={minimum} leaves no values inside the low-velocity band "
          f"for range ({lower}, {upper})."
        )
      if self.rel_inside_low < 1.0 and not (has_below_band or has_above_band):
        raise ValueError(
          f"{name}={minimum} leaves no values outside the low-velocity band "
          f"for range ({lower}, {upper})."
        )
    if self.init_velocity_prob != 0.0:
      raise ValueError(
        "DualBandVelocityCommandCfg post-processes commands after the parent "
        "initializes velocity; set init_velocity_prob=0.0."
      )
    if self.rel_forward_envs != 0.0:
      raise ValueError(
        "DualBandVelocityCommandCfg replaces rel_forward_envs with "
        "rel_straight_envs; set rel_forward_envs=0.0."
      )

  def build(self, env: ManagerBasedRlEnv) -> DualBandVelocityCommand:
    return DualBandVelocityCommand(self, env)



class WbcCommand (CommandTerm) :
  cfg: WbcCommandCfg

  def __init__(self, cfg: WbcCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.ee_right_position_command = torch.zeros((self.num_envs, 3,), device=self.device)
    self.ee_left_position_command = torch.zeros((self.num_envs, 3,), device=self.device)
    self.base_height_command = torch.zeros((self.num_envs, 1), device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat(
      [self.ee_right_position_command, self.ee_left_position_command, self.base_height_command], dim=-1
    ) 

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)

    self.ee_right_position_command[env_ids, 0] = r.uniform_(*self.cfg.tracked_body_ranges[self.cfg.tracked_body_names[0]].x)
    self.ee_right_position_command[env_ids, 1] = r.uniform_(*self.cfg.tracked_body_ranges[self.cfg.tracked_body_names[0]].y)
    self.ee_right_position_command[env_ids, 2] = r.uniform_(*self.cfg.tracked_body_ranges[self.cfg.tracked_body_names[0]].z)

    self.ee_left_position_command[env_ids, 0] = r.uniform_(*self.cfg.tracked_body_ranges[self.cfg.tracked_body_names[1]].x)
    self.ee_left_position_command[env_ids, 1] = r.uniform_(*self.cfg.tracked_body_ranges[self.cfg.tracked_body_names[1]].y)
    self.ee_left_position_command[env_ids, 2] = r.uniform_(*self.cfg.tracked_body_ranges[self.cfg.tracked_body_names[1]].z)

    self.base_height_command[env_ids, 0] = r.uniform_(*self.cfg.tracked_body_ranges[self.cfg.tracked_body_names[2]].z)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    pass

@dataclass(kw_only=True)
class WbcCommandCfg(CommandTermCfg) :

  tracked_body_names: tuple[str, ...]

  @dataclass
  class Ranges:
    x: tuple[float, float] = (0.0, 0.0)
    y: tuple[float, float] = (0.0, 0.0)
    z: tuple[float, float] = (0.0, 0.0)

  tracked_body_ranges: dict[str, Ranges]

  viz = None

  def build(self, env: ManagerBasedRlEnv) -> WbcCommand:
    return WbcCommand(self, env)

  def __post_init__(self):
    pass




class WbcPoseCommand(CommandTerm):
  """Smooth two-hand pose commands and base-height targets.

  Command layout (15 dims):
    [right_pos(3), right_quat(4),
     left_pos(3), left_quat(4), base_height(1)]

  Positions and orientations are expressed in the base frame.
  Quaternions use (w, x, y, z).
  """

  cfg: WbcPoseCommandCfg

  def __init__(self, cfg: WbcPoseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]

    # Current commanded poses.
    self.ee_right_pose_command = torch.zeros(
      self.num_envs, 7, device=self.device
    )
    self.ee_left_pose_command = torch.zeros(
      self.num_envs, 7, device=self.device
    )
    self.ee_right_pose_command[:, 3] = 1.0
    self.ee_left_pose_command[:, 3] = 1.0

    self.base_height_command = torch.zeros(
      self.num_envs, 1, device=self.device
    )

    # Destination poses.
    self._target_right = self.ee_right_pose_command.clone()
    self._target_left = self.ee_left_pose_command.clone()
    self._target_base_height = self.base_height_command.clone()

    # Starting poses for each transition.
    self._start_right = self.ee_right_pose_command.clone()
    self._start_left = self.ee_left_pose_command.clone()
    self._start_base_height = self.base_height_command.clone()

    # Independent transition timer for each environment.
    self._transition_elapsed = torch.zeros(
      self.num_envs, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return torch.cat(
      [
        self.ee_right_pose_command,
        self.ee_left_pose_command,
        self.base_height_command,
      ],
      dim=-1,
    )

  def _update_metrics(self) -> None:
    pass

  def _sample(
    self, n: int, rng: tuple[float, float]
  ) -> torch.Tensor:
    return torch.empty(n, device=self.device).uniform_(*rng)

  def _sample_hand_pose(
    self, n: int, ranges: WbcPoseCommandCfg.Ranges
  ) -> torch.Tensor:
    xy = torch.stack(
      (self._sample(n, ranges.x), self._sample(n, ranges.y)),
      dim=-1,
    )
    xy = push_xy_outside_radius(xy, self.cfg.min_xy_radius)
    z = self._sample(n, ranges.z)

    quat = _quat_from_rpy(
      self._sample(n, ranges.roll),
      self._sample(n, ranges.pitch),
      self._sample(n, ranges.yaw),
    )
    quat = quat / quat.norm(dim=-1, keepdim=True)

    return torch.cat([xy, z.unsqueeze(-1), quat], dim=-1)

  @staticmethod
  def _quat_slerp(
    q0: torch.Tensor,
    q1: torch.Tensor,
    alpha: torch.Tensor,
  ) -> torch.Tensor:
    """Shortest-path SLERP for normalized (w, x, y, z) quaternions."""

    dot = (q0 * q1).sum(dim=-1, keepdim=True)

    # q and -q represent the same rotation. Choose the shorter path.
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)

    # alpha has shape (N,); expand to (N, 1).
    t = alpha.unsqueeze(-1)

    # Use normalized linear interpolation for nearly identical rotations.
    near = dot > 0.9995

    theta = torch.acos(dot.clamp(max=1.0 - 1e-7))
    sin_theta = torch.sin(theta).clamp_min(1e-7)

    w0 = torch.sin((1.0 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta

    slerp = w0 * q0 + w1 * q1
    lerp = (1.0 - t) * q0 + t * q1

    result = torch.where(near, lerp, slerp)
    return result / result.norm(dim=-1, keepdim=True).clamp_min(1e-8)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    """Choose new destinations without jumping the current commands."""

    n = len(env_ids)
    names = self.cfg.tracked_body_names
    ranges = self.cfg.tracked_body_ranges

    # Save the current command as the starting point of the transition.
    self._start_right[env_ids] = self.ee_right_pose_command[env_ids]
    self._start_left[env_ids] = self.ee_left_pose_command[env_ids]
    self._start_base_height[env_ids] = self.base_height_command[env_ids]

    # Sample destination poses.
    self._target_right[env_ids] = self._sample_hand_pose(
      n, ranges[names[0]]
    )
    self._target_left[env_ids] = self._sample_hand_pose(
      n, ranges[names[1]]
    )
    self._target_base_height[env_ids, 0] = self._sample(
      n, ranges[names[2]].z
    )

    # Restart the transition only for the resampled environments.
    self._transition_elapsed[env_ids] = 0.0

    # If duration is zero, apply the targets immediately.
    if self.cfg.transition_duration_s <= 0.0:
      self.ee_right_pose_command[env_ids] = self._target_right[env_ids]
      self.ee_left_pose_command[env_ids] = self._target_left[env_ids]
      self.base_height_command[env_ids] = (
        self._target_base_height[env_ids]
      )

  def _update_command(
    self, env_ids: torch.Tensor | None
  ) -> None:
    """Advance commands smoothly toward their targets."""

    if self.cfg.transition_duration_s <= 0.0:
      return

    if env_ids is None:
      ids = torch.arange(self.num_envs, device=self.device)
    else:
      ids = env_ids

    if len(ids) == 0:
      return

    # ManagerBasedRlEnv's simulation/control timestep.
    dt = self._env.step_dt
    self._transition_elapsed[ids] += dt

    # Linear progress from 0 to 1 over the configured duration.
    alpha = (
      self._transition_elapsed[ids] / self.cfg.transition_duration_s
    ).clamp(0.0, 1.0)

    t = alpha.unsqueeze(-1)

    # Hand positions: linear interpolation.
    self.ee_right_pose_command[ids, :3] = (
      (1.0 - t) * self._start_right[ids, :3]
      + t * self._target_right[ids, :3]
    )
    self.ee_left_pose_command[ids, :3] = (
      (1.0 - t) * self._start_left[ids, :3]
      + t * self._target_left[ids, :3]
    )

    # Hand orientations: spherical quaternion interpolation.
    self.ee_right_pose_command[ids, 3:7] = self._quat_slerp(
      self._start_right[ids, 3:7],
      self._target_right[ids, 3:7],
      alpha,
    )
    self.ee_left_pose_command[ids, 3:7] = self._quat_slerp(
      self._start_left[ids, 3:7],
      self._target_left[ids, 3:7],
      alpha,
    )

    # Base height: linear interpolation.
    self.base_height_command[ids] = (
      (1.0 - t) * self._start_base_height[ids]
      + t * self._target_base_height[ids]
    )

    # Guarantee exact targets at the end of the transition.
    finished = alpha >= 1.0
    if finished.any():
      done_ids = ids[finished]
      self.ee_right_pose_command[done_ids] = self._target_right[done_ids]
      self.ee_left_pose_command[done_ids] = self._target_left[done_ids]
      self.base_height_command[done_ids] = (
        self._target_base_height[done_ids]
      )

  def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    viz = self.cfg.viz

    # Robot base pose in world coordinates.
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_mat_ws = matrix_from_quat(
      self.robot.data.root_link_quat_w
    ).cpu().numpy()

    # Commanded hand poses in the base frame.
    right = self.ee_right_pose_command.cpu().numpy()
    left = self.ee_left_pose_command.cpu().numpy()

    right_mat_bs = matrix_from_quat(
      self.ee_right_pose_command[:, 3:7]
    ).cpu().numpy()
    left_mat_bs = matrix_from_quat(
      self.ee_left_pose_command[:, 3:7]
    ).cpu().numpy()

    base_heights = self.base_height_command.cpu().numpy()

    # Actual body poses in world coordinates.
    body_pos_ws = self.robot.data.body_link_pos_w.cpu().numpy()
    body_mat_ws = matrix_from_quat(
      self.robot.data.body_link_quat_w
    ).cpu().numpy()

    names = self.cfg.tracked_body_names
    right_body_ids = self.robot.find_bodies(names[0])[0]
    left_body_ids = self.robot.find_bodies(names[1])[0]

    if len(right_body_ids) == 0 or len(left_body_ids) == 0:
      raise ValueError(
        f"Could not find tip links: right={names[0]!r}, "
        f"left={names[1]!r}"
      )

    right_body_id = right_body_ids[0]
    left_body_id = left_body_ids[0]

    # Position cross marker, defined in the frame being drawn.
    s = viz.cross_size
    cross = [
      np.array(v, dtype=float) * s
      for v in (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
      )
    ]

    # RGB coordinate axes: X red, Y green, Z blue.
    axis_colors = (
      (1.0, 0.1, 0.1, 1.0),
      (0.1, 1.0, 0.1, 1.0),
      (0.1, 0.3, 1.0, 1.0),
    )

    def draw_frame(
      position_w: np.ndarray,
      rotation_w: np.ndarray,
      color: tuple[float, float, float, float],
    ) -> None:
      # Cross centered on the frame origin.
      for d in cross:
        visualizer.add_arrow(
          position_w,
          position_w + d,
          color=color,
          width=viz.arrow_width,
        )

      # Orientation axes in world coordinates.
      for axis, axis_color in enumerate(axis_colors):
        visualizer.add_arrow(
          position_w,
          position_w + viz.arrow_length * rotation_w[:, axis],
          color=axis_color,
          width=viz.arrow_width * 2,
        )

    for i in env_indices:
      base_pos_w = base_pos_ws[i]
      base_rot_wb = base_mat_ws[i]

      # Skip uninitialized robots.
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      # Inverse rotation: world coordinates -> base coordinates.
      base_rot_bw = base_rot_wb.T

      # -------------------------------------------------------
      # 1. Commanded hand frames.
      #
      # Commands are stored in the base frame.
      # Transform them into world coordinates for drawing.
      # -------------------------------------------------------
      for pose_b, rot_bh, color in (
        (right[i], right_mat_bs[i], viz.right_color),
        (left[i], left_mat_bs[i], viz.left_color),
      ):
        pos_b = pose_b[:3]
        rot_bh = rot_bh

        pos_w = base_pos_w + base_rot_wb @ pos_b
        rot_wh = base_rot_wb @ rot_bh

        draw_frame(pos_w, rot_wh, color)

      # -------------------------------------------------------
      # 2. Actual hand frames.
      #
      # Convert actual world poses into the base frame first.
      # Then transform back to world coordinates for rendering.
      # This keeps the relative-pose calculation explicit.
      # -------------------------------------------------------
      for body_id, color in (
        (right_body_id, viz.right_color),
        (left_body_id, viz.left_color),
      ):
        actual_pos_w = body_pos_ws[i, body_id]
        actual_rot_w = body_mat_ws[i, body_id]

        # Actual pose expressed in the base frame.
        actual_pos_b = base_rot_bw @ (
          actual_pos_w - base_pos_w
        )
        actual_rot_b = base_rot_bw @ actual_rot_w

        # Convert to world coordinates for the viewer.
        actual_pos_w_draw = (
          base_pos_w + base_rot_wb @ actual_pos_b
        )
        actual_rot_w_draw = base_rot_wb @ actual_rot_b

        draw_frame(
          actual_pos_w_draw,
          actual_rot_w_draw,
          color,
        )

      # -------------------------------------------------------
      # 3. Base-height target arrow.
      # -------------------------------------------------------
      start_w = base_pos_w.copy()
      end_w = base_pos_w + base_rot_wb @ np.array(
        [0.0, 0.0, float(base_heights[i, 0])],
        dtype=float,
      )

      visualizer.add_arrow(
        start_w,
        end_w,
        color=viz.base_color,
        width=viz.arrow_width * 2,
      )


@dataclass(kw_only=True)
class WbcPoseCommandCfg(CommandTermCfg):

  tracked_body_names: tuple[str, ...]
  """(right_hand, left_hand, base). Only `z` is used for the base entry."""

  @dataclass
  class Ranges:
    x: tuple[float, float] = (0.0, 0.0)
    y: tuple[float, float] = (0.0, 0.0)
    z: tuple[float, float] = (0.0, 0.0)
    # Radians, base frame (x forward). Defaults keep the hand facing forward
    # or sideways, with everything in between.
    yaw: tuple[float, float] = (0.0, 0.0)#(-math.pi / 2 - math.pi / 6, -math.pi / 2 + math.pi / 6)
    pitch: tuple[float, float] = (0.0, 0.0)
    roll: tuple[float, float] = (0.0, 0.0)

  tracked_body_ranges: dict[str, Ranges]

  min_xy_radius: float = 0.2
  """Minimum horizontal distance between hand targets and the base origin."""

  transition_duration_s: float = 2.0
  """Time in seconds to move from the current command to its target."""

  entity_name: str = "robot"

  @dataclass
  class VizCfg:
    cross_size: float = 0.03
    arrow_length: float = 0.15
    arrow_width: float = 0.008
    right_color: tuple[float, float, float, float] = (1.0, 0.2, 0.2, 0.8)
    left_color: tuple[float, float, float, float] = (0.2, 0.4, 1.0, 0.8)
    base_color: tuple[float, float, float, float] = (0.2, 0.9, 0.3, 0.8)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> WbcPoseCommand:
    return WbcPoseCommand(self, env)

  def __post_init__(self):
    if len(self.tracked_body_names) != 3:
      raise ValueError(
        "tracked_body_names must be (right_hand, left_hand, base); "
        f"got {self.tracked_body_names}."
      )
    missing = [n for n in self.tracked_body_names if n not in self.tracked_body_ranges]
    if missing:
      raise ValueError(f"tracked_body_ranges is missing entries for {missing}.")