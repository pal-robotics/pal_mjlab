from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import math

import torch
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

from mjlab.managers.command_manager import (
  CommandTerm,
  CommandTermCfg
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


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
  """Two-hand pose command (position + orientation) and base height.

  Hand poses are expressed in the base frame. The hand pointing axis (local +x)
  never points behind the robot, and the hand xy position is kept at least
  `min_xy_radius` away from the base origin.

  Command layout (15 dims):
    [right_pos(3), right_quat(4), left_pos(3), left_quat(4), base_height(1)]
  with quaternions in (w, x, y, z).
  """

  cfg: WbcPoseCommandCfg

  def __init__(self, cfg: WbcPoseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.ee_right_pose_command = torch.zeros(self.num_envs, 7, device=self.device)
    self.ee_left_pose_command = torch.zeros(self.num_envs, 7, device=self.device)
    self.ee_right_pose_command[:, 3] = 1.0  # identity quaternion
    self.ee_left_pose_command[:, 3] = 1.0
    self.base_height_command = torch.zeros(self.num_envs, 1, device=self.device)

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

  def _sample(self, n: int, rng: tuple[float, float]) -> torch.Tensor:
    return torch.empty(n, device=self.device).uniform_(*rng)

  def _sample_hand_pose(
    self, n: int, ranges: WbcPoseCommandCfg.Ranges
  ) -> torch.Tensor:
    xy = torch.stack((self._sample(n, ranges.x), self._sample(n, ranges.y)), dim=-1)
    xy = push_xy_outside_radius(xy, self.cfg.min_xy_radius)
    z = self._sample(n, ranges.z)

    quat = _quat_from_rpy(
      self._sample(n, ranges.roll),
      self._sample(n, ranges.pitch),
      self._sample(n, ranges.yaw),
    )
    return torch.cat([xy, z.unsqueeze(-1), quat], dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    names = self.cfg.tracked_body_names
    ranges = self.cfg.tracked_body_ranges

    self.ee_right_pose_command[env_ids] = self._sample_hand_pose(n, ranges[names[0]])
    self.ee_left_pose_command[env_ids] = self._sample_hand_pose(n, ranges[names[1]])
    self.base_height_command[env_ids, 0] = self._sample(n, ranges[names[2]].z)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    pass


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
    yaw: tuple[float, float] = (-math.pi / 2, math.pi / 2)
    pitch: tuple[float, float] = (-math.pi / 4, math.pi / 4)
    roll: tuple[float, float] = (-math.pi / 4, math.pi / 4)

  tracked_body_ranges: dict[str, Ranges]

  min_xy_radius: float = 0.2
  """Minimum horizontal distance between hand targets and the base origin."""

  viz = None

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

    half_pi = math.pi / 2
    for body in self.tracked_body_names[:2]:
      r = self.tracked_body_ranges[body]
      for name in ("yaw", "pitch"):
        lo, hi = getattr(r, name)
        if lo < -half_pi or hi > half_pi:
          raise ValueError(
            f"{body}.{name}={(lo, hi)} must lie within [-pi/2, pi/2]."
          )