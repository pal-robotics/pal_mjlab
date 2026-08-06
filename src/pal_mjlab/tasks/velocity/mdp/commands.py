from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class CommandHistory:
  """Per-episode record of the commands sampled by each environment.

  Entry ``i`` of environment ``e`` holds the command at the start of segment ``i`` and
  the episode time at which that segment started. Segment ``i`` ends when segment
  ``i + 1`` starts; the last valid segment is still open.

  The record is a plain container: it owns the buffers and the append bookkeeping, but
  knows nothing about the environment or about what the commands mean. ``capacity`` is
  the worst-case number of segments an episode can hold, so appends never overflow.
  """

  def __init__(
    self, num_envs: int, capacity: int, command_dim: int, device: str
  ) -> None:
    self.commands = torch.zeros(num_envs, capacity, command_dim, device=device)
    """Command at the start of each segment, shape (num_envs, capacity, command_dim)."""
    self.start_times = torch.zeros(num_envs, capacity, device=device)
    """Episode time at the start of each segment, shape (num_envs, capacity)."""
    self.lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
    """Number of valid segments per environment, shape (num_envs,)."""
    self._pending = torch.zeros(num_envs, dtype=torch.bool, device=device)

  def clear(self, env_ids: torch.Tensor) -> None:
    """Drop the record of the given environments and cancel any pending append."""
    self.commands[env_ids] = 0.0
    self.start_times[env_ids] = 0.0
    self.lengths[env_ids] = 0
    self._pending[env_ids] = False

  def mark_pending(self, env_ids: torch.Tensor) -> None:
    """Mark the given environments for an append by the next :meth:`flush`."""
    self._pending[env_ids] = True

  def flush(self, commands: torch.Tensor, times: torch.Tensor) -> None:
    """Append the current command and time for every environment marked pending."""
    env_ids = self._pending.nonzero(as_tuple=False).flatten()
    if len(env_ids) == 0:
      return
    slots = self.lengths[env_ids]
    self.commands[env_ids, slots] = commands[env_ids]
    self.start_times[env_ids, slots] = times[env_ids]
    self.lengths[env_ids] += 1
    self._pending[env_ids] = False

  def durations(self, end_time: torch.Tensor | float) -> torch.Tensor:
    """Return how long each recorded segment lasted, shape (num_envs, capacity).

    A segment ends when the next one starts. The last segment is still open, so the
    caller decides when it ends: pass the current episode time to measure the episode
    as it happened, or the full episode length to hold the last command to the end.

    Args:
      end_time: When the last segment ends, per environment or shared by all of them.

    Returns:
      Segment durations, zero past :attr:`lengths`.
    """
    device = self.start_times.device
    end_times = torch.zeros_like(self.start_times)
    end_times[:, :-1] = self.start_times[:, 1:]
    last_slot = (self.lengths - 1).clamp(min=0)
    end_times[torch.arange(len(self.lengths), device=device), last_slot] = end_time
    slots = torch.arange(self.start_times.shape[1], device=device)
    return (end_times - self.start_times) * (slots < self.lengths.unsqueeze(-1))


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
    self.command_history = CommandHistory(
      num_envs=self.num_envs,
      # Worst case one command per shortest resampling interval, plus the one sampled
      # on episode reset.
      capacity=math.ceil(env.max_episode_length_s / cfg.resampling_time_range[0]) + 1,
      command_dim=self.command.shape[-1],
      device=self.device,
    )

  @property
  def episode_time(self) -> torch.Tensor:
    """Time elapsed in the current episode per environment, shape (num_envs,)."""
    return self._env.episode_length_buf * self._env.step_dt

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

    # The sampled command is not the tracked one yet: standing environments are zeroed,
    # heading environments get a closed-loop yaw rate and world-frame environments get
    # rotated, all in `_update_command`. Record only once the command has settled.
    self.command_history.mark_pending(env_ids)

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    assert isinstance(env_ids, torch.Tensor)
    # Clear first: the parent then resamples, marking the new episode's first append.
    self.command_history.clear(env_ids)
    return super().reset(env_ids)

  def compute(self, dt: float) -> None:
    super().compute(dt)
    self.command_history.flush(self.command, self.episode_time)


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
