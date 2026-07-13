from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class GatedUniformVelocityCommand(UniformVelocityCommand):
  """Uniform velocity command with configurable floors for nonzero magnitudes."""

  cfg: GatedUniformVelocityCommandCfg  # type: ignore[assignment]

  def __init__(self, cfg: GatedUniformVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._minimum_magnitudes = torch.tensor(
      [cfg.min_lin_vel_x, cfg.min_lin_vel_y, cfg.min_ang_vel_z],
      device=self.device,
      dtype=self.vel_command_b.dtype,
    )

  def _gate(self, command: torch.Tensor) -> torch.Tensor:
    """Raise nonzero component magnitudes while preserving their sampled signs."""
    return torch.sign(command) * torch.maximum(command.abs(), self._minimum_magnitudes)

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
    self.vel_command_b[moving_ids] = self._gate(self.vel_command_b[moving_ids])

  def _update_command(self) -> None:
    super()._update_command()
    moving_ids = (~self.is_standing_env).nonzero(as_tuple=False).flatten()
    self.vel_command_b[moving_ids] = self._gate(self.vel_command_b[moving_ids])


@dataclass(kw_only=True)
class GatedUniformVelocityCommandCfg(UniformVelocityCommandCfg):
  """Per-axis magnitude floors; omitted axes default to zero and stay ungated."""

  rel_straight_envs: float = 0.0
  """Fraction of resampled commands constrained to signed x-only motion."""

  min_lin_vel_x: float = 0.0
  """Magnitude floor for x commands."""

  min_lin_vel_y: float = 0.0
  """Magnitude floor for y commands."""

  min_ang_vel_z: float = 0.0
  """Magnitude floor for yaw commands."""

  def __post_init__(self) -> None:
    super().__post_init__()
    minima = {
      "min_lin_vel_x": self.min_lin_vel_x,
      "min_lin_vel_y": self.min_lin_vel_y,
      "min_ang_vel_z": self.min_ang_vel_z,
    }
    for name, value in minima.items():
      if value < 0.0:
        raise ValueError(f"{name} must be non-negative; got {value}.")
    if self.init_velocity_prob != 0.0:
      raise ValueError(
        "GatedUniformVelocityCommandCfg post-processes commands after the parent "
        "initializes velocity; set init_velocity_prob=0.0."
      )
    if self.rel_forward_envs != 0.0:
      raise ValueError(
        "GatedUniformVelocityCommandCfg replaces rel_forward_envs with "
        "rel_straight_envs; set rel_forward_envs=0.0."
      )

  def build(self, env: ManagerBasedRlEnv) -> GatedUniformVelocityCommand:
    return GatedUniformVelocityCommand(self, env)
