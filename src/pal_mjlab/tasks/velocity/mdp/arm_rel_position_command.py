from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from .safe_position_sampling import push_xy_outside_radius, validate_base_position

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class UniformHandPositionCommand(CommandTerm):
  cfg: UniformHandPositionCommandCfg

  def __init__(self, cfg: UniformHandPositionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.hand_position_command = torch.zeros(self.num_envs, 3, device=self.device)

    self.is_base_position_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return self.hand_position_command

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    r = self.cfg.ranges
    x = torch.empty(n, device=self.device).uniform_(*r.x)
    y = torch.empty(n, device=self.device).uniform_(*r.y)
    xy = push_xy_outside_radius(torch.stack((x, y), dim=-1), self.cfg.min_xy_radius)

    self.hand_position_command[env_ids, :2] = xy
    self.hand_position_command[env_ids, 2] = torch.empty(n, device=self.device).uniform_(*r.z)

    self.is_base_position_env[env_ids] = (
      torch.empty(n, device=self.device).uniform_(0.0, 1.0) <= self.cfg.rel_base_position
    )

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    base_position_ids = self.is_base_position_env.nonzero(as_tuple=False).flatten()
    self.hand_position_command[base_position_ids] = torch.tensor(self.cfg.base_position, device=self.device)

@dataclass(kw_only=True)
class UniformHandPositionCommandCfg(CommandTermCfg):

  @dataclass
  class Ranges:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

  ranges: Ranges

  base_position : tuple[float, float, float]

  rel_base_position: float = 0.1
  min_xy_radius: float = 0.2
  
  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformHandPositionCommand:
    return UniformHandPositionCommand(self, env)

  def __post_init__(self):
    validate_base_position(self.base_position, self.min_xy_radius)
