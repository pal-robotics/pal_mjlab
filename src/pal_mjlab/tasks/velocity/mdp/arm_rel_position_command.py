from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class UniformHandPositionCommand(CommandTerm):
  cfg: UniformHandPositionCommandCfg

  def __init__(self, cfg: UniformHandPositionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.hand_position_command = torch.zeros(self.num_envs, 3, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.hand_position_command

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.hand_position_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.x)
    self.hand_position_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.y)
    self.hand_position_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.z)

  def _update_command(self) -> None:
    pass


@dataclass(kw_only=True)
class UniformHandPositionCommandCfg(CommandTermCfg):

  @dataclass
  class Ranges:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformHandPositionCommand:
    return UniformHandPositionCommand(self, env)

  def __post_init__(self):
    pass