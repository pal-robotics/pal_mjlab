from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg


if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class UniformBoxHeightCommand(CommandTerm):
  cfg: UniformBoxHeightCommandCfg

  def __init__(self, cfg: UniformBoxHeightCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.box: Entity = env.scene[cfg.entity_name]

    self.box_height_command = torch.zeros(self.num_envs, device=self.device)

    self.metrics["error_box_height"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.box_height_command

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_box_height"] += (
      torch.norm(
        self.box_height_command - self.box.data.root_link_pos_w[:, 2], dim=-1
      )
      / max_command_step
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.box_height_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)

  def _update_command(self) -> None:
    pass

@dataclass(kw_only=True)
class UniformBoxHeightCommandCfg(CommandTermCfg):
  entity_name: str

  @dataclass
  class Ranges:
    height: tuple[float, float]

  ranges: Ranges

  def build(self, env: ManagerBasedRlEnv) -> UniformBoxHeightCommand:
    return UniformBoxHeightCommand(self, env)

  def __post_init__(self):
    pass
