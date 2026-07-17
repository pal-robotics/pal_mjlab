from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg


if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class UniformGripperManipulationCommand(CommandTerm):
  cfg: UniformGripperManipulationCommandCfg

  def __init__(self, cfg: UniformGripperManipulationCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.box: Entity = env.scene[cfg.entity_name]

    self.rel_box_position_command = torch.zeros((self.num_envs, 3,), device=self.device)

    self.metrics["error_rel_box_pos_x"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_rel_box_pos_y"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_rel_box_pos_z"] = torch.zeros(self.num_envs, device=self.device)


  @property
  def command(self) -> torch.Tensor:
    return self.rel_box_position_command

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    error = self.rel_box_position_command - self.box.data.root_link_pos_w
    self.metrics["error_rel_box_pos_x"] += (
      torch.norm(
        error[:, 0], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_rel_box_pos_y"] += (
      torch.norm(
        error[:, 1], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_rel_box_pos_z"] += (
      torch.norm(
        error[:, 2], dim=-1
      )
      / max_command_step
    )


  def _resample_command(self, env_ids: torch.Tensor) -> None:
    r = torch.empty(len(env_ids), device=self.device)
    self.rel_box_position_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.x)
    self.rel_box_position_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.y)
    self.rel_box_position_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.z)

  def _update_command(self) -> None:
    pass

@dataclass(kw_only=True)
class UniformGripperManipulationCommandCfg(CommandTermCfg):
  entity_name: str

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

  def build(self, env: ManagerBasedRlEnv) -> UniformGripperManipulationCommand:
    return UniformGripperManipulationCommand(self, env)

  def __post_init__(self):
    pass