"""Extension of mjlab's joint position action"""
from __future__ import annotations

from mjlab.envs.mdp.actions.actions import JointPositionActionCfg, JointPositionAction

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers import SceneEntityCfg


if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class PelvisActionCfg (JointPositionActionCfg) :

    command_name: str
    """Name of the command that will feed the targets to the action"""

    def build(self, env: ManagerBasedRlEnv) -> PelvisAction:
        return PelvisAction(self, env)

class PelvisAction(JointPositionAction):

  cfg: PelvisActionCfg
  
  @property
  def action_dim(self) -> int:
    return 0

  def process_actions(self, actions: torch.Tensor):
    """Process raw actions by applying scale, offset, and optional clip."""
    del actions

    command_actions = self._env.command_manager.get_command(self.cfg.command_name)

    self._raw_actions = command_actions

    self._processed_actions = self._raw_actions * self._scale + self._offset
    if self.cfg.clip is not None:
      self._processed_actions = torch.clamp(
        self._processed_actions,
        min=self._clip[:, :, 0],
        max=self._clip[:, :, 1],
      )
  
  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Reset raw actions to zero for specified environments."""
    self._raw_actions[env_ids] = 0.0