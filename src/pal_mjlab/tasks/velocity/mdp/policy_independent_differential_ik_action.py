"""Extension of mjlab's differential IK to read a specific command"""
from __future__ import annotations

from mjlab.envs.mdp.actions.differential_ik import DifferentialIKActionCfg, DifferentialIKAction

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  apply_delta_pose,
  quat_apply,
  combine_frame_transforms
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv



@dataclass(kw_only=True)
class PolicyIndependentDifferentialIKActionCfg (DifferentialIKActionCfg) :

    command_name: str
    """Name of the command that will feed the targets to the action"""

    asset_cfg: SceneEntityCfg
    """Robot cfg"""

    def build(self, env: ManagerBasedRlEnv) -> PolicyIndependentDifferentialIKAction:
        return PolicyIndependentDifferentialIKAction(self, env)

class PolicyIndependentDifferentialIKAction(DifferentialIKAction):

    cfg: PolicyIndependentDifferentialIKActionCfg
   
    @property
    def action_dim(self) -> int:
        return 0

    def process_actions(self, actions: torch.Tensor) -> None:
        del actions

        command_actions = self._env.command_manager.get_command(self.cfg.command_name)
        asset: Entity = self._env.scene[self.cfg.asset_cfg.name]

        self._raw_actions[:] = command_actions
    
        frame_pos, frame_quat = self._get_frame_pose()

        asset_pos_w = asset.data.root_link_pos_w
        asset_quat_w = asset.data.root_link_quat_w

        if self._action_dim == 3:
          if self.cfg.use_relative_mode:
            self._desired_pos[:] = frame_pos + command_actions * self.cfg.delta_pos_scale
          else:
            self._desired_pos[:] = asset_pos_w + quat_apply(asset_quat_w, command_actions)
          self._desired_quat[:] = frame_quat
        elif self._action_dim == 6:
          delta = command_actions.clone()
          delta[:, :3] *= self.cfg.delta_pos_scale
          delta[:, 3:] *= self.cfg.delta_ori_scale
          target_pos, target_quat = apply_delta_pose(frame_pos, frame_quat, delta)
          self._desired_pos[:] = target_pos
          self._desired_quat[:] = target_quat
        else:
          assert self._action_dim == 7
          target_pos, target_quat = combine_frame_transforms(
                asset_pos_w, asset_quat_w,
                command_actions[:, :3], command_actions[:, 3:7],
            )
          self._desired_pos[:] = target_pos
          self._desired_quat[:] = target_quat