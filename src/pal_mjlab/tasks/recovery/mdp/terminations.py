"""Useful methods for MDP terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.nan_guard import NanGuard

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")




def over_gyro(
  env: ManagerBasedRlEnv,
  limit_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
):
  """Terminate when the asset's angular velocity exceeds the limit."""
  asset: Entity = env.scene[asset_cfg.name]
  angular_velocity = asset.data.root_link_vel_w[:, 3:6]
#   print(torch.norm(angular_velocity, dim=1))
  return torch.norm(angular_velocity, dim=1) > limit_vel
