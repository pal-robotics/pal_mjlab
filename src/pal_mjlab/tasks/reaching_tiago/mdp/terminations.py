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

def object_out_of_reach(env, max_radius, object_name="cube") -> torch.Tensor:
    obj = env.scene[object_name]
    pos = obj.data.root_link_pos_w[:, :2]  # x, y in workspace
    dist_xy = torch.norm(pos, dim=1)
    return dist_xy > max_radius
