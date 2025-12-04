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

def object_out_of_bounds_box(
    env: ManagerBasedRlEnv,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    object_name: str = "cube",
) -> torch.Tensor:
    """Return True if object center leaves the [x_min, x_max] x [y_min, y_max] box."""
    obj: Entity = env.scene[object_name]
    pos = obj.data.root_link_pos_w[:, :2]   # [N, 2] -> x, y

    x = pos[:, 0]
    y = pos[:, 1]

    out_x = (x < x_min) | (x > x_max)
    out_y = (y < y_min) | (y > y_max)

    return out_x | out_y   # [N] bool
