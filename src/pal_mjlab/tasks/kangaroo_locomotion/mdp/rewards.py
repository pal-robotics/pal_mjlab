from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def torso_height(
    env: ManagerBasedRlEnv,
    z_des: float,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    z = asset.data.root_link_pos_w[:, 2]
    z_err = z - z_des

    # As before: penalize being too low more than being too high
    z_err_scaled = torch.where(z_err < 0, z_err, z_err * 0.25)

    # Squared error
    z_err_squared = torch.square(z_err_scaled)

    # Height penalty: 0 when perfect, >0 as we deviate
    penalty = z_err_squared / (std**2)

    env.extras["log"]["Metrics/mean_height"] = torch.mean(z)
    env.extras["log"]["Metrics/mean_height_penalty"] = torch.mean(penalty)

    return penalty
