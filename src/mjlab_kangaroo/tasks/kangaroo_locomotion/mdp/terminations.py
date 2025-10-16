"""Useful methods for MPD terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def illegal_contacts(
    env: ManagerBasedRlEnv,
    sensor_names: list[str],
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate when the asset's designed sensors touch the ground."""
    asset: Entity = env.scene[asset_cfg.name]
    terminate = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for sensor_name in sensor_names:
        contact_force = asset.data.sensor_data[sensor_name]
        # print(contact_force)
        terminate |= (contact_force.abs() >= threshold).any(dim=1)
    return terminate
