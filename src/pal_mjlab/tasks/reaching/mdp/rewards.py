
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_conjugate,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_mul,
    quat_unique,
    sample_uniform,
    quat_apply,
)
if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def position_command_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    site_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)

    des_pos_b = command.command[:, :3]

    root_pos_w = asset.data.body_link_pos_w[:, 0]  # Root body position
    root_quat_w = asset.data.body_link_quat_w[:, 0]  # Root body quaternion
    
    des_pos_w = root_pos_w + quat_apply(root_quat_w, des_pos_b)

    curr_pos_w = asset.data.site_pos_w[:, asset.site_names.index(site_name)]

    return torch.norm(curr_pos_w - des_pos_w, dim=1)
