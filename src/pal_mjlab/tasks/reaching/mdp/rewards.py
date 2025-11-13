
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


from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.third_party.isaaclab.isaaclab.utils.string import (
  resolve_matching_names_values,
)


def position_command_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    site_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)

    des_pos_b = command.command[:, :3]

    root_pos_w = asset.data.site_pos_w[:, 0]  # Root site position
    root_quat_w = asset.data.site_quat_w[:, 0]  # Root site quaternion

    # Transform position: p_w = p_root + R_root * p_b
    pos_rotated = quat_mul(
        quat_mul(root_quat_w, torch.cat([torch.zeros(env.num_envs, 1, device=env.device), 
                                            des_pos_b], dim=1)),
        quat_conjugate(root_quat_w)
    )[:, 1:]  # Extract xyz from quaternion product
    des_pos_w = root_pos_w + pos_rotated


    # Get current site pose
    current_site_pos_w = asset.data.site_pos_w[:, asset.site_names.index(site_name)]
    current_site_quat_w = asset.data.site_quat_w[:, asset.site_names.index(site_name)]

    # Compute position error
    pos_error = current_site_pos_w - des_pos_w

    return torch.norm(pos_error, dim=1)

def position_command_error_tanh(
    env: ManagerBasedRlEnv,
    command_name: str,
    site_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)

    des_pos_b = command.command[:, :3]

    root_pos_w = asset.data.site_pos_w[:, 0]  # Root site position
    root_quat_w = asset.data.site_quat_w[:, 0]  # Root site quaternion

    # Transform position: p_w = p_root + R_root * p_b
    pos_rotated = quat_mul(
        quat_mul(root_quat_w, torch.cat([torch.zeros(env.num_envs, 1, device=env.device), 
                                            des_pos_b], dim=1)),
        quat_conjugate(root_quat_w)
    )[:, 1:]  # Extract xyz from quaternion product
    des_pos_w = root_pos_w + pos_rotated


    # Get current site pose
    current_site_pos_w = asset.data.site_pos_w[:, asset.site_names.index(site_name)]
    current_site_quat_w = asset.data.site_quat_w[:, asset.site_names.index(site_name)]

    # Compute position error
    pos_error = current_site_pos_w - des_pos_w
    distance = torch.norm(pos_error, dim=1)

    ########################3
    return 1 - torch.tanh(distance / std)
    # return torch.exp(-distance / 0.08)


class action_rate_l2_louis:

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]

    _, joint_names = asset.find_joints(
      cfg.params["asset_cfg"].joint_names,
    )
    self._joint_ids = [asset.actuator_names.index(jname) for jname in joint_names 
                if jname in asset.actuator_names]

  def __call__(
    self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    # print(env.action_manager.action[:, self._joint_ids])

    return torch.sum(
        torch.square(env.action_manager.action[:, self._joint_ids] - env.action_manager.prev_action[:, self._joint_ids]), dim=1
    )