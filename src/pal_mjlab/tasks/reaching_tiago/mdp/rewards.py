from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity

from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
    quat_conjugate,
    quat_mul,
    quat_error_magnitude
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

def orientation_command_error(
    env: "ManagerBasedRlEnv",
    command_name: str,
    site_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    # Get robot entity and command
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)

    # Desired orientation in base frame (qw, qx, qy, qz)
    des_quat_b = command.command[:, 3:] 

    # Root (base) orientation in world frame
    root_quat_w = asset.data.site_quat_w[:, 0] 

    # Transform desired orientation from base -> world:
    # q_des_w = q_root_w ⊗ q_des_b
    des_quat_w = quat_mul(root_quat_w, des_quat_b)  

    # Current site orientation in world frame
    site_idx = asset.site_names.index(site_name)
    current_quat_w = asset.data.site_quat_w[:, site_idx]  

    # Quaternion error magnitude (angle between quaternions)
    ori_error = quat_error_magnitude(des_quat_w, current_quat_w)  

    return ori_error

def stand_still_joint_deviation_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    abs_error = torch.abs(error)

    # amount beyond the 0.1 margin
    excess = torch.relu(abs_error - 0.001) 

    # per-env penalty: sum of excess across all monitored joints
    penalty = torch.sum(excess, dim=1)
    return penalty

def joint_velocity_hinge_penalty(
  env: ManagerBasedRlEnv,
  max_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Quadratic hinge penalty on joint velocities exceeding a symmetric limit.
  Penalizes only the amount by which |v| exceeds max_vel. Returns a negative
  penalty, shaped as the negative squared L2 norm of the excess velocities.
  """
  robot: Entity = env.scene[asset_cfg.name]
  joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  excess = (joint_vel.abs() - max_vel).clamp_min(0.0)
  return (excess**2).sum(dim=-1)
