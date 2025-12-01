from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity

from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from pal_mjlab.tasks.reaching_tiago.mdp.commands import LiftingCommand
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
    quat_conjugate,
    quat_mul,
    quat_error_magnitude
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

    root_pos_w = asset.data.site_pos_w[:, 0]  # Root site position
    root_quat_w = asset.data.site_quat_w[:, 0]  # Root site quaternion

    # Transform position: p_w = p_root + R_root * p_b
    pos_rotated = quat_mul(
        quat_mul(
            root_quat_w,
            torch.cat(
                [torch.zeros(env.num_envs, 1, device=env.device), des_pos_b], dim=1
            ),
        ),
        quat_conjugate(root_quat_w),
    )[:, 1:]  # Extract xyz from quaternion product
    des_pos_w = root_pos_w + pos_rotated

    current_site_pos_w = asset.data.site_pos_w[:, asset.site_names.index(site_name)]

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

    pos_rotated = quat_mul(
        quat_mul(
            root_quat_w,
            torch.cat(
                [torch.zeros(env.num_envs, 1, device=env.device), des_pos_b], dim=1
            ),
        ),
        quat_conjugate(root_quat_w),
    )[:, 1:]  # Extract xyz from quaternion product
    des_pos_w = root_pos_w + pos_rotated

    current_site_pos_w = asset.data.site_pos_w[:, asset.site_names.index(site_name)]

    pos_error = current_site_pos_w - des_pos_w
    distance = torch.norm(pos_error, dim=1)

    return 1 - torch.tanh(distance / std)

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
    excess = torch.relu(abs_error - 0.005) 

    # per-env penalty: sum of excess across all monitored joints
    penalty = torch.sum(excess, dim=1)
    return penalty

def staged_position_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  reaching_std: float,
  bringing_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Curriculum reward that gates lifting bonus on reaching progress.

  Returns reaching * (1 + bringing), where both terms are Gaussian kernels
  over position error. Ensures learning signal for approach before lift.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos_w = obj.data.root_link_pos_w
  reach_error = torch.sum(torch.square(ee_pos_w - obj_pos_w), dim=-1)
  reaching = torch.exp(-reach_error / reaching_std**2)
  position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
  bringing = torch.exp(-position_error / bringing_std**2)
  return reaching * (1.0 + bringing)


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


class action_rate_l2_louis:
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]

        _, joint_names = asset.find_joints(
            cfg.params["asset_cfg"].joint_names,
        )
        self._joint_ids = [
            asset.actuator_names.index(jname)
            for jname in joint_names
            if jname in asset.actuator_names
        ]

    def __call__(
        self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        return torch.sum(
            torch.square(
                env.action_manager.action[:, self._joint_ids]
                - env.action_manager.prev_action[:, self._joint_ids]
            ),
            dim=1,
        )
