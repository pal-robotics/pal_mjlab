from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity

from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
    quat_conjugate,
    quat_mul,
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

    joint_names = asset.joint_names  # either 1D list/array or (1, N)

    # # If it's (1, N), flatten it:
    # if hasattr(joint_names, "shape") and len(joint_names.shape) == 2:
    #     joint_names = joint_names[0]

    # default_q = asset.data.default_joint_pos[0]  # (num_joints,)
    # current_q = asset.data.joint_pos[0]          # (num_joints,)

    # print(asset.actuator_names)
    # print(asset.joint_names)

    # for name, q_def, q in zip(joint_names, default_q, current_q):
    #     print(f"{name:30s}  default={float(q_def): .5f}  current={float(q): .5f}")

    return 1 - torch.tanh(distance / std)


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
        # print(asset.actuator_names)
        # print(joint_names)
        # print(self._joint_ids)
        

    def __call__(
        self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]

        return torch.sum(
            torch.square(
                env.action_manager.action[:, self._joint_ids]
                - env.action_manager.prev_action[:, self._joint_ids]
            ),
            dim=1,
        )

