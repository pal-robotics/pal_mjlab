from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

_ACTUATED_LEG_JOINT_RE_1 = (
    r"|.*_leg_length_slider$"
)

_ACTUATED_LEG_JOINT_RE_2 = (
    r".*_hip_z_slider$"
    r"|.*_hip_xy_slider_l$"
    r"|.*_hip_xy_slider_r$"
    r"|.*_ankle_xy_slider_l$"
    r"|.*_ankle_xy_slider_r$"
)


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
    excess = torch.relu(abs_error - 0.25)

    # per-env penalty: sum of excess across all monitored joints
    penalty = torch.sum(excess, dim=1)
    return penalty

def  joint_vel_limit (
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    limit_scale : float = 1.0,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    limit_leg_length = limit_scale * 0.0026     # 0.26 cm
    limit_leg_sliders = limit_scale * 0.0020    # 0.20 cm

    # Only work with the actuated joints
    leg_ids_1, _ = asset.find_joints_by_actuator_names(_ACTUATED_LEG_JOINT_RE_1)
    leg_ids_2, _ = asset.find_joints_by_actuator_names(_ACTUATED_LEG_JOINT_RE_2)

    joint_velocity_1 = asset.data.joint_vel[:,leg_ids_1]
    joint_velocity_2 = asset.data.joint_vel[:,leg_ids_2]

    # Flat penalty in between the limits, quadratic outside
    over_limit_1 = torch.where(joint_velocity_1 > limit_leg_length, joint_velocity_1 - limit_leg_length, torch.zeros_like(joint_velocity_1)) + torch.where(joint_velocity_1 < -limit_leg_length, -limit_leg_length - joint_velocity_1, torch.zeros_like(joint_velocity_1))
    over_limit_squared_1 = torch.square(over_limit_1)

    over_limit_2 = torch.where(joint_velocity_2 > limit_leg_sliders, joint_velocity_2 - limit_leg_sliders, torch.zeros_like(joint_velocity_2)) + torch.where(joint_velocity_2 < -limit_leg_sliders, -limit_leg_sliders - joint_velocity_2, torch.zeros_like(joint_velocity_2))
    over_limit_squared_2 = torch.square(over_limit_2)

    penalty = torch.sum(over_limit_squared_1, dim=1) + torch.sum(over_limit_squared_2, dim=1)
    return penalty # Positive, scaled to negative with weight




