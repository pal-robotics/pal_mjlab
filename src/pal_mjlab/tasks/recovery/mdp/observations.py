from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedEnv
    
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.sensor import BuiltinSensor
_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

def imu_projected_gravity(
    env: ManagerBasedEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Get projected gravity from IMU sensor orientation (accounts for IMU mounting)."""
    sensor = env.scene[sensor_name]
    assert isinstance(sensor, BuiltinSensor)

    # Get IMU orientation (already includes mounting offset)
    imu_quat = sensor.data  # or however you access orientation

    # Gravity in world frame
    gravity_w = torch.tensor([[0.0, 0.0, -1.0]], device=imu_quat.device).expand(imu_quat.shape[0], -1)
    # print(f"imu proj{quat_apply_inverse(imu_quat, gravity_w)}")
    # asset: Entity = env.scene[_DEFAULT_ASSET_CFG.name]
    # print(f"proj{asset.data.projected_gravity_b}")
    # Project to IMU frame (same as your C++ code)
    return quat_apply_inverse(imu_quat, gravity_w)

def torso_height_obs(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2:3]

def head_to_foot_delta_xyz(
    env: ManagerBasedEnv,
    head_name: str = "head",  # either link or site
    left_foot_name: str = "leg_left_foot_link",
    right_foot_name: str = "leg_right_foot_link",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    
    # Determine if head_name is a site or a body
    if head_name in asset.site_names:
        head_ids, _ = asset.find_sites(head_name)
        head_pos = asset.data.site_pos_w[:, head_ids[0], :]
    elif head_name in asset.body_names:
        head_ids, _ = asset.find_bodies(head_name)
        head_pos = asset.data.body_link_pos_w[:, head_ids[0], :]
    else:
        raise ValueError(f"'{head_name}' not found in sites or bodies")
    
    # Get foot positions (assuming they are bodies)
    left_foot_ids, _ = asset.find_bodies(left_foot_name)
    right_foot_ids, _ = asset.find_bodies(right_foot_name)
    
    left_foot_pos = asset.data.body_link_pos_w[:, left_foot_ids[0], :]
    right_foot_pos = asset.data.body_link_pos_w[:, right_foot_ids[0], :]
    
    # Compute differences
    left_diff = head_pos - left_foot_pos
    right_diff = head_pos - right_foot_pos
    
    return torch.cat([left_diff, right_diff], dim=-1)

def head_pos(
    env: ManagerBasedEnv,
    head_name: str = "head",  # either link or site
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    
    # Determine if head_name is a site or a body
    if head_name in asset.site_names:
        head_ids, _ = asset.find_sites(head_name)
        head_pos = asset.data.site_pos_w[:, head_ids[0], :]
    elif head_name in asset.body_names:
        head_ids, _ = asset.find_bodies(head_name)
        head_pos = asset.data.body_link_pos_w[:, head_ids[0], :]
    else:
        raise ValueError(f"'{head_name}' not found in sites or bodies")

    return head_pos