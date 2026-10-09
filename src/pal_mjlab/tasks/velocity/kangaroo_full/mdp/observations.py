"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Root state.
##

def imu_projected_gravity(
    env: ManagerBasedRlEnv,
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