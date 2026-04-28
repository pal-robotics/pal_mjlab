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


_JOINT_NAMES_FOR_DP = (
    "arm_left_1_joint",
    "arm_left_2_joint",
    "arm_left_3_joint",
    "arm_left_4_joint",
    "arm_right_1_joint",
    "arm_right_2_joint",
    "arm_right_3_joint",
    "arm_right_4_joint",
    "leg_left_1_joint",
    "leg_left_2_joint",
    "leg_left_3_joint",
    "leg_left_4_joint",
    "leg_left_5_joint",
    "leg_left_femur_joint",
    "leg_left_knee_joint",
    "leg_left_length_joint",
    "leg_right_1_joint",
    "leg_right_2_joint",
    "leg_right_3_joint",
    "leg_right_4_joint",
    "leg_right_5_joint",
    "leg_right_femur_joint",
    "leg_right_knee_joint",
    "leg_right_length_joint",
    "pelvis_1_joint",
    "pelvis_2_joint",
)

# WITH NO KNEE NOR FEMUR
_JOINT_NAMES_FOR_DP_2 = (
    "arm_left_1_joint",
    "arm_left_2_joint",
    "arm_left_3_joint",
    "arm_left_4_joint",
    "arm_right_1_joint",
    "arm_right_2_joint",
    "arm_right_3_joint",
    "arm_right_4_joint",
    "leg_left_1_joint",
    "leg_left_2_joint",
    "leg_left_3_joint",
    "leg_left_4_joint",
    "leg_left_5_joint",
    "leg_left_length_joint",
    "leg_right_1_joint",
    "leg_right_2_joint",
    "leg_right_3_joint",
    "leg_right_4_joint",
    "leg_right_5_joint",
    "leg_right_length_joint",
    "pelvis_1_joint",
    "pelvis_2_joint",
)


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
  gravity_w = torch.tensor([[0.0, 0.0, -1.0]], device=imu_quat.device).expand(
    imu_quat.shape[0], -1
  )
  # print(f"imu proj{quat_apply_inverse(imu_quat, gravity_w)}")
  # asset: Entity = env.scene[_DEFAULT_ASSET_CFG.name]
  # print(f"proj{asset.data.projected_gravity_b}")
  # Project to IMU frame (same as your C++ code)
  return quat_apply_inverse(imu_quat, gravity_w)

def joint_pos_dp(
  env: ManagerBasedRlEnv,
  biased: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  joint_pos = asset.data.joint_pos_biased if biased else asset.data.joint_pos

  sim_names = asset.joint_names

  canonical_names = _JOINT_NAMES_FOR_DP

  name_to_idx = {name: i for i, name in enumerate(sim_names)}

  ordered_ids = []
  for name in canonical_names:
    if name not in name_to_idx:
      raise ValueError(f"Joint not found in sim: {name}")
    ordered_ids.append(name_to_idx[name])

  return joint_pos[:, ordered_ids]

def joint_vel_dp(
  env: ManagerBasedRlEnv,
  biased: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  joint_vel = asset.data.joint_vel_biased if biased else asset.data.joint_vel

  sim_names = asset.joint_names

  canonical_names = _JOINT_NAMES_FOR_DP

  name_to_idx = {name: i for i, name in enumerate(sim_names)}

  ordered_ids = []
  for name in canonical_names:
    if name not in name_to_idx:
      raise ValueError(f"Joint not found in sim: {name}")
    ordered_ids.append(name_to_idx[name])

  return joint_vel[:, ordered_ids]
