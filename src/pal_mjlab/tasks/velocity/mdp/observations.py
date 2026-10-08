"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
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
  gravity_w = torch.tensor([[0.0, 0.0, -1.0]], device=imu_quat.device).expand(
    imu_quat.shape[0], -1
  )
  # print(f"imu proj{quat_apply_inverse(imu_quat, gravity_w)}")
  # asset: Entity = env.scene[_DEFAULT_ASSET_CFG.name]
  # print(f"proj{asset.data.projected_gravity_b}")
  # Project to IMU frame (same as your C++ code)
  return quat_apply_inverse(imu_quat, gravity_w)


def ref_base_lin_vel_b(env: ManagerBasedRlEnv, command_name : str) -> torch.Tensor:

  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  ref = torch.zeros((env.num_envs, 3,), device=env.device)
  ref[:, :2] = command[:, :2]

  return ref

def ref_base_ang_vel_b(env: ManagerBasedRlEnv, command_name : str) -> torch.Tensor:

  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  ref = torch.zeros((env.num_envs, 3,), device=env.device)
  ref[:, 2] = command[:, 2]
  
  return ref

def ref_gravity_b(env: ManagerBasedRlEnv) -> torch.Tensor:

  ref = torch.zeros((env.num_envs, 3,), device=env.device)
  ref[:, 2] = -1.0
  
  return ref


def ref_base_height(env: ManagerBasedRlEnv, command_name : str) -> torch.Tensor:

  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  
  return command[:, 6]

def ref_body_pos_b(env: ManagerBasedRlEnv, command_name : str) -> torch.Tensor:

  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  
  return command[:, :6]