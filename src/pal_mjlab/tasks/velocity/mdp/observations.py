"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, matrix_from_quat

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


def ref_base_height(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  return command[:, 6:7].clone()

def ref_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:

  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  
  return command[:, :6].clone()

def ref_body_pos_pose_b(
  env: ManagerBasedRlEnv, command_name: str, num_bodies: int = 2
) -> torch.Tensor:
  #return torch.tensor([[ 0.2573, -0.5842, -0.1350,  0.2442,  0.5224, -0.2373]], device=env.device)
  """Position-only part of the pose command. (N, 3 * n)"""
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  ref = command[:, : 7 * num_bodies].view(env.num_envs, num_bodies, 7)
  return ref[..., :3].reshape(env.num_envs, -1)


def ref_body_ori_pose_b(
  env: ManagerBasedRlEnv, command_name: str, num_bodies: int = 2
) -> torch.Tensor:
  #return torch.tensor([[-0.4289,  0.6891,  0.5673,  0.7086, -0.7030,  0.1514,  0.4920,  0.7081, 0.7329, -0.6509,  0.4699,  0.2739]], device=env.device)
  """Hand target orientations as 6D rotation (first two columns of R), base frame. (N, 6 * n)"""
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  ref = command[:, : 7 * num_bodies].view(env.num_envs, num_bodies, 7)
  mat = matrix_from_quat(ref[..., 3:7])  # (N, n, 3, 3)
  return mat[..., :2].reshape(env.num_envs, -1)


def ref_base_height_pose(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  #return torch.tensor([[0.95]], device=env.device)
  """Commanded base height (last command entry). (N, 1)"""
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  return command[:, -1:].clone()