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


# Command properties

def ref_base_height(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)

  base_height = command.ref_base_height

  return base_height

def ref_base_lin_vel_b(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)

  base_lin_vel_b = command.ref_base_lin_vel_b

  return base_lin_vel_b

def ref_base_ang_vel_b(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)

  base_ang_vel_b = command.ref_base_ang_vel_b

  return base_ang_vel_b

def ref_gravity_b(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)

  gravity_b = command.ref_gravity_b

  return gravity_b

def ref_joint_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)

  joint_pos = command.joint_pos

  return joint_pos

def ref_joint_vel(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)

  joint_vel = command.joint_vel

  return joint_vel

def _body_lin_vel_in_anchor_frame(
  anchor_quat_w: torch.Tensor,
  body_lin_vel_w: torch.Tensor,
) -> torch.Tensor:
  num_envs, num_bodies, _ = body_lin_vel_w.shape
  quat = anchor_quat_w[:, None, :].expand(num_envs, num_bodies, 4).reshape(-1, 4)
  vel = body_lin_vel_w.reshape(-1, 3)
  vel_b = quat_apply_inverse(quat, vel)
  return vel_b.view(num_envs, num_bodies * 3)


def _body_ang_vel_in_anchor_frame(
  anchor_quat_w: torch.Tensor,
  body_ang_vel_w: torch.Tensor,
) -> torch.Tensor:
  num_envs, num_bodies, _ = body_ang_vel_w.shape
  quat = anchor_quat_w[:, None, :].expand(num_envs, num_bodies, 4).reshape(-1, 4)
  vel = body_ang_vel_w.reshape(-1, 3)
  vel_b = quat_apply_inverse(quat, vel)
  return vel_b.view(num_envs, num_bodies * 3)

def motion_body_lin_vel(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Actual keybody linear velocities in the robot anchor frame."""
  command = env.command_manager.get_term(command_name)
  return _body_lin_vel_in_anchor_frame(
    command.robot_anchor_quat_w, command.robot_body_lin_vel_w
  )


def motion_body_ang_vel(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Actual keybody angular velocities in the robot anchor frame."""
  command = env.command_manager.get_term(command_name)
  return _body_ang_vel_in_anchor_frame(
    command.robot_anchor_quat_w, command.robot_body_ang_vel_w
  )


def ref_base_lin_acc_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference anchor linear acceleration in anchor frame (critic privileged)."""
  return env.command_manager.get_term(command_name).ref_base_lin_acc_b


def ref_base_ang_acc_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference anchor angular acceleration in anchor frame (critic privileged)."""
  return env.command_manager.get_term(command_name).ref_base_ang_acc_b
