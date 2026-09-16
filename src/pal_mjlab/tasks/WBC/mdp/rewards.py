from __future__ import annotations

from typing import TYPE_CHECKING, cast

# from mjlab.tests.test_runner import env
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg


from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def motion_joint_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float | None = None,
  *,
  kappa: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  jnt_ids = asset_cfg.joint_ids
  ref_joint = command.joint_pos[:, jnt_ids]
  robot_joint = command.robot_joint_pos[:, jnt_ids]
  sq_err = torch.square(ref_joint - robot_joint)

  return torch.exp(-kappa * torch.sum(sq_err, dim=-1) / std**2)


def motion_joint_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float | None = 1.0,
  *,
  kappa: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  jnt_ids = asset_cfg.joint_ids
  sq_err = torch.square(
    command.joint_vel[:, jnt_ids] - command.robot_joint_vel[:, jnt_ids]
  )

  return torch.exp(-kappa * torch.sum(sq_err, dim=-1) / std**2)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  *,
  axes: str = "xy",
) -> torch.Tensor:
  angmom = env.scene[sensor_name].data
  if axes == "xy":
    return torch.sum(torch.square(angmom[..., :2]), dim=-1)
  if axes == "xyz":
    return torch.sum(torch.square(angmom), dim=-1)
  raise ValueError(f"Unsupported axes {axes!r}; use 'xy' or 'xyz'.")