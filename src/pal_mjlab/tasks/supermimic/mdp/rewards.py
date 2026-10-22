"""Tracking reward of SuperMimic (Table A.2): BeyondMimic's terms without self-collision.

Each term is ``exp(-error / std**2)`` of a squared error from ``tracking.py``; the error
functions are the ones imagination will differentiate through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from . import tracking
from .commands import MotionLibraryCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "motion_root_pos_exp",
  "motion_root_ori_exp",
  "motion_body_pos_exp",
  "motion_body_ori_exp",
  "motion_body_lin_vel_exp",
  "motion_body_ang_vel_exp",
  "joint_pos_limits_hard",
]

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _exp_term(
  env: ManagerBasedRlEnv, command_name: str, error_fn, std: float
) -> torch.Tensor:
  command = cast(MotionLibraryCommand, env.command_manager.get_term(command_name))
  return torch.exp(-error_fn(command.robot_bodies, command.ref_bodies) / std**2)


def motion_root_pos_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  return _exp_term(env, command_name, tracking.anchor_pos_error, std)


def motion_root_ori_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  return _exp_term(env, command_name, tracking.anchor_ori_error, std)


def motion_body_pos_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  return _exp_term(env, command_name, tracking.body_pos_error, std)


def motion_body_ori_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  return _exp_term(env, command_name, tracking.body_ori_error, std)


def motion_body_lin_vel_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  return _exp_term(env, command_name, tracking.body_lin_vel_error, std)


def motion_body_ang_vel_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  return _exp_term(env, command_name, tracking.body_ang_vel_error, std)


def joint_pos_limits_hard(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Distance outside the hard joint limits (D12: references often exceed soft limits)."""
  asset: Entity = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  limits = asset.data.joint_pos_limits[:, ids]
  return tracking.joint_limit_violation(
    asset.data.joint_pos[:, ids], limits[..., 0], limits[..., 1]
  )
