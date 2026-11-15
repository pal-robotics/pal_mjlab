"""Metric terms of the stand-up task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .contacts import GroundContactCfg, StandCfg, get_ground_contact, is_standing

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def standing_metric(env: ManagerBasedRlEnv, stand_cfg: StandCfg) -> torch.Tensor:
  """1 while the robot is up."""
  return is_standing(env, stand_cfg).float()


def peak_body_ground_load(
  env: ManagerBasedRlEnv, contact_cfg: GroundContactCfg
) -> torch.Tensor:
  """Largest substep ground force on a body other than a foot, in robot weights."""
  contact = get_ground_contact(env, contact_cfg)
  load = (contact.peak_force / contact.weight)[:, ~contact.is_foot].max(dim=-1).values
  return torch.where(env.episode_length_buf <= 1, torch.zeros_like(load), load)


def peak_joint_speed(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Speed of the fastest of ``asset_cfg``'s joints."""
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_vel[:, asset_cfg.joint_ids].abs().max(dim=-1).values
