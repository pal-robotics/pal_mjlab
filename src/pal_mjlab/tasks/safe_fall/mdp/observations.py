"""Privileged (critic) observation terms for the safe-fall task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .impact import ImpactCfg, get_impact_tracker

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def body_ground_load(env: ManagerBasedRlEnv, impact_cfg: ImpactCfg) -> torch.Tensor:
  """[B, N] ground load per body in robot weights, log-compressed."""
  tracker = get_impact_tracker(env, impact_cfg)
  just_reset = (env.episode_length_buf == 0).unsqueeze(-1)
  load = torch.log1p(tracker.peak_force / tracker.weight_force)
  return torch.where(just_reset, torch.zeros_like(load), load)


def impact_running_max(env: ManagerBasedRlEnv, impact_cfg: ImpactCfg) -> torch.Tensor:
  """[B, 1] largest weighted impulse so far in the episode.

  The damage reward depends on this value, so the critic needs it for the
  reward to be Markov.
  """
  return get_impact_tracker(env, impact_cfg).episode_running_max().unsqueeze(-1)


def base_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """[B, 1] height of the root link above its environment origin."""
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return height.unsqueeze(-1)


def episode_time_left(env: ManagerBasedRlEnv) -> torch.Tensor:
  """[B, 1] remaining fraction of the episode, 1 at the start."""
  left = 1.0 - env.episode_length_buf.float() / env.max_episode_length
  return left.clamp(0.0, 1.0).unsqueeze(-1)
