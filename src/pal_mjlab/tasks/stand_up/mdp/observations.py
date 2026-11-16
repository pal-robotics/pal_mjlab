"""Observation terms of the stand-up task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .contacts import (
  GroundContactCfg,
  StandCfg,
  base_height,
  get_ground_contact,
  torso_up,
)
from .events import get_start_action

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def last_action_from_start(env: ManagerBasedRlEnv) -> torch.Tensor:
  """The last action; before the first step of an episode, the action that
  holds the pose the robot starts in."""
  first = (env.episode_length_buf == 0).unsqueeze(-1)
  return torch.where(first, get_start_action(env), env.action_manager.action)


def body_ground_load(
  env: ManagerBasedRlEnv, contact_cfg: GroundContactCfg
) -> torch.Tensor:
  """[B, N] ground force per body in robot weights, log-compressed."""
  contact = get_ground_contact(env, contact_cfg)
  load = torch.log1p(contact.peak_force / contact.weight)
  # The force history of a freshly reset environment is empty.
  just_reset = (env.episode_length_buf == 0).unsqueeze(-1)
  return torch.where(just_reset, torch.zeros_like(load), load)


def base_height_above_ground(
  env: ManagerBasedRlEnv, stand_cfg: StandCfg
) -> torch.Tensor:
  return base_height(env, stand_cfg).unsqueeze(-1)


def torso_tilt_cos(env: ManagerBasedRlEnv, stand_cfg: StandCfg) -> torch.Tensor:
  return torso_up(env, stand_cfg).unsqueeze(-1)
