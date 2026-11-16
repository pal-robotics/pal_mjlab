"""Termination terms of the stand-up task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .contacts import StandCfg, get_ground_contact, is_standing, torso_up

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.termination_manager import TerminationTermCfg


class fell_after_standing:
  """The robot is down again after it stood for ``min_standing_s``.

  Down: a body other than a foot touches the ground, or the torso is tilted
  more than ``max_tilt`` [rad]. Getting up from a fall is already trained by
  the fallen starts; this ends the episodes that lose the standing they had.
  """

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    del cfg
    zeros = torch.zeros(env.num_envs, device=env.device)
    self._standing_time = zeros
    self._stood = zeros.bool()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._standing_time[env_ids] = 0.0
    self._stood[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    stand_cfg: StandCfg,
    min_standing_s: float = 0.5,
    max_tilt: float = math.radians(45.0),
  ) -> torch.Tensor:
    standing = is_standing(env, stand_cfg)
    self._standing_time = torch.where(
      standing, self._standing_time + env.step_dt, torch.zeros_like(self._standing_time)
    )
    self._stood |= self._standing_time >= min_standing_s
    contact = get_ground_contact(env, stand_cfg.contact_cfg)
    down = contact.body_on_ground() | (torso_up(env, stand_cfg) < math.cos(max_tilt))
    return self._stood & down
