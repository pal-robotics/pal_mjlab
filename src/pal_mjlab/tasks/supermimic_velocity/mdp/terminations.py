"""The velocity task's failure terminations, as single definitions
(``@imaginable_termination``): SuperMimic's imagination ends imagined episodes where the
simulator would, with a smooth probability whose gradient keeps the policy away."""

from __future__ import annotations

import math

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from pal_mjlab.rl.supermimic.imagination import Batch, capsule_bottoms, imaginable_termination

__all__ = ["bad_orientation", "root_height_below", "links_touch_ground"]


@imaginable_termination
def bad_orientation(b: Batch, limit_angle: float, width: float = 0.05) -> torch.Tensor:
  """mjlab's ``bad_orientation``: the root's tilt from upright beyond ``limit_angle``
  (rad). Tested as the cosine of the tilt (-gz) below cos(limit), the same condition:
  ``acos``'s gradient is infinite when upright. Smoothed over ``width`` (in cosine)."""
  quat = b.qpos[:, 3:7]
  gravity = torch.zeros_like(quat[:, :3])
  gravity[:, 2] = -1.0
  cos_tilt = -quat_apply_inverse(quat, gravity)[:, 2]
  return torch.sigmoid((math.cos(limit_angle) - cos_tilt) / width)


@imaginable_termination
def root_height_below(b: Batch, minimum_height: float, width: float = 0.01) -> torch.Tensor:
  """The root below ``minimum_height`` (m) above the terrain, smoothed over ``width``."""
  return torch.sigmoid((minimum_height - b.state[:, 0]) / width)


@imaginable_termination
def links_touch_ground(
  b: Batch, bodies: tuple[str, ...], contact_height: float = 0.0, width: float = 0.002
) -> torch.Tensor:
  """A collision capsule of ``bodies`` within ``contact_height`` (m) of the terrain: the
  geometric version of a contact-sensor termination (mjlab's ``illegal_contact``),
  smoothed over ``width`` (m). On the Kangaroo's knee linkage, at 0 m it fires exactly
  where the contact sensor does (all 360 contacts, no false alarm in 128k samples). The bodies must be FK bodies (``ImaginationCfg.bodies``;
  a passive linkage's need ``passive_joints``)."""
  lowest = capsule_bottoms(b, tuple(bodies)).min(dim=1).values
  return torch.sigmoid((contact_height - lowest) / width)
