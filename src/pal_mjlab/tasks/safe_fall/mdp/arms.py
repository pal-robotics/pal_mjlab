"""Where the arms are relative to the body, for the safe-fall task.

An arm held away from the body is a long lever on a weak shoulder and tends to
hit the ground first and alone. These helpers measure how compact the arms
are without prescribing a pose: any configuration that keeps the elbows and
the forearm tips close to the body's long axis counts as compact.

They also measure how far each forearm is from its guard zone in front of the
chest, where it ends up between the torso's fragile side and the ground.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_IDS_ATTR = "_safe_fall_arm_ids"


@dataclass
class ArmSpreadCfg:
  """Points on the arms and the body axis their distance is measured from."""

  torso_body: str
  """Body whose frame defines the axis."""
  axis_start: tuple[float, float, float]
  axis_end: tuple[float, float, float]
  """End points of the body axis segment, in the torso frame. It should run
  from below the hips to the shoulders, so that arms hanging along the body
  count as close."""
  elbow_bodies: tuple[str, ...]
  """Bodies whose origin is at an elbow."""
  tip_sites: tuple[str, ...]
  """Sites at the forearm tips."""
  entity_name: str = "robot"


def _resolve_ids(
  env: ManagerBasedRlEnv,
  asset: Entity,
  cfg: object,
  torso_body: str,
  elbow_bodies: tuple[str, ...],
  tip_sites: tuple[str, ...],
) -> tuple[int, list[int], list[int]]:
  """Torso body id, elbow body ids and tip site ids of ``cfg``, cached on env."""
  cache: dict[int, tuple] | None = getattr(env, _IDS_ATTR, None)
  if cache is None:
    cache = {}
    setattr(env, _IDS_ATTR, cache)
  ids = cache.get(id(cfg))
  if ids is None:
    torso_id = asset.find_bodies((torso_body,), preserve_order=True)[0][0]
    elbow_ids = asset.find_bodies(elbow_bodies, preserve_order=True)[0]
    tip_ids = asset.find_sites(tip_sites, preserve_order=True)[0]
    ids = (torso_id, elbow_ids, tip_ids)
    cache[id(cfg)] = ids
  return ids


def arm_axis_distances(
  env: ManagerBasedRlEnv, cfg: ArmSpreadCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  """Distance of each elbow and each forearm tip from the body axis segment.

  Returns ``(elbow_dist, tip_dist)`` of shapes [B, num_elbows], [B, num_tips].
  """
  asset: Entity = env.scene[cfg.entity_name]
  torso_id, elbow_ids, tip_ids = _resolve_ids(
    env, asset, cfg, cfg.torso_body, cfg.elbow_bodies, cfg.tip_sites
  )

  torso_pos = asset.data.body_link_pos_w[:, torso_id]
  torso_quat = asset.data.body_link_quat_w[:, torso_id]
  start = torch.tensor(cfg.axis_start, device=env.device)
  axis = torch.tensor(cfg.axis_end, device=env.device) - start

  def distance(points_w: torch.Tensor) -> torch.Tensor:
    rel = points_w - torso_pos.unsqueeze(1)
    quat = torso_quat.unsqueeze(1).expand(-1, rel.shape[1], -1)
    local = quat_apply_inverse(quat, rel) - start
    along = (torch.sum(local * axis, dim=-1) / torch.sum(axis * axis)).clamp(0.0, 1.0)
    return torch.linalg.vector_norm(local - along.unsqueeze(-1) * axis, dim=-1)

  elbow_dist = distance(asset.data.body_link_pos_w[:, elbow_ids])
  tip_dist = distance(asset.data.site_pos_w[:, tip_ids])
  return elbow_dist, tip_dist


@dataclass
class ForearmGuardCfg:
  """Where the forearms belong in a fall: in front of the chest, as a bumper.

  Each arm has its own zone, on its own side, so the two never compete for
  the same place. A zone is a pair of boxes in the torso frame (x forwards, y
  to the left, z up), one for the forearm tip and one for the forearm
  midpoint. They are given for an arm on the left and mirrored in y for an
  arm on the right. The boxes are large on purpose: the forearm may point
  forwards or lie across the chest, as long as it is in front of it.
  """

  torso_body: str
  """Body whose frame the zones are defined in."""
  tip_box_min: tuple[float, float, float]
  tip_box_max: tuple[float, float, float]
  mid_box_min: tuple[float, float, float]
  mid_box_max: tuple[float, float, float]
  elbow_bodies: tuple[str, ...]
  """Bodies whose origin is at an elbow, one per arm."""
  tip_sites: tuple[str, ...]
  """Sites at the forearm tips, in the same order as ``elbow_bodies``."""
  arm_sides: tuple[float, ...]
  """+1 for an arm on the left, -1 for one on the right, in the same order."""
  entity_name: str = "robot"


def forearm_points(
  env: ManagerBasedRlEnv, cfg: ForearmGuardCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  """World positions [B, num_arms, 3] of the elbows and the forearm tips."""
  asset: Entity = env.scene[cfg.entity_name]
  _, elbow_ids, tip_ids = _resolve_ids(
    env, asset, cfg, cfg.torso_body, cfg.elbow_bodies, cfg.tip_sites
  )
  return asset.data.body_link_pos_w[:, elbow_ids], asset.data.site_pos_w[:, tip_ids]


def forearm_guard_distances(
  env: ManagerBasedRlEnv, cfg: ForearmGuardCfg
) -> torch.Tensor:
  """Distance [m] of each forearm's midpoint and tip from its own zone.

  Returns [B, 2 * num_arms] (midpoints first, then tips); 0 inside the zone.
  """
  asset: Entity = env.scene[cfg.entity_name]
  torso_id, _, _ = _resolve_ids(
    env, asset, cfg, cfg.torso_body, cfg.elbow_bodies, cfg.tip_sites
  )
  elbows, tips = forearm_points(env, cfg)
  torso_pos = asset.data.body_link_pos_w[:, torso_id].unsqueeze(1)
  torso_quat = asset.data.body_link_quat_w[:, torso_id].unsqueeze(1)
  # Into the torso frame, with the right arms mirrored onto the left side.
  mirror = torch.ones(len(cfg.arm_sides), 3, device=env.device)
  mirror[:, 1] = torch.tensor(cfg.arm_sides, device=env.device)

  def distance(points_w: torch.Tensor, box_min: tuple, box_max: tuple) -> torch.Tensor:
    quat = torso_quat.expand(-1, points_w.shape[1], -1)
    local = quat_apply_inverse(quat, points_w - torso_pos) * mirror
    lower = torch.tensor(box_min, device=env.device)
    upper = torch.tensor(box_max, device=env.device)
    nearest = torch.maximum(torch.minimum(local, upper), lower)
    return torch.linalg.vector_norm(local - nearest, dim=-1)

  mid_dist = distance(0.5 * (elbows + tips), cfg.mid_box_min, cfg.mid_box_max)
  tip_dist = distance(tips, cfg.tip_box_min, cfg.tip_box_max)
  return torch.cat([mid_dist, tip_dist], dim=1)
