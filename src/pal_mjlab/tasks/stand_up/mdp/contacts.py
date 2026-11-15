"""Ground contact bookkeeping and the definition of standing."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_CONTACT_ATTR = "_stand_up_ground_contacts"
_BODY_ID_ATTR = "_stand_up_body_ids"

GRAVITY = 9.81


@dataclass
class GroundContactCfg:
  sensor_name: str
  """Contact sensor with one primary per body against the terrain, "force" in
  its fields, ``reduce="netforce"`` and a history of one control step."""
  foot_body_regex: str
  """Bodies that carry the soles."""
  entity_name: str = "robot"
  contact_force_threshold: float = 5.0
  """Force [N] above which a body counts as touching the ground."""


class GroundContact:
  """Per-body ground forces over the last control step."""

  def __init__(self, env: ManagerBasedRlEnv, cfg: GroundContactCfg):
    self._env = env
    self.cfg = cfg
    self._sensor: ContactSensor = env.scene[cfg.sensor_name]
    names = [n.split("/")[-1] for n in self._sensor.primary_names]
    self.is_foot = torch.tensor(
      [re.fullmatch(cfg.foot_body_regex, n) is not None for n in names],
      device=env.device,
    )
    asset: Entity = env.scene[cfg.entity_name]
    body_mass = torch.as_tensor(env.sim.mj_model.body_mass, dtype=torch.float32)
    self.weight = float(body_mass[asset.indexing.body_ids.cpu()].sum()) * GRAVITY
    zeros = torch.zeros(env.num_envs, len(names), device=env.device)
    self.peak_force = zeros  # [B, N] largest substep force per body.
    self.mean_force = zeros  # [B, N] force per body averaged over the step.
    self._last_update_step = -1

  def update(self) -> None:
    """Refresh from the sensor, at most once per control step."""
    if self._env.common_step_counter == self._last_update_step:
      return
    self._last_update_step = self._env.common_step_counter
    force_history = self._sensor.data.force_history  # [B, N, H, 3]
    assert force_history is not None
    magnitude = torch.linalg.vector_norm(force_history, dim=-1)
    self.peak_force = magnitude.max(dim=-1).values
    self.mean_force = magnitude.mean(dim=-1)

  def touching(self) -> torch.Tensor:
    return self.peak_force > self.cfg.contact_force_threshold

  def feet_on_ground(self) -> torch.Tensor:
    """[B] whether every foot touches the ground."""
    return self.touching()[:, self.is_foot].all(dim=-1)

  def body_on_ground(self) -> torch.Tensor:
    """[B] whether anything but a foot touches the ground."""
    return self.touching()[:, ~self.is_foot].any(dim=-1)

  def feet_load_fraction(self) -> torch.Tensor:
    """[B] share of the ground force that goes through the feet. The total
    counts as at least half the robot's weight."""
    feet = self.mean_force[:, self.is_foot].sum(dim=-1)
    total = self.mean_force.sum(dim=-1).clamp(min=0.5 * self.weight)
    return (feet / total).clamp(0.0, 1.0)


def get_ground_contact(env: ManagerBasedRlEnv, cfg: GroundContactCfg) -> GroundContact:
  """The (lazily created, up to date) tracker for ``cfg.sensor_name``."""
  trackers: dict[str, GroundContact] | None = getattr(env, _CONTACT_ATTR, None)
  if trackers is None:
    trackers = {}
    setattr(env, _CONTACT_ATTR, trackers)
  tracker = trackers.get(cfg.sensor_name)
  if tracker is None:
    tracker = GroundContact(env, cfg)
    trackers[cfg.sensor_name] = tracker
  tracker.update()
  return tracker


@dataclass
class StandCfg:
  """What counts as standing."""

  contact_cfg: GroundContactCfg
  torso_body: str
  target_height: float
  """Base height [m] of the standing pose."""
  min_height: float
  """Base height [m] of a robot lying or sitting on the ground."""
  stand_height: float
  """Base height [m] above which the robot counts as standing."""
  max_tilt: float = math.radians(15.0)
  entity_name: str = "robot"


def torso_body_id(env: ManagerBasedRlEnv, asset: Entity, body_name: str) -> int:
  cache: dict[str, int] | None = getattr(env, _BODY_ID_ATTR, None)
  if cache is None:
    cache = {}
    setattr(env, _BODY_ID_ATTR, cache)
  if body_name not in cache:
    cache[body_name] = asset.find_bodies((body_name,), preserve_order=True)[0][0]
  return cache[body_name]


def base_height(env: ManagerBasedRlEnv, cfg: StandCfg) -> torch.Tensor:
  asset: Entity = env.scene[cfg.entity_name]
  return asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def torso_up(env: ManagerBasedRlEnv, cfg: StandCfg) -> torch.Tensor:
  """[B] cosine of the torso's tilt from upright."""
  asset: Entity = env.scene[cfg.entity_name]
  quat = asset.data.body_link_quat_w[:, torso_body_id(env, asset, cfg.torso_body)]
  return -quat_apply_inverse(quat, asset.data.gravity_vec_w)[:, 2]


def height_progress(env: ManagerBasedRlEnv, cfg: StandCfg) -> torch.Tensor:
  """[B] in [0, 1]: 0 with the base on the ground, 1 at standing height."""
  span = cfg.target_height - cfg.min_height
  return ((base_height(env, cfg) - cfg.min_height) / span).clamp(0.0, 1.0)


def upright_progress(env: ManagerBasedRlEnv, cfg: StandCfg) -> torch.Tensor:
  """[B] in [0, 1]: 0 upside down, 0.25 horizontal, 1 upright."""
  return torch.square(0.5 * (1.0 + torso_up(env, cfg)))


def is_standing(env: ManagerBasedRlEnv, cfg: StandCfg) -> torch.Tensor:
  """[B] base high, torso upright, on the feet only."""
  contact = get_ground_contact(env, cfg.contact_cfg)
  return (
    (base_height(env, cfg) > cfg.stand_height)
    & (torso_up(env, cfg) > math.cos(cfg.max_tilt))
    & contact.feet_on_ground()
    & ~contact.body_on_ground()
  )
