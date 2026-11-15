"""Shared per-episode impact bookkeeping for the safe-fall task.

One ``ImpactTracker`` per environment instance turns the per-substep ground
contact forces of every robot body into the quantities that rewards, critic
observations and metrics all need. It is updated at most once per control step,
no matter how many terms read it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.sensor.contact_sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_TRACKER_ATTR = "_safe_fall_impact_trackers"

GRAVITY = 9.81


@dataclass
class ImpactCfg:
  """How ground contact on each body is measured and weighted."""

  sensor_name: str
  """Contact sensor with one primary per body, ``fields`` containing "force",
  ``reduce="netforce"`` and ``history_length`` equal to the decimation."""
  fragility: dict[str, float] = field(default_factory=dict)
  """Regex on the body name -> fragility weight. First match wins; bodies that
  match nothing get weight 0 (their contact is free)."""
  entity_name: str = "robot"
  contact_force_threshold: float = 5.0
  """Net force [N] above which a body counts as touching the ground."""


class ImpactTracker:
  """Per-body ground impulses and the running maximum of the weighted impulse.

  Impulses are normalized by ``total_mass * 1 m/s``: a value of 1 is the
  impulse that would change the velocity of the whole robot by 1 m/s. A body
  that statically carries the full robot weight shows ``g * step_dt`` per
  control step (about 0.2 at 50 Hz).

  The running maximum implements the objective of Ha & Liu (2015): the cost of
  an episode is its largest single impact, not the sum.
  """

  def __init__(self, env: ManagerBasedRlEnv, cfg: ImpactCfg):
    self._env = env
    self.cfg = cfg
    self._group_masks: dict[str, torch.Tensor] = {}
    sensor: ContactSensor = env.scene[cfg.sensor_name]
    self._sensor = sensor
    if sensor.cfg.history_length <= 0 or "force" not in sensor.cfg.fields:
      raise ValueError(
        f"Contact sensor '{cfg.sensor_name}' needs fields including 'force' and "
        "history_length > 0 to integrate impulses over the physics substeps."
      )

    self.body_names: list[str] = [n.split("/")[-1] for n in sensor.primary_names]
    weights = []
    for name in self.body_names:
      for pattern, weight in cfg.fragility.items():
        if re.fullmatch(pattern, name):
          weights.append(float(weight))
          break
      else:
        weights.append(0.0)
    self.weights = torch.tensor(weights, device=env.device, dtype=torch.float32)

    asset: Entity = env.scene[cfg.entity_name]
    self.body_ids = torch.tensor(
      asset.find_bodies(tuple(self.body_names), preserve_order=True)[0],
      device=env.device,
      dtype=torch.long,
    )
    body_mass = torch.as_tensor(env.sim.mj_model.body_mass, dtype=torch.float32)
    self.total_mass = float(body_mass[asset.indexing.body_ids.cpu()].sum())
    self.weight_force = self.total_mass * GRAVITY
    self._impulse_scale = env.physics_dt / self.total_mass

    num_envs, num_bodies = env.num_envs, len(self.body_names)
    zeros = torch.zeros(num_envs, num_bodies, device=env.device)
    self.step_impulse = zeros
    """[B, N] normalized ground impulse per body over the last control step."""
    self.peak_force = zeros
    """[B, N] largest substep ground force [N] per body over the last step."""
    self.running_max = torch.zeros(num_envs, device=env.device)
    """[B] largest fragility-weighted step impulse so far in the episode."""
    self.increment = torch.zeros(num_envs, device=env.device)
    """[B] growth of ``running_max`` during the last control step."""
    self._last_update_step = -1

  def group_mask(self, body_regex: str) -> torch.Tensor:
    """Boolean mask [N] of the tracked bodies whose name matches the regex."""
    mask = self._group_masks.get(body_regex)
    if mask is None:
      mask = torch.tensor(
        [re.fullmatch(body_regex, n) is not None for n in self.body_names],
        device=self._env.device,
      )
      self._group_masks[body_regex] = mask
    return mask

  def update(self) -> None:
    """Refresh from the sensor. No-op if already done this control step."""
    env = self._env
    if env.common_step_counter == self._last_update_step:
      return
    self._last_update_step = env.common_step_counter

    force_history = self._sensor.data.force_history  # [B, N, H, 3]
    assert force_history is not None
    force_mag = torch.linalg.vector_norm(force_history, dim=-1)  # [B, N, H]
    self.step_impulse = force_mag.sum(dim=-1) * self._impulse_scale
    self.peak_force = force_mag.max(dim=-1).values

    # Rewards run after episode_length_buf was incremented, so the first
    # control step of an episode has length 1. Start its maximum from zero.
    new_episode = env.episode_length_buf <= 1
    previous = torch.where(
      new_episode, torch.zeros_like(self.running_max), self.running_max
    )
    weighted = (self.step_impulse * self.weights).max(dim=-1).values
    self.running_max = torch.maximum(previous, weighted)
    self.increment = self.running_max - previous

  def touching(self) -> torch.Tensor:
    """[B, N] whether each body pressed on the ground during the last step."""
    return self.peak_force > self.cfg.contact_force_threshold

  def episode_running_max(self) -> torch.Tensor:
    """[B] running maximum, zero for environments that were just reset."""
    just_reset = self._env.episode_length_buf == 0
    return torch.where(just_reset, torch.zeros_like(self.running_max), self.running_max)


def get_impact_tracker(env: ManagerBasedRlEnv, cfg: ImpactCfg) -> ImpactTracker:
  """Return the (lazily created, up to date) tracker for ``cfg.sensor_name``."""
  trackers: dict[str, ImpactTracker] | None = getattr(env, _TRACKER_ATTR, None)
  if trackers is None:
    trackers = {}
    setattr(env, _TRACKER_ATTR, trackers)
  tracker = trackers.get(cfg.sensor_name)
  if tracker is None:
    tracker = ImpactTracker(env, cfg)
    trackers[cfg.sensor_name] = tracker
  tracker.update()
  return tracker
