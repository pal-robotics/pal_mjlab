"""Counting recovery steps, for the safe-fall task.

The fallback policy may take over because actuators are degraded, so it must
not depend on running a kick off with many quick steps. A step or two to
orient the fall is fine; this module counts them so that more can be charged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.sensor.contact_sensor import ContactSensor

from .impact import ImpactCfg, get_impact_tracker

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_COUNTERS_ATTR = "_safe_fall_step_counters"


@dataclass
class StepCountCfg:
  feet_sensor_name: str
  """Contact sensor of the feet against the ground, with ``track_air_time``."""
  impact_cfg: ImpactCfg
  """Ground contact of the rest of the body. A foot touching down while a
  fragile body is on the ground is part of a tumble, not a step."""
  free_steps: int = 2
  """Touchdowns per episode that are not counted as excess."""
  min_air_time: float = 0.02
  """Shortest time [s] a foot must have been off the ground for its touchdown
  to count. Filters out contact chatter."""


class StepCounter:
  """Per-episode count of foot touchdowns, shared by rewards and metrics."""

  def __init__(self, env: ManagerBasedRlEnv, cfg: StepCountCfg):
    self._env = env
    self.cfg = cfg
    self._sensor: ContactSensor = env.scene[cfg.feet_sensor_name]
    self.count = torch.zeros(env.num_envs, device=env.device)
    """[B] touchdowns so far in the episode."""
    self.excess_increment = torch.zeros(env.num_envs, device=env.device)
    """[B] touchdowns beyond the free ones during the last control step."""
    self._last_update_step = -1

  def update(self) -> None:
    """Refresh from the sensor. No-op if already done this control step."""
    env = self._env
    if env.common_step_counter == self._last_update_step:
      return
    self._last_update_step = env.common_step_counter

    landed = self._sensor.compute_first_contact(env.step_dt)  # [B, feet]
    last_air_time = self._sensor.data.last_air_time
    assert last_air_time is not None
    stepped = landed & (last_air_time >= self.cfg.min_air_time - 1e-6)

    tracker = get_impact_tracker(env, self.cfg.impact_cfg)
    on_the_ground = (tracker.touching() & (tracker.weights > 0.0)).any(dim=-1)
    new = stepped.sum(dim=-1).float() * (~on_the_ground)

    # Same episode-start convention as the impact tracker.
    new_episode = env.episode_length_buf <= 1
    previous = torch.where(new_episode, torch.zeros_like(self.count), self.count)
    self.count = previous + new
    free = float(self.cfg.free_steps)
    self.excess_increment = (self.count - free).clamp(min=0.0) - (
      previous - free
    ).clamp(min=0.0)


def get_step_counter(env: ManagerBasedRlEnv, cfg: StepCountCfg) -> StepCounter:
  """Return the (lazily created, up to date) counter for ``cfg``'s sensor."""
  counters: dict[str, StepCounter] | None = getattr(env, _COUNTERS_ATTR, None)
  if counters is None:
    counters = {}
    setattr(env, _COUNTERS_ATTR, counters)
  counter = counters.get(cfg.feet_sensor_name)
  if counter is None:
    counter = StepCounter(env, cfg)
    counters[cfg.feet_sensor_name] = counter
  counter.update()
  return counter
