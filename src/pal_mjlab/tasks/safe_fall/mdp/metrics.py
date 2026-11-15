"""Logged (not rewarded) metrics for the safe-fall task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor
from mjlab.sensor.contact_sensor import ContactSensor

from .arms import ArmSpreadCfg, arm_axis_distances
from .impact import ImpactCfg, get_impact_tracker
from .rewards import safe_state_flags
from .steps import StepCountCfg, get_step_counter

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def impact_running_max_metric(
  env: ManagerBasedRlEnv, impact_cfg: ImpactCfg
) -> torch.Tensor:
  """Largest fragility-weighted impulse of the episode (use ``reduce="max"``)."""
  return get_impact_tracker(env, impact_cfg).running_max


def group_impulse(
  env: ManagerBasedRlEnv, impact_cfg: ImpactCfg, body_regex: str
) -> torch.Tensor:
  """Largest unweighted step impulse among the bodies matching ``body_regex``."""
  tracker = get_impact_tracker(env, impact_cfg)
  mask = tracker.group_mask(body_regex)
  return (tracker.step_impulse * mask).max(dim=-1).values


def group_contact(
  env: ManagerBasedRlEnv, impact_cfg: ImpactCfg, body_regex: str
) -> torch.Tensor:
  """1 if any body matching ``body_regex`` touches the ground."""
  tracker = get_impact_tracker(env, impact_cfg)
  mask = tracker.group_mask(body_regex)
  return (tracker.touching() & mask).any(dim=-1).float()


def sensor_norm(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Norm of a builtin vector sensor, e.g. the IMU accelerometer."""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  return torch.linalg.vector_norm(sensor.data, dim=-1)


def contact_force_above(
  env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float
) -> torch.Tensor:
  """1 if any contact of the sensor exceeded the force during the last step."""
  sensor: ContactSensor = env.scene[sensor_name]
  force_history = sensor.data.force_history  # [B, N, H, 3]
  assert force_history is not None
  peak = torch.linalg.vector_norm(force_history, dim=-1).amax(dim=(1, 2))
  return (peak > force_threshold).float()


def arm_tip_distance(env: ManagerBasedRlEnv, arm_cfg: ArmSpreadCfg) -> torch.Tensor:
  """Largest distance [m] of a forearm tip from the body axis."""
  return arm_axis_distances(env, arm_cfg)[1].max(dim=-1).values


def steps_taken(env: ManagerBasedRlEnv, step_cfg: StepCountCfg) -> torch.Tensor:
  """Foot touchdowns so far in the episode (use ``reduce="max"``)."""
  return get_step_counter(env, step_cfg).count


def safe_state_metric(
  env: ManagerBasedRlEnv,
  impact_cfg: ImpactCfg,
  feet_sensor_name: str,
  kind: str,
  leg_length_max: float = 0.22,
  lin_speed_max: float = 0.15,
  ang_speed_max: float = 0.3,
  low_height: float = 0.6,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Safe-state indicators.

  ``kind`` is "squat" (clean feet-only squat), "rest" (stopped and low in any
  posture) or "unsafe" (neither). Use ``reduce="last"`` for success rates and
  ``reduce="mean"`` on "unsafe" for the fraction of the episode spent unsafe.
  """
  squat, rest = safe_state_flags(
    env,
    impact_cfg,
    feet_sensor_name,
    leg_length_max,
    lin_speed_max,
    ang_speed_max,
    low_height,
    asset_cfg,
  )
  if kind == "squat":
    return squat.float()
  if kind == "rest":
    return rest.float()
  if kind == "unsafe":
    return (~rest).float()
  raise ValueError(f"Unknown kind '{kind}'.")
