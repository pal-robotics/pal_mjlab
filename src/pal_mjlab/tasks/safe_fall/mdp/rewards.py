"""Reward terms for the safe-fall task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from .arms import (
  ForearmGuardCfg,
  forearm_guard_distances,
  forearm_points,
)
from .impact import ImpactCfg, get_impact_tracker
from .steps import StepCountCfg, get_step_counter

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


##
# Damage: the objective of Ha & Liu (2015).
##


def max_impulse_increment(
  env: ManagerBasedRlEnv, impact_cfg: ImpactCfg
) -> torch.Tensor:
  """Growth of the episode's largest fragility-weighted ground impulse.

  Summed over an episode this is exactly the largest single weighted impulse,
  so the return penalizes the worst impact instead of the total. The value is
  divided by the step duration to cancel the reward manager's dt scaling: the
  weight is then "reward per unit of normalized impulse".
  """
  tracker = get_impact_tracker(env, impact_cfg)
  if env.cfg.scale_rewards_by_dt:
    return tracker.increment / env.step_dt
  return tracker.increment


def contact_load(env: ManagerBasedRlEnv, impact_cfg: ImpactCfg) -> torch.Tensor:
  """Fragility-weighted ground load, in units of the robot's weight.

  Small penalty that keeps every contact soft, not only the largest one, and
  discourages resting on fragile bodies.
  """
  tracker = get_impact_tracker(env, impact_cfg)
  load = tracker.peak_force / tracker.weight_force
  return torch.sum(load * tracker.weights, dim=-1)


def pre_impact_velocity(
  env: ManagerBasedRlEnv,
  impact_cfg: ImpactCfg,
  height_scale: float = 0.2,
  exclude_body_regex: str | None = None,
) -> torch.Tensor:
  """Downward speed of fragile bodies that are close to the ground.

  Dense signal that arrives before the impact does: a body that approaches
  the floor slowly cannot hit it hard. Height is measured at the body's
  center of mass, so ``height_scale`` should be of the order of a link radius.

  ``exclude_body_regex`` leaves bodies out. The arms are excluded in the
  Kangaroo task: for them this term is cheapest to satisfy by holding the arms
  up and away from the body, which is the opposite of a safe posture.
  """
  tracker = get_impact_tracker(env, impact_cfg)
  asset: Entity = env.scene[impact_cfg.entity_name]
  weights = tracker.weights
  if exclude_body_regex is not None:
    weights = weights * (~tracker.group_mask(exclude_body_regex))
  height = asset.data.body_com_pos_w[:, tracker.body_ids, 2]
  height = height - env.scene.env_origins[:, 2:3]
  down_speed = (-asset.data.body_com_lin_vel_w[:, tracker.body_ids, 2]).clamp(min=0.0)
  proximity = torch.exp(-height.clamp(min=0.0) / height_scale)
  return torch.sum(weights * torch.square(down_speed) * proximity, dim=-1)


def guard_coverage(
  env: ManagerBasedRlEnv, guard_cfg: ForearmGuardCfg, std: float
) -> torch.Tensor:
  """[B] in [0, 1]: how well the forearms cover the chest.

  Mean over both forearms' midpoints and tips of a kernel on their distance
  from their guard zones: 1 with all four points inside.
  """
  distance = forearm_guard_distances(env, guard_cfg)
  return torch.exp(-torch.square(distance) / std**2).mean(dim=-1)


def forearm_guard_distance(
  env: ManagerBasedRlEnv, guard_cfg: ForearmGuardCfg
) -> torch.Tensor:
  """Mean distance [m] of the forearms' midpoints and tips from their zones.

  The fallback policy only runs once something has gone wrong, and a robot
  that looks stable in simulation may still go down on the real one. The
  guard is therefore not tied to how far the robot has fallen: the forearms
  go in front of the chest at once and stay there, also at rest. A fall
  towards the front then lands on them instead of on the torso's fragile
  side. Linear in the distance, so an arm that is far away (behind the back,
  out to the side) is pulled in as strongly as one that is nearly there.
  """
  return forearm_guard_distances(env, guard_cfg).mean(dim=-1)


def elbow_first(
  env: ManagerBasedRlEnv,
  guard_cfg: ForearmGuardCfg,
  height_scale: float = 0.15,
  drop_scale: float = 0.2,
) -> torch.Tensor:
  """Elbows that are about to reach the ground before their forearm.

  Per arm: how close the elbow is to the ground (a kernel of scale
  ``height_scale`` [m]) times how far the forearm tip is above the elbow,
  saturating at ``drop_scale`` [m]; summed over the arms. The elbow holds an
  actuator and the forearm is a plastic shell, so the forearm and its tip
  should touch down first. With the tip level with the elbow or below it the
  term is zero, whatever the direction of the fall.
  """
  elbows, tips = forearm_points(env, guard_cfg)
  ground = env.scene.env_origins[:, 2:3]
  elbow_height = (elbows[..., 2] - ground).clamp(min=0.0)
  tip_above = ((tips[..., 2] - elbows[..., 2]) / drop_scale).clamp(0.0, 1.0)
  return torch.sum(torch.exp(-elbow_height / height_scale) * tip_above, dim=-1)


def backward_lean(
  env: ManagerBasedRlEnv, torso_body: str, entity_name: str = "robot"
) -> torch.Tensor:
  """[B] in [0, 1]: how far the torso leans backward (1 = flat on its back).

  The component of gravity along the torso's backward axis. Going down
  backward ends on the pelvis and its actuators; going down forward ends on
  the knees and the forearms. Penalized from the first step, so that the
  robot leans and turns forward before anything touches the ground.
  """
  asset: Entity = env.scene[entity_name]
  torso_id = _body_id(env, asset, torso_body)
  quat = asset.data.body_link_quat_w[:, torso_id]
  down = torch.zeros_like(quat[:, :3])
  down[:, 2] = -1.0
  return (-quat_apply_inverse(quat, down)[:, 0]).clamp(min=0.0)


def _body_id(env: ManagerBasedRlEnv, asset: Entity, body: str) -> int:
  cache: dict[str, int] | None = getattr(env, "_safe_fall_body_ids", None)
  if cache is None:
    cache = {}
    env._safe_fall_body_ids = cache  # type: ignore[attr-defined]
  if body not in cache:
    cache[body] = asset.find_bodies((body,), preserve_order=True)[0][0]
  return cache[body]


def excess_steps(env: ManagerBasedRlEnv, step_cfg: StepCountCfg) -> torch.Tensor:
  """Foot touchdowns beyond the free ones of the episode.

  Summed over an episode this is the number of steps taken beyond
  ``step_cfg.free_steps``. Like ``max_impulse_increment`` it is divided by the
  step duration, so the weight is "reward per excess step".
  """
  counter = get_step_counter(env, step_cfg)
  if env.cfg.scale_rewards_by_dt:
    return counter.excess_increment / env.step_dt
  return counter.excess_increment


def foot_slide(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  impact_cfg: ImpactCfg,
  max_speed: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared horizontal speed of the feet that are on the ground.

  Closes the loophole around ``excess_steps``: shuffling the feet along the
  ground without lifting them. The speed is clipped at ``max_speed``, so the
  slide a strong kick forces on the soles costs little, and the term is off
  once a fragile body is on the ground, where the feet are dragged along.
  ``asset_cfg`` selects one site per foot, in the sensor's order.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  in_contact = (sensor.data.found > 0).float()  # [B, feet]
  speed = torch.linalg.vector_norm(
    asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2], dim=-1
  ).clamp(max=max_speed)
  tracker = get_impact_tracker(env, impact_cfg)
  on_the_ground = (tracker.touching() & (tracker.weights > 0.0)).any(dim=-1)
  return torch.sum(torch.square(speed) * in_contact, dim=-1) * (~on_the_ground)


class actuator_overload:
  """Soft count of actuators working near their force limit.

  Each actuator contributes 0 below ``threshold`` (as a fraction of its force
  limit) and ramps linearly to 1 at the limit. Protects the gearboxes and the
  leg linear actuators from being used as rigid struts during an impact.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    asset: Entity = env.scene[asset_cfg.name]
    ctrl_ids = asset.indexing.ctrl_ids.cpu().numpy()
    force_range = env.sim.mj_model.actuator_forcerange[ctrl_ids]
    limit = torch.as_tensor(force_range[:, 1], dtype=torch.float32, device=env.device)
    if torch.any(limit <= 0.0):
      raise ValueError("actuator_overload needs a force limit on every actuator.")
    self._limit = limit

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    threshold: float = 0.9,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    usage = asset.data.actuator_force.abs() / self._limit
    return torch.sum(((usage - threshold) / (1.0 - threshold)).clamp(0.0, 1.0), dim=-1)


def joint_limit_strike(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared joint speed towards a limit while beyond the soft limit.

  Penalizes slamming into an end stop (in particular the leg-length stroke at
  the bottom of a fast squat) without penalizing sitting near the limit.
  """
  asset: Entity = env.scene[asset_cfg.name]
  limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids]
  pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  into_lower = (-vel).clamp(min=0.0) * (pos < limits[..., 0])
  into_upper = vel.clamp(min=0.0) * (pos > limits[..., 1])
  return torch.sum(torch.square(into_lower) + torch.square(into_upper), dim=-1)


##
# Getting low and coming to rest.
##


def crouch(
  env: ManagerBasedRlEnv,
  target_length: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Kernel on the leg-length joints reaching the squat length."""
  asset: Entity = env.scene[asset_cfg.name]
  length = asset.data.joint_pos[:, asset_cfg.joint_ids]
  error = torch.mean(torch.square(length - target_length), dim=-1)
  return torch.exp(-error / std**2)


def com_speed_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared linear speed of the root center of mass."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_com_lin_vel_w), dim=-1)


def motion_at_rest(
  env: ManagerBasedRlEnv,
  lin_speed_std: float = 0.3,
  ang_speed_std: float = 0.6,
  low_height: float = 0.5,
  height_std: float = 0.15,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared speed of the selected joints once the base is low and still.

  The base being at rest is measured as in ``safe_state``. Keeps the arms
  from wandering about after the robot has come down, without slowing them
  while it is falling.
  """
  asset: Entity = env.scene[asset_cfg.name]
  joint_speed_sq = torch.sum(
    torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1
  )
  return joint_speed_sq * _base_at_rest(
    env, asset, lin_speed_std, ang_speed_std, low_height, height_std
  )


def _base_at_rest(
  env: ManagerBasedRlEnv,
  asset: Entity,
  lin_speed_std: float,
  ang_speed_std: float,
  low_height: float,
  height_std: float,
) -> torch.Tensor:
  """[B] in [0, 1]: kernels on the base's speed and on its height above low."""
  lin_speed_sq = torch.sum(torch.square(asset.data.root_link_lin_vel_w), dim=-1)
  ang_speed_sq = torch.sum(torch.square(asset.data.root_link_ang_vel_w), dim=-1)
  still = torch.exp(-lin_speed_sq / lin_speed_std**2 - ang_speed_sq / ang_speed_std**2)
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  low = torch.exp(-torch.square((height - low_height).clamp(min=0.0)) / height_std**2)
  return still * low


def safe_state_flags(
  env: ManagerBasedRlEnv,
  impact_cfg: ImpactCfg,
  feet_sensor_name: str,
  leg_length_max: float = 0.22,
  lin_speed_max: float = 0.15,
  ang_speed_max: float = 0.3,
  low_height: float = 0.6,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return (squat, rest) boolean flags, each of shape [B].

  - ``rest``: the base is low and not moving. The robot has stopped, in any
    posture, and is no longer a hazard.
  - ``squat``: ``rest`` and additionally the legs are short, both feet are on
    the ground and no other body touches it. This is the clean outcome.
  """
  tracker = get_impact_tracker(env, impact_cfg)
  asset: Entity = env.scene[asset_cfg.name]

  lin_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=-1)
  ang_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=-1)
  still = (lin_speed < lin_speed_max) & (ang_speed < ang_speed_max)
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  rest = still & (height < low_height)

  feet: ContactSensor = env.scene[feet_sensor_name]
  assert feet.data.found is not None
  feet_down = (feet.data.found > 0).all(dim=-1)
  body_touching = (tracker.touching() & (tracker.weights > 0.0)).any(dim=-1)
  legs_short = (
    asset.data.joint_pos[:, asset_cfg.joint_ids].mean(dim=-1) < leg_length_max
  )
  squat = rest & feet_down & ~body_touching & legs_short
  return squat, rest


def safe_state(
  env: ManagerBasedRlEnv,
  impact_cfg: ImpactCfg,
  lin_speed_std: float = 0.3,
  ang_speed_std: float = 0.6,
  low_height: float = 0.5,
  height_std: float = 0.15,
  rest_fraction: float = 0.5,
  guard_cfg: ForearmGuardCfg | None = None,
  guard_std: float = 0.2,
  unguarded_fraction: float = 1.0,
  contact_rest_fractions: dict[str, float] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Smooth measure of being low, still and protected.

  The product of three kernels in [0, 1]: base linear speed, base angular
  speed, and base height above ``low_height``. It is scaled by
  ``rest_fraction`` while any fragile body touches the ground: below 1 a
  squat on the feet is worth more than lying still, at 1 every way of coming
  to rest is worth the same and only the damage terms tell them apart. Paid
  every step, so reaching the safe state earlier is worth more.

  With ``guard_cfg`` the result is also scaled by the chest guard: from
  ``unguarded_fraction`` with the forearms away from the chest to 1 with them
  covering it (see ``guard_coverage``). A robot at rest with its arms behind
  its back is not in a safe state: the next disturbance finds the chest open.

  ``contact_rest_fractions`` maps a body regex to the fraction of the result
  that is left while a matching body touches the ground; the smallest one
  applies. For the Kangaroo: kneeling is the place to rest, lying with the
  torso on the ground is worth less, and sitting or lying on the pelvis and
  its leg actuators least of all, so that anything which keeps them off the
  ground is better, also standing on all fours.
  """
  tracker = get_impact_tracker(env, impact_cfg)
  asset: Entity = env.scene[asset_cfg.name]

  at_rest = _base_at_rest(
    env, asset, lin_speed_std, ang_speed_std, low_height, height_std
  )

  body_touching = (tracker.touching() & (tracker.weights > 0.0)).any(dim=-1)
  posture = torch.where(body_touching, rest_fraction, 1.0)
  value = at_rest * posture
  if contact_rest_fractions:
    touching = tracker.touching()
    factor = torch.ones_like(value)
    for body_regex, fraction in contact_rest_fractions.items():
      resting_on = (touching & tracker.group_mask(body_regex)).any(dim=-1)
      factor = torch.where(resting_on, factor.clamp(max=fraction), factor)
    value = value * factor
  if guard_cfg is not None:
    covered = guard_coverage(env, guard_cfg, guard_std)
    value = value * (unguarded_fraction + (1.0 - unguarded_fraction) * covered)
  return value
