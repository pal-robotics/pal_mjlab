from __future__ import annotations

import math
from typing import TYPE_CHECKING

# from mjlab.tests.test_runner import env
import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  KANGAROO_KNEE_TO_JOINT_4_OFFSET,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def torso_height(
  env: ManagerBasedRlEnv,
  z_des: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  z = asset.data.root_link_pos_w[:, 2]
  z_err = z - z_des

  # As before: penalize being too low more than being too high
  z_err_scaled = torch.where(z_err < 0, z_err, z_err * 0.25)

  # Squared error
  z_err_squared = torch.square(z_err_scaled)

  # Height penalty: 0 when perfect, >0 as we deviate
  penalty = z_err_squared / (std**2)

  env.extras["log"]["Metrics/mean_height"] = torch.mean(z)
  env.extras["log"]["Metrics/mean_height_penalty"] = torch.mean(penalty)

  return penalty


def stand_still_joint_deviation_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  error = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  abs_error = torch.abs(error)

  # amount beyond the 0.1 margin
  excess = torch.relu(abs_error - 0.25)

  # per-env penalty: sum of excess across all monitored joints
  penalty = torch.sum(excess, dim=1)
  return penalty


def _resolve_envelope_limits(
  limits: dict[str, dict[str, float]],
  required: tuple[str, ...],
) -> dict[str, dict[str, float]]:
  """Copy limit parameters, check required names, and add distance scales.

  Parameter fields are not validated. For each sloped boundary, store
  ``distance_scale = 1 / sqrt(1 + slope**2)``. Multiplying the signed constraint
  residual by this factor gives its distance to the boundary line in the two
  joint-angle coordinates [rad], with knee angle and leg length held fixed.
  Negative values satisfy the constraint; positive values violate it.
  """
  missing = [name for name in required if name not in limits]
  if missing:
    raise KeyError(f"missing envelope limits {missing}; expected {list(required)}")
  resolved = {name: dict(params) for name, params in limits.items()}
  for params in resolved.values():
    if "slope" in params:
      params["distance_scale"] = 1.0 / math.hypot(1.0, params["slope"])
  return resolved


def _split_joint_groups(
  asset: Entity, joint_names_group: list[list[str]], roles: tuple[str, ...]
) -> list[list[int]]:
  """Resolve per-leg joint names into one list of joint IDs per role.

  Each input group contains one joint per role, in role order. The output
  preserves leg order within each role: groups [[left_j2, left_j3],
  [right_j2, right_j3]] produce IDs [[left_j2_id, right_j2_id],
  [left_j3_id, right_j3_id]].
  """
  ids_per_role: list[list[int]] = [[] for _ in roles]
  for group in joint_names_group:
    if len(group) != len(roles):
      raise ValueError(f"each joint group must be {list(roles)}, got {group}")
    group_ids, _ = asset.find_joints(group, preserve_order=True)
    for role_ids, joint_id in zip(ids_per_role, group_ids, strict=True):
      role_ids.append(joint_id)
  return ids_per_role


class ankle_envelope_limits:
  """Penalize ankle configurations outside the fitted envelope.

  ``joint_names_group`` contains [joint 4, joint 5, leg length, knee] per leg.
  Joint angles are in radians; leg length is in meters. Joint ranges come from
  the model, and fitted coefficients come from ``limits``. See
  ``KANGAROO_ANKLE_ENVELOPE_LIMITS`` for geometry and coefficient conventions.

  ``j4_knee_rel`` is joint 4 normalized to the knee by subtracting the
  knee-dependent angular offset. It remains an angle in radians.

  ``shrink`` is a fraction in [0, 1]: 0.02 retains 98% of the linear extent
  about the center. The center is recomputed for each knee angle and leg length,
  at joint 5 = 0 and the midpoint of the feasible joint 4 interval. The joint 5
  range must contain zero.

  For each leg, square the largest positive constraint violation (zero inside).
  Return the sum across legs [rad^2], with shape (num_envs,). The metric
  ``Metrics/joint_envelope_ankle`` averages the sum of unsquared violations
  across environments [rad]. These violations are boundary distances, not the
  Euclidean distance to the feasible region.
  """

  _LIMIT_NAMES = (
    "joint_4_knee_relative",
    "tibia_bars",
    "crank_collision_limits",
    "actuator_min",
    "actuator_max",
  )

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    asset: Entity = env.scene[params["asset_cfg"].name]
    self._j4_ids, self._j5_ids, self._leg_length_ids, self._knee_ids = (
      _split_joint_groups(
        asset, params["joint_names_group"], ("joint 4", "joint 5", "leg length", "knee")
      )
    )
    self._limits = limits = _resolve_envelope_limits(
      params["limits"], self._LIMIT_NAMES
    )
    # Per-leg tensors from the model's joint ranges (MJCF `range`).
    joint_limits = asset.data.default_joint_pos_limits[0]
    j4_min, j4_max = joint_limits[self._j4_ids].unbind(-1)
    # Names refer to actuator travel: maximum travel bounds joint 4 from below;
    # minimum travel bounds it from above. Offsets are evaluated at joint 5 = 0.
    limits["actuator_max"]["offset"] = j4_min
    limits["actuator_min"]["offset"] = j4_max
    j5_min, j5_max = joint_limits[self._j5_ids].unbind(-1)
    limits["joint5"] = {"j5_min": j5_min, "j5_max": j5_max}

  def violation(
    self,
    j4: torch.Tensor,
    j5: torch.Tensor,
    knee: torch.Tensor,
    leg_length: torch.Tensor,
    shrink: float,
  ) -> torch.Tensor:
    """Return the largest signed constraint violation per leg [rad].

    Inputs and output have shape (..., n_legs), preserving group order.
    ``j4``, ``j5``, and ``knee`` are angles [rad]; ``leg_length`` is in meters.
    Values <= 0 are feasible after shrinking. If the lower and upper joint 4
    bounds conflict, the result is at least their positive gap.
    """
    limits = self._limits
    j4_knee_rel_limit, tibia_bars, crank_collision = (
      limits["joint_4_knee_relative"],
      limits["tibia_bars"],
      limits["crank_collision_limits"],
    )
    actuator_min, actuator_max, joint5 = (
      limits["actuator_min"],
      limits["actuator_max"],
      limits["joint5"],
    )

    abs_j5 = j5.abs()
    # Normalize joint 4 to the knee using the model's angular offset.
    j4_knee_rel = j4 - (KANGAROO_KNEE_TO_JOINT_4_OFFSET - 0.5 * knee)
    crank_j4_max = crank_collision["offset"] + crank_collision["z"] * leg_length

    # Each residual is <= 0 inside; scale sloped residuals to boundary distances [rad].
    distances = torch.stack(
      [
        # Lower bound on joint 4 normalized to the knee.
        j4_knee_rel_limit["x_min"] - j4_knee_rel,
        # Tibia bar clearance: lower bound on joint 4 normalized to the knee.
        (
          (tibia_bars["offset"] + tibia_bars["slope"] * abs_j5 - j4_knee_rel)
          * tibia_bars["distance_scale"]
        ),
        # Crank collision clearance: upper bound on joint 4 at this leg length.
        (
          (j4 - crank_j4_max + crank_collision["slope"] * abs_j5)
          * crank_collision["distance_scale"]
        ),
        # Minimum actuator travel: upper bound on joint 4.
        (
          (j4 - actuator_min["offset"] + actuator_min["slope"] * abs_j5)
          * actuator_min["distance_scale"]
        ),
        # Maximum actuator travel: lower bound on joint 4.
        (
          (actuator_max["offset"] + actuator_max["slope"] * abs_j5 - j4)
          * actuator_max["distance_scale"]
        ),
        # Joint 5 upper bound.
        j5 - joint5["j5_max"],
        # Joint 5 lower bound.
        joint5["j5_min"] - j5,
      ],
      dim=-1,
    )

    # At this knee angle and leg length, center the feasible j4 interval at j5 = 0.
    # Add back the knee-dependent offset to express normalized bounds in j4.
    low = torch.clamp(
      max(j4_knee_rel_limit["x_min"], tibia_bars["offset"]) + (j4 - j4_knee_rel),
      min=actuator_max["offset"],
    )
    high = torch.clamp(crank_j4_max, max=actuator_min["offset"])
    j4_center = 0.5 * (low + high)
    j4_knee_rel_center = j4_knee_rel + (j4_center - j4)
    center_distances = torch.stack(
      [
        # Lower bound on joint 4 normalized to the knee, at the center.
        j4_knee_rel_limit["x_min"] - j4_knee_rel_center,
        # Tibia bar clearance at joint 5 = 0.
        (tibia_bars["offset"] - j4_knee_rel_center) * tibia_bars["distance_scale"],
        # Crank collision clearance at joint 5 = 0.
        (j4_center - crank_j4_max) * crank_collision["distance_scale"],
        # Minimum actuator travel at joint 5 = 0.
        (j4_center - actuator_min["offset"]) * actuator_min["distance_scale"],
        # Maximum actuator travel at joint 5 = 0.
        (actuator_max["offset"] - j4_center) * actuator_max["distance_scale"],
        # Joint 5 upper bound, evaluated at zero.
        torch.zeros_like(j4_center) - joint5["j5_max"],
        # Joint 5 lower bound, evaluated at zero.
        torch.zeros_like(j4_center) + joint5["j5_min"],
      ],
      dim=-1,
    )

    # Move each boundary toward the center by shrink times its distance from it.
    # Preserve a positive penalty when the lower bound exceeds the upper bound.
    return torch.maximum((distances - shrink * center_distances).amax(-1), low - high)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    joint_names_group: list[list[str]],
    limits: dict[str, dict[str, float]],
    shrink: float = 0.0,
  ) -> torch.Tensor:
    del joint_names_group, limits  # Resolved in __init__.
    joint_pos = env.scene[asset_cfg.name].data.joint_pos
    max_violation = self.violation(
      joint_pos[:, self._j4_ids],
      joint_pos[:, self._j5_ids],
      joint_pos[:, self._knee_ids],
      joint_pos[:, self._leg_length_ids],
      shrink,
    )
    violation = torch.clamp(max_violation, min=0.0)
    env.extras["log"]["Metrics/joint_envelope_ankle"] = torch.mean(violation.sum(dim=1))
    return torch.sum(torch.square(violation), dim=1)


class hip_envelope_limits:
  r"""Penalize hip configurations outside the fitted envelope.

  ``joint_names_group`` contains [joint 2, joint 3] per leg, both in radians.
  Joint ranges come from the model, and actuator coefficients from ``limits``.
  See ``KANGAROO_HIP_ENVELOPE_LIMITS`` for the actuator fit and its assumptions.

  Feasible space with symmetric joint 3 bounds (schematic, not to scale)::

                          j2 [rad]
                             ^
                     +-------+-------+  j2_max
                    /                 \
       actuator_max/                   \ actuator_max
                  +                     +
           j3_min |      feasible       | j3_max
                  |          o center   |
                  +                     +
       actuator_min\                   / actuator_min
                    \                 /
                     +---------------+  j2_min
                             0                  -> j3 [rad]

  ``shrink`` is a fraction in [0, 1]: 0.02 retains 98% of the linear extent
  about the fixed center at joint 3 = 0 and the midpoint of the feasible joint 2
  interval. The joint 3 range must contain zero; its bounds may be asymmetric.

  For each leg, square the largest positive constraint violation (zero inside).
  Return the sum across legs [rad^2], with shape (num_envs,). The metric
  ``Metrics/joint_envelope_hip`` averages the sum of unsquared violations across
  environments [rad]. These violations are boundary distances, not the Euclidean
  distance to the feasible region. Conflicting joint 2 bounds impose a minimum
  violation equal to their positive gap.
  """

  _LIMIT_NAMES = ("actuator_min", "actuator_max")

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    asset: Entity = env.scene[params["asset_cfg"].name]
    self._j2_ids, self._j3_ids = _split_joint_groups(
      asset, params["joint_names_group"], ("joint 2", "joint 3")
    )
    self._limits = limits = _resolve_envelope_limits(
      params["limits"], self._LIMIT_NAMES
    )
    # Per-leg tensors from the model's joint ranges (MJCF `range`).
    joint_limits = asset.data.default_joint_pos_limits[0]
    self._j2_min, self._j2_max = joint_limits[self._j2_ids].unbind(-1)
    self._j3_min, self._j3_max = joint_limits[self._j3_ids].unbind(-1)

    # The center is fixed: j3 = 0, j2 at the midpoint of its feasible interval.
    actuator_min, actuator_max = limits["actuator_min"], limits["actuator_max"]
    low = torch.clamp(self._j2_min, min=actuator_min["offset"])
    high = torch.clamp(self._j2_max, max=actuator_max["offset"])
    self._bound_gap = low - high  # > 0: the envelope is empty.
    j2_center = 0.5 * (low + high)
    self._center_distances = torch.stack(
      [
        # Joint 2 lower bound, evaluated at the center.
        self._j2_min - j2_center,
        # Joint 2 upper bound, evaluated at the center.
        j2_center - self._j2_max,
        # Minimum actuator travel at joint 3 = 0.
        (actuator_min["offset"] - j2_center) * actuator_min["distance_scale"],
        # Maximum actuator travel at joint 3 = 0.
        (j2_center - actuator_max["offset"]) * actuator_max["distance_scale"],
        # Joint 3 upper bound, evaluated at zero.
        -self._j3_max,
        # Joint 3 lower bound, evaluated at zero.
        self._j3_min,
      ],
      dim=-1,
    )  # (n_legs, 6)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    joint_names_group: list[list[str]],
    limits: dict[str, dict[str, float]],
    shrink: float = 0.0,
  ) -> torch.Tensor:
    del joint_names_group, limits  # Resolved in __init__.
    joint_pos = env.scene[asset_cfg.name].data.joint_pos
    j2 = joint_pos[:, self._j2_ids]  # (num_envs, n_legs)
    j3 = joint_pos[:, self._j3_ids]
    abs_j3 = j3.abs()
    actuator_min, actuator_max = (
      self._limits["actuator_min"],
      self._limits["actuator_max"],
    )
    # Mirroring j3 swaps the two actuators, so their bounds depend on abs(j3).
    # Each residual is <= 0 inside; scale sloped residuals to boundary distances [rad].
    distances = torch.stack(
      [
        # Joint 2 lower bound.
        self._j2_min - j2,
        # Joint 2 upper bound.
        j2 - self._j2_max,
        # Minimum actuator travel: lower bound on joint 2.
        (
          (actuator_min["offset"] + actuator_min["slope"] * abs_j3 - j2)
          * actuator_min["distance_scale"]
        ),
        # Maximum actuator travel: upper bound on joint 2.
        (
          (j2 - actuator_max["offset"] + actuator_max["slope"] * abs_j3)
          * actuator_max["distance_scale"]
        ),
        # Joint 3 upper bound.
        j3 - self._j3_max,
        # Joint 3 lower bound.
        self._j3_min - j3,
      ],
      dim=-1,
    )
    # Move each boundary toward the center by shrink times its distance from it.
    max_violation = (distances - shrink * self._center_distances.to(distances)).amax(-1)
    max_violation = torch.maximum(max_violation, self._bound_gap.to(max_violation))
    violation = torch.clamp(max_violation, min=0.0)
    env.extras["log"]["Metrics/joint_envelope_hip"] = torch.mean(violation.sum(dim=1))
    return torch.sum(torch.square(violation), dim=1)


class joint_vel_limits:
  """Penalize joint velocities if they cross soft limits."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    joint_ids, resolved_names = asset.find_joints(
      cfg.params["asset_cfg"].joint_names,
    )

    index_list, _, limits = resolve_matching_names_values(
      data=cfg.params["velocity_limits"],
      list_of_strings=resolved_names,
    )
    limits = torch.tensor(limits, device=env.device, dtype=torch.float32)
    self._soft_vel_limits = 0.9 * limits
    # Only the joints covered by velocity_limits; may be a subset of asset_cfg joints.
    self._joint_ids = [joint_ids[i] for i in index_list]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    velocity_limits,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del velocity_limits  # Resolved in __init__.
    asset: Entity = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, self._joint_ids]

    out_of_limits = -(joint_vel - self._soft_vel_limits[:, 0]).clip(max=0.0)
    out_of_limits += (joint_vel - self._soft_vel_limits[:, 1]).clip(min=0.0)

    penalty = torch.sum(out_of_limits, dim=1)

    env.extras["log"]["Metrics/joint_vel_max"] = torch.max(torch.abs(joint_vel)).item()
    env.extras["log"]["Metrics/joint_vel_limit_violation"] = torch.mean(penalty).item()

    return penalty


class sound_suppression:
  """Soft-landing reward that penalizes impulsive contact forces at footstrike.

  Ref: https://arxiv.org/abs/2512.16705 (Disney Olaf paper)
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    # self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.prev_site_velocities = torch.zeros(
      (env.num_envs, len(self.site_names), 3),
      device=env.device,
      dtype=torch.float32,
    )
    self.step_dt = env.step_dt
    self.delta_vel_max = (
      torch.ones(1, device=env.device, dtype=torch.float32)
      * cfg.params["delta_vel_max"]
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    metrics_sensor_name: str,
    delta_vel_max: float,
  ) -> torch.Tensor:
    # Calculate and set the metrics
    contact_sensor: ContactSensor = env.scene[metrics_sensor_name]
    sensor_data = contact_sensor.data
    assert sensor_data.force is not None
    forces = sensor_data.force  # [B, N, 3]
    force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
    landing_impact = force_magnitude * first_contact.float()  # [B, N]
    cost = torch.sum(landing_impact, dim=1)  # [B]
    num_landings = torch.sum(first_contact.float())
    mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
    env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force

    asset: Entity = env.scene[asset_cfg.name]
    site_velocities = asset.data.site_lin_vel_w[:, asset_cfg.site_ids]
    change_in_site_velocities = site_velocities - self.prev_site_velocities
    self.prev_site_velocities = site_velocities.clone()
    # Calculate the squared sum of change in velocity along the projected gravity direction
    gravity_vector_w = asset.data.gravity_vec_w

    term = change_in_site_velocities * gravity_vector_w.unsqueeze(1)

    squared_term_along_gravity_vector_w = torch.square(term)

    change_in_velocities_along_gravity_vector_w = squared_term_along_gravity_vector_w[
      ..., 2
    ]  # shape (N, M)

    env.extras["log"]["Metrics/delta_v_max"] = torch.max(
      change_in_velocities_along_gravity_vector_w
    ).item()

    cost = torch.sum(
      torch.min(
        self.delta_vel_max,
        change_in_velocities_along_gravity_vector_w,
      ),
      dim=1,
    )

    # print(f"The z_component : {change_in_velocities_along_gravity_vector_w} and max_component : {max_of_all_sites} and "
    # "cost : {cost} and the term is : {term} and squared_term_along_gravity_vector_w : {squared_term_along_gravity_vector_w}")

    # return the squared sum of change in velocity along the projected gravity direction
    return cost


def track_body_ang_vel_z_exp(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track commanded yaw rate (z, body frame) with exponential kernel."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  z_error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])
  return torch.exp(-z_error / std**2)


def body_ang_vel_xy_l2_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize roll/pitch rates in the body frame."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel_b = asset.data.root_link_ang_vel_b  # body frame for consistency
  return torch.sum(torch.square(ang_vel_b[:, :2]), dim=1)
