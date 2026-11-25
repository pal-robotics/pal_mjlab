from __future__ import annotations

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


class joint_envelope_limits:
  """Penalize configurations outside a feasible joint space ``A x <= b``.

  x holds the joint positions of one leg, one per role, and the same limits apply
  to both legs. Rows of A and b are normalized, so that A x - b is the signed
  distance [rad] to every edge. In case of overlapping constraints, the violations
  of all of them are summed. The parameters are:

    asset_cfg: joints of both legs, one pattern per role, in the column order of A
      (``preserve_order=True``).
    A, b: the limits, (rows, roles) and (rows,).
    metrics_name: name of the envelope in the metrics.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    asset_cfg: SceneEntityCfg = params["asset_cfg"]
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    self.A = torch.tensor(params["A"], dtype=torch.float32, device=joint_pos.device)
    self.b = torch.tensor(params["b"], dtype=torch.float32, device=joint_pos.device)

    n_rows, self._n_roles = self.A.shape
    if self.b.shape != (n_rows,):
      raise ValueError(
        f"b must have one bound per row of A, got {tuple(self.b.shape)} for "
        f"{n_rows} rows"
      )
    n_joints = joint_pos.shape[1]
    if n_joints != 2 * self._n_roles:
      raise ValueError(
        f"asset_cfg must select {2 * self._n_roles} joints, one per column of A for "
        f"both legs, got {n_joints}: {asset_cfg.joint_names}"
      )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    A: list[list[float]],
    b: list[float],
    metrics_name: str,
  ) -> torch.Tensor:
    del A, b  # Resolved in __init__.
    asset: Entity = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    joint_pos = joint_pos.view(joint_pos.shape[0], self._n_roles, -1)
    X = joint_pos.transpose(1, 2)  # (num_envs, n_legs, n_roles)

    # Distance past every edge [rad], 0 where the edge is satisfied.
    v = (X @ self.A.T - self.b).clamp(min=0.0)  # (num_envs, n_legs, n_rows)

    violation = v.amax(dim=(1, 2))
    log = env.extras["log"]
    log[f"Metrics/joint_envelope_{metrics_name}_violation_mean"] = violation.mean()
    log[f"Metrics/joint_envelope_{metrics_name}_violation_max"] = violation.max()

    return v.sum(dim=(1, 2))


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
