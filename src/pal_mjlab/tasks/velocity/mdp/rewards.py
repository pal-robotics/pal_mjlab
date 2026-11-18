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

from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  KANGAROO_TIBIA_PITCH_OFFSET,
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


class ankle_envelope_limits:
  """Penalize ankle configurations outside the feasible joint space of the ankle.

  The space is bounded by the joint 5 range, the foot-tibia and crank-bar
  collisions and the actuator strokes. The collisions and actuators are modelled
  as affine lines, symmetric in joint 5. All limits are solved together as
  ``A x <= b`` in x = (j4, j5, j4n, L), where j4n = j4 - (70 deg + femur - knee)
  is the foot angle relative to the tibia instead of the hip. In case of
  overlapping constraints, the violations of all of them are summed and each |j5|
  pair (foot-tibia, crank-bar) counts once, as its larger row. Joint limits come
  from the model, the other limits are described by:

    j4n_min: lowest foot angle relative to the tibia [rad].
    ft_b0j5: j4n where the affine lines approximating the foot-tibia collision
      meet, at j5 = 0 [rad].
    ft_dj4ndj5: rate of change of j4n with respect to |j5|, the slope of the
      foot-tibia collision [rad/rad].
    act_a0: stroke of both actuators at j4 = j5 = 0 [m].
    act_dadj4, act_dadj5: rates of change of the actuator with respect to
      joints 4 and 5 [m/rad].
    min, max: actuator stroke limits [m].
    cb_b0l: value of j4 at the crank-bar collision when L = 0, j5 = 0 [rad].
    cb_dj4dl, cb_dj4dj5: rates of change of j4 with respect to leg length
      [rad/m] and |j5| [rad/rad], approximating the reprojected crank-bar
      collision with a plane.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    asset_cfg: SceneEntityCfg = params["asset_cfg"]
    asset: Entity = env.scene[asset_cfg.name]
    device = asset.data.joint_pos.device

    # Joint 5 range from the model, one value per leg.
    joint_limits = asset.data.default_joint_pos_limits[0, asset_cfg.joint_ids]
    n_joints = joint_limits.shape[0]
    if n_joints != 10:
      raise ValueError(
        "asset_cfg must select 10 joints, [joint 4, joint 5, leg length, femur, "
        f"knee] for both legs, got {n_joints}: {asset_cfg.joint_names}"
      )
    joint_limits = joint_limits.to(torch.float32).view(5, 2, 2)  # (joint, leg, 2)
    j5_min, j5_max = joint_limits[1].unbind(-1)

    limits = params["limits"]
    foot_tibia = limits["foot_tibia_collision"]
    act = limits["actuator"]
    crank = limits["crank_bar_collision"]

    shrink = params.get("shrink", {})
    margin_joint = shrink.get("joint", 0.0)
    margin_act = shrink.get("actuator", 0.0)
    margin_crank = shrink.get("crank", 0.0)

    j4n_min = foot_tibia["j4n_min"]
    ft_b0j5, ft_dj4ndj5 = foot_tibia["b0j5"], foot_tibia["dj4ndj5"]
    act_a0, act_dadj4, act_dadj5 = act["a0"], act["dadj4"], act["dadj5"]
    cb_b0l, cb_dj4dl, cb_dj4dj5 = crank["b0l"], crank["dj4dl"], crank["dj4dj5"]

    # The collision lines are lines in joint space, so their margin in rad is a
    # distance there: b - margin * |(a_j4, a_j5)| before normalising.
    margin_ft = margin_joint * (1 + ft_dj4ndj5**2) ** 0.5
    margin_cb = margin_crank * (1 + cb_dj4dj5**2) ** 0.5
    act_min, act_max = act["min"] + margin_act, act["max"] - margin_act

    # fmt: off
    A = torch.tensor([
      # j4           j5            j4n            L
      [ 0.0,         0.0,         -1.0,           0.0       ],  #  0 j4n >= j4n_min
      [ 0.0,         1.0,          0.0,           0.0       ],  #  1 j5 <= j5_max
      [ 0.0,        -1.0,          0.0,           0.0       ],  #  2 j5 >= j5_min
      [ 0.0,         ft_dj4ndj5,  -1.0,           0.0       ],  #  3 j4n >= ft_b0j5 + ft_dj4ndj5 j5
      [ 0.0,        -ft_dj4ndj5,  -1.0,           0.0       ],  #  4 j4n >= ft_b0j5 - ft_dj4ndj5 j5
      [ act_dadj4,   act_dadj5,    0.0,           0.0       ],  #  5 a4 <= act_max
      [-act_dadj4,  -act_dadj5,    0.0,           0.0       ],  #  6 a4 >= act_min
      [ act_dadj4,  -act_dadj5,    0.0,           0.0       ],  #  7 a5 <= act_max
      [-act_dadj4,   act_dadj5,    0.0,           0.0       ],  #  8 a5 >= act_min
      [ 1.0,         cb_dj4dj5,    0.0,          -cb_dj4dl  ],  #  9 j4 <= cb_b0l + cb_dj4dl L - cb_dj4dj5 j5
      [ 1.0,        -cb_dj4dj5,    0.0,          -cb_dj4dl  ],  # 10 j4 <= cb_b0l + cb_dj4dl L + cb_dj4dj5 j5
    ], dtype=torch.float32, device=device)

    one = torch.ones_like(j5_min)
    b = torch.stack([
      -(j4n_min + margin_joint) * one,  #  0
      j5_max - margin_joint,            #  1
      -(j5_min + margin_joint),         #  2
      (-ft_b0j5 - margin_ft) * one,     #  3
      (-ft_b0j5 - margin_ft) * one,     #  4
      (act_max - act_a0) * one,         #  5
      (act_a0 - act_min) * one,         #  6
      (act_max - act_a0) * one,         #  7
      (act_a0 - act_min) * one,         #  8
      (cb_b0l - margin_cb) * one,       #  9
      (cb_b0l - margin_cb) * one,       # 10
    ], dim=-1)
    # fmt: on

    # Normalise by dividing every row and its bound by the length of the row's
    # gradient in the (j4, j5) plane, so that A @ x - b is the signed distance
    # [rad]. j4 and j4n are collapsed into a single value (since they contribute to
    # different constraints) so the norm is correct. Also skip leg length as it
    # is not needed for the normalization.
    n = torch.stack((A[:, 0] + A[:, 2], A[:, 1]), dim=-1).norm(dim=1)
    self.A, self.b = A / n[:, None], b / n

    # Penalty terms: rows that are limits of their own, and the |j5| pairs
    # (foot-tibia, crank-bar) that count once each.
    self._single = [0, 1, 2, 5, 6, 7, 8]
    self._j5_positive = [3, 9]  # +j5 row of each pair
    self._j5_negative = [4, 10]  # -j5 row of each pair

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    limits: dict[str, dict[str, float]],
    shrink: dict[str, float] | None = None,
  ) -> torch.Tensor:
    del limits, shrink  # Resolved in __init__.
    joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    joint_pos = joint_pos.view(joint_pos.shape[0], 5, -1)  # (num_envs, joint, leg)
    j4, j5, leg_length, femur, knee = joint_pos.unbind(1)  # each (num_envs, n_legs)

    # Joint 4 position relative to the tibia, from the robot's geometry.
    j4n = j4 - (KANGAROO_TIBIA_PITCH_OFFSET + femur - knee)
    X = torch.stack((j4, j5, j4n, leg_length), dim=-1)  # (num_envs, n_legs, 4)

    # Distance past every edge [rad], 0 where the edge is satisfied.
    v = (X @ self.A.T - self.b).clamp(min=0.0)  # (num_envs, n_legs, 11)

    violation = v.amax(dim=(1, 2))
    env.extras["log"]["Metrics/joint_envelope_ankle_violation_mean"] = violation.mean()
    env.extras["log"]["Metrics/joint_envelope_ankle_violation_max"] = violation.max()

    # Sum the violations of the non-symmetric rows and the larger row of each
    # symmetric pair.
    return v[..., self._single].sum((1, 2)) + torch.maximum(
      v[..., self._j5_positive], v[..., self._j5_negative]
    ).sum((1, 2))


class hip_envelope_limits:
  """Penalize hip configurations outside the feasible joint space of hip xy.

  The space is mostly bounded by the joint 3 range and the actuator strokes;
  the actuators are modelled as affine lines, symmetric along both joints. All
  limits are solved together as ``A x <= b`` in x = (j2, j3). In case of
  overlapping constraints, the violations of all of them are summed. Joint
  limits come from the model, the actuators are described by:

    a0: stroke of both actuators at j2 = j3 = 0 [m].
    dadj2, dadj3: rates of change of the actuator with respect to joints 2 and
      3, representing linearized reprojections of the actuator limits onto joint
      space [m/rad].
    min, max: actuator stroke limits [m].
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    asset_cfg: SceneEntityCfg = params["asset_cfg"]
    asset: Entity = env.scene[asset_cfg.name]
    device = asset.data.joint_pos.device

    # Joint 2 and joint 3 ranges from the model, one value per leg.
    joint_limits = asset.data.default_joint_pos_limits[0, asset_cfg.joint_ids]
    n_joints = joint_limits.shape[0]
    if n_joints != 4:
      raise ValueError(
        "asset_cfg must select 4 joints, [joint 2, joint 3] for both legs, "
        f"got {n_joints}: {asset_cfg.joint_names}"
      )
    joint_limits = joint_limits.to(torch.float32).view(2, 2, 2)  # (joint, leg, 2)
    j2_min, j2_max = joint_limits[0].unbind(-1)
    j3_min, j3_max = joint_limits[1].unbind(-1)
    one = torch.ones_like(j2_min)

    act = params["limits"]["actuator"]

    shrink = params.get("shrink", {})
    margin_joint = shrink.get("joint", 0.0)
    margin_act = shrink.get("actuator", 0.0)

    a0, dadj2, dadj3 = act["a0"], act["dadj2"], act["dadj3"]
    act_min, act_max = act["min"] + margin_act, act["max"] - margin_act

    # fmt: off
    A = torch.tensor([
      # j2       j3
      [-1.0,     0.0   ],  # 0 j2 >= j2_min
      [ 1.0,     0.0   ],  # 1 j2 <= j2_max
      [ 0.0,     1.0   ],  # 2 j3 <= j3_max
      [ 0.0,    -1.0   ],  # 3 j3 >= j3_min
      [ dadj2,   dadj3 ],  # 4 a2 <= act_max
      [-dadj2,  -dadj3 ],  # 5 a2 >= act_min
      [ dadj2,  -dadj3 ],  # 6 a3 <= act_max
      [-dadj2,   dadj3 ],  # 7 a3 >= act_min
    ], dtype=torch.float32, device=device)
    b = torch.stack([
      -(j2_min + margin_joint),  # 0
      j2_max - margin_joint,     # 1
      j3_max - margin_joint,     # 2
      -(j3_min + margin_joint),  # 3
      (act_max - a0) * one,      # 4
      (a0 - act_min) * one,      # 5
      (act_max - a0) * one,      # 6
      (a0 - act_min) * one,      # 7
    ], dim=-1)
    # fmt: on

    # Normalise by dividing every row and its bound by the row's length, so that
    # A @ x - b is the signed distance [rad] to the edge in the (j2, j3) plane.
    n = A.norm(dim=1)
    self.A, self.b = A / n[:, None], b / n

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    limits: dict[str, dict[str, float]],
    shrink: dict[str, float] | None = None,
  ) -> torch.Tensor:
    del limits, shrink  # Resolved in __init__.
    joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    joint_pos = joint_pos.view(joint_pos.shape[0], 2, -1)  # (num_envs, joint, leg)
    j2, j3 = joint_pos.unbind(1)  # each (num_envs, n_legs)
    X = torch.stack((j2, j3), dim=-1)  # (num_envs, n_legs, 2)

    v = (X @ self.A.T - self.b).clamp(min=0.0)  # (num_envs, n_legs, 8)

    violation = v.amax(dim=(1, 2))
    env.extras["log"]["Metrics/joint_envelope_hip_violation_mean"] = violation.mean()
    env.extras["log"]["Metrics/joint_envelope_hip_violation_max"] = violation.max()
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
