from __future__ import annotations

from typing import TYPE_CHECKING

# from mjlab.tests.test_runner import env
import mujoco
import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.tasks.velocity.mdp.rewards import feet_air_time as _mjlab_feet_air_time
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

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


class joint_limits_convex_hull:
  """
  joint_limits_convex_hull is mainly to penalize the commands that are outside the convex hull of the joint limits.
  This is a more flexible way to enforce joint limits, especially for complex robots where the feasible joint space
  may not be a simple box defined by min/max limits on each joint. By using the convex hull of the joint limits,
  we can capture the true feasible joint space and penalize any commands that fall outside of it.
  This can help improve the realism and safety of the robot's movements.
  """

  def __init__(
    self,
    cfg: RewardTermCfg,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    self.convex_hull = None
    self.equations = None
    self.equation_coeff_A = None
    self.equation_coeff_b = None
    self.original_equation_coeff_b = None  # Store original b coefficients

    for joint_group in cfg.params["joint_names_group"]:
      asset: Entity = env.scene[asset_cfg.name]
      target_ids, target_names = asset.find_joints(joint_group)
      # print("Target joint names:", target_names)
      # print("Target joint ids:", target_ids)
      joint_pos = asset.data.joint_pos[:, target_ids]

      if self.convex_hull is None:
        self.convex_hull = ConvexHull(cfg.params["hull_points"].cpu().numpy())
        self.equations = torch.from_numpy(self.convex_hull.equations).to(
          device=joint_pos.device, dtype=joint_pos.dtype
        )
        self.equation_coeff_A = self.equations[:, :-1].to(
          device=joint_pos.device, dtype=joint_pos.dtype
        )  # Normals
        self.equation_coeff_b = self.equations[:, -1].to(
          device=joint_pos.device, dtype=joint_pos.dtype
        )  # Offsets
        print("Convex hull equations device", self.equations.device)
        # Apply margin reduction to shrink the convex hull
        self._reduce_convex_hull(cfg.params["margin"])
        # Plot comparison of original and reduced hulls
        self._plot_hull_comparison(
          cfg.params["hull_points"].cpu().numpy(),
          cfg.params["margin"],
          f"/tmp/convex_hull_comparison_{cfg.params['metrics_suffix']}.png",
        )

  def _reduce_convex_hull(self, margin: float) -> None:
    """
    Reduce the convex hull by moving each hyperplane inward by the specified margin.
    This shrinks the feasible region to create a safety buffer.

    Args:
        margin: The distance by which to shrink each hyperplane (must be non-negative)
    """
    if self.convex_hull is None or margin <= 0:
      return

    # Store original b coefficients before modification
    if self.original_equation_coeff_b is None:
      self.original_equation_coeff_b = self.equation_coeff_b.clone()

    # Get the normal vectors (normalized)
    normals = self.equation_coeff_A
    # Compute the norm of each normal vector
    norm_magnitudes = torch.norm(normals, dim=1, keepdim=True)
    # Normalize the vectors
    # normalized_normals = normals / (norm_magnitudes + 1e-8)
    # Reduce the feasible region by increasing b (making constraint tighter)
    # For equations Ax + b <= 0, increasing b shrinks the region
    self.equation_coeff_b = self.equation_coeff_b + margin * norm_magnitudes.squeeze(1)

  def _plot_hull_comparison(
    self, hull_points: np.ndarray, margin: float, save_path: str
  ) -> None:
    """
    Plot and save comparison of original and margin-reduced convex hulls.

    Args:
        hull_points: Original points defining the convex hull
        margin: The margin used to reduce the hull
        save_path: Path where to save the plot image
    """
    if self.convex_hull is None:
      print("Convex hull not yet initialized")
      return

    ndim = hull_points.shape[1]

    if ndim == 2:
      self._plot_2d_comparison(hull_points, margin, save_path)
    elif ndim == 3:
      self._plot_3d_comparison(hull_points, margin, save_path)
    else:
      print(f"Convex hull visualization not supported for {ndim}D points")

  def _plot_2d_comparison(
    self, hull_points: np.ndarray, margin: float, save_path: str
  ) -> None:
    """Plot 2D comparison of original and reduced convex hulls."""
    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot original hull points
    ax.scatter(
      hull_points[:, 0],
      hull_points[:, 1],
      c="blue",
      s=100,
      label="Original hull vertices",
      zorder=5,
      edgecolors="black",
      linewidths=1.5,
    )

    # Plot original hull edges
    for simplex in self.convex_hull.simplices:
      ax.plot(
        hull_points[simplex, 0],
        hull_points[simplex, 1],
        "b-",
        linewidth=2.5,
        label="Original hull" if simplex[0] == self.convex_hull.simplices[0][0] else "",
        alpha=0.8,
      )

    # Generate sample points to visualize the reduced hull
    if margin > 0 and self.original_equation_coeff_b is not None:
      # Create a grid of points to test
      x_min, x_max = hull_points[:, 0].min() - 0.2, hull_points[:, 0].max() + 0.2
      y_min, y_max = hull_points[:, 1].min() - 0.2, hull_points[:, 1].max() + 0.2
      xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200)
      )
      grid_points = np.c_[xx.ravel(), yy.ravel()]

      # Check which points are inside the reduced hull
      A = self.equation_coeff_A.cpu().numpy()
      b_reduced = self.equation_coeff_b.cpu().numpy()

      # For 2D: Ax + b <= 0
      dots = grid_points @ A.T + b_reduced
      inside_reduced = np.all(dots <= 1e-6, axis=1)

      # Plot the reduced hull boundary
      inside_reduced_grid = inside_reduced.reshape(xx.shape)
      ax.contour(
        xx,
        yy,
        inside_reduced_grid,
        levels=[0.5],
        colors="red",
        linewidths=2.5,
        linestyles="--",
      )

      # Add a dummy line for legend
      ax.plot([], [], "r--", linewidth=2.5, label=f"Reduced hull (margin={margin:.3f})")

    ax.set_xlabel("Joint 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("Joint 2", fontsize=12, fontweight="bold")
    ax.set_title(
      f"Convex Hull Comparison: Original vs Reduced (margin={margin:.3f})",
      fontsize=14,
      fontweight="bold",
    )
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"2D Convex hull comparison saved to: {save_path}")

  def _plot_3d_comparison(
    self, hull_points: np.ndarray, margin: float, save_path: str
  ) -> None:
    """Plot 3D comparison of original and reduced convex hulls."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot original hull points
    ax.scatter(
      hull_points[:, 0],
      hull_points[:, 1],
      hull_points[:, 2],
      c="blue",
      s=100,
      label="Original hull vertices",
      edgecolors="black",
      linewidths=1.5,
      zorder=10,
    )

    # Plot original hull surface
    for simplex in self.convex_hull.simplices:
      triangle = hull_points[simplex]
      from mpl_toolkits.mplot3d.art3d import Poly3DCollection

      poly = Poly3DCollection(
        [triangle], alpha=0.3, facecolor="blue", edgecolor="darkblue", linewidth=1.5
      )
      ax.add_collection3d(poly)

    # For the reduced hull, we'll visualize it by showing points that are inside
    if margin > 0 and self.original_equation_coeff_b is not None:
      # Generate points slightly inside the reduced hull boundary
      reduced_points = []
      for simplex in self.convex_hull.simplices:
        triangle = hull_points[simplex]
        # Get the centroid of the whole hull
        centroid = hull_points.mean(axis=0)
        # Move triangle vertices toward centroid
        for vertex in triangle:
          direction = centroid - vertex
          # Move approximately by margin distance
          new_point = vertex + direction * (margin / np.linalg.norm(direction)) * 0.8
          reduced_points.append(new_point)

      reduced_points = np.array(reduced_points)

      # Plot reduced hull surface approximation
      from scipy.spatial import ConvexHull as CH

      try:
        reduced_hull = CH(reduced_points)
        for simplex in reduced_hull.simplices:
          triangle = reduced_points[simplex]
          poly = Poly3DCollection(
            [triangle],
            alpha=0.3,
            facecolor="red",
            edgecolor="darkred",
            linewidth=1.5,
            linestyle="--",
          )
          ax.add_collection3d(poly)
      except Exception:
        # If reduced hull cannot be computed, just plot points
        ax.scatter(
          reduced_points[:, 0],
          reduced_points[:, 1],
          reduced_points[:, 2],
          c="red",
          s=20,
          alpha=0.5,
          label=f"Reduced hull region (margin={margin:.3f})",
        )

    ax.set_xlabel("Joint 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("Joint 2", fontsize=12, fontweight="bold")
    ax.set_zlabel("Joint 3", fontsize=12, fontweight="bold")
    ax.set_title(
      f"3D Convex Hull Comparison: Original vs Reduced (margin={margin:.3f})",
      fontsize=14,
      fontweight="bold",
    )

    # Create custom legend
    from matplotlib.patches import Patch

    legend_elements = [
      Patch(facecolor="blue", edgecolor="darkblue", alpha=0.3, label="Original hull"),
      Patch(
        facecolor="red",
        edgecolor="darkred",
        alpha=0.3,
        label=f"Reduced hull (margin={margin:.3f})",
      ),
    ]
    ax.legend(handles=legend_elements, fontsize=11, loc="best")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"3D Convex hull comparison saved to: {save_path}")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    metrics_suffix: str,
    margin: float,
    joint_names_group: list[list[str]],
    hull_points: torch.Tensor,
  ) -> torch.Tensor:
    penalty = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    metrics_violation_dist = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.float32
    )
    for joint_group in joint_names_group:
      asset: Entity = env.scene[asset_cfg.name]
      target_ids, target_names = asset.find_joints(joint_group)
      # print("Target joint names:", target_names)
      # print("Target joint ids:", target_ids)
      joint_pos = asset.data.joint_pos[:, target_ids]

      # Calculate distance using A*x + b
      # equation_coeff_A has shape (K, N), joint_pos has shape (M, N)
      # equation_coeff_b has shape (K,)
      # Result has shape (M, K)
      dot_product_res = (
        torch.matmul(joint_pos, self.equation_coeff_A.T) + self.equation_coeff_b
      )
      # For those that are within the polygon return 0.0, but for others return the squared distance to the polygon
      violation_dist = torch.clamp(dot_product_res, min=0.0).max(dim=1)[0]
      penalty += torch.square(violation_dist)
      metrics_violation_dist += violation_dist

    env.extras["log"][f"Metrics/joint_limits_hull_{metrics_suffix}"] = torch.mean(
      metrics_violation_dist
    )
    return penalty


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


def _speed_scaled_std(
  command: torch.Tensor, std: float, std_per_mps: float
) -> torch.Tensor:
  """Per-env kernel width: ``max(std, std_per_mps * |command_xy|)``.

  Keeps the kernel tight at low commanded speed and widens it linearly with the
  commanded planar speed, so a high-speed command still gives a usable gradient
  (e.g. a 4 m/s command with ``std_per_mps=0.25`` gives sigma=1.0, where a fixed
  ``sqrt(0.1)`` std gives ~0 reward to anything slower than 3.5 m/s).
  """
  speed = torch.linalg.norm(command[:, :2], dim=1)
  return torch.clamp(std_per_mps * speed, min=std)


def track_linear_velocity_speed_scaled(
  env: ManagerBasedRlEnv,
  std: float,
  std_per_mps: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """``track_linear_velocity`` with a std that grows with the commanded speed.

  Same error as the mjlab term (xy tracking + z velocity, body frame); the kernel
  width is ``max(std, std_per_mps * |command_xy|)``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  sigma = _speed_scaled_std(command, std, std_per_mps)
  return torch.exp(-(xy_error + z_error) / sigma**2)


def track_angular_velocity_speed_scaled(
  env: ManagerBasedRlEnv,
  std: float,
  std_per_mps: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """``track_angular_velocity`` with a std that grows with the commanded speed.

  Same error as the mjlab term (yaw-rate tracking + roll/pitch rates, body frame).
  Roll/pitch oscillation naturally grows when running, so the kernel width is
  ``max(std, std_per_mps * |command_xy|)`` (scaled by the *linear* command).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  sigma = _speed_scaled_std(command, std, std_per_mps)
  return torch.exp(-(z_error + xy_error) / sigma**2)


def forward_progress_gate(
  env: ManagerBasedRlEnv,
  command_name: str,
  gate_start_speed: float = 0.5,
  gate_full_speed: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Fraction of the commanded planar velocity actually achieved, in [0, 1].

  ``progress = clamp(<v_xy, cmd_xy> / |cmd_xy|^2, 0, 1)`` (body frame). The gate is
  1 (no effect) for commanded speeds below ``gate_start_speed``, so standing,
  turning on the spot and slow walking are unchanged, and blends linearly into
  ``progress`` up to ``gate_full_speed``. Used to stop "style" rewards (air time,
  arm swing) from paying a robot that steps in place instead of moving.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  cmd_xy = command[:, :2]
  speed_sq = torch.sum(torch.square(cmd_xy), dim=1)
  along = torch.sum(asset.data.root_link_lin_vel_b[:, :2] * cmd_xy, dim=1)
  progress = (along / speed_sq.clamp(min=1e-6)).clamp(0.0, 1.0)
  span = max(gate_full_speed - gate_start_speed, 1e-6)
  blend = ((torch.sqrt(speed_sq) - gate_start_speed) / span).clamp(0.0, 1.0)
  return 1.0 - blend * (1.0 - progress)


def feet_air_time_progress_gated(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_threshold: float = 0.5,
  gate_start_speed: float = 0.5,
  gate_full_speed: float = 1.0,
) -> torch.Tensor:
  """mjlab ``feet_air_time`` multiplied by :func:`forward_progress_gate`."""
  reward = _mjlab_feet_air_time(
    env,
    sensor_name=sensor_name,
    threshold_min=threshold_min,
    threshold_max=threshold_max,
    command_name=command_name,
    command_threshold=command_threshold,
  )
  return reward * forward_progress_gate(
    env, command_name, gate_start_speed, gate_full_speed
  )


def body_ang_vel_xy_l2_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize roll/pitch rates in the body frame."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel_b = asset.data.root_link_ang_vel_b  # body frame for consistency
  return torch.sum(torch.square(ang_vel_b[:, :2]), dim=1)


def _segment_segment_distance(
  p1: torch.Tensor, q1: torch.Tensor, p2: torch.Tensor, q2: torch.Tensor
) -> torch.Tensor:
  """Minimum distance between segments [p1, q1] and [p2, q2] (broadcast over leading dims).

  Clamped closest-point computation (Ericson, Real-Time Collision Detection 5.1.9).
  """
  eps = 1e-9
  d1 = q1 - p1
  d2 = q2 - p2
  r = p1 - p2
  a = (d1 * d1).sum(-1)
  e = (d2 * d2).sum(-1)
  f = (d2 * r).sum(-1)
  c = (d1 * r).sum(-1)
  b = (d1 * d2).sum(-1)
  denom = a * e - b * b
  # Closest point on infinite line 1 to line 2 (0 if parallel), clamped to segment 1.
  s = torch.where(
    denom > eps, ((b * f - c * e) / denom.clamp(min=eps)).clamp(0.0, 1.0), 0.0
  )
  t = (b * s + f) / e.clamp(min=eps)
  # If t falls outside [0, 1], clamp it and recompute s for that endpoint.
  t_clamped = t.clamp(0.0, 1.0)
  s = torch.where(
    t != t_clamped, ((b * t_clamped - c) / a.clamp(min=eps)).clamp(0.0, 1.0), s
  )
  closest1 = p1 + d1 * s.unsqueeze(-1)
  closest2 = p2 + d2 * t_clamped.unsqueeze(-1)
  return torch.linalg.norm(closest1 - closest2, dim=-1)


class feet_proximity:
  """Penalize the two feet coming closer than a safety margin.

  Computes the exact minimum surface distance between the capsule geoms of the
  left and the right foot. The cost is ``relu(margin - clearance) / margin``:
  0 while the feet are at least ``margin`` apart, 1 at contact, >1 when
  penetrating. Unlike a contact count it is graded, so the policy gets a signal
  before the feet touch and learns to keep a safety gap (e.g. to absorb hip-roll
  backlash on the real robot).

  With ``margin_per_mps > 0`` the margin grows with the actual planar base speed,
  ``margin + margin_per_mps * |v_xy|``, since the legs flex (and the feet swing
  closer) more when running than when walking.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name]
    mj_model = env.sim.mj_model
    self._asset = asset

    def _resolve(names) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
      local_ids, resolved = asset.find_geoms(names)
      assert local_ids, f"feet_proximity: no geoms match {names}"
      global_ids = asset.indexing.geom_ids[local_ids].cpu().numpy()
      for gid, name in zip(global_ids, resolved, strict=True):
        assert mj_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CAPSULE, (
          f"feet_proximity only supports capsules, '{name}' is not one."
        )
      size = torch.tensor(mj_model.geom_size[global_ids], device=env.device)
      ids = torch.tensor(global_ids, device=env.device, dtype=torch.long)
      return ids, size[:, 0].float(), size[:, 1].float()

    self._left_ids, self._left_radius, self._left_half = _resolve(
      cfg.params["left_geom_names"]
    )
    self._right_ids, self._right_radius, self._right_half = _resolve(
      cfg.params["right_geom_names"]
    )

  def _segments(self, env: ManagerBasedRlEnv, ids: torch.Tensor, half: torch.Tensor):
    pos = env.sim.data.geom_xpos[:, ids]  # [B, G, 3]
    xmat = env.sim.data.geom_xmat[:, ids].reshape(pos.shape[0], -1, 3, 3)
    axis = xmat[..., :, 2] * half.unsqueeze(-1)  # Capsule axis is the local z.
    return pos - axis, pos + axis

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    left_geom_names,
    right_geom_names,
    margin: float,
    margin_per_mps: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del left_geom_names, right_geom_names, asset_cfg  # Resolved in __init__.
    lp, lq = self._segments(env, self._left_ids, self._left_half)
    rp, rq = self._segments(env, self._right_ids, self._right_half)
    # All left/right pairs: [B, L, R].
    dist = _segment_segment_distance(
      lp.unsqueeze(2), lq.unsqueeze(2), rp.unsqueeze(1), rq.unsqueeze(1)
    )
    clearance = dist - self._left_radius[:, None] - self._right_radius[None, :]
    min_clearance = clearance.flatten(1).min(dim=1).values  # [B]

    log = env.extras.setdefault("log", {})
    log["Metrics/feet_min_clearance_mean"] = min_clearance.mean().item()
    log["Metrics/feet_min_clearance_min"] = min_clearance.min().item()

    if margin_per_mps > 0.0:
      speed = torch.linalg.norm(self._asset.data.root_link_lin_vel_b[:, :2], dim=1)
      margin_t = margin + margin_per_mps * speed
    else:
      margin_t = torch.full_like(min_clearance, margin)
    return torch.clamp(margin_t - min_clearance, min=0.0) / margin_t


class arm_swing:
  """Reward a human-like arm swing, counter-phase to the legs and scaled by speed.

  Each shoulder pitch is asked to follow the opposite leg: when the left leg swings
  forward the left arm swings back and the right arm forward. The gait phase is read
  from the hip pitch difference, so the arms stay synchronized with whatever gait the
  policy learns (no clock needed), and the swing amplitude is set by the command.

  Optionally the elbows follow too, as in human gait: they bend more on the forward
  swing and open on the back swing (elbow_swing_ratio), and they carry a speed
  dependent extra bend so running arms are more flexed (elbow_bend_per_mps):

    phase    = clip((hip_flex_L - hip_flex_R) / (2 * hip_flexion_ref), -1, 1)
    A        = min(amplitude_per_mps * |v_x_cmd|, max_amplitude)
    shoulder = (-A * phase, +A * phase)  (left, right flexion offsets)
    elbow    = elbow_swing_ratio * shoulder
               + min(elbow_bend_per_mps * |v_x_cmd|, max_elbow_bend)

  Optionally the torso counter-rotates against the hips, as the human thorax does
  against the pelvis: the shoulder that swings its arm forward also comes forward.
  The yaw is positive counter-clockwise seen from above (left shoulder back):

    torso    = min(torso_yaw_per_mps * |v_x_cmd|, max_torso_yaw) * phase

    reward   = exp(-mean((err / std_j)^2)), only when |v_xy_cmd| > command_threshold
               with std_j = std for the arm joints and torso_yaw_std for the torso.

  Flexion is "forward" (shoulder) / "bending" (elbow); the *_signs params map joint
  angles to it, and offsets are measured from the default joint positions.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name]
    p = cfg.params
    elbow_names = tuple(p.get("elbow_joint_names") or ())
    torso_names = (p["torso_yaw_joint_name"],) if p.get("torso_yaw_joint_name") else ()
    names = (
      *p["hip_joint_names"],
      *p["shoulder_joint_names"],
      *elbow_names,
      *torso_names,
    )
    ids, _ = asset.find_joints(names, preserve_order=True)
    assert len(ids) == len(names), f"Expected joints {names}, got {ids}."
    self._hip_ids = ids[:2]
    self._shoulder_ids = ids[2:4]
    self._elbow_ids = ids[4 : 4 + len(elbow_names)]
    self._torso_ids = ids[4 + len(elbow_names) :]

    def _t(x):
      return torch.tensor(x, device=env.device, dtype=torch.float32)

    self._hip_sign = _t(p["hip_flexion_signs"])
    self._shoulder_sign = _t(p["shoulder_flexion_signs"])
    self._elbow_sign = _t(p.get("elbow_flexion_signs", (1.0, 1.0)))
    self._torso_sign = p.get("torso_yaw_sign", 1.0)
    # Left arm opposes the left leg, right arm follows it (i.e. opposes the right leg).
    self._side = _t((-1.0, 1.0))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    hip_joint_names,
    shoulder_joint_names,
    hip_flexion_signs,
    shoulder_flexion_signs,
    std: float,
    amplitude_per_mps: float = 0.2,
    max_amplitude: float = 0.5,
    hip_flexion_ref: float = 0.25,
    elbow_joint_names=None,
    elbow_flexion_signs=(1.0, 1.0),
    elbow_swing_ratio: float = 0.5,
    elbow_bend_per_mps: float = 0.1,
    max_elbow_bend: float = 0.4,
    torso_yaw_joint_name: str | None = None,
    torso_yaw_sign: float = 1.0,
    torso_yaw_per_mps: float = 0.05,
    max_torso_yaw: float = 0.15,
    torso_yaw_std: float = 0.05,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    # Resolved in __init__.
    del hip_joint_names, shoulder_joint_names, elbow_joint_names, torso_yaw_joint_name
    del hip_flexion_signs, shoulder_flexion_signs, elbow_flexion_signs, torso_yaw_sign
    asset: Entity = env.scene[asset_cfg.name]
    q = asset.data.joint_pos
    q0 = asset.data.default_joint_pos
    command = env.command_manager.get_command(command_name)
    assert command is not None
    speed_x = torch.abs(command[:, 0])

    hip_flex = (q[:, self._hip_ids] - q0[:, self._hip_ids]) * self._hip_sign
    phase = torch.clamp(
      (hip_flex[:, 0] - hip_flex[:, 1]) / (2.0 * hip_flexion_ref), -1.0, 1.0
    )
    amplitude = torch.clamp(amplitude_per_mps * speed_x, max=max_amplitude)
    shoulder_target = (amplitude * phase).unsqueeze(1) * self._side  # [B, 2]
    shoulder_flex = (
      q[:, self._shoulder_ids] - q0[:, self._shoulder_ids]
    ) * self._shoulder_sign
    errors = [(shoulder_flex - shoulder_target) / std]

    log = env.extras.setdefault("log", {})
    log["Metrics/arm_swing_shoulder_error"] = (errors[0] * std).abs().mean().item()
    log["Metrics/arm_swing_target_amplitude"] = amplitude.mean().item()

    if self._elbow_ids:
      bend = torch.clamp(elbow_bend_per_mps * speed_x, max=max_elbow_bend)
      elbow_target = elbow_swing_ratio * shoulder_target + bend.unsqueeze(1)
      elbow_flex = (q[:, self._elbow_ids] - q0[:, self._elbow_ids]) * self._elbow_sign
      elbow_error = elbow_flex - elbow_target
      errors.append(elbow_error / std)
      log["Metrics/arm_swing_elbow_error"] = elbow_error.abs().mean().item()

    if self._torso_ids:
      yaw_amplitude = torch.clamp(torso_yaw_per_mps * speed_x, max=max_torso_yaw)
      torso_target = (yaw_amplitude * phase).unsqueeze(1)
      torso_yaw = (q[:, self._torso_ids] - q0[:, self._torso_ids]) * self._torso_sign
      torso_error = torso_yaw - torso_target
      errors.append(torso_error / torso_yaw_std)
      log["Metrics/arm_swing_torso_yaw_error"] = torso_error.abs().mean().item()

    normalized_error_sq = torch.mean(torch.square(torch.cat(errors, dim=1)), dim=1)
    moving = (torch.norm(command[:, :2], dim=1) > command_threshold).float()
    return torch.exp(-normalized_error_sq) * moving


class arm_swing_progress_gated(arm_swing):
  """``arm_swing`` multiplied by :func:`forward_progress_gate`.

  The arm swing is cosmetic: it must not pay a robot that swings its arms while
  stepping in place instead of tracking a high-speed command.
  """

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    gate_start_speed: float = 0.5,
    gate_full_speed: float = 1.0,
    **kwargs,
  ) -> torch.Tensor:
    reward = super().__call__(env, **kwargs)
    return reward * forward_progress_gate(
      env, kwargs["command_name"], gate_start_speed, gate_full_speed
    )
