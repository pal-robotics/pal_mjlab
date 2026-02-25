from __future__ import annotations

from typing import TYPE_CHECKING

# from mjlab.tests.test_runner import env
import torch
from mjlab.entity import Entity
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
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
    _, resolved_names = asset.find_joints(
      cfg.params["asset_cfg"].joint_names,
    )

    _, _, limits = resolve_matching_names_values(
      data=cfg.params["velocity_limits"],
      list_of_strings=resolved_names,
    )
    limits = torch.tensor(limits, device=env.device, dtype=torch.float32)
    self._soft_vel_limits = 0.9 * limits

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    velocity_limits,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del velocity_limits  # Resolved in __init__.
    asset: Entity = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]

    out_of_limits = -(joint_vel - self._soft_vel_limits[:, 0]).clip(max=0.0)
    out_of_limits += (joint_vel - self._soft_vel_limits[:, 1]).clip(min=0.0)

    penalty = torch.sum(out_of_limits, dim=1)

    env.extras["log"]["Metrics/joint_vel_max"] = torch.max(torch.abs(joint_vel)).item()
    env.extras["log"]["Metrics/joint_vel_limit_violation"] = torch.mean(penalty).item()

    return penalty


class disney_soft_landing:
    """Penalize high impact forces at landing to encourage soft footfalls similar to Disney Olaf."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        # self.sensor_name = cfg.params["sensor_name"]
        self.site_names = cfg.params["asset_cfg"].site_names
        self.prev_site_velocities = torch.zeros(
            (env.num_envs, len(self.site_names), 3),
            device=env.device,
            dtype=torch.float32,
        )
        self.step_dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg,
        metrics_sensor_name: str,
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
        self.prev_site_velocities = site_velocities
        # print(change_in_site_velocities)
        # Calculate the squared sum of change in velocity along the projected gravity direction
        projected_gravity = asset.data.projected_gravity_b

        # User asked for "change in velocities along projected gravity but only z element"
        # We interpret this as:
        # 1. Project change in velocity onto the gravity direction vector?
        #    OR perform element-wise multiplication and take the Z component?
        # Based on "only z element", likely just the Z-component of the element-wise product.
        # But usually "along projected gravity" implies a dot product.
        # However, if projected_gravity is [0, 0, -1] (in body frame aligned with world),
        # then dot product is -vz.
        # We will compute the element-wise product, then take the Z component, then square it.

        # projected_gravity is (N, 3). change is (N, M, 3).
        # We need to broadcast projected_gravity to (N, 1, 3).

        term = change_in_site_velocities * projected_gravity.unsqueeze(1)
        # term shape: (N, M, 3)
        # take Z component: term[..., 2] -> shape (N, M)
        z_component = term[..., 2]

        squared_term_along_proj_gravity = torch.square(term)
        squared_change_in_site_velocities = torch.square(change_in_site_velocities)

        # print(f"Projected gravity: {projected_gravity}, change_in_site_velocities: {change_in_site_velocities}, term: {term}, z_component: {z_component}, squared_term_along_proj_gravity: {squared_term_along_proj_gravity}")

        # Get the maximum component of the 3 components squared
        max_component_x = squared_change_in_site_velocities[..., 0]
        max_component_y = squared_change_in_site_velocities[..., 1]
        max_component_z = squared_change_in_site_velocities[..., 2]

        max_value = torch.max(
            torch.max(max_component_x, max_component_y), max_component_z
        )
        max_of_all_sites = torch.max(max_value, dim=1).values  # shape (N,)
        # print(f"Max value x: {max_component_x}, y: {max_component_y}, z: {max_component_z}, max_value: {max_value} and max_of_all_sites : {max_of_all_sites}")

        change_in_velocities_along_projected_gravity = squared_term_along_proj_gravity[
            ..., 2
        ]  # shape (N, M)

        # calculate the cummulative sum of all the sites that is min(max_of_all_sites, change_in_velocities_along_projected_gravity)
        cost = torch.sum(
            torch.min(
                max_of_all_sites.unsqueeze(1),
                change_in_velocities_along_projected_gravity,
            ),
            dim=1,
        )

        # print(f"The z_component : {change_in_velocities_along_projected_gravity} and max_component : {max_of_all_sites} and cost : {cost} and the term is : {term} and squared_term_along_proj_gravity : {squared_term_along_proj_gravity}")

        # print(change_in_velocities_along_projected_gravity)
        # return the squared sum of change in velocity along the projected gravity direction
        return cost
