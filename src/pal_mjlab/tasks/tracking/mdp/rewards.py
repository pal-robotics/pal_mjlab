from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_conjugate, quat_mul
from scipy.spatial import ConvexHull

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class site_distance_convex_hull:
  """
  site_distance_convex_hull is to penalize the relative distance between two sites
  (e.g., left and right foot) that are outside the convex hull in the base frame.
  """

  def __init__(
    self,
    cfg: RewardTermCfg,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ):
    self.convex_hull = None
    self.equations = None
    self.equation_coeff_A = None
    self.equation_coeff_b = None
    self.original_equation_coeff_b = None

    # Initialize hull from points
    self.convex_hull = ConvexHull(cfg.params["hull_points"].cpu().numpy())
    device = env.device
    self.equations = torch.from_numpy(self.convex_hull.equations).to(
      device=device, dtype=torch.float32
    )
    self.equation_coeff_A = self.equations[:, :-1]
    self.equation_coeff_b = self.equations[:, -1]

    if "margin" in cfg.params and cfg.params["margin"] > 0:
      self._reduce_convex_hull(cfg.params["margin"])

  def _reduce_convex_hull(self, margin: float) -> None:
    """
    Reduce the convex hull by moving each hyperplane inward by the specified margin.
    """
    if self.convex_hull is None or margin <= 0:
      return

    if self.original_equation_coeff_b is None:
      self.original_equation_coeff_b = self.equation_coeff_b.clone()

    normals = self.equation_coeff_A
    norm_magnitudes = torch.norm(normals, dim=1, keepdim=True)
    self.equation_coeff_b = self.equation_coeff_b + margin * norm_magnitudes.squeeze(1)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    site_names: list[str],
    hull_points: torch.Tensor,
    metrics_suffix: str = "feet",
    margin: float = 0.0,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    # Get site indices
    site_ids = [asset.site_names.index(name) for name in site_names]

    # Get site positions in world frame
    pos_l_w = asset.data.site_pos_w[:, site_ids[0]]
    pos_r_w = asset.data.site_pos_w[:, site_ids[1]]

    # Relative position in world frame
    diff_w = pos_l_w - pos_r_w

    # Transform to base frame
    root_quat_w = asset.data.site_quat_w[:, 0]

    # Rotate diff_w to base frame: v_b = R_root^T * v_w
    q_inv = quat_conjugate(root_quat_w)
    p_quat = torch.cat(
      [torch.zeros(env.num_envs, 1, device=env.device), diff_w], dim=1
    )
    diff_b = quat_mul(quat_mul(q_inv, p_quat), root_quat_w)[:, 1:]

    # We only care about XY for the hull
    diff_xy = diff_b[:, :2]

    # Calculate distance using A*x + b
    dot_product_res = (
      torch.matmul(diff_xy, self.equation_coeff_A.T) + self.equation_coeff_b
    )
    violation_dist = torch.clamp(dot_product_res, min=0.0).max(dim=1)[0]

    penalty = torch.square(violation_dist)

    env.extras["log"][f"Metrics/site_distance_hull_{metrics_suffix}"] = torch.mean(
      violation_dist
    )

    return penalty
