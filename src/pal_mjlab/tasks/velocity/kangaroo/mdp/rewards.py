from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.managers.reward_manager import RewardTermCfg
# from mjlab.tests.test_runner import env
import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

from scipy.spatial import ConvexHull
import numpy as np

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
        cfg: RewardTermCfg, env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        self.convex_hull = None
        self.equations = None
        self.equation_coeff_A = None
        self.equation_coeff_b = None


    def __call__(self, env: ManagerBasedRlEnv,
                asset_cfg: SceneEntityCfg,
                joint_names_group: list[list[str]],
                hull_points: torch.Tensor) -> torch.Tensor:

        penalty = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        metrics_violation_dist = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        for joint_group in joint_names_group:
            # print("Processing joint group:", joint_group)
            asset: Entity = env.scene[asset_cfg.name]
            target_ids, target_names = asset.find_joints(joint_group)
            # print("Target joint names:", target_names)
            # print("Target joint ids:", target_ids)
            joint_pos = asset.data.joint_pos[:, target_ids]

            if self.convex_hull is None:
                self.convex_hull = ConvexHull(hull_points.cpu().numpy())
                self.equations = torch.from_numpy(self.convex_hull.equations).to(device=joint_pos.device, dtype=joint_pos.dtype)
                self.equation_coeff_A = self.equations[:, :-1].to(device=joint_pos.device, dtype=joint_pos.dtype) # Normals
                self.equation_coeff_b = self.equations[:, -1].to(device=joint_pos.device, dtype=joint_pos.dtype)  # Offsets
                print("Convex hull equations device", self.equations.device)

            M = joint_pos.shape[0]
            ones = torch.ones((M, 1), dtype=joint_pos.dtype, device=joint_pos.device)
            # print("self.equation_coeff_A device", self.equation_coeff_A.device)
            # print("self.equation_coeff_b device", self.equation_coeff_b.device)
            query_points_homo = torch.cat([joint_pos, ones], dim=1)
            # print("query_points_homo", query_points_homo)

            dot_product_res = torch.matmul(query_points_homo, self.equations.T)
            # dot_product_res = joint_pos @ self.equation_coeff_A.T + self.equation_coeff_b

            # torch.all(dot_products <= 1e-9, dim=1)
            # For those that are within the polygon return 0.0, but for others return the squared distance to the polygon
            inside = torch.all(dot_product_res <= 1e-9, dim=1)
            violation_dist = torch.clamp(dot_product_res, min=0.0).max(dim=1)[0]
            penalty += torch.square(violation_dist)
            metrics_violation_dist += violation_dist
            # penalty = torch.where(inside, torch.zeros_like(inside, dtype=torch.float32), torch.square(dot_product_res))

        env.extras["log"][f"Metrics/convex_joint_limits_hull_hip_{asset_cfg.name}"] = torch.mean(metrics_violation_dist)
        return penalty