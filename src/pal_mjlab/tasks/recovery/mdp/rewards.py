from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.string import (
    resolve_matching_names_values,
)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def orientation(
    env: ManagerBasedRlEnv, std: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    ori_err = asset.data.gravity_vec_w - asset.data.projected_gravity_b
    ori_err_squarred = torch.sum(torch.square(ori_err), dim=1)

    env.extras["log"]["Metrics/mean_orientation_error"] = ori_err

    return torch.exp(-ori_err_squarred / std**2)


def torso_height(
    env: ManagerBasedRlEnv,
    z_des: float,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    z = asset.data.root_link_pos_w[:, 2]
    z_err = z - z_des
    z_err_scaled = torch.where(z_err < 0, z_err, z_err * 0.25)
    z_err_squared = torch.square(z_err_scaled)

    env.extras["log"]["Metrics/mean_height"] = torch.mean(z)

    return torch.exp(-z_err_squared / std**2)


def head_height(
    env: ManagerBasedRlEnv,
    z_des: float,
    std: float,
    head_name: str = "head",  # either link or site,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    # Determine if head_name is a site or a body
    if head_name in asset.site_names:
        head_ids, _ = asset.find_sites(head_name)
        z = asset.data.site_pos_w[:, head_ids[0], 2]
    elif head_name in asset.body_names:
        head_ids, _ = asset.find_bodies(head_name)
        z = asset.data.body_link_pos_w[:, head_ids[0], 2]
    else:
        raise ValueError(f"'{head_name}' not found in sites or bodies")
    
    z_err = z - z_des
    z_err_scaled = torch.where(z_err < 0, z_err, z_err * 0.25)
    z_err_squared = torch.square(z_err_scaled)

    env.extras["log"]["Metrics/mean_head_height"] = torch.mean(z)

    return torch.exp(-z_err_squared / std**2)


def getup_posture(
    env: ManagerBasedRlEnv,
    z_min: float = 0.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    pos_err = torch.sum(
        torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1
    )
    rew = torch.exp(-0.5 * pos_err)

    z = asset.data.root_link_pos_w[:, 2]
    gate = z > z_min

    env.extras["log"]["Metrics/mean_position_error"] = pos_err

    return torch.where(gate, rew, torch.zeros_like(rew))


def joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def power_limit(
    env: ManagerBasedRlEnv,
    max_power: float = 400.0,  # Watts
    soft_ratio: float = 0.9,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize when instantaneous power exceeds limit."""
    asset: Entity = env.scene[asset_cfg.name]

    power = torch.sum(
        torch.abs(asset.data.actuator_force * asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1
    )
    soft_limit = soft_ratio * max_power
    over_limit = torch.clamp(power - soft_limit, min=0.0)
    penalty = (over_limit / (max_power - soft_limit)).pow(2)

    env.extras["log"]["Metrics/mean_power"] = power.mean()

    return penalty


class joint_vel_limits:
    """Penalize joint velocities that exceed softened per-joint limits."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]

        joint_ids, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

        _, _, limits = resolve_matching_names_values(
            data=cfg.params["velocity_limits"],
            list_of_strings=joint_names,
        )
        self.limits = torch.tensor(limits, device=env.device, dtype=torch.float32)

        self.joint_ids = joint_ids
        self.asset_name = cfg.params["asset_cfg"].name

        self.soft_ratio = float(cfg.params.get("soft_ratio", 0.9))
        self.clip_max = float(cfg.params.get("clip_max", 1.0))  # rad/s cap per joint

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        velocity_limits,
        soft_ratio: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        del velocity_limits, soft_ratio, asset_cfg

        asset: Entity = env.scene[self.asset_name]
        vel = torch.abs(asset.data.joint_vel[:, self.joint_ids])
        soft_limits = self.limits * self.soft_ratio
        over = (vel - soft_limits).clamp(min=0.0, max=self.clip_max)

        return over.sum(dim=1)
    

def joint_velocity_hinge_penalty(
  env: ManagerBasedRlEnv,
  max_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Quadratic hinge penalty on joint velocities exceeding a symmetric limit.

  Penalizes only the amount by which |v| exceeds max_vel. Returns a negative
  penalty, shaped as the negative squared L2 norm of the excess velocities.
  """
  robot: Entity = env.scene[asset_cfg.name]
  joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  excess = (joint_vel.abs() - max_vel).clamp_min(0.0)
  return (excess**2).sum(dim=-1)


class variable_posture_standup:
    """Penalize deviation from standing pose with height-dependent tolerance.
    
    Uses per-joint standard deviations to control how much each joint can deviate
    from the standing pose. Smaller std = stricter (less deviation allowed), larger
    std = more forgiving. The reward is: exp(-mean(error² / std²))
    
    Three height regimes (based on head height):
      - std_fallen (height < rising_threshold): Loose tolerance for exploration.
      - std_rising (rising_threshold <= height < standing_threshold): Moderate.
      - std_standing (height >= standing_threshold): Tight tolerance for good posture.
    
    Tune std values per joint based on how much freedom that joint needs at each
    height. Map joint name patterns to std values, e.g. {".*knee.*": 0.5}.
    """
    
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        
        # Standing pose (not default fallen pose!)
        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None
        self.default_joint_pos = default_joint_pos
        
        # Standard deviations for fallen state (loose)
        _, _, std_fallen = resolve_matching_names_values(
            data=cfg.params["std_fallen"],
            list_of_strings=joint_names,
        )
        self.std_fallen = torch.tensor(
            std_fallen, device=env.device, dtype=torch.float32
        )
        
        # Standard deviations for rising state (moderate)
        _, _, std_rising = resolve_matching_names_values(
            data=cfg.params["std_rising"],
            list_of_strings=joint_names,
        )
        self.std_rising = torch.tensor(
            std_rising, device=env.device, dtype=torch.float32
        )
        
        # Standard deviations for standing state (tight)
        _, _, std_standing = resolve_matching_names_values(
            data=cfg.params["std_standing"],
            list_of_strings=joint_names,
        )
        self.std_standing = torch.tensor(
            std_standing, device=env.device, dtype=torch.float32
        )
        
        # Target standing height
        self.z_des = cfg.params["z_des"]
    
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std_fallen,
        std_rising,
        std_standing,
        z_des,
        asset_cfg: SceneEntityCfg,
        head_name: str = "head",
        rising_threshold: float = 0.6,    # Height ratio for rising state
        standing_threshold: float = 0.9,  # Height ratio for standing state
    ) -> torch.Tensor:
        del std_fallen, std_rising, std_standing, z_des  # Unused (loaded in __init__)
        
        asset: Entity = env.scene[asset_cfg.name]
        
        # Get head height
        if head_name in asset.site_names:
            head_ids, _ = asset.find_sites(head_name)
            z = asset.data.site_pos_w[:, head_ids[0], 2]
        elif head_name in asset.body_names:
            head_ids, _ = asset.find_bodies(head_name)
            z = asset.data.body_link_pos_w[:, head_ids[0], 2]
        else:
            raise ValueError(f"'{head_name}' not found in sites or bodies")
        
        # Compute height ratio
        height_ratio = z / self.z_des
        
        # Create masks for different height regimes
        fallen_mask = (height_ratio < rising_threshold).float()
        rising_mask = (
            (height_ratio >= rising_threshold) & (height_ratio < standing_threshold)
        ).float()
        standing_mask = (height_ratio >= standing_threshold).float()
        
        # Interpolate standard deviation based on height
        std = (
            self.std_fallen * fallen_mask.unsqueeze(1)
            + self.std_rising * rising_mask.unsqueeze(1)
            + self.std_standing * standing_mask.unsqueeze(1)
        )
        
        # Get current and desired joint positions
        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        
        # Compute error with height-dependent tolerance
        error_squared = torch.square(current_joint_pos - desired_joint_pos)
        
        return torch.exp(-torch.mean(error_squared / (std**2), dim=1))
