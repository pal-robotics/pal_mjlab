from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds
from mjlab.managers.scene_entity_config import SceneEntityCfg


def randomize_box_size(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Randomizes the box geometry dimensions per-environment at startup."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)

    asset = env.scene[asset_cfg.name]
    geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids]

    num_envs = len(env_ids)
    box_sizes = torch.zeros(num_envs, 3, device=env.device)

    # 50% chance of standard/cube-like shape
    # 50% chance of narrow-but-long shape
    choice = torch.rand(num_envs, device=env.device) < 0.5

    x_std = torch.full((num_envs,), 0.025, device=env.device)
    x_narrow = torch.rand(num_envs, device=env.device) * (0.02 - 0.01) + 0.01
    box_sizes[:, 0] = torch.where(choice, x_std, x_narrow)

    y_std = torch.full((num_envs,), 0.025, device=env.device)
    y_narrow = torch.rand(num_envs, device=env.device) * (0.06 - 0.04) + 0.04
    box_sizes[:, 1] = torch.where(choice, y_std, y_narrow)

    # Z tallness always uniform(0.02, 0.035)
    box_sizes[:, 2] = torch.rand(num_envs, device=env.device) * (0.035 - 0.02) + 0.02

    # Map box_sizes to env.sim.model.geom_size
    env_grid, geom_grid = torch.meshgrid(env_ids, geom_ids, indexing="ij")
    expanded_sizes = box_sizes.unsqueeze(1).expand(-1, len(geom_ids), -1)

    env.sim.model.geom_size[env_grid, geom_grid] = expanded_sizes

    # Recompute broadphase bounds
    _recompute_geom_bounds(env, env_ids, asset_cfg)
