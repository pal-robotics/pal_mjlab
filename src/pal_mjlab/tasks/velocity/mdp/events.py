"""Useful methods for custom MDP Events."""

import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.envs.mdp.events import reset_root_state_uniform, resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_root_state_terrain_yaw(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  terrain_yaw: dict[str, tuple[float, ...]],
  yaw_noise: float = 0.3,
  velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Like ``reset_root_state_uniform``, but with a terrain-dependent yaw.

  Envs on the sub-terrains named in ``terrain_yaw`` (curriculum generator only)
  spawn facing one of the given base yaws (picked uniformly) +- ``yaw_noise``,
  e.g. facing a straight staircase. Every other env uses ``pose_range`` as is.
  """
  env_ids = resolve_env_ids(env, env_ids)
  reset_root_state_uniform(env, env_ids, pose_range, velocity_range, asset_cfg)

  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_generator is None:
    return
  if not terrain.cfg.terrain_generator.curriculum:
    return  # terrain_types only maps to sub-terrain columns in curriculum mode.

  names = list(terrain.cfg.terrain_generator.sub_terrains)
  terrain_types = terrain.terrain_types[env_ids]
  for name, yaws in terrain_yaw.items():
    ids = env_ids[terrain_types == names.index(name)]
    if len(ids) == 0:
      continue
    choice = torch.randint(len(yaws), (len(ids),), device=ids.device)
    for i, yaw in enumerate(yaws):
      sub_ids = ids[choice == i]
      if len(sub_ids) == 0:
        continue
      sub_pose_range = {**pose_range, "yaw": (yaw - yaw_noise, yaw + yaw_noise)}
      reset_root_state_uniform(env, sub_ids, sub_pose_range, velocity_range, asset_cfg)
