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
  yaw_half_range: float = 0.3,
  velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Like ``reset_root_state_uniform``, but with a terrain-dependent yaw.

  Envs on the sub-terrains named in ``terrain_yaw`` (curriculum generator only)
  spawn facing one of the given base yaws (picked uniformly), with the yaw
  sampled uniformly from ``(yaw - yaw_half_range, yaw + yaw_half_range)``,
  e.g. facing a straight staircase. Every other env uses ``pose_range`` as is.
  """
  # Validate the names first, so a typo fails on the first reset (at env
  # creation) instead of being silently skipped below.
  terrain = env.scene.terrain
  generator = None if terrain is None else terrain.cfg.terrain_generator
  names = [] if generator is None else list(generator.sub_terrains)
  unknown = sorted(set(terrain_yaw) - set(names))
  if unknown:
    raise ValueError(
      f"Unknown sub-terrain(s) {unknown} in 'terrain_yaw'. Available: {names}."
    )

  env_ids = resolve_env_ids(env, env_ids)
  reset_root_state_uniform(env, env_ids, pose_range, velocity_range, asset_cfg)

  if terrain is None or generator is None:
    return
  if not generator.curriculum:
    return  # terrain_types only maps to sub-terrain columns in curriculum mode.

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
      sub_pose_range = {
        **pose_range,
        "yaw": (yaw - yaw_half_range, yaw + yaw_half_range),
      }
      reset_root_state_uniform(env, sub_ids, sub_pose_range, velocity_range, asset_cfg)
