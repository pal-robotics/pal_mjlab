"""Useful methods for custom MDP Events."""

import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.envs.mdp.events import reset_root_state_uniform, resolve_env_ids
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pal_mjlab.actuators import BacklashPdActuator

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def reset_root_state_terrain_yaw(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  terrain_yaw: dict[str, tuple[float, ...]],
  yaw_offset_range: tuple[float, float] = (0.0, 0.0),
  velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Like ``reset_root_state_uniform``, but with a terrain-dependent yaw.

  Envs on the sub-terrains named in ``terrain_yaw`` (curriculum generator only)
  spawn facing one of the given base yaws (picked uniformly), with the yaw
  sampled uniformly from ``(yaw + yaw_offset_range[0], yaw + yaw_offset_range[1])``,
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
        "yaw": (yaw + yaw_offset_range[0], yaw + yaw_offset_range[1]),
      }
      reset_root_state_uniform(env, sub_ids, sub_pose_range, velocity_range, asset_cfg)


def randomize_backlash(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  backlash_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Sample a per-env, per-joint total backlash (rad) for every BacklashPdActuator."""
  env_ids = resolve_env_ids(env, env_ids)
  asset = env.scene[asset_cfg.name]
  actuators = [a for a in asset.actuators if isinstance(a, BacklashPdActuator)]
  assert actuators, "randomize_backlash: the asset has no BacklashPdActuator."
  for act in actuators:
    num_targets = act.target_ids.numel()
    samples = torch.empty((len(env_ids), num_targets), device=env.device).uniform_(
      *backlash_range
    )
    act.set_backlash(env_ids, samples)
