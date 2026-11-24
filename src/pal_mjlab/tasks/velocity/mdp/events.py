"""Useful methods for custom MDP Events."""

import torch
from mjlab.actuator.pd_actuator import IdealPdActuator
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
  target_names_expr: tuple[str, ...] | None = None,
) -> None:
  """Sample a per-env, per-joint total backlash (rad) for every BacklashPdActuator.

  With ``target_names_expr`` only the actuators configured with exactly these
  target expressions are randomized (e.g. a different range per joint group).
  """
  env_ids = resolve_env_ids(env, env_ids)
  asset = env.scene[asset_cfg.name]
  actuators = [
    a
    for a in asset.actuators
    if isinstance(a, BacklashPdActuator)
    and (target_names_expr is None or a.cfg.target_names_expr == target_names_expr)
  ]
  assert actuators, (
    f"randomize_backlash: no BacklashPdActuator matching {target_names_expr}."
  )
  for act in actuators:
    num_targets = act.target_ids.numel()
    samples = torch.empty((len(env_ids), num_targets), device=env.device).uniform_(
      *backlash_range
    )
    act.set_backlash(env_ids, samples)


def randomize_pd_gains_by_target(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  kp_range: tuple[float, float],
  kd_range: tuple[float, float],
  target_names_exprs: tuple[tuple[str, ...], ...],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Scale the default PD gains of explicit PD actuators, per env and per joint.

  Actuators are selected by their configured ``target_names_expr`` (each entry of
  ``target_names_exprs`` must match exactly one explicit PD actuator).
  """
  env_ids = resolve_env_ids(env, env_ids)
  asset = env.scene[asset_cfg.name]
  for expr in target_names_exprs:
    matches = [
      a
      for a in asset.actuators
      if isinstance(a, IdealPdActuator) and a.cfg.target_names_expr == expr
    ]
    assert len(matches) == 1, (
      f"randomize_pd_gains_by_target: {len(matches)} explicit PD actuators "
      f"match {expr}, expected 1."
    )
    act = matches[0]
    assert act.default_stiffness is not None and act.default_damping is not None
    shape = (len(env_ids), act.target_ids.numel())
    kp = torch.empty(shape, device=env.device).uniform_(*kp_range)
    kd = torch.empty(shape, device=env.device).uniform_(*kd_range)
    act.set_gains(
      env_ids,
      kp=act.default_stiffness[env_ids] * kp,
      kd=act.default_damping[env_ids] * kd,
    )
