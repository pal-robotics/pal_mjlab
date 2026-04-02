from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from pal_mjlab.tasks.velocity.mdp.commands import (
  UniformVelocityCommandWithProgressTracking,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class RewardParamStage(TypedDict):
  step: int
  params: dict[str, object]


class EventParamStage(TypedDict):
  step: int
  params: dict[str, object]


def _deep_update(target: dict, source: dict) -> None:
  """Recursively merge ``source`` into ``target`` in-place.

  Dict-valued keys are merged rather than replaced, so callers can update a
  subset of a nested mapping without losing other entries.  All other value
  types are overwritten directly.
  """
  for key, value in source.items():
    if isinstance(value, dict) and isinstance(target.get(key), dict):
      _deep_update(target[key], value)
    else:
      target[key] = value


def reward_params(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  param_stages: list[RewardParamStage],
) -> dict[str, torch.Tensor]:
  """Update a reward term's params based on training step stages.

  Each stage specifies a ``step`` threshold and a ``params`` dict with keys
  matching the reward function's keyword arguments.  When
  ``env.common_step_counter`` exceeds a stage's ``step``, the corresponding
  params are applied.  Later stages in the list take precedence when multiple
  thresholds are exceeded.

  When a param value is itself a dict (e.g. ``std_walking`` in a posture
  reward that maps joint-name patterns to std values), the stage value is
  **deep-merged** into the existing dict so that only the specified keys are
  updated and the rest are preserved.  Pass the full dict in the stage to
  replace it entirely.

  Example — scalar param::

    CurriculumTermCfg(
      func=reward_params,
      params={
        "reward_name": "track_linear_velocity",
        "param_stages": [
          {"step": 0,    "params": {"std": 0.5}},
          {"step": 1000, "params": {"std": 0.3}},
        ],
      },
    )

  Example — dict-valued param::

    CurriculumTermCfg(
      func=reward_params,
      params={
        "reward_name": "base_height",
        "param_stages": [
          {"step": 0,    "params": {"joint": {"leg_right_knee_joint": 0.5}}},
          {"step": 1000, "params": {"joint": {"leg_right_knee_joint": 0.3}}},
        ],
      },
    )
  """
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in param_stages:
    if env.common_step_counter > stage["step"]:
      _deep_update(reward_term_cfg.params, stage["params"])
  return {
    k: torch.tensor(v) if not isinstance(v, torch.Tensor) else v
    for k, v in reward_term_cfg.params.items()
    if isinstance(v, (int, float, torch.Tensor))
  }


def event_params(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  event_name: str,
  param_stages: list[EventParamStage],
) -> dict[str, torch.Tensor]:
  """Update an event term's params based on training step stages.

  Each stage specifies a ``step`` threshold and a ``params`` dict with keys
  matching the event function's keyword arguments inside ``term_cfg.params``.

  When ``env.common_step_counter`` exceeds a stage's ``step``, the
  corresponding params are applied. Later stages take precedence when multiple
  thresholds are exceeded.

  Nested dict values are deep-merged, which is useful for event params such as
  ``velocity_range`` where only a subset of axes should be updated.

  Example::

    CurriculumTermCfg(
      func=event_params,
      params={
        "event_name": "push_robot",
        "param_stages": [
          {
            "step": 0,
            "params": {
              "velocity_range": {
                "x": (-0.10, 0.10),
                "y": (-0.10, 0.10),
                "yaw": (-0.10, 0.10),
              },
            },
          },
          {
            "step": 5000 * 24,
            "params": {
              "velocity_range": {
                "x": (-0.20, 0.20),
                "y": (-0.20, 0.20),
                "yaw": (-0.20, 0.20),
              },
            },
          },
          {
            "step": 10000 * 24,
            "params": {
              "velocity_range": {
                "x": (-0.35, 0.35),
                "y": (-0.25, 0.25),
                "yaw": (-0.35, 0.35),
              },
            },
          },
        ],
      },
    )

  Notes
  -----
  - This updates only ``term_cfg.params``.
  - It does not modify top-level event config fields such as ``mode`` or
    ``interval_range_s``.
  """
  del env_ids  # Unused.
  event_term_cfg = env.event_manager.get_term_cfg(event_name)

  applied_stage = -1
  for stage_idx, stage in enumerate(param_stages):
    if env.common_step_counter > stage["step"]:
      _deep_update(event_term_cfg.params, stage["params"])
      applied_stage = stage_idx

  # Return something numeric for curriculum logging/debugging.
  return {"stage": torch.tensor(applied_stage, dtype=torch.int64)}


def rough_terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
) -> dict[str, torch.Tensor]:
  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command_term = env.command_manager.get_term(command_name)
  command_term = cast(UniformVelocityCommandWithProgressTracking, command_term)
  command_term.flush_episode_progress(env_ids, env.step_dt)

  progress_ratio = command_term.get_episode_progress_ratio(env_ids)
  desired_progress = command_term.episode_desired_progress[env_ids]
  achieved_progress = command_term.episode_achieved_progress[env_ids]

  # Both a min desired and achieved make sense, as even with a good ratio, for low 
  # achieved (because the command was small), we don't really have a good metric
  # of the policy actually progressing over the rough terrain
  promote_threshold = 0.60
  demote_threshold = 0.25
  min_required_desired_progress = 0.5
  min_required_achieved_progress = max(0.5, terrain_generator.size[0] * 0.2)

  valid = desired_progress >= min_required_desired_progress

  move_up = (
    valid
    & (progress_ratio >= promote_threshold)
    & (achieved_progress >= min_required_achieved_progress)
  )
  move_down = valid & (progress_ratio < demote_threshold)

  terrain.update_env_origins(env_ids, move_up, move_down)

  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
    "progress_ratio_mean": torch.mean(progress_ratio),
    "desired_progress_mean": torch.mean(desired_progress),
    "achieved_progress_mean": torch.mean(achieved_progress),
    "valid_fraction": valid.float().mean(),
    "move_up_fraction": move_up.float().mean(),
    "move_down_fraction": move_down.float().mean(),
  }

  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]

  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result
