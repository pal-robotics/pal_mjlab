from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

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