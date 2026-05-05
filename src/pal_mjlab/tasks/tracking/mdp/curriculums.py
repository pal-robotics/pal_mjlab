from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from pal_mjlab.tasks.tracking.mdp.commands import PalMotionCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class TrajectoryFractionStage(TypedDict):
  step: int
  fraction: float


def motion_trajectory_fraction(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  fraction_stages: list[TrajectoryFractionStage],
) -> dict[str, torch.Tensor]:
  del env_ids
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(PalMotionCommandCfg, command_term.cfg)
  for stage in fraction_stages:
    if env.common_step_counter >= stage["step"]:
      cfg.max_time_fraction = stage["fraction"]
  return {"max_time_fraction": torch.tensor(cfg.max_time_fraction)}
