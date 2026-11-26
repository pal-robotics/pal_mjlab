from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from pal_mjlab.rl.supermimic.imagination import sim_batch

from . import tracking
from .commands import MotionLibraryCommand
from .rewards import tracked_bodies

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import TerminationTermCfg

__all__ = ["motion_end", "anchor_failure", "anchor_failure_held"]


def motion_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """The target was the clip's last frame: the reference ends, and so does the return."""
  return cast(
    MotionLibraryCommand, env.command_manager.get_term(command_name)
  ).at_clip_end


def anchor_failure(
  env: ManagerBasedRlEnv, z_threshold: float, tilt_threshold: float
) -> torch.Tensor:
  """SuperMimic's termination: root height error > z_threshold or tilt error > tilt_threshold.

  On the simulator's batch (``imagination.py``), like the tracking rewards: terminations
  run on the same state, so they share its FK and reference. (Imagination does not compute
  it: the world model's learned head predicts failures.)"""
  robot, ref = tracked_bodies(sim_batch(env))
  return tracking.anchor_failure(robot, ref, z_threshold, tilt_threshold)


class anchor_failure_held:
  """``anchor_failure`` that has lasted ``seconds``: replay (``scripts/play.py``) shows a
  fall as the physics plays it out instead of resetting on its first frame. Training and
  the benchmark use ``anchor_failure``."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    self._steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._steps[slice(None) if env_ids is None else env_ids] = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    z_threshold: float,
    tilt_threshold: float,
    seconds: float,
  ) -> torch.Tensor:
    failed = anchor_failure(env, z_threshold, tilt_threshold)
    self._steps = torch.where(failed, self._steps + 1, torch.zeros_like(self._steps))
    return self._steps * env.step_dt >= seconds
