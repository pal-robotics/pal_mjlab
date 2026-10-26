from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from . import tracking
from .commands import MotionLibraryCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers import TerminationTermCfg

__all__ = ["motion_end", "anchor_failure", "anchor_failure_held"]


def motion_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """The target was the clip's last frame: the reference ends, so does the return (D8)."""
  return cast(
    MotionLibraryCommand, env.command_manager.get_term(command_name)
  ).at_clip_end


def anchor_failure(
  env: ManagerBasedRlEnv, command_name: str, z_threshold: float, tilt_threshold: float
) -> torch.Tensor:
  """SuperMimic's termination: root height error > z_threshold or tilt error > tilt_threshold."""
  command = cast(MotionLibraryCommand, env.command_manager.get_term(command_name))
  return tracking.anchor_failure(
    command.robot_bodies, command.ref_bodies, z_threshold, tilt_threshold
  )


class anchor_failure_held:
  """``anchor_failure`` that has lasted ``seconds``: replay (``play.py``, D45) shows a fall as the
  physics plays it out instead of resetting on its first frame. Training and the benchmark keep
  ``anchor_failure``."""

  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    self._steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._steps[slice(None) if env_ids is None else env_ids] = 0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    z_threshold: float,
    tilt_threshold: float,
    seconds: float,
  ) -> torch.Tensor:
    failed = anchor_failure(env, command_name, z_threshold, tilt_threshold)
    self._steps = torch.where(failed, self._steps + 1, torch.zeros_like(self._steps))
    return self._steps * env.step_dt >= seconds
