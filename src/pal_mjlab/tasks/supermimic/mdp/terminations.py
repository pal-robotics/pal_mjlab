from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from . import tracking
from .commands import MotionLibraryCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["motion_end", "anchor_failure"]


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
