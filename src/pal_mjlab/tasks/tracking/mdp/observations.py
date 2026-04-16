from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Return the normalized phase of the motion clip as [sin(2πφ), cos(2πφ)]."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  # Normalized phase φ in [0, 1): current_step / total_steps
  # command.time_steps has shape (num_envs,)
  phase = command.time_steps.float() / command.motion.time_step_total
  phase_rad = 2.0 * math.pi * phase

  # Return encoded periodic phase as (num_envs, 2)
  return torch.stack([torch.sin(phase_rad), torch.cos(phase_rad)], dim=-1)
