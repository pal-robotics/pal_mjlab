from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

def motion_time_out(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Terminate the episode when the motion command reaches the end of the clip."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  # Command time_steps are incremented before terminations are checked.
  # When time_steps reaches time_step_total, the command automatically resamples and resets time_steps to 0.
  # Therefore, we must trigger the timeout on the last frame before it resets.
  return command.time_steps >= (command.motion.time_step_total - 1)
