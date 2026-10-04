"""Useful methods for custom MDP Curriculums."""

import torch
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def standing_envs_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  standing_stages: list[dict],
) -> torch.Tensor:
  """Update the relative number of standing environments based on training progress.

  Args:
      env: The RL environment
      env_ids: Environment IDs (unused, but required by curriculum interface)
      command_name: Name of the velocity command term
      standing_stages: List of dicts with 'step' and 'rel_standing_envs' keys
          Example: [
              {"step": 0, "rel_standing_envs": 0.02},
              {"step": 1000, "rel_standing_envs": 0.1},
              {"step": 2000, "rel_standing_envs": 0.2},
          ]

  Returns:
      Current rel_standing_envs value as a tensor
  """
  del env_ids  # Unused

  from typing import cast

  from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None, f"Command term '{command_name}' not found"

  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)

  # Update rel_standing_envs based on current step
  for stage in standing_stages:
    if env.common_step_counter > stage["step"]:
      cfg.rel_standing_envs = stage["rel_standing_envs"]

  return torch.tensor([cfg.rel_standing_envs])
