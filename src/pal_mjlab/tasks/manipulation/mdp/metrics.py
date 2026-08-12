"""Useful methods for MDP Metrics — manipulation tasks."""

from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv

from pal_mjlab.tasks.manipulation.mdp.commands import LiftingCommand


def object_height(
  env: ManagerBasedRlEnv,
  command_name: str = "lift_height",
) -> torch.Tensor:
  """Height of the object bottom surface above the world origin (Z)."""
  command: LiftingCommand = env.command_manager.get_term(command_name)
  return command.object_bottom_z


def position_error(
  env: ManagerBasedRlEnv,
  command_name: str = "lift_height",
) -> torch.Tensor:
  """L2 distance between the object center and the target position."""
  command: LiftingCommand = env.command_manager.get_term(command_name)
  return torch.norm(command.target_pos - command.object_pos_w, dim=-1)


def episode_success(
  env: ManagerBasedRlEnv,
  command_name: str = "lift_height",
) -> torch.Tensor:
  """1.0 if the episode was successful (object reached goal, fell, and was released)."""
  command: LiftingCommand = env.command_manager.get_term(command_name)
  return command.episode_success


def reached(
  env: ManagerBasedRlEnv,
  command_name: str = "lift_height",
) -> torch.Tensor:
  """1.0 if the object has been held within the goal threshold for the required dwell time."""
  command: LiftingCommand = env.command_manager.get_term(command_name)
  return command.reached.float()


def grasped_distance(
  env: ManagerBasedRlEnv,
  command_name: str = "lift_height",
) -> torch.Tensor:
  """Cumulative distance (m) the object has traveled while grasped by both fingers."""
  command: LiftingCommand = env.command_manager.get_term(command_name)
  return command.grasped_distance
