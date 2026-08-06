"""Tests for the command-record-driven terrain level curriculum."""

import math
from unittest.mock import MagicMock

import pytest
import torch
from conftest import make_mock_rl_env
from pal_mjlab.tasks.velocity.mdp.curriculums import (
  commanded_displacement,
  terrain_levels_vel_with_history,
)
from test_command_history import advance
from test_dual_band_velocity_command import make_command

TERRAIN_SIZE = 3.0
EPISODE_LENGTH_S = 4.0
STEP_DT = 0.25


def test_straight_command_displaces_along_its_own_heading():
  commands = torch.tensor([[[0.5, 0.0, 0.0]]])
  durations = torch.tensor([[2.0]])

  displacement = commanded_displacement(commands, durations)

  assert torch.allclose(displacement, torch.tensor([[1.0, 0.0]]))


def test_lateral_command_displaces_sideways():
  commands = torch.tensor([[[0.0, 0.5, 0.0]]])
  durations = torch.tensor([[2.0]])

  displacement = commanded_displacement(commands, durations)

  assert torch.allclose(displacement, torch.tensor([[0.0, 1.0]]), atol=1e-6)


def test_full_circle_command_returns_to_its_start():
  # A full turn at constant forward speed traces a closed circle.
  commands = torch.tensor([[[0.5, 0.0, 2.0 * math.pi]]])
  durations = torch.tensor([[1.0]])

  displacement = commanded_displacement(commands, durations)

  assert torch.allclose(displacement, torch.zeros(1, 2), atol=1e-6)


def test_half_circle_command_spans_the_diameter():
  # Turning pi over the segment traces a semicircle of radius v / w.
  speed, yaw_rate = 0.5, math.pi
  commands = torch.tensor([[[speed, 0.0, yaw_rate]]])
  durations = torch.tensor([[1.0]])

  displacement = commanded_displacement(commands, durations)

  assert torch.allclose(
    displacement, torch.tensor([[0.0, 2.0 * speed / yaw_rate]]), atol=1e-6
  )


def test_segments_chain_through_the_accumulated_heading():
  # Drive forward, turn a quarter circle in place, then drive forward again.
  commands = torch.tensor(
    [[[1.0, 0.0, 0.0], [0.0, 0.0, math.pi / 2.0], [1.0, 0.0, 0.0]]]
  )
  durations = torch.tensor([[1.0, 1.0, 1.0]])

  displacement = commanded_displacement(commands, durations)

  assert torch.allclose(displacement, torch.tensor([[1.0, 1.0]]), atol=1e-6)


def test_segments_past_the_record_length_do_not_contribute():
  commands = torch.tensor([[[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]])
  durations = torch.tensor([[2.0, 0.0]])

  displacement = commanded_displacement(commands, durations)

  assert torch.allclose(displacement, torch.tensor([[1.0, 0.0]]))


def make_curriculum_env(num_envs: int, *, lin_vel_x: tuple[float, float]):
  """Build a mock env with a real command term and a mock curriculum terrain."""
  env = make_mock_rl_env(num_envs, step_dt=STEP_DT, episode_length_s=EPISODE_LENGTH_S)
  command = make_command(
    num_envs,
    env=env,
    lin_vel_x=lin_vel_x,
    resampling_time_range=(1.0, 1.0),
  )
  env.command_manager.get_term.return_value = command

  terrain = MagicMock()
  terrain.cfg.terrain_generator.size = (TERRAIN_SIZE, TERRAIN_SIZE)
  terrain.cfg.terrain_generator.sub_terrains = {"flat": None}
  terrain.terrain_levels = torch.zeros(num_envs, dtype=torch.long)
  terrain.terrain_types = torch.zeros(num_envs, dtype=torch.long)
  terrain.terrain_origins = torch.zeros(2, 1, 3)
  env.scene.terrain = terrain
  return env, command, terrain


def run_episode(env, command, walked_distance: float, num_steps: int):
  """Reset, run the command term, and place the robot at ``walked_distance``."""
  command.reset(torch.arange(env.num_envs))
  command.compute(0.0)
  advance(env, command, num_steps=num_steps, dt=STEP_DT)
  env.scene[  # type: ignore[index]
    "robot"
  ].data.root_link_pos_w[:, 0] = walked_distance


def moves(terrain) -> tuple[torch.Tensor, torch.Tensor]:
  _, move_up, move_down = terrain.update_env_origins.call_args.args
  return move_up, move_down


def test_meeting_the_commanded_distance_promotes():
  env, command, terrain = make_curriculum_env(2, lin_vel_x=(0.5, 0.5))
  # 0.5 m/s over a 4 s episode asks for 2 m; promotion needs 80% of that, capped at
  # half a sub-terrain (1.5 m).
  run_episode(env, command, walked_distance=1.6, num_steps=16)

  terrain_levels_vel_with_history(env, torch.arange(2), "twist")

  move_up, move_down = moves(terrain)
  assert torch.all(move_up)
  assert not torch.any(move_down)


def test_falling_well_short_demotes():
  env, command, terrain = make_curriculum_env(2, lin_vel_x=(0.5, 0.5))
  # Demotion sits at half the 1.5 m promotion distance.
  run_episode(env, command, walked_distance=0.5, num_steps=16)

  terrain_levels_vel_with_history(env, torch.arange(2), "twist")

  move_up, move_down = moves(terrain)
  assert not torch.any(move_up)
  assert torch.all(move_down)


def test_landing_between_the_thresholds_keeps_the_level():
  env, command, terrain = make_curriculum_env(2, lin_vel_x=(0.5, 0.5))
  run_episode(env, command, walked_distance=1.0, num_steps=16)

  terrain_levels_vel_with_history(env, torch.arange(2), "twist")

  move_up, move_down = moves(terrain)
  assert not torch.any(move_up)
  assert not torch.any(move_down)


def test_standing_commands_keep_the_level():
  env, command, terrain = make_curriculum_env(2, lin_vel_x=(0.0, 0.0))
  # Drifting past half a sub-terrain must not promote an env asked to stand still.
  run_episode(env, command, walked_distance=2.0, num_steps=16)

  terrain_levels_vel_with_history(env, torch.arange(2), "twist")

  move_up, move_down = moves(terrain)
  assert not torch.any(move_up)
  assert not torch.any(move_down)


def test_early_termination_is_judged_against_a_full_episode():
  env, command, terrain = make_curriculum_env(2, lin_vel_x=(0.5, 0.5))
  # Terminating four steps in does not shrink the expectation: the last command is held
  # to the end of the episode, so 2 m is still asked for and 0.2 m is a failure.
  run_episode(env, command, walked_distance=0.2, num_steps=4)

  terrain_levels_vel_with_history(env, torch.arange(2), "twist")

  move_up, move_down = moves(terrain)
  assert not torch.any(move_up)
  assert torch.all(move_down)


def test_empty_record_freezes_levels_on_the_initial_reset():
  env, command, terrain = make_curriculum_env(2, lin_vel_x=(0.5, 0.5))
  # The curriculum runs before the command manager reset, so on the very first reset
  # the record is still empty.
  env.scene["robot"].data.root_link_pos_w[:, 0] = 5.0

  terrain_levels_vel_with_history(env, torch.arange(2), "twist")

  move_up, move_down = moves(terrain)
  assert not torch.any(move_up)
  assert not torch.any(move_down)


def test_reports_distance_statistics():
  env, command, terrain = make_curriculum_env(2, lin_vel_x=(0.5, 0.5))
  run_episode(env, command, walked_distance=1.6, num_steps=16)

  result = terrain_levels_vel_with_history(env, torch.arange(2), "twist")

  assert result["walked_distance"].item() == pytest.approx(1.6)
  assert result["expected_distance"].item() == pytest.approx(2.0)
  assert set(result) == {
    "mean",
    "max",
    "walked_distance",
    "expected_distance",
    "flat",
  }
