"""Tests for the per-episode command record kept by the velocity command."""

import torch
from conftest import make_mock_rl_env
from pal_mjlab.tasks.velocity.mdp.commands import (
  CommandHistory,
  DualBandVelocityCommand,
)
from test_dual_band_velocity_command import make_command


def advance(env, command: DualBandVelocityCommand, num_steps: int, dt: float) -> None:
  """Step the command term, keeping the episode clock in sync like the env does."""
  for _ in range(num_steps):
    env.episode_length_buf += 1
    command.compute(dt)


def test_flush_appends_only_environments_marked_pending():
  history = CommandHistory(num_envs=3, capacity=4, command_dim=2, device="cpu")
  commands = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
  times = torch.tensor([0.5, 1.5, 2.5])

  history.mark_pending(torch.tensor([0, 2]))
  history.flush(commands, times)

  assert torch.equal(history.lengths, torch.tensor([1, 0, 1]))
  assert torch.equal(history.commands[0, 0], commands[0])
  assert torch.equal(history.commands[2, 0], commands[2])
  assert torch.count_nonzero(history.commands[1]) == 0
  assert torch.equal(history.start_times[:, 0], torch.tensor([0.5, 0.0, 2.5]))

  # Flushing again without marking anything leaves the record untouched.
  history.flush(commands * 2.0, times + 1.0)

  assert torch.equal(history.lengths, torch.tensor([1, 0, 1]))
  assert torch.equal(history.commands[0, 0], commands[0])


def test_flush_appends_in_order():
  history = CommandHistory(num_envs=1, capacity=3, command_dim=2, device="cpu")
  env_ids = torch.tensor([0])

  for step in range(3):
    history.mark_pending(env_ids)
    history.flush(torch.full((1, 2), float(step)), torch.tensor([float(step)]))

  assert history.lengths.item() == 3
  assert torch.equal(history.start_times[0], torch.tensor([0.0, 1.0, 2.0]))
  assert torch.equal(history.commands[0, :, 0], torch.tensor([0.0, 1.0, 2.0]))


def test_clear_drops_only_selected_environments():
  history = CommandHistory(num_envs=2, capacity=4, command_dim=2, device="cpu")
  history.mark_pending(torch.arange(2))
  history.flush(torch.ones(2, 2), torch.ones(2))

  history.clear(torch.tensor([0]))

  assert torch.equal(history.lengths, torch.tensor([0, 1]))
  assert torch.count_nonzero(history.commands[0]) == 0
  assert history.start_times[0, 0].item() == 0.0
  assert torch.equal(history.commands[1, 0], torch.ones(2))


def test_clear_cancels_pending_appends():
  history = CommandHistory(num_envs=1, capacity=2, command_dim=2, device="cpu")
  history.mark_pending(torch.tensor([0]))

  history.clear(torch.tensor([0]))
  history.flush(torch.ones(1, 2), torch.ones(1))

  assert history.lengths.item() == 0


def test_capacity_covers_the_worst_case_episode():
  env = make_mock_rl_env(4, episode_length_s=20.0)
  command = make_command(4, env=env, resampling_time_range=(3.0, 8.0))

  # ceil(20 / 3) resamples during the episode, plus the one sampled on reset.
  assert command.command_history.commands.shape == (4, 8, 3)


def test_reset_records_the_initial_command_at_time_zero():
  env = make_mock_rl_env(4, step_dt=0.25)
  command = make_command(4, env=env)

  command.reset(torch.arange(4))
  command.compute(0.0)

  history = command.command_history
  assert torch.equal(history.lengths, torch.full((4,), 1))
  assert torch.equal(history.start_times[:, 0], torch.zeros(4))
  assert torch.equal(history.commands[:, 0], command.command)


def test_resample_appends_with_the_episode_time():
  env = make_mock_rl_env(4, step_dt=0.25)
  command = make_command(4, env=env, resampling_time_range=(1.0, 1.0))
  command.reset(torch.arange(4))
  command.compute(0.0)
  initial_command = command.command.clone()

  advance(env, command, num_steps=4, dt=0.25)

  history = command.command_history
  assert torch.equal(history.lengths, torch.full((4,), 2))
  assert torch.equal(history.start_times[:, :2], torch.tensor([[0.0, 1.0]] * 4))
  assert torch.equal(history.commands[:, 0], initial_command)
  assert torch.equal(history.commands[:, 1], command.command)


def test_reset_clears_the_previous_episode():
  env = make_mock_rl_env(2, step_dt=0.25)
  command = make_command(2, env=env, resampling_time_range=(1.0, 1.0))
  command.reset(torch.arange(2))
  command.compute(0.0)
  advance(env, command, num_steps=4, dt=0.25)

  # The env zeroes the episode clock after the command manager reset.
  command.reset(torch.tensor([0]))
  env.episode_length_buf[0] = 0
  command.compute(0.25)

  history = command.command_history
  assert torch.equal(history.lengths, torch.tensor([1, 2]))
  assert history.start_times[0, 0].item() == 0.0
  assert torch.count_nonzero(history.start_times[0, 1:]) == 0
  assert torch.count_nonzero(history.commands[0, 1:]) == 0


def test_environments_record_independently():
  env = make_mock_rl_env(2, step_dt=0.25)
  command = make_command(2, env=env, resampling_time_range=(1.0, 1.0))
  command.reset(torch.arange(2))
  command.compute(0.0)

  # Only the first env is due for a resample on the next step.
  command.time_left = torch.tensor([0.25, 0.75])
  advance(env, command, num_steps=1, dt=0.25)

  history = command.command_history
  assert torch.equal(history.lengths, torch.tensor([2, 1]))
  assert history.start_times[0, 1].item() == 0.25
  assert history.start_times[1, 1].item() == 0.0


def test_standing_environments_record_zero_commands():
  env = make_mock_rl_env(64, step_dt=0.25)
  command = make_command(
    64,
    env=env,
    lin_vel_y=(-1.0, 1.0),
    ang_vel_z=(-1.0, 1.0),
    rel_standing_envs=1.0,
  )

  command.reset(torch.arange(64))
  command.compute(0.0)

  assert torch.count_nonzero(command.command_history.commands) == 0


def test_full_episode_fills_the_record_without_overflowing():
  env = make_mock_rl_env(2, step_dt=0.25, episode_length_s=2.0)
  command = make_command(2, env=env, resampling_time_range=(1.0, 1.0))
  capacity = command.command_history.commands.shape[1]

  command.reset(torch.arange(2))
  command.compute(0.0)
  advance(env, command, num_steps=8, dt=0.25)

  assert capacity == 3
  assert torch.equal(command.command_history.lengths, torch.full((2,), capacity))
