"""Tests for dual-band velocity command sampling."""

from unittest.mock import Mock

import pytest
import torch
from pal_mjlab.tasks.velocity.mdp.commands import (
  DualBandVelocityCommand,
  DualBandVelocityCommandCfg,
)


def make_command(
  num_envs: int,
  *,
  lin_vel_x: tuple[float, float] = (-1.0, 1.0),
  lin_vel_y: tuple[float, float] = (0.0, 0.0),
  ang_vel_z: tuple[float, float] = (0.0, 0.0),
  heading: tuple[float, float] | None = None,
  **kwargs,
) -> DualBandVelocityCommand:
  env = Mock()
  env.num_envs = num_envs
  env.device = "cpu"
  robot = Mock()
  robot.data.heading_w = torch.zeros(num_envs)
  env.scene = {"robot": robot}

  cfg = DualBandVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0, 1.0),
    heading_command=heading is not None,
    ranges=DualBandVelocityCommandCfg.Ranges(
      lin_vel_x=lin_vel_x,
      lin_vel_y=lin_vel_y,
      ang_vel_z=ang_vel_z,
      heading=heading,
    ),
    **kwargs,
  )
  return DualBandVelocityCommand(cfg, env)


@pytest.mark.parametrize(
  ("limits", "minimum", "expected_negative_fraction", "expected_negative_mean"),
  [
    ((-1.0, 1.0), 0.2, 0.5, -0.6),
    ((-0.4, 1.0), 0.2, 0.2, -0.3),
  ],
)
def test_samples_uniformly_outside_low_velocity_band(
  limits: tuple[float, float],
  minimum: float,
  expected_negative_fraction: float,
  expected_negative_mean: float,
):
  torch.manual_seed(0)
  num_envs = 50_000
  command = make_command(
    num_envs,
    lin_vel_x=limits,
    min_lin_vel_x=minimum,
  )

  command._resample_command(torch.arange(num_envs))
  samples = command.vel_command_b[:, 0]
  negative = samples < 0.0

  assert torch.all(samples.abs() >= minimum)
  assert negative.float().mean().item() == pytest.approx(
    expected_negative_fraction, abs=0.01
  )
  assert samples[negative].mean().item() == pytest.approx(
    expected_negative_mean, abs=0.01
  )
  assert samples[~negative].mean().item() == pytest.approx(0.6, abs=0.01)
  assert torch.isclose(samples.abs(), torch.tensor(minimum)).sum() < 10


def test_straight_commands_preserve_intentional_zeros():
  torch.manual_seed(0)
  num_envs = 2_000
  command = make_command(
    num_envs,
    lin_vel_y=(-1.0, 1.0),
    ang_vel_z=(-1.0, 1.0),
    rel_straight_envs=1.0,
    min_lin_vel_x=0.3,
    min_lin_vel_y=0.2,
    min_ang_vel_z=0.1,
  )

  command._resample_command(torch.arange(num_envs))

  assert torch.all(command.vel_command_b[:, 0].abs() >= 0.3)
  assert torch.any(command.vel_command_b[:, 0] < 0.0)
  assert torch.any(command.vel_command_b[:, 0] > 0.0)
  assert torch.count_nonzero(command.vel_command_b[:, 1:]) == 0


def test_low_band_straight_commands_preserve_intentional_zeros():
  command = make_command(
    1_000,
    lin_vel_y=(-1.0, 1.0),
    ang_vel_z=(-1.0, 1.0),
    rel_straight_envs=1.0,
    rel_inside_low=1.0,
    min_lin_vel_x=0.3,
    min_lin_vel_y=0.2,
    min_ang_vel_z=0.1,
  )

  command._resample_command(torch.arange(command.num_envs))

  assert torch.all(command.vel_command_b[:, 0].abs() < 0.3)
  assert torch.count_nonzero(command.vel_command_b[:, 1:]) == 0


def test_standing_commands_remain_zero():
  command = make_command(
    100,
    lin_vel_y=(-1.0, 1.0),
    ang_vel_z=(-1.0, 1.0),
    rel_standing_envs=1.0,
    rel_inside_low=1.0,
    min_lin_vel_x=0.3,
    min_lin_vel_y=0.2,
    min_ang_vel_z=0.1,
  )

  command._resample_command(torch.arange(command.num_envs))
  command._update_command()

  assert torch.count_nonzero(command.vel_command_b) == 0


def test_world_frame_stores_resampled_command():
  command = make_command(
    1_000,
    lin_vel_y=(-1.0, 1.0),
    rel_world_envs=1.0,
    min_lin_vel_x=0.3,
    min_lin_vel_y=0.2,
  )

  command._resample_command(torch.arange(command.num_envs))

  assert torch.equal(command.vel_command_w, command.vel_command_b)


def test_heading_update_can_enter_low_velocity_band():
  command = make_command(
    10,
    ang_vel_z=(-1.0, 1.0),
    heading=(-torch.pi, torch.pi),
    rel_heading_envs=1.0,
    heading_control_stiffness=0.5,
    min_ang_vel_z=0.3,
  )
  command._resample_command(torch.arange(command.num_envs))
  command.heading_target.fill_(0.1)

  command._update_command()

  assert torch.allclose(command.vel_command_b[:, 2], torch.full((10,), 0.05))


def test_fixed_zero_range_is_valid_with_nonzero_minimum():
  command = make_command(
    10,
    lin_vel_x=(0.0, 0.0),
    min_lin_vel_x=0.3,
  )

  command._resample_command(torch.arange(command.num_envs))

  assert torch.count_nonzero(command.vel_command_b[:, 0]) == 0


def test_rel_inside_low_controls_fraction_and_ignores_unconfigured_axis():
  torch.manual_seed(0)
  num_envs = 50_000
  command = make_command(
    num_envs,
    lin_vel_y=(-1.0, 1.0),
    rel_inside_low=0.25,
    min_lin_vel_x=0.2,
    min_lin_vel_y=0.0,
  )

  command._resample_command(torch.arange(num_envs))
  x_samples = command.vel_command_b[:, 0]
  y_samples = command.vel_command_b[:, 1]
  inside_low = x_samples.abs() < 0.2

  assert inside_low.float().mean().item() == pytest.approx(0.25, abs=0.01)
  assert x_samples[inside_low].abs().mean().item() == pytest.approx(0.1, abs=0.01)
  assert torch.all(x_samples[~inside_low].abs() >= 0.2)
  assert y_samples[inside_low].abs().mean().item() == pytest.approx(0.5, abs=0.01)
  assert y_samples[~inside_low].abs().mean().item() == pytest.approx(0.5, abs=0.01)


def test_positive_only_range_samples_inside_and_above_low_band():
  torch.manual_seed(0)
  num_envs = 50_000
  command = make_command(
    num_envs,
    lin_vel_x=(0.1, 0.6),
    rel_inside_low=0.25,
    min_lin_vel_x=0.2,
  )

  command._resample_command(torch.arange(num_envs))
  samples = command.vel_command_b[:, 0]
  inside_low = samples < 0.2

  assert inside_low.float().mean().item() == pytest.approx(0.25, abs=0.01)
  assert torch.all((samples[inside_low] >= 0.1) & (samples[inside_low] < 0.2))
  assert torch.all((samples[~inside_low] >= 0.2) & (samples[~inside_low] <= 0.6))


def test_all_low_commands_allow_range_entirely_inside_band():
  command = make_command(
    1_000,
    lin_vel_x=(-0.2, 0.2),
    rel_inside_low=1.0,
    min_lin_vel_x=0.2,
  )

  command._resample_command(torch.arange(command.num_envs))

  assert torch.all(command.vel_command_b[:, 0].abs() < 0.2)


def test_rejects_minimum_without_outside_sampling_interval():
  with pytest.raises(ValueError, match="outside the low-velocity band"):
    make_command(
      1,
      lin_vel_x=(-0.1, 0.1),
      min_lin_vel_x=0.1,
    )


def test_rejects_minimum_above_range_magnitude_early():
  with pytest.raises(
    ValueError,
    match=(
      r"min_lin_vel_x=1.1 exceeds the maximum magnitude 1.0 available in range "
      r"\(-0.5, 1.0\)"
    ),
  ):
    make_command(
      1,
      lin_vel_x=(-0.5, 1.0),
      rel_inside_low=1.0,
      min_lin_vel_x=1.1,
    )


def test_rejects_low_fraction_without_inside_sampling_interval():
  with pytest.raises(ValueError, match="inside the low-velocity band"):
    make_command(
      1,
      lin_vel_x=(0.5, 1.0),
      rel_inside_low=0.1,
      min_lin_vel_x=0.2,
    )


@pytest.mark.parametrize("rel_inside_low", [-0.1, 1.1])
def test_rejects_invalid_low_fraction(rel_inside_low: float):
  with pytest.raises(ValueError, match="rel_inside_low must be between 0 and 1"):
    make_command(1, rel_inside_low=rel_inside_low)
