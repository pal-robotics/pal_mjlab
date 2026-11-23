"""Tests for the speed-coupled velocity command (lateral/yaw shrink at high speed)."""

import math
from unittest.mock import Mock

import pytest
import torch
from pal_mjlab.tasks.velocity.mdp.commands import (
  SpeedCoupledVelocityCommand,
  SpeedCoupledVelocityCommandCfg,
)


def make_command(
  num_envs: int,
  *,
  heading: tuple[float, float] | None = None,
  **kwargs,
) -> SpeedCoupledVelocityCommand:
  env = Mock()
  env.num_envs = num_envs
  env.device = "cpu"
  robot = Mock()
  robot.data.heading_w = torch.zeros(num_envs)
  env.scene = {"robot": robot}

  cfg = SpeedCoupledVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(1.0, 1.0),
    heading_command=heading is not None,
    ranges=SpeedCoupledVelocityCommandCfg.Ranges(
      lin_vel_x=(-2.0, 5.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-1.0, 1.0),
      heading=heading,
    ),
    coupling_start=1.5,
    coupling_end=4.0,
    min_scale=0.25,
    **kwargs,
  )
  return SpeedCoupledVelocityCommand(cfg, env)


def test_coupling_scale_is_piecewise_linear():
  command = make_command(1)
  vx = torch.tensor([0.0, 1.5, -1.5, 2.75, 4.0, 5.0, -2.0])
  expected = torch.tensor([1.0, 1.0, 1.0, 0.625, 0.25, 0.25, 0.85])
  torch.testing.assert_close(command._coupling_scale(vx), expected)


def test_lateral_and_yaw_shrink_with_forward_speed():
  torch.manual_seed(0)
  num_envs = 50_000
  command = make_command(num_envs)
  command._resample_command(torch.arange(num_envs))
  cmd = command.vel_command_b
  vx = cmd[:, 0].abs()

  # Bound: |vy|, |wz| <= scale(vx) * 1.0 for every env.
  scale = command._coupling_scale(cmd[:, 0])
  assert torch.all(cmd[:, 1].abs() <= scale + 1e-6)
  assert torch.all(cmd[:, 2].abs() <= scale + 1e-6)

  slow = vx < 1.5
  fast = vx >= 4.0
  # Slow commands keep the full uniform range (mean |u| = 0.5), fast ones a quarter.
  assert cmd[slow, 1].abs().mean().item() == pytest.approx(0.5, abs=0.02)
  assert cmd[fast, 1].abs().mean().item() == pytest.approx(0.125, abs=0.01)
  assert cmd[fast, 2].abs().max().item() <= 0.25 + 1e-6


def test_heading_yaw_rate_is_clipped_at_high_speed():
  num_envs = 2
  command = make_command(
    num_envs, heading=(-math.pi, math.pi), rel_heading_envs=1.0, rel_standing_envs=0.0
  )
  command.is_heading_env[:] = True
  command.heading_target[:] = torch.tensor([3.0, 3.0])  # Large heading error.
  command.vel_command_b[:, 0] = torch.tensor([1.0, 4.5])
  command.lateral_yaw_scale[:] = command._coupling_scale(command.vel_command_b[:, 0])

  command._update_command()

  # Slow env: full +-1.0 rad/s clip; fast env: +-0.25 rad/s.
  assert command.vel_command_b[0, 2].item() == pytest.approx(1.0)
  assert command.vel_command_b[1, 2].item() == pytest.approx(0.25)


@pytest.mark.parametrize(
  ("kwargs", "match"),
  [
    ({"coupling_start": 3.0, "coupling_end": 2.0}, "coupling_start"),
    ({"min_scale": 1.5}, "min_scale"),
  ],
)
def test_invalid_cfg_raises(kwargs, match):
  base = {"coupling_start": 1.5, "coupling_end": 4.0, "min_scale": 0.25, **kwargs}
  with pytest.raises(ValueError, match=match):
    SpeedCoupledVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(1.0, 1.0),
      ranges=SpeedCoupledVelocityCommandCfg.Ranges(
        lin_vel_x=(0.0, 1.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
      ),
      **base,
    )
