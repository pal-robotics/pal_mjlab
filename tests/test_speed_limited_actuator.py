"""Tests for the transmission speed limit of the PAL PD actuators."""

import pytest
import torch
from pal_mjlab.actuators import (
  BacklashPdActuatorCfg,
  SpeedLimitedPdActuatorCfg,
  speed_limit_clip,
)


def _clip(torque, vel, limit=10.0, ratio=0.8, force=100.0):
  t = lambda x: torch.tensor([float(x)])  # noqa: E731
  return speed_limit_clip(t(torque), t(force), t(limit), t(ratio), t(vel)).item()


@pytest.mark.parametrize(
  ("torque", "vel", "expected"),
  [
    (500.0, 0.0, 100.0),  # Standing: full effort limit.
    (500.0, 8.0, 100.0),  # Up to 80 % of the limit: still full.
    (500.0, 9.0, 50.0),  # Half-way between 80 % and 100 %.
    (500.0, 10.0, 0.0),  # At the limit: cannot drive any faster.
    (500.0, 11.0, -50.0),  # Above it: the motor brakes.
    (-500.0, 10.0, -100.0),  # Braking against the motion is unaffected.
    (-500.0, -9.0, -50.0),  # Symmetric in the negative direction.
    (30.0, 9.0, 30.0),  # Small torques inside the envelope pass through.
  ],
)
def test_speed_limit_clip(torque, vel, expected):
  assert _clip(torque, vel) == pytest.approx(expected, abs=1e-4)


def test_infinite_limits_reduce_to_plain_clamp():
  assert _clip(500.0, 50.0, limit=float("inf")) == pytest.approx(100.0)
  assert _clip(500.0, 50.0, force=float("inf")) == pytest.approx(500.0)


def test_cfg_validation():
  base = dict(target_names_expr=("j",), stiffness=1.0, damping=0.1, effort_limit=1.0)
  with pytest.raises(ValueError):
    SpeedLimitedPdActuatorCfg(**base, velocity_limit=0.0)
  with pytest.raises(ValueError):
    SpeedLimitedPdActuatorCfg(**base, velocity_limit=1.0, full_torque_ratio=1.0)
  # Backlash default: no speed limit.
  assert BacklashPdActuatorCfg(**base).velocity_limit == float("inf")
