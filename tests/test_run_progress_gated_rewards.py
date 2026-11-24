"""Tests for the street-running reward changes (progress gate, speed-dependent margin)."""

from unittest.mock import Mock

import pytest
import torch
from pal_mjlab.tasks.velocity.mdp import rewards as r


def make_env(commands, lin_vel):
  env = Mock()
  env.num_envs = len(commands)
  env.device = "cpu"
  env.extras = {"log": {}}
  robot = Mock()
  robot.data.root_link_lin_vel_b = torch.tensor(lin_vel, dtype=torch.float32)
  env.scene = {"robot": robot}
  cmd = torch.tensor(commands, dtype=torch.float32)
  env.command_manager.get_command = Mock(return_value=cmd)
  return env


def test_forward_progress_gate():
  env = make_env(
    commands=[
      [4.0, 0.0, 0.0],  # Stepping in place on a fast command -> 0.
      [4.0, 0.0, 0.0],  # 3 of 4 m/s -> 0.75.
      [4.0, 0.0, 0.0],  # Overshoot is clamped -> 1.
      [0.3, 0.0, 0.5],  # Below gate_start_speed: unchanged -> 1.
      [0.75, 0.0, 0.0],  # Half-way through the blend, not moving -> 0.5.
      [-2.0, 0.0, 0.0],  # Backwards, 1 of 2 m/s -> 0.5.
      [0.0, 3.0, 0.0],  # Lateral, moving forward instead -> 0.
    ],
    lin_vel=[
      [0.0, 0.0, 0.0],
      [3.0, 0.0, 0.0],
      [5.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [-1.0, 0.0, 0.0],
      [3.0, 0.0, 0.0],
    ],
  )
  gate = r.forward_progress_gate(env, "twist", 0.5, 1.0)
  torch.testing.assert_close(gate, torch.tensor([0.0, 0.75, 1.0, 1.0, 0.5, 0.5, 0.0]))


def test_feet_air_time_progress_gated():
  env = make_env(
    commands=[[4.8, 0.0, 0.0], [4.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
    lin_vel=[[0.0, 0.0, 0.0], [3.6, 0.0, 0.0], [0.0, 0.0, 0.0]],
  )
  sensor = Mock()
  # Both feet in the rewarded air-time window for every env.
  sensor.data.current_air_time = torch.full((3, 2), 0.2)
  env.scene["feet_ground_contact"] = sensor

  value = r.feet_air_time_progress_gated(
    env,
    sensor_name="feet_ground_contact",
    command_name="twist",
    threshold_min=0.1,
    threshold_max=0.4,
    command_threshold=0.1,
  )
  # In place: 0; 3.6 of 4.5 m/s: 2 * 0.8; zero command: off as in mjlab.
  torch.testing.assert_close(value, torch.tensor([0.0, 1.6, 0.0]))


def test_arm_swing_progress_gated(monkeypatch):
  seen = {}

  def fake_call(self, env, **kwargs):
    seen.update(kwargs)
    return torch.ones(env.num_envs)

  monkeypatch.setattr(r.arm_swing, "__call__", fake_call)
  term = object.__new__(r.arm_swing_progress_gated)
  env = make_env(
    commands=[[4.8, 0.0, 0.0], [4.0, 0.0, 0.0]],
    lin_vel=[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
  )
  value = term(
    env, command_name="twist", std=0.15, gate_start_speed=0.5, gate_full_speed=1.0
  )
  torch.testing.assert_close(value, torch.tensor([0.0, 1.0]))
  # Gate params are consumed and not forwarded to the parent reward.
  assert "gate_start_speed" not in seen and "gate_full_speed" not in seen


def _proximity_term(gap_y: float, speed: float):
  """Two vertical capsules (r=0.05, half=0.1) whose surfaces are gap_y apart."""
  term = object.__new__(r.feet_proximity)
  term._left_ids = torch.tensor([0])
  term._right_ids = torch.tensor([1])
  term._left_radius = term._right_radius = torch.tensor([0.05])
  term._left_half = term._right_half = torch.tensor([0.1])
  term._asset = Mock()
  term._asset.data.root_link_lin_vel_b = torch.tensor([[speed, 0.0, 0.0]])
  env = Mock()
  env.extras = {"log": {}}
  y = (gap_y + 0.1) / 2
  env.sim.data.geom_xpos = torch.tensor([[[0.0, y, 0.0], [0.0, -y, 0.0]]])
  env.sim.data.geom_xmat = torch.eye(3).reshape(1, 1, 9).repeat(1, 2, 1)
  return term, env


@pytest.mark.parametrize(
  ("gap", "speed", "margin_per_mps", "expected"),
  [
    (0.05, 0.0, 0.0, 0.0),  # Outside the fixed 4 cm margin.
    (0.02, 0.0, 0.0, 0.5),  # (0.04 - 0.02) / 0.04.
    (0.05, 5.0, 0.008, 0.375),  # Margin 8 cm at 5 m/s: (0.08 - 0.05) / 0.08.
    (0.05, 0.0, 0.008, 0.0),  # Standing: still the 4 cm margin.
  ],
)
def test_feet_proximity_speed_dependent_margin(gap, speed, margin_per_mps, expected):
  term, env = _proximity_term(gap, speed)
  value = term(
    env,
    left_geom_names=(),
    right_geom_names=(),
    margin=0.04,
    margin_per_mps=margin_per_mps,
  )
  assert value.item() == pytest.approx(expected, abs=1e-5)
