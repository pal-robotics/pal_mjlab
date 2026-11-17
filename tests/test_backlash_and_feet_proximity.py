"""Tests for the hip roll backlash actuator and the feet proximity distance."""

import torch
from mjlab.actuator.actuator import ActuatorCmd
from pal_mjlab.actuators import BacklashPdActuator
from pal_mjlab.tasks.velocity.mdp.rewards import _segment_segment_distance


def _torque(pos_error: float, backlash: float, kp: float = 100.0) -> float:
  params = {
    "stiffness": torch.tensor([[kp]]),
    "damping": torch.tensor([[0.0]]),
    "force_limit": torch.tensor([[1e3]]),
    "backlash": torch.tensor([[backlash]]),
  }
  z = torch.zeros(1, 1)
  cmd = ActuatorCmd(torch.tensor([[pos_error]]), z, z, z, z)
  return BacklashPdActuator.control_law(params, cmd).item()


def test_backlash_dead_zone():
  # Free inside +-backlash/2, PD on the excess outside, plain PD without backlash.
  assert _torque(0.009, backlash=0.02) == 0.0
  assert _torque(-0.009, backlash=0.02) == 0.0
  assert abs(_torque(0.015, backlash=0.02) - 0.5) < 1e-5
  assert abs(_torque(-0.015, backlash=0.02) + 0.5) < 1e-5
  assert abs(_torque(0.015, backlash=0.0) - 1.5) < 1e-5


def test_segment_distance_matches_brute_force():
  g = torch.Generator().manual_seed(0)
  a, b, c, d = (torch.randn(500, 3, generator=g) for _ in range(4))
  ts = torch.linspace(0, 1, 1001)
  pa = a[:, None] + (b - a)[:, None] * ts[None, :, None]
  pc = c[:, None] + (d - c)[:, None] * ts[None, :, None]
  brute = torch.cdist(pa, pc).flatten(1).min(1).values
  exact = _segment_segment_distance(a, b, c, d)
  assert (exact <= brute + 1e-5).all()
  assert (brute - exact).max() < 2e-3  # Brute force sampling resolution.


def test_segment_distance_parallel_and_crossing():
  z = torch.zeros(3)
  # Parallel segments 0.1 apart.
  dist = _segment_segment_distance(
    z,
    torch.tensor([1.0, 0, 0]),
    torch.tensor([0.5, 0.1, 0]),
    torch.tensor([2.0, 0.1, 0]),
  )
  assert abs(dist.item() - 0.1) < 1e-6
  # Crossing segments 0.2 apart in z.
  dist = _segment_segment_distance(
    torch.tensor([-1.0, 0, 0]),
    torch.tensor([1.0, 0, 0]),
    torch.tensor([0, -1.0, 0.2]),
    torch.tensor([0, 1.0, 0.2]),
  )
  assert abs(dist.item() - 0.2) < 1e-6
