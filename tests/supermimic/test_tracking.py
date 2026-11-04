"""Properties of the pure tracking terms (seeded random poses)."""

import math

import pytest
import torch
from pal_mjlab.tasks.supermimic.kinematics import (
  quat_error_magnitude,
  quat_from_axis_angle,
  quat_mul,
  quat_rotate,
  rot6d,
)
from pal_mjlab.tasks.supermimic.mdp import tracking
from pal_mjlab.tasks.supermimic.mdp.tracking import BodyState


def _random_bodies(batch: int, bodies: int, seed: int) -> BodyState:
  g = torch.Generator().manual_seed(seed)
  q = torch.randn(batch, bodies, 4, generator=g, dtype=torch.float64)
  return BodyState(
    pos=torch.randn(batch, bodies, 3, generator=g, dtype=torch.float64),
    quat=q / q.norm(dim=-1, keepdim=True),
    lin_vel=torch.randn(batch, bodies, 3, generator=g, dtype=torch.float64),
    ang_vel=torch.randn(batch, bodies, 3, generator=g, dtype=torch.float64),
  )


ERRORS = (
  tracking.anchor_pos_error,
  tracking.anchor_ori_error,
  tracking.body_pos_error,
  tracking.body_ori_error,
  tracking.body_lin_vel_error,
  tracking.body_ang_vel_error,
)


@pytest.mark.parametrize("seed", range(3))
def test_perfect_tracking_has_zero_error(seed):
  s = _random_bodies(16, 5, seed)
  for fn in ERRORS:
    assert torch.allclose(fn(s, s), torch.zeros(16, dtype=torch.float64), atol=1e-10), (
      fn
    )
  assert not tracking.anchor_failure(s, s, 0.25, 0.8).any()


@pytest.mark.parametrize("seed", range(3))
def test_body_terms_ignore_global_drift(seed):
  """Moving the whole reference in xy and yaw only changes the root terms."""
  robot = _random_bodies(8, 6, seed)
  g = torch.Generator().manual_seed(100 + seed)
  yaw = torch.rand(8, 1, generator=g, dtype=torch.float64) * 2 * math.pi
  z = torch.zeros(8, 1, 3, dtype=torch.float64)
  z[..., 2] = 1.0
  rot = quat_from_axis_angle(z.expand(8, 6, 3), yaw.expand(8, 6))
  shift = torch.cat(
    [torch.randn(8, 1, 2, generator=g, dtype=torch.float64), z[..., :1] * 0], -1
  )
  ref = BodyState(
    pos=quat_rotate(rot, robot.pos) + shift,
    quat=quat_mul(rot, robot.quat),
    lin_vel=robot.lin_vel,
    ang_vel=robot.ang_vel,
  )
  assert torch.allclose(
    tracking.body_pos_error(robot, ref), torch.zeros(8, dtype=torch.float64), atol=1e-10
  )
  assert torch.allclose(
    tracking.body_ori_error(robot, ref), torch.zeros(8, dtype=torch.float64), atol=1e-10
  )
  assert (tracking.anchor_pos_error(robot, ref) > 0).all()


def test_anchor_failure_thresholds():
  robot = _random_bodies(1, 2, 0)
  ident = torch.tensor([1.0, 0, 0, 0], dtype=torch.float64)
  robot = robot._replace(quat=ident.expand(1, 2, 4).clone())
  up = robot._replace(pos=robot.pos + torch.tensor([0, 0, 0.3], dtype=torch.float64))
  assert tracking.anchor_failure(robot, up, 0.25, 0.8).all()
  slightly_up = robot._replace(
    pos=robot.pos + torch.tensor([0, 0, 0.2], dtype=torch.float64)
  )
  assert not tracking.anchor_failure(robot, slightly_up, 0.25, 0.8).any()
  x = torch.tensor([1.0, 0, 0], dtype=torch.float64).expand(1, 2, 3)
  tilted = robot._replace(
    quat=quat_from_axis_angle(
      x, torch.full((1, 2), math.radians(85.0), dtype=torch.float64)
    )
  )
  assert tracking.anchor_failure(robot, tilted, 0.25, 0.8).all()  # 1 - cos(85°) > 0.8
  tilted = robot._replace(
    quat=quat_from_axis_angle(
      x, torch.full((1, 2), math.radians(70.0), dtype=torch.float64)
    )
  )
  assert not tracking.anchor_failure(robot, tilted, 0.25, 0.8).any()


def test_rot6d_matches_rotated_axes():
  q = _random_bodies(32, 1, 3).quat[:, 0]
  ex = torch.tensor([1.0, 0, 0], dtype=torch.float64).expand(32, 3)
  ey = torch.tensor([0, 1.0, 0], dtype=torch.float64).expand(32, 3)
  expected = torch.cat([quat_rotate(q, ex), quat_rotate(q, ey)], dim=-1)
  assert torch.allclose(rot6d(q), expected, atol=1e-12)


def test_orientation_error_gradient_is_finite_at_zero():
  q = _random_bodies(4, 1, 4).quat[:, 0].clone().requires_grad_()
  quat_error_magnitude(q, q.detach()).square().sum().backward()
  assert torch.isfinite(q.grad).all()


def test_joint_limit_violation():
  lower, upper = torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 2.0])
  q = torch.tensor([[1.5, -0.25], [0.0, 1.0]])
  assert torch.allclose(
    tracking.joint_limit_violation(q, lower, upper), torch.tensor([0.75, 0.0])
  )
