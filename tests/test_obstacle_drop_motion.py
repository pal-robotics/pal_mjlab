"""Tests for retiming a drop-from-obstacle motion to another obstacle height."""

import pytest
import torch
from pal_mjlab.tasks.tracking.mdp.obstacle_commands import (
  detect_flight_frames,
  make_height_variant,
)

DT = 0.02
H_REF = 0.6
G = 9.81


def _drop_motion():
  """Root + one foot: stand on a box, hop up, free-fall to the ground, stand."""
  stand, n_after = 20, 30
  v0 = 1.0  # Takeoff vertical velocity.
  foot_off = 0.9  # Root height above the foot.
  # Flight until the foot reaches the ground: H_REF + v0 t - g t^2 / 2 = 0.
  t_land = (v0 + (v0**2 + 2 * G * H_REF) ** 0.5) / G
  n_flight = int(t_land / DT)
  t = torch.arange(n_flight) * DT
  foot_z = torch.cat(
    [
      torch.full((stand,), H_REF),
      H_REF + v0 * t - 0.5 * G * t**2,
      torch.zeros(n_after),
    ]
  ).clamp(min=0.0)
  vz = torch.cat([torch.zeros(stand), v0 - G * t, torch.zeros(n_after)])
  n = foot_z.shape[0]
  pos = torch.zeros(n, 2, 3)
  pos[:, 0, 2] = foot_z + foot_off
  pos[:, 1, 2] = foot_z
  pos[:, :, 1] = torch.linspace(0.0, -0.5, n)[:, None]
  lin_vel = torch.zeros(n, 2, 3)
  lin_vel[:, :, 2] = vz[:, None]
  quat = torch.zeros(n, 2, 4)
  quat[..., 0] = 1.0
  return dict(
    joint_pos=torch.linspace(0, 1, n)[:, None].repeat(1, 3),
    joint_vel=torch.ones(n, 3) / (n * DT),
    body_pos_w=pos,
    body_quat_w=quat,
    body_lin_vel_w=lin_vel,
    body_ang_vel_w=torch.zeros(n, 2, 3),
  ), (stand, stand + n_flight)


def test_detect_flight_frames():
  motion, (takeoff, touchdown) = _drop_motion()
  t0, t1 = detect_flight_frames(motion["body_pos_w"][:, 1:, 2], tolerance=0.02)
  assert takeoff <= t0 <= takeoff + 2
  assert touchdown - 2 <= t1 <= touchdown + 1


def test_reference_height_is_identity():
  motion, flight = _drop_motion()
  v = make_height_variant(**motion, flight_frames=flight, height_delta=0.0, dt=DT)
  for name, x in motion.items():
    torch.testing.assert_close(getattr(v, name), x)


@pytest.mark.parametrize("dh", [-0.3, -0.1, 0.1, 0.3])
def test_height_variant(dh):
  motion, flight = _drop_motion()
  v = make_height_variant(**motion, flight_frames=flight, height_delta=dh, dt=DT)
  ref_pos = motion["body_pos_w"]
  n_land = ref_pos.shape[0] - flight[1]

  # Starts on the new obstacle, lands where the reference lands.
  torch.testing.assert_close(v.body_pos_w[0, 1, 2], torch.tensor(H_REF + dh))
  torch.testing.assert_close(v.body_pos_w[-n_land:], ref_pos[-n_land:])
  assert v.flight_frames[0] == flight[0]
  assert v.num_frames == ref_pos.shape[0] + (v.flight_frames[1] - flight[1])

  # Higher obstacle -> longer flight and faster impact (free fall: v^2 = 2 g h).
  assert (v.flight_frames[1] > flight[1]) == (dh > 0)
  impact = -v.body_lin_vel_w[v.flight_frames[1] - 1, 0, 2]
  ref_impact = -motion["body_lin_vel_w"][flight[1] - 1, 0, 2]
  assert (impact > ref_impact) == (dh > 0)

  # The position track is continuous and its velocities are consistent with it.
  z = v.body_pos_w[:, 0, 2]
  assert (z[1:] - z[:-1]).abs().max() < 0.25
  a, b = v.flight_frames
  fd = (z[a + 2 : b - 1] - z[a : b - 3]) / (2 * DT)  # Stop before touchdown.
  torch.testing.assert_close(
    fd, v.body_lin_vel_w[a + 1 : b - 2, 0, 2], atol=0.05, rtol=0
  )
