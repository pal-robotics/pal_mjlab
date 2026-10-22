"""Finite-difference velocities of the motion-library converter follow MuJoCo's qvel layout."""

import mujoco
import numpy as np
import pytest
from pal_mjlab.scripts.bones_to_library import clip_qpos_qvel


@pytest.mark.parametrize("seed", range(5))
def test_constant_motion_is_recovered_exactly(seed):
  """Constant body-frame angular velocity, linear root motion and linear joint ramps
  are differentiated exactly, including the one-sided first and last frames."""
  rng = np.random.default_rng(seed)
  fps, n, nj = 50.0, 40, 32
  t = np.arange(n) / fps
  omega_b = rng.normal(0, 2.0, 3)
  v_w = rng.normal(0, 1.0, 3)
  joint_rate = rng.normal(0, 1.0, nj)

  q0 = rng.normal(size=4)
  q0 /= np.linalg.norm(q0)
  quat = np.zeros((n, 4))
  for k in range(n):
    # q(t) = q0 * exp(omega_b t): rotation at constant body-frame rate.
    dq = np.zeros(4)
    mujoco.mju_axisAngle2Quat(
      dq, omega_b / np.linalg.norm(omega_b), np.linalg.norm(omega_b) * t[k]
    )
    mujoco.mju_mulQuat(quat[k], q0, dq)
  quat[1::2] *= -1.0  # sign flips must not matter
  root_pos = rng.normal(size=3) + v_w * t[:, None]
  dof = rng.normal(size=nj) + joint_rate * t[:, None]

  qpos, qvel = clip_qpos_qvel(root_pos, quat, dof, fps)
  assert qpos.shape == (n, 7 + nj) and qvel.shape == (n, 6 + nj)
  np.testing.assert_allclose(qvel[:, :3], np.broadcast_to(v_w, (n, 3)), atol=1e-9)
  np.testing.assert_allclose(qvel[:, 3:6], np.broadcast_to(omega_b, (n, 3)), atol=1e-9)
  np.testing.assert_allclose(
    qvel[:, 6:], np.broadcast_to(joint_rate, (n, nj)), atol=1e-9
  )
  # Stored quaternions are unit and hemisphere-continuous.
  np.testing.assert_allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1.0, atol=1e-12)
  assert np.all(np.sum(qpos[1:, 3:7] * qpos[:-1, 3:7], axis=1) > 0)
