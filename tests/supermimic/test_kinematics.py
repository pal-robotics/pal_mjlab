"""Torch FK parity with MuJoCo, and the Kangaroo arm-axis fix on real BONES clips; for both
Kangaroo variants (7-DoF arms with grippers, 5-DoF arms with hands)."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  get_kangaroo_grippers_spec,
  get_kangaroo_hands_spec,
)
from pal_mjlab.tasks.supermimic.kinematics import (
  TorchKinematics,
  compile_kinematic_model,
)

FIXTURES = Path(__file__).parent / "fixtures"
VARIANTS = {  # model, a hand body, real clips (fixtures/make_fixtures.py)
  "grippers": (get_kangaroo_grippers_spec, "arm_left_7_link", "bones_3clips.npz"),
  "hands": (get_kangaroo_hands_spec, "hand_left_center_link", "bones_3clips_hands.npz"),
}


@pytest.fixture(scope="module", params=list(VARIANTS))
def variant(request) -> str:
  return request.param


@pytest.fixture(scope="module")
def model(variant) -> mujoco.MjModel:
  return compile_kinematic_model(VARIANTS[variant][0]())


def _names(model, obj, count):
  return [mujoco.mj_id2name(model, obj, i) for i in range(count)]


def _random_states(model, n, seed):
  """Seeded random configurations: any root pose, joints within range, random velocities."""
  rng = np.random.default_rng(seed)
  qpos = np.zeros((n, model.nq))
  qpos[:, :3] = rng.uniform(-2, 2, (n, 3))
  quat = rng.normal(size=(n, 4))
  qpos[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
  lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
  qpos[:, 7:] = rng.uniform(lo, hi, (n, model.nq - 7))
  qvel = rng.normal(0, 2.0, (n, model.nv))
  return qpos, qvel


def test_fk_matches_mujoco(model):
  bodies = _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)[1:]
  sites = [s for s in _names(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite) if s]
  fk = TorchKinematics(model, bodies, sites, dtype=torch.float64)
  qpos, qvel = _random_states(model, 64, seed=0)
  out = fk(torch.from_numpy(qpos), torch.from_numpy(qvel))

  data = mujoco.MjData(model)
  res = np.zeros(6)
  for i in range(len(qpos)):
    data.qpos[:], data.qvel[:] = qpos[i], qvel[i]
    mujoco.mj_forward(model, data)
    for k, name in enumerate(bodies):
      b = model.body(name).id
      np.testing.assert_allclose(out.body_pos[i, k], data.xpos[b], atol=1e-10)
      q_ref, q = data.xquat[b], out.body_quat[i, k].numpy()
      assert min(np.abs(q - q_ref).max(), np.abs(q + q_ref).max()) < 1e-10, name
      mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, res, 0)
      np.testing.assert_allclose(
        out.body_ang_vel[i, k], res[:3], atol=1e-9, err_msg=name
      )
      np.testing.assert_allclose(
        out.body_lin_vel[i, k], res[3:], atol=1e-9, err_msg=name
      )
    for k, name in enumerate(sites):
      s = model.site(name).id
      np.testing.assert_allclose(out.site_pos[i, k], data.site_xpos[s], atol=1e-10)
      mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, s, res, 0)
      np.testing.assert_allclose(
        out.site_ang_vel[i, k], res[:3], atol=1e-9, err_msg=name
      )
      np.testing.assert_allclose(
        out.site_lin_vel[i, k], res[3:], atol=1e-9, err_msg=name
      )


def test_fk_is_differentiable(model, variant):
  fk = TorchKinematics(
    model, (VARIANTS[variant][1], "leg_right_5_link"), ("imu",), dtype=torch.float64
  )
  qpos, qvel = _random_states(model, 2, seed=1)
  qpos_t = torch.from_numpy(qpos).requires_grad_()
  qvel_t = torch.from_numpy(qvel).requires_grad_()

  def f(qp, qv):
    o = fk(qp, qv)
    return o.body_pos, o.body_lin_vel, o.body_ang_vel, o.site_quat

  assert torch.autograd.gradcheck(f, (qpos_t, qvel_t))


def test_bones_clips_match_gmr_model(model, variant):
  """Real fixture: BONES retargets are defined in GMR's model of the variant (kangaroo_7dof,
  kangaroo_hands of GMR branch bones-seed-kangaroo-5dof).

  With arm_*_5/7 at axis +z (as in GMR and the robot URDF) the clips load unchanged:
  every shared body lands where GMR put it and every frame is inside the hard ranges.
  With the old -z axis the arm bodies are off by up to ~0.18 m and arm_5 leaves its range.
  With hands the shared bodies include the hand and its centre (the tracked hand frame).
  """
  fx = np.load(FIXTURES / VARIANTS[variant][2])
  joint_names = _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)[1:]
  assert list(fx["joint_names"]) == joint_names
  qpos = np.concatenate(
    [fx["root_pos"], fx["root_rot_xyzw"][:, [3, 0, 1, 2]], fx["dof_pos"]], axis=1
  ).astype(np.float64)
  fk = TorchKinematics(model, list(fx["body_names"]), dtype=torch.float64)
  out = fk(
    torch.from_numpy(qpos), torch.zeros(len(qpos), model.nv, dtype=torch.float64)
  )
  err = np.abs(out.body_pos.numpy() - fx["gmr_body_pos"]).max()
  assert err < 1e-5, f"body positions differ from the GMR model by {err:.3f} m"

  lo, hi = model.jnt_range[1:, 0], model.jnt_range[1:, 1]
  actuated = [
    i for i, n in enumerate(joint_names) if "femur" not in n and "knee" not in n
  ]
  q = fx["dof_pos"][:, actuated]
  assert np.all((q >= lo[actuated] - 1e-6) & (q <= hi[actuated] + 1e-6))


def test_compiled_fk_matches_eager(model, variant):
  """Regression: torch.compile once skipped the free joint (int vs mujoco enum compare)."""
  fk = TorchKinematics(
    model, ("base_link", VARIANTS[variant][1], "leg_right_5_link"), ("imu",)
  )
  compiled = torch.compile(fk.forward)
  for seed in range(3):
    qpos, qvel = _random_states(model, 8, seed)
    qpos_t, qvel_t = torch.from_numpy(qpos).float(), torch.from_numpy(qvel).float()
    eager, comp = fk(qpos_t, qvel_t), compiled(qpos_t, qvel_t)
    for a, b in zip(eager, comp, strict=True):
      torch.testing.assert_close(a, b, atol=1e-5, rtol=0)
