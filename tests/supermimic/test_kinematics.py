"""Torch FK parity with MuJoCo on the Kangaroo, in the model's and in a library's joint layout."""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from mjlab.scene import Scene
from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import (
  TRACKED_BODIES,
  pal_kangaroo_grippers_supermimic_env_cfg,
)
from pal_mjlab.tasks.supermimic.kinematics import TorchKinematics

LIBRARY = Path(__file__).parent / "fixtures" / "library_tiny"
BODIES = ("robot/base_link", "robot/arm_left_7_link", "robot/leg_right_5_link")
IMU = ("robot/imu",)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  return Scene(pal_kangaroo_grippers_supermimic_env_cfg().scene, "cpu").compile()


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
  bodies = [model.body(b).name for b in range(model.nbody)]
  bodies = [b for b in bodies if b.startswith("robot/")]
  sites = [model.site(s).name for s in range(model.nsite)]
  fk = TorchKinematics(model, bodies, sites, dtype=torch.float64)
  qpos, qvel = _random_states(model, 64, seed=0)
  out = fk(torch.from_numpy(qpos), torch.from_numpy(qvel))

  data = mujoco.MjData(model)
  res, quat = np.zeros(6), np.zeros(4)
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
      mujoco.mju_mat2Quat(quat, data.site_xmat[s])
      q = out.site_quat[i, k].numpy()
      assert min(np.abs(q - quat).max(), np.abs(q + quat).max()) < 1e-10, name
      mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, s, res, 0)
      np.testing.assert_allclose(
        out.site_ang_vel[i, k], res[:3], atol=1e-9, err_msg=name
      )


def test_fk_in_a_library_layout(model):
  """With the library's joints as the layout, the tracked bodies land where the model's
  own layout puts them; a body on a joint the library lacks is refused."""
  joints = [
    f"robot/{n}"
    for n in json.loads((LIBRARY / "clips.json").read_text())["joint_names"]
  ]
  bodies = [f"robot/{b}" for b in TRACKED_BODIES]
  full = TorchKinematics(model, bodies, IMU, dtype=torch.float64)
  packed = TorchKinematics(model, bodies, IMU, joints, dtype=torch.float64)
  qpos, qvel = _random_states(model, 16, seed=3)
  q_adr = [int(model.joint(j).qposadr[0]) for j in joints]
  v_adr = [int(model.joint(j).dofadr[0]) for j in joints]
  qpos_t, qvel_t = torch.from_numpy(qpos), torch.from_numpy(qvel)
  a = full(qpos_t, qvel_t)
  b = packed(
    torch.cat([qpos_t[:, :7], qpos_t[:, q_adr]], dim=1),
    torch.cat([qvel_t[:, :6], qvel_t[:, v_adr]], dim=1),
  )
  for x, y in zip(a, b, strict=True):
    torch.testing.assert_close(x, y, atol=1e-12, rtol=0)

  with pytest.raises(ValueError, match="leg_left_femur_joint"):
    TorchKinematics(model, ["robot/leg_left_knee_link"], joint_names=joints)


def test_fk_is_differentiable(model):
  fk = TorchKinematics(model, BODIES[1:], IMU, dtype=torch.float64)
  qpos, qvel = _random_states(model, 2, seed=1)
  qpos_t = torch.from_numpy(qpos).requires_grad_()
  qvel_t = torch.from_numpy(qvel).requires_grad_()

  def f(qp, qv):
    o = fk(qp, qv)
    return o.body_pos, o.body_lin_vel, o.body_ang_vel, o.site_quat

  assert torch.autograd.gradcheck(f, (qpos_t, qvel_t))


def test_traced_fk_matches_eager(model):
  """Regression: tracing once skipped the free joint (an int compared with a mujoco enum).
  ``aot_eager`` captures the graph without generating code."""
  fk = TorchKinematics(model, BODIES, IMU)
  traced = torch.compile(fk.forward, backend="aot_eager")
  for seed in range(3):
    qpos, qvel = _random_states(model, 8, seed)
    qpos_t, qvel_t = torch.from_numpy(qpos).float(), torch.from_numpy(qvel).float()
    for a, b in zip(fk(qpos_t, qvel_t), traced(qpos_t, qvel_t), strict=True):
      torch.testing.assert_close(a, b, atol=1e-5, rtol=0)
