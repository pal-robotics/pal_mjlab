"""Batched, differentiable forward kinematics of a MuJoCo tree in torch.

SuperMimic's reward and observation heads are analytic: they need body and IMU poses and
velocities of *predicted* states, with gradients. The same code computes the reference
bodies from the packed motion library (qpos/qvel only), so sim, library and imagination share
one source of truth.

Follows ``mj_kinematics`` (free, hinge and slide joints, ``qpos - qpos0``) and reports
velocities of the body frame origin in world orientation, like
``mj_objectVelocity(mjOBJ_XBODY, flg_local=0)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import mujoco
import numpy as np
import torch


def compile_kinematic_model(spec: mujoco.MjSpec) -> mujoco.MjModel:
  """Compile a robot spec on its own, outside a scene.

  Contact excludes are dropped: the PAL XMLs exclude pairs with the scene's ``terrain``
  body (and kangaroo_grippers names geoms in two excludes), which only resolves inside the
  mjlab scene. Kinematics does not depend on them.
  """
  for exclude in list(spec.excludes):
    spec.delete(exclude)
  return spec.compile()


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack(
    (
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ),
    dim=-1,
  )


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Rotate ``v`` by the unit quaternion ``q`` (wxyz)."""
  w, xyz = q[..., :1], q[..., 1:]
  t = 2.0 * torch.linalg.cross(xyz, v, dim=-1)
  return v + w * t + torch.linalg.cross(xyz, t, dim=-1)


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
  half = 0.5 * angle.unsqueeze(-1)
  return torch.cat((torch.cos(half), torch.sin(half) * axis), dim=-1)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
  return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """Express world vectors ``v`` in the frame of ``q``."""
  return quat_rotate(quat_conj(q), v)


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
  """Heading (yaw-only) part of a quaternion."""
  w, x, y, z = q.unbind(-1)
  yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  zero = torch.zeros_like(yaw)
  return torch.stack([torch.cos(0.5 * yaw), zero, zero, torch.sin(0.5 * yaw)], dim=-1)


def quat_error_magnitude(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Rotation angle between two orientations, in [0, pi]."""
  d = quat_mul(a, quat_conj(b))
  # eps keeps the gradient finite when the orientations coincide (BPTT goes through it).
  sin_half = torch.sqrt(d[..., 1:].square().sum(-1) + 1e-12)
  return 2.0 * torch.atan2(sin_half, d[..., 0].abs())


def matrix_from_rot6d(r6: torch.Tensor) -> torch.Tensor:
  """Rotation matrix (..., 3, 3) from the first two columns, Gram-Schmidt orthonormalized."""
  a, b = r6[..., :3], r6[..., 3:]
  x = torch.nn.functional.normalize(a, dim=-1)
  y = torch.nn.functional.normalize(b - (x * b).sum(-1, keepdim=True) * x, dim=-1)
  z = torch.linalg.cross(x, y, dim=-1)
  return torch.stack([x, y, z], dim=-1)


def quat_from_matrix(m: torch.Tensor) -> torch.Tensor:
  """Unit quaternion (wxyz) of a rotation matrix, robust for any angle.

  Computes the four candidate solutions and keeps the best-conditioned one (largest
  denominator), which stays accurate and differentiable near 180 degrees.
  """
  m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
  m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
  m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]
  four = torch.stack(
    [
      1 + m00 + m11 + m22,
      1 + m00 - m11 - m22,
      1 - m00 + m11 - m22,
      1 - m00 - m11 + m22,
    ],
    dim=-1,
  )
  r = torch.sqrt(four.clamp_min(1e-12))  # 2|w|, 2|x|, 2|y|, 2|z|
  candidates = torch.stack(
    [
      torch.stack([r[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
      torch.stack([m21 - m12, r[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
      torch.stack([m02 - m20, m10 + m01, r[..., 2] ** 2, m12 + m21], dim=-1),
      torch.stack([m10 - m01, m20 + m02, m21 + m12, r[..., 3] ** 2], dim=-1),
    ],
    dim=-2,
  ) / (2.0 * r[..., :, None])
  best = four.argmax(dim=-1)
  q = torch.gather(
    candidates, -2, best[..., None, None].expand(*best.shape, 1, 4)
  ).squeeze(-2)
  return torch.nn.functional.normalize(q, dim=-1)


def yaw_angle(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rot6d(q: torch.Tensor) -> torch.Tensor:
  """First two columns of the rotation matrix (continuous 6D representation)."""
  w, x, y, z = q.unbind(-1)
  return torch.stack(
    (
      1 - 2 * (y * y + z * z),
      2 * (x * y + w * z),
      2 * (x * z - w * y),
      2 * (x * y - w * z),
      1 - 2 * (x * x + z * z),
      2 * (y * z + w * x),
    ),
    dim=-1,
  )


class Kinematics(NamedTuple):
  """World-frame poses and velocities, shapes (B, n, 3) / (B, n, 4) (quat wxyz)."""

  body_pos: torch.Tensor
  body_quat: torch.Tensor
  body_lin_vel: torch.Tensor
  body_ang_vel: torch.Tensor
  site_pos: torch.Tensor
  site_quat: torch.Tensor
  site_lin_vel: torch.Tensor
  site_ang_vel: torch.Tensor


# Joint types as plain ints: comparing an int with a pybind11 mujoco enum is not traced
# correctly by torch.compile (the free-joint branch was silently skipped).
_FREE = int(mujoco.mjtJoint.mjJNT_FREE)
_SLIDE = int(mujoco.mjtJoint.mjJNT_SLIDE)
_HINGE = int(mujoco.mjtJoint.mjJNT_HINGE)


@dataclass(frozen=True)
class _Joint:
  kind: int
  qadr: int
  dadr: int
  axis: torch.Tensor  # (3,) in the body frame
  pos: torch.Tensor  # (3,) anchor in the body frame
  qpos0: float


@dataclass(frozen=True)
class _Body:
  parent: int  # index into the evaluated body list, -1 for the world
  pos: torch.Tensor  # (3,) offset in the parent frame
  quat: torch.Tensor  # (4,) orientation in the parent frame
  joints: tuple[_Joint, ...]


class TorchKinematics:
  """FK for a fixed set of bodies and sites of one MuJoCo model.

  Only the requested bodies, the sites' bodies and their ancestors are evaluated. All
  model constants are tensors built once here, so ``forward`` is plain tensor code
  (eager or ``torch.compile``).
  """

  def __init__(
    self,
    model: mujoco.MjModel,
    body_names: Sequence[str],
    site_names: Sequence[str] = (),
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
  ) -> None:
    self.device, self.dtype = torch.device(device), dtype
    body_ids = [_require(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in body_names]
    site_ids = [_require(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in site_names]
    needed: set[int] = set()
    for b in body_ids + [int(model.site_bodyid[s]) for s in site_ids]:
      while b > 0:
        needed.add(b)
        b = int(model.body_parentid[b])
    order = sorted(needed)  # MuJoCo body ids are topologically sorted
    index = {b: k for k, b in enumerate(order)}

    bodies = []
    for b in order:
      joints = []
      for j in range(model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]):
        kind = int(model.jnt_type[j])
        if kind not in (_FREE, _SLIDE, _HINGE):
          raise NotImplementedError("only free, slide and hinge joints are supported")
        if kind == _FREE and model.body_parentid[b] != 0:
          raise NotImplementedError("free joints only on children of the world")
        qadr = int(model.jnt_qposadr[j])
        joints.append(
          _Joint(
            kind=kind,
            qadr=qadr,
            dadr=int(model.jnt_dofadr[j]),
            axis=self._t(model.jnt_axis[j]),
            pos=self._t(model.jnt_pos[j]),
            qpos0=float(model.qpos0[qadr]),
          )
        )
      parent = int(model.body_parentid[b])
      bodies.append(
        _Body(
          parent=index.get(parent, -1),
          pos=self._t(model.body_pos[b]),
          quat=self._t(model.body_quat[b]),
          joints=tuple(joints),
        )
      )
    self._bodies = bodies
    self._out_bodies = [index[b] for b in body_ids]
    self._site_body = [index[int(model.site_bodyid[s])] for s in site_ids]
    self._site_pos = self._t(np.asarray(model.site_pos[site_ids]).reshape(-1, 3))
    self._site_quat = self._t(np.asarray(model.site_quat[site_ids]).reshape(-1, 4))
    self.body_names = tuple(body_names)
    self.site_names = tuple(site_names)

  def _t(self, x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=self.dtype, device=self.device)

  def forward(self, qpos: torch.Tensor, qvel: torch.Tensor) -> Kinematics:
    """qpos (B, nq), qvel (B, nv) in MuJoCo layout (free joint: lin vel world, ang vel local)."""
    batch = qpos.shape[0]
    pos: list[torch.Tensor] = []
    quat: list[torch.Tensor] = []
    lin: list[torch.Tensor] = []
    ang: list[torch.Tensor] = []
    zero3 = qpos.new_zeros(batch, 3)
    ident = torch.cat([qpos.new_ones(batch, 1), qpos.new_zeros(batch, 3)], dim=-1)
    for body in self._bodies:
      if body.parent < 0:
        p_par, q_par, v_par, w_par = zero3, ident, zero3, zero3
      else:
        p_par, q_par = pos[body.parent], quat[body.parent]
        v_par, w_par = lin[body.parent], ang[body.parent]

      if body.joints and body.joints[0].kind == _FREE:
        jnt = body.joints[0]
        q = qpos[:, jnt.qadr + 3 : jnt.qadr + 7]
        q = q / q.norm(dim=-1, keepdim=True)
        pos.append(qpos[:, jnt.qadr : jnt.qadr + 3])
        quat.append(q)
        lin.append(qvel[:, jnt.dadr : jnt.dadr + 3])
        ang.append(quat_rotate(q, qvel[:, jnt.dadr + 3 : jnt.dadr + 6]))
        continue

      q = quat_mul(q_par, body.quat.expand(batch, 4))
      p = p_par + quat_rotate(q_par, body.pos.expand(batch, 3))
      # Joint contributions, applied in order; velocities are accumulated as
      # (axis, anchor, rate) and turned into the origin velocity at the end.
      w = w_par
      screws = []
      for jnt in body.joints:
        axis_w = quat_rotate(q, jnt.axis.expand(batch, 3))
        anchor = p + quat_rotate(q, jnt.pos.expand(batch, 3))
        val = qpos[:, jnt.qadr] - jnt.qpos0
        rate = qvel[:, jnt.dadr : jnt.dadr + 1]
        if jnt.kind == _SLIDE:
          p = p + axis_w * val.unsqueeze(-1)
          screws.append((axis_w, None, rate))
        else:  # hinge
          q = quat_mul(q, quat_from_axis_angle(jnt.axis.expand(batch, 3), val))
          p = anchor - quat_rotate(q, jnt.pos.expand(batch, 3))
          screws.append((axis_w, anchor, rate))
          w = w + axis_w * rate
      v = v_par + torch.linalg.cross(w_par, p - p_par, dim=-1)
      for axis_w, anchor, rate in screws:
        if anchor is None:
          v = v + axis_w * rate
        else:
          v = v + torch.linalg.cross(axis_w * rate, p - anchor, dim=-1)
      pos.append(p)
      quat.append(q)
      lin.append(v)
      ang.append(w)

    ob = self._out_bodies
    body_pos = torch.stack([pos[k] for k in ob], dim=1)
    body_quat = torch.stack([quat[k] for k in ob], dim=1)
    body_lin = torch.stack([lin[k] for k in ob], dim=1)
    body_ang = torch.stack([ang[k] for k in ob], dim=1)

    if self._site_body:
      sb = self._site_body
      bp = torch.stack([pos[k] for k in sb], dim=1)
      bq = torch.stack([quat[k] for k in sb], dim=1)
      bv = torch.stack([lin[k] for k in sb], dim=1)
      bw = torch.stack([ang[k] for k in sb], dim=1)
      offset = quat_rotate(bq, self._site_pos.expand(batch, -1, -1))
      site_pos = bp + offset
      site_quat = quat_mul(bq, self._site_quat.expand(batch, -1, -1))
      site_lin = bv + torch.linalg.cross(bw, offset, dim=-1)
      site_ang = bw
    else:
      empty3, empty4 = qpos.new_zeros(batch, 0, 3), qpos.new_zeros(batch, 0, 4)
      site_pos, site_quat, site_lin, site_ang = empty3, empty4, empty3, empty3
    return Kinematics(
      body_pos, body_quat, body_lin, body_ang, site_pos, site_quat, site_lin, site_ang
    )

  __call__ = forward


def _require(model: mujoco.MjModel, obj: mujoco.mjtObj, name: str) -> int:
  idx = mujoco.mj_name2id(model, obj, name)
  if idx < 0:
    raise KeyError(f"{name!r} not found in model")
  return idx
