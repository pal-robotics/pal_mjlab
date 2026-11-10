"""Batched, differentiable forward kinematics of a MuJoCo tree in torch.

SuperMimic's reward and observation heads are analytic: they need body and IMU poses and
velocities of *predicted* states, with gradients. The same code computes the reference
bodies from the motion library (qpos/qvel only), so sim, library and imagination share
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
from mjlab.utils.lab_api.math import quat_apply, quat_conjugate, quat_mul


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
  half = 0.5 * angle.unsqueeze(-1)
  return torch.cat((torch.cos(half), torch.sin(half) * axis), dim=-1)


def quat_error_magnitude(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Rotation angle between two orientations, in [0, pi]."""
  d = quat_mul(a, quat_conjugate(b))
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
  site_quat: torch.Tensor
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

  Args:
    model: The compiled model.
    body_names: Bodies to report, in output order.
    site_names: Sites to report, in output order.
    joint_names: Joint layout of the inputs: the floating base, then these joints in
      this order, one column each (``qpos`` (B, 7 + n), ``qvel`` (B, 6 + n)). A requested
      body may only depend on these joints. None for the model's own layout.
    device: Device of the constants and outputs.
    dtype: Floating point type of the constants and outputs.
  """

  def __init__(
    self,
    model: mujoco.MjModel,
    body_names: Sequence[str],
    site_names: Sequence[str] = (),
    joint_names: Sequence[str] | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
  ) -> None:
    self.device, self.dtype = torch.device(device), dtype
    column = None if joint_names is None else {n: i for i, n in enumerate(joint_names)}
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
        name = model.joint(j).name
        if column is None:
          qadr, dadr = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        elif kind == _FREE:
          qadr = dadr = 0
        elif name in column:
          qadr, dadr = 7 + column[name], 6 + column[name]
        else:
          raise ValueError(
            f"Body {model.body(b).name!r} depends on joint {name!r}, "
            "which is not in the joint layout."
          )
        joints.append(
          _Joint(
            kind=kind,
            qadr=qadr,
            dadr=dadr,
            axis=self._t(model.jnt_axis[j]),
            pos=self._t(model.jnt_pos[j]),
            qpos0=float(model.qpos0[model.jnt_qposadr[j]]),
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
    self._site_quat = self._t(np.asarray(model.site_quat[site_ids]).reshape(-1, 4))
    self.body_names = tuple(body_names)
    self.site_names = tuple(site_names)
    self.joint_names = None if joint_names is None else tuple(joint_names)

  def _t(self, x) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=self.dtype, device=self.device)

  def forward(self, qpos: torch.Tensor, qvel: torch.Tensor) -> Kinematics:
    """Kinematics of ``qpos`` and ``qvel`` in the joint layout (free joint as in MuJoCo:
    linear velocity in the world frame, angular velocity in the body frame)."""
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
        ang.append(quat_apply(q, qvel[:, jnt.dadr + 3 : jnt.dadr + 6]))
        continue

      q = quat_mul(q_par, body.quat.expand(batch, 4))
      p = p_par + quat_apply(q_par, body.pos.expand(batch, 3))
      # Joint contributions, applied in order; velocities are accumulated as
      # (axis, anchor, rate) and turned into the origin velocity at the end.
      w = w_par
      screws = []
      for jnt in body.joints:
        axis_w = quat_apply(q, jnt.axis.expand(batch, 3))
        anchor = p + quat_apply(q, jnt.pos.expand(batch, 3))
        val = qpos[:, jnt.qadr] - jnt.qpos0
        rate = qvel[:, jnt.dadr : jnt.dadr + 1]
        if jnt.kind == _SLIDE:
          p = p + axis_w * val.unsqueeze(-1)
          screws.append((axis_w, None, rate))
        else:  # hinge
          q = quat_mul(q, quat_from_axis_angle(jnt.axis.expand(batch, 3), val))
          p = anchor - quat_apply(q, jnt.pos.expand(batch, 3))
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
      bq = torch.stack([quat[k] for k in sb], dim=1)
      site_quat = quat_mul(bq, self._site_quat.expand(batch, -1, -1))
      site_ang = torch.stack([ang[k] for k in sb], dim=1)
    else:
      site_quat, site_ang = qpos.new_zeros(batch, 0, 4), qpos.new_zeros(batch, 0, 3)
    return Kinematics(body_pos, body_quat, body_lin, body_ang, site_quat, site_ang)

  __call__ = forward


def _require(model: mujoco.MjModel, obj: mujoco.mjtObj, name: str) -> int:
  idx = mujoco.mj_name2id(model, obj, name)
  if idx < 0:
    raise KeyError(f"{name!r} not found in model")
  return idx
