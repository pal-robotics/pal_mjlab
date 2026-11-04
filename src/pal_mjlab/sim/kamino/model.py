"""Build a Kamino model from a compiled MuJoCo model.

Kamino simulates every moving body in maximal coordinates, so welded bodies are
merged first. Tree joints and ``<connect>`` loops then all become Kamino joints;
nothing distinguishes a loop closure from any other joint.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import mujoco
import numpy as np
import warp as wp

from pal_mjlab.third_party.kamino._newton import (
  MAXVAL,
  BodyFlags,
  JointTargetMode,
  JointType,
)
from pal_mjlab.third_party.kamino._src.core.conversions import (
  convert_joints,
  convert_rigid_bodies,
)
from pal_mjlab.third_party.kamino._src.core.geometry import GeometriesModel
from pal_mjlab.third_party.kamino._src.core.gravity import GravityModel
from pal_mjlab.third_party.kamino._src.core.model import (
  ModelKamino,
  ModelKaminoInfo,
)
from pal_mjlab.third_party.kamino._src.core.size import SizeKamino
from pal_mjlab.third_party.kamino._src.core.time import TimeModel

# MJCF range of the generated linkage joints, which never reach it. Treating them
# as unlimited saves one potential constraint row per joint in the dense problem.
_NO_LIMIT = np.pi - 1e-3
_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
_DOF_DEFAULTS = {
  "armature": 0.0,
  "damping": 0.0,
  "lower": -MAXVAL,
  "upper": MAXVAL,
  "mode": JointTargetMode.NONE,
  "ke": 0.0,
  "kd": 0.0,
  "effort": MAXVAL,
}


@dataclass
class KaminoLayout:
  """Where each mjw quantity lives in Kamino's arrays, per world."""

  body_mj: np.ndarray
  """MuJoCo body of each Kamino body."""
  body_com: np.ndarray
  """COM of each Kamino body in its frame, (nworld, nbody, 3)."""
  geom_body: np.ndarray
  """Kamino body of each MuJoCo geom, -1 for the world."""
  tree_joints: np.ndarray
  """Hinge and slide joints as rows (qpos adr, dof adr, Kamino coord, Kamino dof)."""
  actuator_coords: np.ndarray
  """Kamino coordinate each actuator targets."""
  root_body: int
  root_qadr: int
  root_dadr: int
  ncoords: int
  ndofs: int
  loops: np.ndarray
  """Kamino joint of each ``<connect>``."""


def _quat_xyzw(wxyz: np.ndarray) -> np.ndarray:
  return np.asarray([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])


def _rotmat(wxyz: np.ndarray) -> np.ndarray:
  mat = np.zeros(9)
  mujoco.mju_quat2Mat(mat, wxyz)
  return mat.reshape(3, 3)


def _relative(x_a, r_a, x_b, r_b):
  """Pose of frame b in frame a."""
  return r_a.T @ (x_b - x_a), r_a.T @ r_b


def _matrix_to_wxyz(r: np.ndarray) -> np.ndarray:
  quat = np.zeros(4)
  mujoco.mju_mat2Quat(quat, r.reshape(-1))
  return quat


class _JointRows:
  """Joints and DoFs of one world, in the layout of Newton's ``Model`` arrays."""

  def __init__(self, nworld: int):
    self.nworld = nworld
    self.parent, self.child, self.jtype, self.dim = [], [], [], []
    self.nq, self.nv, self.x_p, self.x_c = [], [], [], []
    self.dof: dict[str, list] = {k: [] for k in ("axis", "friction", *_DOF_DEFAULTS)}

  def add(self, jtype, dim, nq, axes, *, parent, child, x_p, x_c, friction=None, **dof):
    self.parent.append(parent)
    self.child.append(child)
    self.jtype.append(int(jtype))
    self.dim.append(dim)
    self.nq.append(nq)
    self.nv.append(len(axes))
    self.x_p.append(x_p)
    self.x_c.append(x_c)
    friction = np.zeros(self.nworld) if friction is None else friction
    for axis in axes:
      self.dof["axis"].append(axis)
      self.dof["friction"].append(friction)
      for name, default in _DOF_DEFAULTS.items():
        self.dof[name].append(dof.get(name, default))

  def as_newton_model(
    self, mj, roots, mass, com, inertia, d0, device
  ) -> SimpleNamespace:
    """What ``convert_rigid_bodies`` and ``convert_joints`` read from a Newton ``Model``."""
    nworld, nbody, njoint = self.nworld, len(roots), len(self.parent)
    self.q_start, self.qd_start = np.cumsum([0, *self.nq]), np.cumsum([0, *self.nv])
    ncoord, ndof = int(self.q_start[-1]), int(self.qd_start[-1])
    joint_q = np.zeros(ncoord)
    joint_q[
      self.q_start[1:][np.isin(self.jtype, (JointType.BALL, JointType.FREE))] - 1
    ] = 1.0

    def arr(values, dtype):
      return wp.array(np.asarray(values), dtype=dtype, device=device)

    def tile(values, dtype):
      values = np.asarray(values)
      return arr(np.tile(values, (nworld, *[1] * (values.ndim - 1))), dtype)

    def world_of(per_world):
      return arr(np.repeat(np.arange(nworld), per_world), wp.int32)

    def world_starts(per_world):
      return arr([*(np.arange(nworld + 1) * per_world), nworld * per_world], wp.int32)

    def body_index(values):
      local = np.asarray(values)
      return arr(
        np.concatenate(
          [np.where(local >= 0, local + w * nbody, -1) for w in range(nworld)]
        ),
        wp.int32,
      )

    def offsets(starts):
      total = starts[-1]
      return arr(
        [*(s for w in range(nworld) for s in starts[:-1] + w * total), nworld * total],
        wp.int32,
      )

    def transforms(poses):
      local = [wp.transformf(wp.vec3f(*p), wp.quatf(*_quat_xyzw(q))) for p, q in poses]
      return wp.array(local * nworld, dtype=wp.transformf, device=device)

    return SimpleNamespace(
      use_coord_layout_targets=True,
      world_count=nworld,
      device=device,
      requires_grad=False,
      body_count=nworld * nbody,
      shape_count=0,
      joint_count=nworld * njoint,
      joint_dof_count=nworld * ndof,
      articulation_count=0,
      body_label=[mj.body(int(b)).name for b in roots] * nworld,
      joint_label=[f"joint_{j}" for j in range(njoint)] * nworld,
      body_world=world_of(nbody),
      body_world_start=world_starts(nbody),
      shape_world_start=arr(np.zeros(nworld + 2), wp.int32),
      body_com=arr(com.reshape(-1, 3), wp.vec3f),
      body_q=transforms([(d0.xpos[b], d0.xquat[b]) for b in roots]),
      body_qd=wp.zeros(nworld * nbody, dtype=wp.spatial_vectorf, device=device),
      body_mass=tile(mass, wp.float32),
      body_inv_mass=tile(1.0 / mass, wp.float32),
      body_inertia=arr(inertia.reshape(-1, 3, 3), wp.mat33f),
      body_inv_inertia=arr(np.linalg.inv(inertia.reshape(-1, 3, 3)), wp.mat33f),
      body_flags=arr(np.full(nworld * nbody, int(BodyFlags.DYNAMIC)), wp.int32),
      joint_world=world_of(njoint),
      joint_world_start=world_starts(njoint),
      joint_parent=body_index(self.parent),
      joint_child=body_index(self.child),
      joint_type=tile(self.jtype, wp.int32),
      joint_dof_dim=tile(self.dim, wp.int32),
      joint_q_start=offsets(self.q_start),
      joint_qd_start=offsets(self.qd_start),
      joint_axis=tile(self.dof["axis"], wp.vec3f),
      joint_X_p=transforms(self.x_p),
      joint_X_c=transforms(self.x_c),
      joint_q=tile(joint_q, wp.float32),
      joint_qd=wp.zeros(nworld * ndof, dtype=wp.float32, device=device),
      joint_armature=tile(self.dof["armature"], wp.float32),
      joint_damping=tile(self.dof["damping"], wp.float32),
      joint_friction=arr(np.stack(self.dof["friction"], 1).reshape(-1), wp.float32),
      joint_target_mode=tile(self.dof["mode"], wp.int32),
      joint_target_ke=tile(self.dof["ke"], wp.float32),
      joint_target_kd=tile(self.dof["kd"], wp.float32),
      joint_effort_limit=tile(self.dof["effort"], wp.float32),
      joint_velocity_limit=arr(np.full(nworld * ndof, MAXVAL), wp.float32),
      joint_limit_lower=tile(self.dof["lower"], wp.float32),
      joint_limit_upper=tile(self.dof["upper"], wp.float32),
    )


def _fuse_welds(mj, d0, body_ipos: np.ndarray, min_mass: float):
  """Merge welded bodies: mass, COM and inertia per world, in the weld root's frame."""
  roots = np.flatnonzero((mj.body_weldid == np.arange(mj.nbody)) & (mj.body_weldid > 0))
  nworld = body_ipos.shape[0]
  mass = np.zeros(len(roots))
  com = np.zeros((nworld, len(roots), 3))
  inertia = np.zeros((nworld, len(roots), 3, 3))
  for i, root in enumerate(roots):
    x_r, r_r = d0.xpos[root], d0.xmat[root].reshape(3, 3)
    members = np.flatnonzero(mj.body_weldid == root)
    masses = mj.body_mass[members]
    mass[i] = masses.sum()
    points = np.zeros((nworld, len(members), 3))
    shape = np.zeros((3, 3))
    for k, b in enumerate(members):
      p_rb, r_rb = _relative(x_r, r_r, d0.xpos[b], d0.xmat[b].reshape(3, 3))
      points[:, k] = p_rb + body_ipos[:, b] @ r_rb.T
      r_bi = r_rb @ _rotmat(mj.body_iquat[b])
      shape += r_bi @ np.diag(mj.body_inertia[b]) @ r_bi.T
    com[:, i] = (masses[None, :, None] * points).sum(1) / max(mass[i], 1e-12)
    d = points - com[:, i, None]
    d2 = (d * d).sum(-1)
    inertia[:, i] = (
      shape
      + np.einsum("k,wk,ab->wab", masses, d2, np.eye(3))
      - np.einsum("k,wka,wkb->wab", masses, d, d)
    )
  return roots, np.maximum(mass, min_mass), com, inertia


def build_model(
  mj: mujoco.MjModel,
  body_ipos: np.ndarray,
  dof_frictionloss: np.ndarray,
  dt: float,
  device: wp.Device,
  min_body_mass: float,
  min_frictionloss: float,
) -> tuple[ModelKamino, KaminoLayout]:
  """Compile ``mj`` to Kamino.

  ``body_ipos`` and ``dof_frictionloss`` are per world, (nworld, nbody, 3) and
  (nworld, nv): domain randomization has already written them at build time.
  ``min_body_mass`` floors the generated 1e-05 kg placeholders and
  ``min_frictionloss`` drops friction rows too small to matter, such as the
  +/-0.008 that the friction event adds to the frictionless linkage.
  """
  nworld = body_ipos.shape[0]
  if np.any(mj.qpos0[7:] != 0.0):
    raise ValueError("Kamino bridge expects joint references at zero.")
  d0 = mujoco.MjData(mj)
  mujoco.mj_kinematics(mj, d0)  # At qpos0, where MJCF frames are defined.

  roots, mass, com, inertia = _fuse_welds(mj, d0, body_ipos, min_body_mass)
  root_index = {int(b): i for i, b in enumerate(roots)}
  index_of_weld = lambda body: root_index.get(int(mj.body_weldid[body]), -1)  # noqa: E731

  rows = _JointRows(nworld)
  actuated = {}
  for a in range(mj.nu):
    if mj.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
      raise ValueError("Kamino bridge expects joint-transmission actuators only.")
    j = int(mj.actuator_trnid[a, 0])
    limit = mj.actuator_forcerange[a, 1] if mj.actuator_forcelimited[a] else MAXVAL
    if mj.jnt_actfrclimited[j]:  # mjw clamps here too, in qfrc_actuator.
      limit = min(limit, mj.jnt_actfrcrange[j, 1])
    actuated[j] = (a, mj.actuator_gainprm[a, 0], -mj.actuator_biasprm[a, 2], limit)

  # Tree joints first, in MuJoCo's order.
  for j in range(mj.njnt):
    b = mj.jnt_bodyid[j]
    weld = index_of_weld(mj.body_parentid[b])
    x_w, r_w = (
      (d0.xpos[roots[weld]], d0.xmat[roots[weld]].reshape(3, 3))
      if weld >= 0
      else (np.zeros(3), np.eye(3))
    )
    p_rel, r_rel = _relative(x_w, r_w, d0.xpos[b], d0.xmat[b].reshape(3, 3))
    frames = dict(
      parent=weld,
      child=root_index[int(b)],
      x_p=(p_rel + r_rel @ mj.jnt_pos[j], _matrix_to_wxyz(r_rel)),
      x_c=(mj.jnt_pos[j].copy(), _IDENTITY),
    )
    if mj.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
      rows.add(JointType.FREE, (3, 3), 7, np.vstack([np.eye(3), np.eye(3)]), **frames)
      continue
    dof = mj.jnt_dofadr[j]
    lo, hi = mj.jnt_range[j]
    bounded = mj.jnt_limited[j] and max(abs(lo), abs(hi)) < _NO_LIMIT
    fric = dof_frictionloss[:, dof]
    _, kp, kv, effort = actuated.get(j, (None, 0.0, 0.0, MAXVAL))
    is_hinge = mj.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    rows.add(
      JointType.REVOLUTE if is_hinge else JointType.PRISMATIC,
      (0, 1) if is_hinge else (1, 0),
      1,
      mj.jnt_axis[j][None],
      armature=mj.dof_armature[dof],
      damping=mj.dof_damping[dof],
      friction=np.where(fric >= min_frictionloss, fric, 0.0),
      lower=lo if bounded else -MAXVAL,
      upper=hi if bounded else MAXVAL,
      mode=JointTargetMode.POSITION_VELOCITY if j in actuated else JointTargetMode.NONE,
      ke=kp,
      kd=kv,
      effort=effort,
      **frames,
    )

  # Then the loops: each <connect> between two sites becomes a ball joint.
  loops = []
  for e in range(mj.neq):
    if (
      mj.eq_type[e] != mujoco.mjtEq.mjEQ_CONNECT
      or mj.eq_objtype[e] != mujoco.mjtObj.mjOBJ_SITE
    ):
      raise ValueError("Kamino bridge expects <connect> between sites only.")
    sites = (mj.eq_obj1id[e], mj.eq_obj2id[e])
    welds = [index_of_weld(mj.site_bodyid[site]) for site in sites]
    anchors = [
      d0.xmat[roots[w]].reshape(3, 3).T @ (d0.site_xpos[site] - d0.xpos[roots[w]])
      for w, site in zip(welds, sites, strict=True)
    ]
    loops.append(len(rows.parent))
    rows.add(
      JointType.BALL,
      (0, 3),
      4,
      np.eye(3),
      parent=welds[0],
      child=welds[1],
      x_p=(anchors[0], _IDENTITY),
      x_c=(anchors[1], _IDENTITY),
    )

  ns = rows.as_newton_model(mj, roots, mass, com, inertia, d0, device)

  size = SizeKamino(num_worlds=nworld)
  with wp.ScopedDevice(device):
    info = ModelKaminoInfo(num_worlds=nworld)
    time = TimeModel(
      dt=wp.full(nworld, dt, dtype=wp.float32),
      inv_dt=wp.full(nworld, 1.0 / dt, dtype=wp.float32),
    )
    gravity = GravityModel(
      vector=wp.full(nworld, wp.vec3f(*mj.opt.gravity), dtype=wp.vec3f)
    )
    bodies = convert_rigid_bodies(ns, size, info)
    joints = convert_joints(ns, size, info, bodies)
    info.geoms_offset = wp.zeros(nworld, dtype=wp.int32)
  model = ModelKamino(
    _model=ns,
    _device=device,
    size=size,
    info=info,
    time=time,
    gravity=gravity,
    bodies=bodies,
    joints=joints,
    geoms=GeometriesModel(),
  )

  # Layout for the bridge's kernels.
  q_start, qd_start = rows.q_start, rows.qd_start
  tree = [
    (mj.jnt_qposadr[j], mj.jnt_dofadr[j], q_start[j], qd_start[j])
    for j in range(mj.njnt)
    if mj.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE
  ]
  actuator_coords = np.zeros(mj.nu, dtype=np.int32)
  for j, (a, *_) in actuated.items():
    actuator_coords[a] = q_start[j]
  free = np.flatnonzero(mj.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
  if len(free) != 1:
    raise ValueError("Kamino bridge expects one floating robot.")
  layout = KaminoLayout(
    body_mj=roots.astype(np.int32),
    body_com=com,
    geom_body=np.asarray([index_of_weld(b) for b in mj.geom_bodyid], dtype=np.int32),
    tree_joints=np.asarray(tree, dtype=np.int32),
    actuator_coords=actuator_coords,
    root_body=root_index[int(mj.jnt_bodyid[free[0]])],
    root_qadr=int(mj.jnt_qposadr[free[0]]),
    root_dadr=int(mj.jnt_dofadr[free[0]]),
    ncoords=int(q_start[-1]),
    ndofs=int(qd_start[-1]),
    loops=np.asarray(loops, dtype=np.int32),
  )
  return model, layout
