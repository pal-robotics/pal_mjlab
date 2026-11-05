"""``Simulation`` whose physics step is Kamino.

``mjw.Data`` stays the interface mjlab reads, so kinematics, collision and sensors all
remain mujoco_warp's. Kamino's maximal state is authoritative between steps: it is
projected to ``qpos``/``qvel`` on commit, and rebuilt from them only in worlds that
mjlab wrote to (resets, pushes).
"""

import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from mjlab.sim.sim import Simulation, _suspend_gc

from pal_mjlab.sim.kamino import KaminoSimulationCfg
from pal_mjlab.sim.kamino.contacts import read_mjw_contacts, write_mjw_forces
from pal_mjlab.sim.kamino.model import build_model
from pal_mjlab.third_party.kamino._src.geometry.contacts import ContactsKamino
from pal_mjlab.third_party.kamino._src.kinematics.resets import (
  reset_body_wrenches,
  reset_joints_state_from_bodies_state,
)
from pal_mjlab.third_party.kamino._src.solver_kamino_impl import SolverKaminoImpl
from pal_mjlab.third_party.kamino.config import (
  ConstrainedDynamicsConfig,
  ConstraintStabilizationConfig,
  PADMMSolverConfig,
  SolverKaminoConfig,
)


@wp.kernel
def _bodies_from_mjw(
  mask: wp.array[wp.bool],
  xpos: wp.array2d[wp.vec3],
  xquat: wp.array2d[wp.quat],
  cvel: wp.array2d[wp.spatial_vector],
  subtree_com: wp.array2d[wp.vec3],
  body_rootid: wp.array[int],
  body_mj: wp.array[int],
  body_com: wp.array[wp.vec3f],
  nbody: int,
  q_i: wp.array[wp.transformf],
  u_i: wp.array[wp.spatial_vectorf],
):
  w, i = wp.tid()
  if not mask[w]:
    return
  b = body_mj[i]
  k = w * nbody + i
  q = xquat[w, b]  # mjw keeps MuJoCo's wxyz in Warp's quat slots.
  quat = wp.quatf(q[1], q[2], q[3], q[0])
  com = xpos[w, b] + wp.quat_rotate(quat, body_com[k])
  omega = wp.spatial_top(cvel[w, b])
  lin = wp.spatial_bottom(cvel[w, b])
  v = lin + wp.cross(omega, com - subtree_com[w, body_rootid[b]])
  q_i[k] = wp.transformf(com, quat)
  u_i[k] = wp.spatial_vectorf(v[0], v[1], v[2], omega[0], omega[1], omega[2])


@wp.kernel
def _targets_to_kamino(
  ctrl: wp.array2d[float],
  coord: wp.array[int],
  ncoords: int,
  q_j_ref: wp.array[wp.float32],
):
  w, a = wp.tid()
  q_j_ref[w * ncoords + coord[a]] = ctrl[w, a]


@wp.kernel
def _tree_joints_from_kamino(
  q_j: wp.array[wp.float32],
  dq_j: wp.array[wp.float32],
  qadr: wp.array[int],
  vadr: wp.array[int],
  cadr: wp.array[int],
  dadr: wp.array[int],
  ncoords: int,
  ndofs: int,
  qpos_next: wp.array2d[float],
  qvel_next: wp.array2d[float],
):
  w, j = wp.tid()
  qpos_next[w, qadr[j]] = q_j[w * ncoords + cadr[j]]
  qvel_next[w, vadr[j]] = dq_j[w * ndofs + dadr[j]]


@wp.kernel
def _root_from_kamino(
  q_i: wp.array[wp.transformf],
  u_i: wp.array[wp.spatial_vectorf],
  body_com: wp.array[wp.vec3f],
  root_body: int,
  nbody: int,
  qadr: int,
  vadr: int,
  qpos_next: wp.array2d[float],
  qvel_next: wp.array2d[float],
):
  w = wp.tid()
  k = w * nbody + root_body
  com = wp.transform_get_translation(q_i[k])
  quat = wp.transform_get_rotation(q_i[k])
  v = wp.spatial_top(u_i[k])
  omega = wp.spatial_bottom(u_i[k])
  origin = com - wp.quat_rotate(quat, body_com[k])
  v_origin = v + wp.cross(omega, origin - com)
  omega_body = wp.quat_rotate_inv(quat, omega)
  for i in range(3):
    qpos_next[w, qadr + i] = origin[i]
    qvel_next[w, vadr + i] = v_origin[i]
    qvel_next[w, vadr + 3 + i] = omega_body[i]
  qpos_next[w, qadr + 3] = quat[3]
  qpos_next[w, qadr + 4] = quat[0]
  qpos_next[w, qadr + 5] = quat[1]
  qpos_next[w, qadr + 6] = quat[2]


@wp.kernel
def _acceleration(
  qvel_next: wp.array2d[float],
  qvel: wp.array2d[float],
  inv_dt: float,
  qacc: wp.array2d[float],
):
  w, i = wp.tid()
  qacc[w, i] = inv_dt * (qvel_next[w, i] - qvel[w, i])


@wp.kernel
def _advance_time(dt: float, time: wp.array[float]):
  time[wp.tid()] += dt


class KaminoSimulation(Simulation):
  """Steps with Kamino, reads and writes through ``mjw.Data``.

  The Kamino model is built on the first ``step()``, because startup events write
  the per-world values it needs (``body_ipos``, ``dof_frictionloss``) after init. Other
  model fields are read from the template, and ``xfrc_applied`` is not read at all.
  """

  cfg: KaminoSimulationCfg

  def __init__(self, **kwargs):
    self._solver = None
    super().__init__(**kwargs)

  def create_graph(self) -> None:
    # Unlike the base class, doesn't capture mjw's solver, which never runs here.
    self.step_graph = None
    self.forward_graph = None
    self.reset_graph = None
    self.sense_graph = None
    if self.use_cuda_graph:
      with _suspend_gc(), wp.ScopedDevice(self.wp_device):
        with wp.ScopedCapture() as capture:
          self._forward()
        self.forward_graph = capture.graph
        with wp.ScopedCapture() as capture:
          mjwarp.reset_data(self.wp_model, self.wp_data, reset=self._reset_mask_wp)
        self.reset_graph = capture.graph
        if self._sensor_context is not None:
          with wp.ScopedCapture() as capture:
            self._sense_kernel()
          self.sense_graph = capture.graph
        if self._solver is not None:
          self._capture_step()

  def _capture_step(self) -> None:
    with _suspend_gc(), wp.ScopedCapture() as capture:
      self._step()
    self.step_graph = capture.graph

  def _forward(self, collide: bool = False) -> None:
    """mjw's forward up to the solver. Without ``collide``, acceleration-stage sensors
    keep the last solved contact forces, which is what the critic should see."""
    m, d = self.wp_model, self.wp_data
    mjwarp.fwd_kinematics(m, d)
    if collide:
      mjwarp.collision(m, d)
    mjwarp.transmission(m, d)
    mjwarp.sensor_pos(m, d)
    mjwarp.fwd_velocity(m, d)
    mjwarp.sensor_vel(m, d)
    mjwarp.fwd_actuation(m, d)

  def _build(self) -> None:
    cfg, m, d, dev = self.cfg, self._mj_model, self.wp_data, self.wp_device
    # mj_forward at init left its constraint counts here; no mjw solver runs anymore.
    for count in (d.ne, d.nf, d.nl, d.nefc):
      count.zero_()

    def per_world(field: wp.array) -> np.ndarray:
      values = field.numpy()
      return np.broadcast_to(values, (self.num_envs, *values.shape[1:]))

    self.kamino, self.layout = build_model(
      m,
      per_world(self.wp_model.body_ipos),
      per_world(self.wp_model.dof_frictionloss),
      cfg.mujoco.timestep,
      dev,
      cfg.min_body_mass,
      cfg.min_frictionloss,
    )
    lay = self.layout
    self.contacts = ContactsKamino(
      capacity=[cfg.max_contacts] * self.num_envs, device=dev, remappable=True
    )
    is_crf = cfg.linear_solver == "CRF"
    self._solver = SolverKaminoImpl(
      self.kamino,
      self.contacts,
      SolverKaminoConfig(
        sparse_jacobian=True,
        sparse_dynamics=is_crf,
        constraints=ConstraintStabilizationConfig(
          alpha=cfg.alpha,
          max_joint_error=cfg.max_joint_error,
          max_limit_error=cfg.max_limit_error,
        ),
        dynamics=ConstrainedDynamicsConfig(
          linear_solver_type=cfg.linear_solver,
          linear_solver_kwargs={"maxiter": cfg.crf_iterations} if is_crf else {},
        ),
        padmm=PADMMSolverConfig(
          primal_tolerance=cfg.tolerance,
          dual_tolerance=cfg.tolerance,
          compl_tolerance=cfg.tolerance,
          max_iterations=cfg.max_iterations,
          rho_0=cfg.rho_0,
          contact_warmstart_method=cfg.contact_warmstart,
          use_graph_conditionals=not is_crf,
        ),
      ),
    )
    self.state_p = self.kamino.state()
    self.state_n = self.kamino.state()
    self.control = self.kamino.control()

    ints = lambda a: wp.array(a, dtype=wp.int32, device=dev)  # noqa: E731
    self._body_mj = ints(lay.body_mj)
    self._geom_body = ints(lay.geom_body)
    self._body_com = wp.array(lay.body_com.reshape(-1, 3), dtype=wp.vec3f, device=dev)
    self._act_coord = ints(lay.actuator_coords)
    self._tree = [ints(lay.tree_joints[:, i]) for i in range(4)]
    like = lambda a: wp.zeros_like(a)  # noqa: E731
    self._qpos_next, self._qvel_next = like(d.qpos), like(d.qvel)
    self._qpos_last, self._qvel_last = like(d.qpos), like(d.qvel)
    self._rows_used = wp.zeros(self.num_envs, dtype=int, device=dev)
    self._mask = wp.ones(self.num_envs, dtype=wp.bool, device=dev)
    self._cold = wp.ones(self.num_envs, dtype=wp.bool, device=dev)
    self._resync()
    if self.use_cuda_graph:
      self._capture_step()

  def _step(self) -> None:
    m, d = self.wp_model, self.wp_data
    cfg, lay, nbody = self.cfg, self.layout, len(self.layout.body_mj)
    d.sensordata.zero_()
    self._forward(collide=True)

    read_mjw_contacts(d, self._geom_body, nbody, self._mj_model.ngeom, self.contacts)
    wp.launch(
      _targets_to_kamino,
      dim=(self.num_envs, self._mj_model.nu),
      inputs=[d.ctrl, self._act_coord, lay.ncoords, self.control.q_j_ref],
    )
    self.state_p.copy_from(self.state_n)
    self._solver.step(self.state_p, self.state_n, self.control, self.contacts)

    wp.launch(
      _tree_joints_from_kamino,
      dim=(self.num_envs, len(lay.tree_joints)),
      inputs=[
        self.state_n.q_j,
        self.state_n.dq_j,
        *self._tree,
        lay.ncoords,
        lay.ndofs,
        self._qpos_next,
        self._qvel_next,
      ],
    )
    wp.launch(
      _root_from_kamino,
      dim=self.num_envs,
      inputs=[
        self.state_n.q_i,
        self.state_n.u_i,
        self._body_com,
        lay.root_body,
        nbody,
        lay.root_qadr,
        lay.root_dadr,
        self._qpos_next,
        self._qvel_next,
      ],
    )
    wp.launch(
      _acceleration,
      dim=(self.num_envs, self._mj_model.nv),
      inputs=[self._qvel_next, d.qvel, 1.0 / cfg.mujoco.timestep, d.qacc],
    )
    write_mjw_forces(d, d.njmax, self._rows_used, self.contacts)
    mjwarp.sensor_acc(m, d)

    # Commit last, as mjw does: acceleration sensors pair qacc with this step's cdof.
    wp.copy(d.qpos, self._qpos_next)
    wp.copy(d.qvel, self._qvel_next)
    wp.copy(self._qpos_last, d.qpos)
    wp.copy(self._qvel_last, d.qvel)
    wp.launch(_advance_time, dim=self.num_envs, inputs=[cfg.mujoco.timestep, d.time])

  def _resync(self) -> None:
    """Rebuild Kamino's bodies from ``mjw.Data`` in the worlds in ``self._mask``."""
    d, nbody = self.wp_data, len(self.layout.body_mj)
    wp.launch(
      _bodies_from_mjw,
      dim=(self.num_envs, nbody),
      inputs=[
        self._mask,
        d.xpos,
        d.xquat,
        d.cvel,
        d.subtree_com,
        self.wp_model.body_rootid,
        self._body_mj,
        self._body_com,
        nbody,
        self.state_n.q_i,
        self.state_n.u_i,
      ],
    )
    reset_joints_state_from_bodies_state(self.kamino, self.state_n, self._mask)
    reset_body_wrenches(self.kamino, self.state_n, self._mask)
    self._solver._reset_solver_data(world_mask=self._cold)
    wp.copy(self._qpos_last, d.qpos)
    wp.copy(self._qvel_last, d.qvel)

  def forward(self) -> None:
    with wp.ScopedDevice(self.wp_device):
      if self.use_cuda_graph and self.forward_graph is not None:
        wp.capture_launch(self.forward_graph)
      else:
        self._forward()
      # All of mjlab's state writes (resets, pushes) land before forward().
      if self._solver is not None:
        moved = (wp.to_torch(self.wp_data.qpos) != wp.to_torch(self._qpos_last)).any(1)
        written = moved | (
          wp.to_torch(self.wp_data.qvel) != wp.to_torch(self._qvel_last)
        ).any(1)
        if written.any():
          wp.to_torch(self._mask).copy_(written)
          # A push only writes velocities, so its contact forces still warm start well.
          wp.to_torch(self._cold).copy_(moved)
          self._resync()

  def step(self) -> None:
    with wp.ScopedDevice(self.wp_device):
      if self._solver is None:
        with torch.no_grad():
          self._build()
      with self.nan_guard.watch(self.data):
        if self.use_cuda_graph and self.step_graph is not None:
          wp.capture_launch(self.step_graph)
        else:
          self._step()
