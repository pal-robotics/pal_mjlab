"""Single-definition terms: one step's tensors, built from the simulator or imagined.

SuperMimic optimizes, in imagination, the same observations and rewards the env computes.
Instead of writing each term twice (once on the simulator, once on predicted states), a
term is written once, as a pure function of a ``Batch`` (one step's tensors), and
``@imaginable`` makes the env's term from it:

  @imaginable
  def action_rate(b: Batch) -> torch.Tensor:
    return (b.action - b.prev_action).square().sum(-1)

``action_rate`` is then an mjlab term ``(env, **params)``: it calls the function on the
batch built from the simulator's true state (``sim_batch``). The heads call the same
function, ``action_rate.imagined``, on batches built from predicted states
(``BatchSource.imagined``). The two agree by construction.

Timing matches mjlab's: rewards are computed before the command updates, so they see the
command state the step targeted; observations after, so they see the advanced one.

What a term can use is what the batch holds: the heading state and root pose, FK of the
configured bodies and sites, the command state and its model, the last two actions and
the sim context ξ. A term that needs anything else (contact forces, say) cannot be
written, as it could not be imagined.
"""

from __future__ import annotations

import copy
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pal_mjlab.utils.math.kinematics import Kinematics, TorchKinematics

from .command_model import CommandModel, CommandState, command_model_of
from .names import imagination_cfg
from .state import heading_pose, heading_state, state_to_qpos_qvel

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class ContextPart(NamedTuple):
  """A named part of the sim context ξ, as its term declares it (``context_layout``)."""

  columns: slice
  """Its columns in ξ."""
  joint_ids: list[int] | None = None
  """For a per-joint part: the robot joint of each column."""


class BatchSpec:
  """Constants of the batch layout, resolved once per env: which robot joints the state
  holds, the FK's bodies and sites, the joints' defaults and limits."""

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    cfg = imagination_cfg(env)
    action = env.action_manager.get_term(cfg.action_term)
    self.entity_name: str = action.cfg.entity_name
    robot = env.scene[self.entity_name]
    ids = action.target_ids
    self.joint_ids: list[int] = [int(i) for i in ids]
    """Robot joint ids the state holds, in state order: the action joints."""
    self.joint_names: list[str] = [robot.joint_names[i] for i in self.joint_ids]
    self.num_robot_joints = len(robot.joint_names)
    self.body_names: tuple[str, ...] = tuple(cfg.bodies)
    self.site_names: tuple[str, ...] = tuple(cfg.sites)
    self._body = {name: i for i, name in enumerate(self.body_names)}
    self._site = {name: i for i, name in enumerate(self.site_names)}
    self.q0 = robot.data.default_joint_pos[0, ids].clone()
    """(nj,) default positions of the state's joints."""
    self.joint_limits = robot.data.joint_pos_limits[0, ids].clone()
    """(nj, 2) position limits of the state's joints."""
    self.qd0 = robot.data.default_joint_vel[0, ids].clone()
    """(nj,) default velocities of the state's joints."""
    self.context_parts: dict[str, ContextPart] = {}
    """ξ's named parts (``ContextPart``), from its term's ``context_layout``."""
    self._index: dict[tuple, slice | torch.Tensor] = {}
    self._device = robot.data.default_joint_pos.device
    idx = robot.indexing
    self.q_adr = torch.cat([idx.free_joint_q_adr, idx.joint_q_adr[ids]])
    self.v_adr = torch.cat([idx.free_joint_v_adr, idx.joint_v_adr[ids]])

  def body(self, name: str) -> int:
    """Index of body ``name`` in ``Batch.kin``'s body tensors."""
    if name not in self._body:
      raise KeyError(
        f"Body {name!r} is not in the imagination's FK bodies {self.body_names} "
        "(ImaginationCfg.bodies)."
      )
    return self._body[name]

  def site(self, name: str) -> int:
    """Index of site ``name`` in ``Batch.kin``'s site tensors."""
    if name not in self._site:
      raise KeyError(
        f"Site {name!r} is not in the imagination's FK sites {self.site_names} "
        "(ImaginationCfg.sites)."
      )
    return self._site[name]

  def joint_cols(self, joint_ids: slice | list[int] | tuple[int, ...]) -> list[int]:
    """State columns (into q, q̇) of robot joints ``joint_ids`` (a resolved
    ``SceneEntityCfg.joint_ids``). The joints must be among the state's."""
    ids = (
      list(range(self.num_robot_joints))[joint_ids]
      if isinstance(joint_ids, slice)
      else [int(i) for i in joint_ids]
    )
    col = {j: c for c, j in enumerate(self.joint_ids)}
    missing = [i for i in ids if i not in col]
    if missing:
      raise ValueError(
        f"Joints {missing} are not in the state (the action joints {self.joint_names}): "
        "a term can only read the joints the world model predicts."
      )
    return [col[i] for i in ids]

  def joint_index(self, joint_ids: slice | list[int] | tuple[int, ...]) -> slice | torch.Tensor:
    """``joint_cols`` as an index into the state's joints (a slice when it is all of them,
    in order), cached: terms call it every step."""
    key = ("joints", str(joint_ids) if isinstance(joint_ids, slice) else tuple(joint_ids))
    if key not in self._index:
      cols = self.joint_cols(joint_ids)
      self._index[key] = (
        slice(None)
        if cols == list(range(len(self.joint_ids)))
        else torch.tensor(cols, device=self._device)
      )
    return self._index[key]

  def context_joint_index(
    self, part: str, joint_ids: slice | list[int] | tuple[int, ...]
  ) -> torch.Tensor:
    """Columns of ξ holding per-joint part ``part`` for robot joints ``joint_ids``."""
    key = ("context", part, str(joint_ids) if isinstance(joint_ids, slice) else tuple(joint_ids))
    if key not in self._index:
      if part not in self.context_parts:
        raise KeyError(
          f"The sim context declares no part {part!r} (its parts: "
          f"{sorted(self.context_parts)}); give its term a context_layout."
        )
      p = self.context_parts[part]
      if p.joint_ids is None:
        raise ValueError(f"Context part {part!r} is not per joint.")
      start = p.columns.start or 0
      where = {j: start + k for k, j in enumerate(p.joint_ids)}
      wanted = [self.joint_ids[c] for c in self.joint_cols(joint_ids)]
      missing = [j for j in wanted if j not in where]
      if missing:
        raise ValueError(f"Context part {part!r} has no column for joints {missing}.")
      self._index[key] = torch.tensor([where[j] for j in wanted], device=self._device)
    return self._index[key]


class Batch(NamedTuple):
  """One step's tensors, (B, ...) each: all a single-definition term can read."""

  state: torch.Tensor
  """Heading state ŝ (``state.py``)."""
  pose: torch.Tensor
  """Root world (x, y, yaw), env-local frame."""
  qpos: torch.Tensor
  """(B, 7 + nj) floating base and state joints, env-local frame."""
  qvel: torch.Tensor
  """(B, 6 + nj) their velocities (MuJoCo layout: root angular velocity in body frame)."""
  kin: Kinematics
  """FK of (qpos, qvel) for ``spec.body_names`` and ``spec.site_names``."""
  c: CommandState
  command_model: CommandModel
  action: torch.Tensor
  """The last action (the one that led to this state)."""
  prev_action: torch.Tensor
  """The action before it."""
  context: torch.Tensor
  """Sim context ξ (``ImaginationCfg.context``), (B, 0) if none."""
  spec: BatchSpec
  memo: dict
  """Per-batch cache: what several terms derive from this batch, computed once (e.g. a
  motion's reference bodies). A batch lives one state (sim) or one imagined step."""


class BatchSource:
  """Builds an env's batches: from the simulator (``sim``) and from predicted states
  (``imagined``), with the same FK and layout."""

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    cfg = imagination_cfg(env)
    self.env = env
    self.spec = BatchSpec(env)
    self.command_model = command_model_of(env, cfg.command)
    prefix = f"{self.spec.entity_name}/"  # scene names carry the entity's prefix
    self.fk = TorchKinematics(
      env.sim.mj_model,
      [prefix + b for b in self.spec.body_names],
      [prefix + s for s in self.spec.site_names],
      [prefix + j for j in self.spec.joint_names],
      device=env.device,
    )
    self.fk_fn: Callable = self.fk.forward
    """FK used for both kinds of batches; the heads may replace it by a compiled one."""
    self._context = None
    if cfg.context is not None:  # not in a manager: resolve its entities on a copy
      self._context = copy.deepcopy(cfg.context)
      for value in self._context.params.values():
        if isinstance(value, SceneEntityCfg):
          value.resolve(env.scene)
      layout = getattr(self._context.func, "context_layout", None)
      if layout is not None:  # ξ's named parts, e.g. the per-joint encoder offsets
        self.spec.context_parts = layout(env, **self._context.params)
    self._key: tuple | None = None
    self._batch: Batch | None = None
    self._kin_key: tuple | None = None

  def context(self) -> torch.Tensor:
    """The simulator's ξ."""
    if self._context is None:
      return torch.zeros(self.env.num_envs, 0, device=self.env.device)
    return self._context.func(self.env, **self._context.params)

  @torch.no_grad()
  def sim(self) -> Batch:
    """The batch of the simulator's current state, cached while its inputs (env step,
    qpos, qvel, command state) are unchanged: an env step's terms share one batch. Its FK
    is cached apart, on qpos and qvel alone: the observations after the command update
    (same state, new command state) reuse it."""
    env, spec = self.env, self.spec
    qpos = env.sim.data.qpos[:, spec.q_adr]  # advanced indexing: a copy
    qvel = env.sim.data.qvel[:, spec.v_adr]
    qpos[:, :3] -= env.scene.env_origins
    c = self.command_model.state()
    tensors = (qpos, qvel, *c.values())
    if self._key is not None and self._key[0] == env.common_step_counter:
      cached = self._key[1:]
      if len(cached) == len(tensors) and all(
        a.shape == b.shape and torch.equal(a, b) for a, b in zip(cached, tensors)
      ):
        assert self._batch is not None
        return self._batch
    am = env.action_manager
    self._batch = Batch(
      state=heading_state(qpos, qvel),
      pose=heading_pose(qpos),
      qpos=qpos,
      qvel=qvel,
      kin=self._sim_kinematics(qpos, qvel),
      c=c,
      command_model=self.command_model,
      action=am.action.clone(),
      prev_action=am.prev_action.clone(),
      context=self.context(),
      spec=spec,
      memo={},
    )
    self._key = (env.common_step_counter, *tensors)
    return self._batch

  def _sim_kinematics(self, qpos: torch.Tensor, qvel: torch.Tensor) -> Kinematics:
    """FK of the simulator's state, reused while qpos and qvel are unchanged within an
    env step (e.g. by the observations computed after the command update)."""
    step = self.env.common_step_counter
    if self._kin_key is not None and self._kin_key[0] == step:
      _, old_qpos, old_qvel, kin = self._kin_key
      if torch.equal(qpos, old_qpos) and torch.equal(qvel, old_qvel):
        return kin
    kin = self.fk_fn(qpos, qvel)
    self._kin_key = (step, qpos, qvel, kin)
    return kin

  def imagined(
    self,
    state: torch.Tensor,
    pose: torch.Tensor,
    c: CommandState,
    action: torch.Tensor,
    prev_action: torch.Tensor,
    context: torch.Tensor,
  ) -> Batch:
    """The batch of a predicted state (differentiable)."""
    qpos, qvel = state_to_qpos_qvel(state, pose)
    return Batch(
      state=state, pose=pose, qpos=qpos, qvel=qvel, kin=self.fk_fn(qpos, qvel), c=c,
      command_model=self.command_model, action=action, prev_action=prev_action,
      context=context, spec=self.spec, memo={},
    )  # fmt: skip


_SOURCES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def batch_source(env: ManagerBasedRlEnv) -> BatchSource:
  """The env's ``BatchSource`` (one per env, built on first use)."""
  source = _SOURCES.get(env)
  if source is None:
    source = _SOURCES[env] = BatchSource(env)
  return source


def sim_batch(env: ManagerBasedRlEnv) -> Batch:
  """The batch of the env's current simulator state."""
  return batch_source(env).sim()


Term = Callable[..., torch.Tensor]


def imaginable(fn: Callable[..., torch.Tensor]) -> Term:
  """The mjlab term ``(env, **params)`` of single-definition term ``fn(b, **params)``.

  The returned term computes ``fn`` on the simulator's batch and carries ``fn`` as
  ``.imagined``, for the heads to call on imagined batches.
  """

  def term(env: ManagerBasedRlEnv, **params) -> torch.Tensor:
    return fn(sim_batch(env), **params)

  # Not functools.wraps: its __wrapped__ would make the term's signature look like fn's.
  for attr in ("__module__", "__name__", "__qualname__", "__doc__"):
    setattr(term, attr, getattr(fn, attr))
  term.imagined = fn  # type: ignore[attr-defined]
  return term


def imagined_fn(func: Callable) -> Callable[..., torch.Tensor] | None:
  """The single definition behind env term function ``func``, None if it has none."""
  return getattr(func, "imagined", None)
