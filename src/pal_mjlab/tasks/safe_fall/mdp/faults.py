"""Actuator faults for the safe-fall task: actuators that go limp.

A failed actuator can no longer be commanded but still reports its state. It
has no brake, so its joint moves freely against the joint's own friction,
damping and reflected rotor inertia. In the model this is the position
actuator with its stiffness and damping set to zero for that environment.
The drive reports the error, so the policy is told which actuators are down.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import mujoco
import torch
from mjlab.entity import Entity
from mjlab.managers.event_manager import EventTermCfg, requires_model_fields

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
  from mjlab.viewer.debug_visualizer import DebugVisualizer

_FAULTS_ATTR = "_safe_fall_actuator_faults"
# Onset step of an actuator that does not fail in this episode.
_NEVER = 1_000_000
# Viewer marks of the actuators that are down: color (RGBA) and sizes [m].
_MARK_COLOR = (1.0, 0.05, 0.05, 0.9)
_MARK_BALL_RADIUS = 0.05
_MARK_BAR_RADIUS = 0.018
_MARK_BAR_HALF_LENGTH = 0.11


@dataclass
class ActuatorFaultCfg:
  """Which actuators fail in an episode, and when."""

  count_probabilities: tuple[float, ...]
  """Probability of 0, 1, 2, ... failed actuators in an episode."""
  joint_weights: dict[str, float]
  """Relative chance of failing per joint, as joint name regex -> weight; the
  first regex that matches a joint is used. A weight of 0 never fails."""
  later_onset_range_s: tuple[float, float] = (0.0, 1.0)
  """The first failure is there from the first step: it is why the fallback
  took over. Each further one starts at a time [s] drawn from this range."""
  action_name: str = "joint_pos"
  entity_name: str = "robot"


class ActuatorFaults:
  """Per-environment fault state of the actuators the policy commands.

  Columns follow the action order. ``onset_step`` [B, A] is the control step
  of the episode from which an actuator is down and ``failed`` [B, A] is
  whether it is down now. ``first_failed`` [B] is the column of the actuator
  that is down from the first step (-1 if none): the one that made the
  fallback take over.
  """

  def __init__(self, env: ManagerBasedRlEnv, cfg: ActuatorFaultCfg):
    self.cfg = cfg
    self._env = env
    device = env.device
    entity: Entity = env.scene[cfg.entity_name]
    self.joint_names: list[str] = list(
      env.action_manager.get_term(cfg.action_name).target_names
    )
    num = len(self.joint_names)

    ctrl_id_of: dict[str, int] = {}
    for actuator in entity.actuators:
      ids = actuator.global_ctrl_ids.tolist()
      ctrl_id_of.update(zip(actuator.target_names, ids, strict=True))
    self._ctrl_ids = torch.tensor(
      [ctrl_id_of[name] for name in self.joint_names], device=device
    )
    # Position actuator: force = kp * (ctrl - q) - kd * qvel, stored as
    # gainprm[0] = kp, biasprm[1] = -kp, biasprm[2] = -kd.
    gainprm = env.sim.get_default_field("actuator_gainprm")
    biasprm = env.sim.get_default_field("actuator_biasprm")
    self._gain = gainprm[self._ctrl_ids, 0].clone()
    self._bias_pos = biasprm[self._ctrl_ids, 1].clone()
    self._bias_vel = biasprm[self._ctrl_ids, 2].clone()

    weights = []
    for name in self.joint_names:
      matches = [
        w for regex, w in cfg.joint_weights.items() if re.fullmatch(regex, name)
      ]
      if not matches:
        raise ValueError(f"No fault weight for joint '{name}'.")
      weights.append(float(matches[0]))
    self.weights = torch.tensor(weights, device=device)
    self._count_probabilities = torch.tensor(cfg.count_probabilities, device=device)
    # No more failures than there are actuators that can fail.
    self.max_count = min(
      len(cfg.count_probabilities) - 1, int((self.weights > 0).sum())
    )

    self.onset_step = torch.full(
      (env.num_envs, num), _NEVER, device=device, dtype=torch.long
    )
    self.failed = torch.zeros(env.num_envs, num, device=device, dtype=torch.bool)
    self.first_failed = torch.full((env.num_envs,), -1, device=device, dtype=torch.long)

  def planned(self) -> torch.Tensor:
    """[B, A] whether each actuator fails at some point of the episode."""
    return self.onset_step < _NEVER

  def sample(self, env_ids: torch.Tensor) -> None:
    """Draw the failures of a new episode for ``env_ids`` and apply them."""
    n = len(env_ids)
    device = self._env.device
    onset = torch.full(
      (n, len(self.joint_names)), _NEVER, device=device, dtype=torch.long
    )
    if self.max_count > 0:
      count = torch.multinomial(self._count_probabilities, n, replacement=True)
      count = count.clamp(max=self.max_count)
      order = torch.multinomial(
        self.weights.expand(n, -1), self.max_count, replacement=False
      )
      lo, hi = self.cfg.later_onset_range_s
      step_dt = self._env.step_dt
      later = torch.rand(n, self.max_count, device=device) * (hi - lo) + lo
      later_step = torch.round(later / step_dt).long()
      later_step[:, 0] = 0
      rows = torch.arange(n, device=device)
      for k in range(self.max_count):
        hit = count > k
        onset[rows[hit], order[hit, k]] = later_step[hit, k]
    self.set_onsets(env_ids, onset)

  def set_onsets(self, env_ids: torch.Tensor, onset_step: torch.Tensor) -> None:
    """Set the onset steps [len(env_ids), A] of a new episode and apply them."""
    self.onset_step[env_ids] = onset_step
    self.failed[env_ids] = onset_step <= 0
    first_onset, first = onset_step.min(dim=-1)
    self.first_failed[env_ids] = torch.where(
      first_onset <= 0, first, torch.full_like(first, -1)
    )
    self._write(env_ids)

  def update(self) -> None:
    """Take down the actuators whose onset step has come."""
    step = self._env.episode_length_buf.unsqueeze(-1)
    due = self.onset_step <= step
    changed = (due != self.failed).any(dim=-1)
    if not bool(changed.any()):
      return
    self.failed = due
    self._write(changed.nonzero().flatten())

  def _write(self, env_ids: torch.Tensor) -> None:
    model = self._env.sim.model
    healthy = (~self.failed[env_ids]).float()
    rows = env_ids.unsqueeze(-1)
    model.actuator_gainprm[rows, self._ctrl_ids, 0] = self._gain * healthy
    model.actuator_biasprm[rows, self._ctrl_ids, 1] = self._bias_pos * healthy
    model.actuator_biasprm[rows, self._ctrl_ids, 2] = self._bias_vel * healthy


def get_actuator_faults(
  env: ManagerBasedRlEnv, cfg: ActuatorFaultCfg
) -> ActuatorFaults:
  """Return the (lazily created) fault state of the environment."""
  faults: ActuatorFaults | None = getattr(env, _FAULTS_ATTR, None)
  if faults is None:
    faults = ActuatorFaults(env, cfg)
    setattr(env, _FAULTS_ATTR, faults)
  return faults


def replace_actuator_fault_cfg(env_cfg: ManagerBasedRlEnvCfg, **changes) -> None:
  """Change the fault configuration in every term of ``env_cfg`` that uses it.

  For evaluations and tests, e.g. ``count_probabilities=(1.0,)`` for no
  failures. All terms must share one configuration, as they share the state.
  """
  term_cfgs = [*env_cfg.events.values(), *env_cfg.metrics.values()]
  for group in env_cfg.observations.values():
    term_cfgs += [t for t in group.terms.values() if t is not None]
  users = [
    t for t in term_cfgs if isinstance(t.params.get("fault_cfg"), ActuatorFaultCfg)
  ]
  if not users:
    raise ValueError("The environment has no actuator fault terms.")
  new_cfg = replace(users[0].params["fault_cfg"], **changes)
  for term_cfg in users:
    term_cfg.params["fault_cfg"] = new_cfg


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def sample_actuator_faults(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, fault_cfg: ActuatorFaultCfg
) -> None:
  """Reset event: draw which actuators fail in the new episode, and when."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  get_actuator_faults(env, fault_cfg).sample(env_ids)


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
class apply_actuator_faults:
  """Step event: take down the actuators that fail later in the episode.

  In a viewer, every actuator that is down is marked in red on the robot: a
  hinge by a ball on the joint with a bar along its axis (the hip and the
  ankle have two joints in one place, the bar tells them apart), a leg-length
  actuator by a bar along the whole leg.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self._fault_cfg: ActuatorFaultCfg = cfg.params["fault_cfg"]
    # The fault state needs the action term, which is created after the
    # events, so the joints to mark are looked up on first use.
    self._joint_ids: torch.Tensor | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    fault_cfg: ActuatorFaultCfg,
  ) -> None:
    del env_ids  # Every environment is checked.
    get_actuator_faults(env, fault_cfg).update()

  def _resolve_joints(self, faults: ActuatorFaults) -> None:
    env = self._env
    entity: Entity = env.scene[faults.cfg.entity_name]
    local_ids = entity.find_joints(faults.joint_names, preserve_order=True)[0]
    self._entity = entity
    self._local_joint_ids = torch.tensor(local_ids, device=env.device)
    self._joint_ids = entity.indexing.joint_ids[self._local_joint_ids]
    joint_types = env.sim.mj_model.jnt_type[self._joint_ids.cpu().numpy()]
    self._is_slide = [int(t) == int(mujoco.mjtJoint.mjJNT_SLIDE) for t in joint_types]

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids  # The reset event draws the new episode's failures.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    faults = get_actuator_faults(self._env, self._fault_cfg)
    failed = faults.failed
    if not bool(failed.any()):
      return
    if self._joint_ids is None:
      self._resolve_joints(faults)
    assert self._joint_ids is not None
    data = self._env.sim.data
    for env_idx in visualizer.get_env_indices(self._env.num_envs):
      for column in failed[env_idx].nonzero().flatten().tolist():
        joint_id = self._joint_ids[column]
        anchor = data.xanchor[env_idx, joint_id].cpu().numpy()
        axis = data.xaxis[env_idx, joint_id].cpu().numpy()
        if self._is_slide[column]:
          # The slide axis runs down the leg; the joint position is its length.
          length = float(
            self._entity.data.joint_pos[env_idx, self._local_joint_ids[column]]
          )
          visualizer.add_cylinder(
            anchor, anchor + length * axis, _MARK_BAR_RADIUS * 2.0, _MARK_COLOR
          )
        else:
          visualizer.add_sphere(anchor, _MARK_BALL_RADIUS, _MARK_COLOR)
          half = _MARK_BAR_HALF_LENGTH * axis
          visualizer.add_cylinder(
            anchor - half, anchor + half, _MARK_BAR_RADIUS, _MARK_COLOR
          )


def actuator_fault_flags(
  env: ManagerBasedRlEnv, fault_cfg: ActuatorFaultCfg
) -> torch.Tensor:
  """[B, A] 1 for each actuator that is down, in action order."""
  return get_actuator_faults(env, fault_cfg).failed.float()


def num_failed_actuators(
  env: ManagerBasedRlEnv, fault_cfg: ActuatorFaultCfg
) -> torch.Tensor:
  """[B] number of actuators that are down."""
  return get_actuator_faults(env, fault_cfg).failed.sum(dim=-1).float()
