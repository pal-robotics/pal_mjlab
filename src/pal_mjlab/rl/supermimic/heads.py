"""Analytic heads of SuperMimic's world model (paper Section 3.2).

The paper's world model is a learned dynamics model plus reward, observation and
termination heads, "each either learned or computed analytically". Here the reward r̂, the
noise-free observation ō and the goal g are computed from a predicted state with forward
kinematics and the training env's own definitions and parameters (read from its config);
the termination d̂ is the learned head of ``WorldModel``. The command's own quantities
(goal, episode end, termination reference, critic features) are the ``CommandModel``'s.

States are heading states ŝ (``state.py``) with world poses p = (x, y, yaw) in the env's
local frame; ξ is the sim context; c is the command state (``CommandModel``). Only the
env terms SuperMimic uses are supported; anything else raises, so
imagination never silently optimizes a different objective than the simulator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import torch
from mjlab.envs.mdp import observations as mj_obs
from mjlab.envs.mdp import rewards as mj_rewards
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.managers.command_manager import CommandTerm

from pal_mjlab.utils.math.kinematics import TorchKinematics

from .command_model import CommandModel, CommandState
from .names import DEFAULT_CRITIC_GROUP, EnvNamesCfg
from .state import joint_state, state_to_qpos_qvel

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from pal_mjlab.tasks.supermimic.mdp.tracking import BodyState


class Kin(NamedTuple):
  """Kinematics of a (predicted) state: tracked bodies and the IMU frame."""

  bodies: BodyState
  imu_quat: torch.Tensor  # world orientation of the IMU site
  imu_ang_vel: torch.Tensor  # world angular velocity of the IMU site


class AnalyticHeads:
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    command_model: CommandModel,
    compile_fk: bool = True,
    names: EnvNamesCfg | None = None,
    critic_group: str = DEFAULT_CRITIC_GROUP,
  ) -> None:
    """``command_model``: the task's command twin (the tracking rewards' reference);
    ``names``: the env's term and group names (``EnvNamesCfg``'s defaults if None);
    ``critic_group``: the critic's observation group. Observation terms are recognized by
    their function, so their names are free."""
    names = names or EnvNamesCfg()
    # TODO(M2): the heads still use the motion-tracking task's terms; the task boundary
    # removes this. Imported here, not at module level: importing pal_mjlab.tasks loads
    # every task, the SuperMimic one included, and that one imports this package.
    from pal_mjlab.tasks.supermimic.mdp import observations as sm_obs
    from pal_mjlab.tasks.supermimic.mdp import rewards as sm_rewards
    from pal_mjlab.tasks.supermimic.mdp import tracking

    self._tracking = tracking
    self.command_model = command_model
    cfg, self.device = env.cfg, env.device
    cmd = cast(CommandTerm, env.command_manager.get_term(names.command))
    robot = env.scene[cmd.cfg.entity_name]

    # Proprioception (Table A.1): q - q0 (with the calibration offset), q̇, IMU angular
    # velocity, projected gravity, previous action; history of h frames per term.
    om = env.observation_manager
    proprio = names.proprio_group
    group = cfg.observations[proprio]
    self.obs_terms: list[str] = list(group.terms)
    self.history = group.history_length or 0
    self.noise: dict[str, tuple[float, float]] = {}
    # Each term's analytic head, by the term's function (the builtin sensor is the IMU's
    # angular velocity, the only one the heads support).
    self._obs_kind: dict[str, str] = {}
    imu_sensors, joint_terms = [], []
    for name, term in group.terms.items():
      func = term.func
      if func is mj_obs.joint_pos_rel:
        assert term.params.get("biased"), "joint positions must include the offset"
        kind = "joint_pos"
        joint_terms.append(name)
      elif func is mj_obs.joint_vel_rel:
        kind = "joint_vel"
        joint_terms.append(name)
      elif func is mj_obs.builtin_sensor:
        kind = "imu_ang_vel"
        imu_sensors.append(term.params["sensor_name"])
      elif func is sm_obs.imu_projected_gravity:
        kind = "imu_projected_gravity"
        imu_sensors.append(term.params["sensor_name"])
      elif func is mj_obs.last_action:
        kind = "actions"
      else:
        raise NotImplementedError(f"observation term {name!r} has no analytic head")
      self._obs_kind[name] = kind
      if group.enable_corruption and term.noise is not None:
        assert isinstance(term.noise, UniformNoiseCfg) and term.noise.operation == "add"
        self.noise[name] = (float(term.noise.n_min), float(term.noise.n_max))
    # The state holds the library's joints, so the env's joint terms must read those.
    self._library_joints = cmd.joint_ids.tolist()
    self._num_joints = len(robot.joint_names)
    for name in joint_terms:
      self._check_joints(om.get_term_cfg(proprio, name).params["asset_cfg"])
    critic = cfg.observations[critic_group].terms
    context = [n for n, t in critic.items() if t.func is sm_obs.sim_context]
    if len(context) != 1:
      raise ValueError(f"The critic group needs one sim_context term, got {context}.")
    self.context_cfg = om.get_term_cfg(critic_group, context[0])
    """The sim context ξ's term: the world model's and the critic's context input."""
    self._check_joints(self.context_cfg.params["bias_asset_cfg"])
    self.q0 = robot.data.default_joint_pos[0, cmd.joint_ids]
    term_dims = dict(
      zip(om.active_terms[proprio], om.group_obs_term_dim[proprio], strict=True)
    )
    self.obs_dims = {
      n: term_dims[n][-1] // max(self.history, 1) for n in self.obs_terms
    }

    # IMU site: both IMU sensors must read the same site.
    mj = env.sim.mj_model
    sites = {mj.site(int(mj.sensor(s).objid[0])).name for s in imu_sensors}
    assert len(sites) == 1, f"IMU sensors on different sites: {sites}"
    self.fk = TorchKinematics(
      mj, cmd.fk.body_names, sorted(sites), cmd.fk.joint_names, device=self.device
    )
    self._fk = torch.compile(self.fk.forward) if compile_fk else self.fk.forward

    # Reward (Table A.2), with the env's weights, stds and dt scaling.
    self.tracking_terms, self.action_rate_weight = [], 0.0
    self.limit_weight = 0.0
    for name, term in cfg.rewards.items():
      if term is None or term.weight == 0.0:
        continue
      if term.func in sm_rewards.TRACKING_ERRORS:
        self.tracking_terms.append(
          (term.weight, term.params["std"], sm_rewards.TRACKING_ERRORS[term.func])
        )
      elif term.func is mj_rewards.action_rate_l2:
        self.action_rate_weight = term.weight
      elif term.func is sm_rewards.joint_pos_limits_hard:
        self.limit_weight = term.weight
        self._check_joints(env.reward_manager.get_term_cfg(name).params["asset_cfg"])
        self.limits = robot.data.joint_pos_limits[0, cmd.joint_ids]
      else:
        raise NotImplementedError(f"reward term {name!r} has no analytic head")
    self.reward_scale = env.step_dt if cfg.scale_rewards_by_dt else 1.0

    # The critic's input layout (``critic_input``), recognized by the terms' functions; the
    # reference-reading terms must read this command.
    self.critic_terms = list(critic)
    layout = [sm_obs.robot_heading_state, sm_obs.motion_goal, sm_obs.sim_context,
              sm_obs.motion_root_pose_error, sm_obs.motion_time_left]  # fmt: skip
    if [t.func for t in critic.values()] != layout:
      raise ValueError(
        f"The critic group {critic_group!r} must hold, in order: heading state, motion "
        f"goal, sim context, root pose error and time left; got {self.critic_terms}."
      )
    for name, term in critic.items():
      if term.params.get("command_name", names.command) != names.command:
        raise ValueError(f"critic term {name!r} reads another command than {names.command!r}")

  def _check_joints(self, asset_cfg) -> None:
    ids = asset_cfg.joint_ids
    ids = list(range(self._num_joints))[ids] if isinstance(ids, slice) else list(ids)
    if ids != self._library_joints:
      raise ValueError("The env's joint terms must read the motion library's joints.")

  # Kinematics -------------------------------------------------------------------

  def kinematics(self, state: torch.Tensor, pose: torch.Tensor) -> Kin:
    qpos, qvel = state_to_qpos_qvel(state, pose)
    k = self._fk(qpos, qvel)
    bodies = self._tracking.BodyState(k.body_pos, k.body_quat, k.body_lin_vel, k.body_ang_vel)
    return Kin(bodies, k.site_quat[:, 0], k.site_ang_vel[:, 0])

  # Observation ō and sensor noise η ------------------------------------------------

  def observation(
    self,
    state: torch.Tensor,
    kin: Kin,
    context: torch.Tensor,
    last_action: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Noise-free proprioceptive frame ō per term, (B, dim) each."""
    q, qd = joint_state(state)
    nj = q.shape[-1]
    bias = context[:, 1 : 1 + nj]  # ξ = (friction, calibration offsets, CoM offset)
    gravity = torch.zeros_like(kin.imu_ang_vel)
    gravity[:, 2] = -1.0
    frame = {
      "joint_pos": q + bias - self.q0,
      "joint_vel": qd,
      "imu_ang_vel": quat_apply_inverse(kin.imu_quat, kin.imu_ang_vel),
      "imu_projected_gravity": quat_apply_inverse(kin.imu_quat, gravity),
      "actions": last_action,
    }
    return {name: frame[self._obs_kind[name]] for name in self.obs_terms}

  def add_noise(
    self, frame: dict[str, torch.Tensor], generator: torch.Generator | None = None
  ) -> dict[str, torch.Tensor]:
    noise = self.sample_noise(len(next(iter(frame.values()))), generator)
    return {n: x + noise[n] if n in noise else x for n, x in frame.items()}

  def sample_noise(
    self, batch: int, generator: torch.Generator | None = None
  ) -> dict[str, torch.Tensor]:
    """Additive sensor noise η per noisy term, (batch, dim) each."""
    out = {}
    for name, (lo, hi) in self.noise.items():
      shape = (batch, self.obs_dims[name])
      u = torch.rand(shape, device=self.device, generator=generator)
      out[name] = lo + (hi - lo) * u
    return out

  def flatten_history(self, history: dict[str, torch.Tensor]) -> torch.Tensor:
    """(B, h, dim) per term -> the env's flat layout: per-term history, oldest first."""
    return torch.cat([history[n].flatten(1) for n in self.obs_terms], dim=-1)

  # Reward r̂ -------------------------------------------------------

  def reward_terms(
    self,
    state: torch.Tensor,
    kin: Kin,
    c: CommandState,
    action: torch.Tensor,
    prev_action: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """The reward split into its bounded tracking part and its two unbounded penalties.

    ``c`` is the command state the step targets. TODO(M2b): the tracking terms read the
    motion command model's reference; twins will take that over.
    """
    ref = self.command_model.reference(c)
    zero = torch.zeros_like(state[:, 0])
    terms = {"tracking": zero, "action_rate": zero, "joint_limit": zero}
    terms["tracking"] = sum(
      w * torch.exp(-err(kin.bodies, ref) / std**2)
      for w, std, err in self.tracking_terms
    )
    if self.action_rate_weight:
      terms["action_rate"] = self.action_rate_weight * (
        action - prev_action
      ).square().sum(-1)
    if self.limit_weight:
      q = joint_state(state)[0]
      violation = self._tracking.joint_limit_violation(
        q, self.limits[:, 0], self.limits[:, 1]
      )
      terms["joint_limit"] = self.limit_weight * violation
    return {k: v * self.reward_scale for k, v in terms.items()}
