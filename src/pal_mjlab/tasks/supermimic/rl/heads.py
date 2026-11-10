"""Analytic heads of SuperMimic's world model (paper Section 3.2).

The paper's world model is a learned dynamics model plus reward, observation and
termination heads, "each either learned or computed analytically". Here the reward r̂, the
noise-free observation ō and the goal g are computed from a predicted state with forward
kinematics and the training env's own definitions and parameters (read from its config);
the termination d̂ is the learned head of ``WorldModel``.

States are heading states ŝ (``state.py``) with world poses p = (x, y, yaw) in the env's
local frame; ξ is the sim context; the reference is a (clip, target frame) pair of the
motion library. Only the env terms SuperMimic uses are supported; anything else raises, so
imagination never silently optimizes a different objective than the simulator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import torch
from mjlab.envs.mdp import observations as mj_obs
from mjlab.envs.mdp import rewards as mj_rewards
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.noise import UniformNoiseCfg

from ..kinematics import TorchKinematics
from ..mdp import observations as sm_obs
from ..mdp import rewards as sm_rewards
from ..mdp import tracking
from ..mdp.commands import MotionLibraryCommand
from ..mdp.tracking import BodyState
from ..state import reference_goal, root_pose_error, state_to_qpos_qvel

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class Kin(NamedTuple):
  """Kinematics of a (predicted) state: tracked bodies and the IMU frame."""

  bodies: BodyState
  imu_quat: torch.Tensor  # world orientation of the IMU site
  imu_ang_vel: torch.Tensor  # world angular velocity of the IMU site


class AnalyticHeads:
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    compile_fk: bool = True,
  ) -> None:
    cfg, self.device = env.cfg, env.device
    cmd = cast(MotionLibraryCommand, env.command_manager.get_term("motion"))
    robot = env.scene[cmd.cfg.entity_name]
    self.library = cmd.library
    self.num_future_frames = cmd.cfg.num_future_frames
    self.time_left_cap = cmd.cfg.time_left_cap_s / env.step_dt

    # Proprioception (Table A.1): q - q0 (with the calibration offset), q̇, IMU angular
    # velocity, projected gravity, previous action; history of h frames per term.
    om = env.observation_manager
    group = cfg.observations["proprio"]
    self.obs_terms: list[str] = list(group.terms)
    self.history = group.history_length or 0
    self.noise: dict[str, tuple[float, float]] = {}
    imu_sensors = []
    for name, term in group.terms.items():
      func = term.func
      if func is mj_obs.joint_pos_rel:
        assert term.params.get("biased"), "joint positions must include the offset"
      elif func is mj_obs.builtin_sensor or func is sm_obs.imu_projected_gravity:
        imu_sensors.append(term.params["sensor_name"])
      elif func not in (mj_obs.joint_vel_rel, mj_obs.last_action):
        raise NotImplementedError(f"observation term {name!r} has no analytic head")
      if group.enable_corruption and term.noise is not None:
        assert isinstance(term.noise, UniformNoiseCfg) and term.noise.operation == "add"
        self.noise[name] = (float(term.noise.n_min), float(term.noise.n_max))
    # The state holds the library's joints, so the env's joint terms must read those.
    self._library_joints = cmd.joint_ids.tolist()
    self._num_joints = len(robot.joint_names)
    self._check_joints(om.get_term_cfg("proprio", "joint_pos").params["asset_cfg"])
    self._check_joints(om.get_term_cfg("proprio", "joint_vel").params["asset_cfg"])
    context = om.get_term_cfg("critic", "sim_context").params
    self._check_joints(context["bias_asset_cfg"])
    self.q0 = robot.data.default_joint_pos[0, cmd.joint_ids]
    term_dims = dict(
      zip(om.active_terms["proprio"], om.group_obs_term_dim["proprio"], strict=True)
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

    # The critic's input layout.
    self.critic_terms = list(cfg.observations["critic"].terms)
    expected = ["heading_state", "motion_goal", "sim_context", "root_pose_error",
                "time_left"]  # fmt: skip
    assert self.critic_terms == expected, self.critic_terms

  def _check_joints(self, asset_cfg) -> None:
    ids = asset_cfg.joint_ids
    ids = list(range(self._num_joints))[ids] if isinstance(ids, slice) else list(ids)
    if ids != self._library_joints:
      raise ValueError("The env's joint terms must read the motion library's joints.")

  # Kinematics -------------------------------------------------------------------

  def kinematics(self, state: torch.Tensor, pose: torch.Tensor) -> Kin:
    qpos, qvel = state_to_qpos_qvel(state, pose)
    k = self._fk(qpos, qvel)
    bodies = BodyState(k.body_pos, k.body_quat, k.body_lin_vel, k.body_ang_vel)
    return Kin(bodies, k.site_quat[:, 0], k.site_ang_vel[:, 0])

  @torch.no_grad()
  def reference(self, frame: torch.Tensor) -> BodyState:
    """Reference bodies at the target frames (local frame, like the logged states)."""
    k = self._fk(self.library.qpos[frame], self.library.qvel[frame])
    return BodyState(k.body_pos, k.body_quat, k.body_lin_vel, k.body_ang_vel)

  def ref_root(self, frame: torch.Tensor) -> torch.Tensor:
    """(B, 2) reference root height and gravity-z: the termination head's goal input."""
    qp = self.library.qpos[frame]
    return torch.stack([qp[:, 2], tracking.projected_gravity_z(qp[:, 3:7])], dim=-1)

  # Observation ō and sensor noise η ------------------------------------------------

  def observation(
    self,
    state: torch.Tensor,
    kin: Kin,
    context: torch.Tensor,
    last_action: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Noise-free proprioceptive frame ō per term, (B, dim) each."""
    nj = (state.shape[-1] - 13) // 2
    q, qd = state[:, 13 : 13 + nj], state[:, 13 + nj :]
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
    return {name: frame[name] for name in self.obs_terms}

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

  # Goal and critic input ----------------------------------------------------------

  def goal(
    self, root_quat: torch.Tensor, clip: torch.Tensor, frame: torch.Tensor
  ) -> torch.Tensor:
    """K future reference frames from the target frame on; past the clip end, the last."""
    k = torch.arange(self.num_future_frames, device=self.device)
    last = self.library.clip_end[clip] - 1
    idx = torch.minimum(frame[:, None] + k, last[:, None])
    qp, qv = self.library.qpos[idx], self.library.qvel[idx]
    return reference_goal(qp, qv, root_quat)

  def critic_input(
    self,
    state: torch.Tensor,
    pose: torch.Tensor,
    goal: torch.Tensor,
    context: torch.Tensor,
    clip: torch.Tensor,
    frame: torch.Tensor,
  ) -> torch.Tensor:
    """(ŝ, g, ξ) plus the privileged root-pose error and time left in the clip."""
    qpos, _ = state_to_qpos_qvel(state, pose)
    ref = self.library.qpos[frame]
    err = root_pose_error(qpos[:, :3], qpos[:, 3:7], ref[:, :3], ref[:, 3:7])
    left = (self.library.clip_end[clip] - 1 - frame).float()
    time_left = (left.clamp(max=self.time_left_cap) / self.time_left_cap)[:, None]
    return torch.cat([state, goal, context, err, time_left], dim=-1)

  # Reward r̂ and termination -------------------------------------------------------

  def reward_terms(
    self,
    state: torch.Tensor,
    kin: Kin,
    ref: BodyState,
    action: torch.Tensor,
    prev_action: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """The reward split into its bounded tracking part and its two unbounded penalties."""
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
      nj = (state.shape[-1] - 13) // 2
      q = state[:, 13 : 13 + nj]
      violation = tracking.joint_limit_violation(
        q, self.limits[:, 0], self.limits[:, 1]
      )
      terms["joint_limit"] = self.limit_weight * violation
    return {k: v * self.reward_scale for k, v in terms.items()}

  def clip_end(self, clip: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """The step that targets the clip's last frame ends the episode."""
    return frame >= self.library.clip_end[clip] - 1

  def next_frame(self, clip: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    return torch.minimum(frame + 1, self.library.clip_end[clip] - 1)
