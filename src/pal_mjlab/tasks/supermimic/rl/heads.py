"""Analytic heads of SuperMimic's world model (paper Section 3.2, D5).

The paper's world model is a learned dynamics model plus reward, observation and
termination heads, "each either learned or computed analytically". Here the reward r̂, the
noise-free observation ō and the goal g are computed from a predicted state with forward
kinematics and the training env's own definitions and parameters (read from its config);
the termination d̂ is the learned head of ``WorldModel`` (D31), and this module gives the
env's failure rule next to it for comparison.

States are heading states ŝ (D4) with world poses p = (x, y, yaw) in the env's local frame;
ξ is the sim context; the reference is a (clip, target frame) pair of the motion library.
Only the env terms SuperMimic uses are supported; anything else raises, so imagination
never silently optimizes a different objective than the simulator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import torch
from mjlab.envs.mdp import observations as mj_obs
from mjlab.envs.mdp import rewards as mj_rewards
from mjlab.utils.noise import UniformNoiseCfg

from ..kinematics import (
  TorchKinematics,
  compile_kinematic_model,
  quat_rotate_inv,
)
from ..mdp import observations as sm_obs
from ..mdp import rewards as sm_rewards
from ..mdp import tracking
from ..mdp.commands import MotionLibraryCommand
from ..mdp.tracking import BodyState
from ..state import reference_goal, root_pose_error, state_to_qpos_qvel

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_TRACKING = {
  sm_rewards.motion_root_pos_exp: tracking.anchor_pos_error,
  sm_rewards.motion_root_ori_exp: tracking.anchor_ori_error,
  sm_rewards.motion_body_pos_exp: tracking.body_pos_error,
  sm_rewards.motion_body_ori_exp: tracking.body_ori_error,
  sm_rewards.motion_body_lin_vel_exp: tracking.body_lin_vel_error,
  sm_rewards.motion_body_ang_vel_exp: tracking.body_ang_vel_error,
}


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
    privileged_critic: bool = True,
  ) -> None:
    cfg, self.device = env.cfg, env.device
    cmd = cast(MotionLibraryCommand, env.command_manager.get_term("motion"))
    robot = env.scene[cmd.cfg.entity_name]
    self.library = cmd.library
    self.num_future_frames = cmd.cfg.num_future_frames
    self.goal_joints = cmd._goal_joint_ids
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
    for name in ("joint_pos", "joint_vel"):
      ids = om.get_term_cfg("proprio", name).params["asset_cfg"].joint_ids
      assert torch.equal(torch.as_tensor(ids, device=self.device), self.goal_joints)
    self.q0 = robot.data.default_joint_pos[0, self.goal_joints]
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
    model = compile_kinematic_model(robot.spec.copy())
    bodies = [f"{cmd.cfg.entity_name}/{b}" for b in cmd.cfg.body_names]
    self.fk = TorchKinematics(model, bodies, sorted(sites), device=self.device)
    self._fk = torch.compile(self.fk.forward) if compile_fk else self.fk.forward
    self.privileged_critic = privileged_critic

    # Reward (Table A.2), with the env's weights, stds and dt scaling.
    self.tracking_terms, self.action_rate_weight = [], 0.0
    self.limit_weight, self.limit_joints = 0.0, None
    for name, term in cfg.rewards.items():
      if term is None or term.weight == 0.0:
        continue
      if term.func in _TRACKING:
        self.tracking_terms.append(
          (term.weight, term.params["std"], _TRACKING[term.func])
        )
      elif term.func is mj_rewards.action_rate_l2:
        self.action_rate_weight = term.weight
      elif term.func is sm_rewards.joint_pos_limits_hard:
        self.limit_weight = term.weight
        ids = env.reward_manager.get_term_cfg(name).params["asset_cfg"].joint_ids
        self.limit_joints = torch.as_tensor(ids, device=self.device)
        self.limits = robot.data.joint_pos_limits[0, self.limit_joints]
      else:
        raise NotImplementedError(f"reward term {name!r} has no analytic head")
    self.reward_scale = env.step_dt if cfg.scale_rewards_by_dt else 1.0

    # Failure rule (paper Appendix A) and the critic's input layout (D7, D22).
    params = cfg.terminations["anchor_failure"].params
    self.z_threshold, self.tilt_threshold = (
      params["z_threshold"],
      params["tilt_threshold"],
    )
    self.critic_terms = list(cfg.observations["critic"].terms)
    expected = ["heading_state", "motion_goal", "sim_context", "root_pose_error",
                "time_left"]  # fmt: skip
    assert self.critic_terms == expected, self.critic_terms

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
    ids = self.goal_joints
    bias = context[:, 1 : 1 + len(ids)]
    gravity = torch.zeros_like(kin.imu_ang_vel)
    gravity[:, 2] = -1.0
    frame = {
      "joint_pos": q[:, ids] + bias - self.q0,
      "joint_vel": qd[:, ids],
      "imu_ang_vel": quat_rotate_inv(kin.imu_quat, kin.imu_ang_vel),
      "imu_projected_gravity": quat_rotate_inv(kin.imu_quat, gravity),
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

  def unflatten_history(self, proprio: torch.Tensor) -> dict[str, torch.Tensor]:
    """Inverse of ``flatten_history``."""
    sizes = [self.history * self.obs_dims[n] for n in self.obs_terms]
    return {
      n: x.unflatten(1, (self.history, self.obs_dims[n]))
      for n, x in zip(self.obs_terms, proprio.split(sizes, dim=1), strict=True)
    }

  # Goal and critic input ----------------------------------------------------------

  def goal(
    self, root_quat: torch.Tensor, clip: torch.Tensor, frame: torch.Tensor
  ) -> torch.Tensor:
    """K future reference frames from the target frame on (D6); past the end, the last."""
    k = torch.arange(self.num_future_frames, device=self.device)
    last = self.library.clip_end[clip] - 1
    idx = torch.minimum(frame[:, None] + k, last[:, None])
    qp, qv = self.library.qpos[idx], self.library.qvel[idx]
    return reference_goal(qp, qv, root_quat, self.goal_joints)

  def critic_input(
    self,
    state: torch.Tensor,
    pose: torch.Tensor,
    goal: torch.Tensor,
    context: torch.Tensor,
    clip: torch.Tensor,
    frame: torch.Tensor,
  ) -> torch.Tensor:
    """(ŝ, g, ξ) plus the privileged root-pose error and time left in the clip (D7);
    with ``privileged_critic=False`` those two are zero (stability ablation A6)."""
    qpos, _ = state_to_qpos_qvel(state, pose)
    ref = self.library.qpos[frame]
    err = root_pose_error(qpos[:, :3], qpos[:, 3:7], ref[:, :3], ref[:, 3:7])
    left = (self.library.clip_end[clip] - 1 - frame).float()
    time_left = (left.clamp(max=self.time_left_cap) / self.time_left_cap)[:, None]
    privileged = torch.cat([err, time_left], dim=-1) * float(self.privileged_critic)
    return torch.cat([state, goal, context, privileged], dim=-1)

  # Reward r̂ and termination -------------------------------------------------------

  def reward(
    self,
    state: torch.Tensor,
    kin: Kin,
    ref: BodyState,
    action: torch.Tensor,
    prev_action: torch.Tensor,
  ) -> torch.Tensor:
    """The env's per-step reward (all terms, dt-scaled) for the state after ``action``."""
    return sum(self.reward_terms(state, kin, ref, action, prev_action).values())

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
      q = state[:, 13 : 13 + nj][:, self.limit_joints]
      violation = tracking.joint_limit_violation(
        q, self.limits[:, 0], self.limits[:, 1]
      )
      terms["joint_limit"] = self.limit_weight * violation
    return {k: v * self.reward_scale for k, v in terms.items()}

  def failure_margin(self, kin: Kin, ref: BodyState) -> torch.Tensor:
    """The env's failure rule as a margin: > 1 means the episode fails."""
    robot = kin.bodies
    z_err = (ref.pos[:, 0, 2] - robot.pos[:, 0, 2]).abs()
    tilt = (
      tracking.projected_gravity_z(ref.quat[:, 0])
      - tracking.projected_gravity_z(robot.quat[:, 0])
    ).abs()
    return torch.maximum(z_err / self.z_threshold, tilt / self.tilt_threshold)

  def clip_end(self, clip: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """The step that targets the clip's last frame ends the episode (D8)."""
    return frame >= self.library.clip_end[clip] - 1

  def next_frame(self, clip: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    return torch.minimum(frame + 1, self.library.clip_end[clip] - 1)
