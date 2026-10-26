"""Motion command backed by a packed motion library (thousands of clips).

Per env it holds a clip and a *target* frame: the reference frame the next action should
reach. Rewards and terminations compare the post-step robot with the target; observations
show the next K frames starting at the target.

Episodes start at frames drawn by the paper's return-based sampler and end at the clip's
last frame (``at_clip_end``): there is no teleport to a new clip mid-episode, so every
transition is physical (D8).

Robot and reference kinematics come from ``TorchKinematics`` on (qpos, qvel), cached per
env step, so all terms read the same tensors.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform

from ..kinematics import (
  TorchKinematics,
  compile_kinematic_model,
  quat_conj,
  quat_mul,
  quat_rotate,
  rot6d,
)
from ..library import MotionLibrary, resolve_library
from ..state import heading_state, root_pose_error
from .sampler import ReturnSampler
from .tracking import BodyState

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer

__all__ = ["MotionLibraryCommand", "MotionLibraryCommandCfg"]

_POSE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


class MotionLibraryCommand(CommandTerm):
  cfg: MotionLibraryCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionLibraryCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    path = resolve_library(cfg.library_path, cfg.dataset, cfg.subset)
    self.library = MotionLibrary(path, device=self.device, clips=cfg.clips)
    if self.library.joint_names != tuple(self.robot.joint_names):
      raise ValueError("Motion library joint order differs from the robot's joints.")
    if abs(self.library.fps * env.step_dt - 1.0) > 1e-6:
      raise ValueError(
        f"Library is {self.library.fps} Hz, env steps at {1 / env.step_dt} Hz."
      )

    # FK on the robot exactly as simulated: its spec as attached to the scene, where the
    # scene prefixes entity names with "<entity_name>/".
    model = compile_kinematic_model(self.robot.spec.copy())
    prefixed = [f"{cfg.entity_name}/{name}" for name in cfg.body_names]
    self._fk = TorchKinematics(model, prefixed, device=self.device)
    self._fk_fn = (
      torch.compile(self._fk.forward) if cfg.compile_kinematics else self._fk.forward
    )
    idx = self.robot.indexing
    self._q_adr = torch.cat([idx.free_joint_q_adr, idx.joint_q_adr]).long()
    self._v_adr = torch.cat([idx.free_joint_v_adr, idx.joint_v_adr]).long()
    goal_ids, self.goal_joint_names = self.robot.find_joints(
      cfg.goal_joint_names, preserve_order=True
    )
    self._goal_joint_ids = torch.tensor(goal_ids, device=self.device)

    lib = self.library
    frames_per_bin = max(1, round(cfg.bin_seconds / env.step_dt))
    self.sampler = ReturnSampler(
      lib.clip_start,
      lib.clip_length,
      frames_per_bin,
      temperature=cfg.sampler_temperature,
      mix=cfg.sampler_mix,
      alpha=cfg.sampler_alpha,
      past_rate=cfg.sampler_past_rate,
    )

    n = self.num_envs
    self.clip = torch.zeros(n, dtype=torch.long, device=self.device)
    self.frame = lib.clip_start[self.clip].clone()  # target frame (global index)
    self.start_frame = self.frame.clone()
    self.start_bin = torch.zeros(n, dtype=torch.long, device=self.device)
    self._forced_clip = torch.full((n,), -1, dtype=torch.long, device=self.device)
    self._ep_return = torch.zeros(n, device=self.device)
    self._ep_steps = torch.zeros(n, dtype=torch.long, device=self.device)
    self._fresh = torch.zeros(n, dtype=torch.bool, device=self.device)
    self._max_step_reward: float | None = None
    self._version = 0
    self._cache_key: tuple[int, int] | None = None
    self._cache: dict[str, object] = {}

    for name in (
      "error_anchor_pos",
      "error_joint_pos",
      "sampler_mean_return",
      "sampler_coverage",
    ):
      self.metrics[name] = torch.zeros(n, device=self.device)
    self._ghost_model = None

  # Commanded goal ---------------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    return self.goal

  @property
  def goal(self) -> torch.Tensor:
    """(E, K * F) the next K reference frames from the target frame on (D6).

    Each frame is BeyondMimic's command without state estimation: reference joint
    positions and velocities, and the reference root orientation in the robot root frame
    (6D). Frames past the clip end repeat the last frame.
    """
    return self._cached("goal", self._compute_goal)

  def _compute_goal(self) -> torch.Tensor:
    k = torch.arange(self.cfg.num_future_frames, device=self.device)
    last = self.library.clip_end[self.clip] - 1
    idx = torch.minimum(self.frame[:, None] + k, last[:, None])
    qp, qv = self.library.qpos[idx], self.library.qvel[idx]  # (E, K, ·)
    ref_quat = qp[..., 3:7]
    robot_quat = self.robot_qpos[:, None, 3:7].expand_as(ref_quat)
    feats = (
      qp[..., 7:][..., self._goal_joint_ids],
      qv[..., 6:][..., self._goal_joint_ids],
      rot6d(quat_mul(quat_conj(robot_quat), ref_quat)),
    )
    return torch.cat(feats, dim=-1).flatten(1)

  # Robot and reference state ----------------------------------------------------

  @property
  def robot_qpos(self) -> torch.Tensor:
    return self._env.sim.data.qpos[:, self._q_adr]

  @property
  def robot_qvel(self) -> torch.Tensor:
    return self._env.sim.data.qvel[:, self._v_adr]

  @property
  def ref_qpos(self) -> torch.Tensor:
    """Reference qpos at the target frame, root position in world (env origin added)."""
    qp = self.library.qpos[self.frame].clone()
    qp[:, :3] += self._env.scene.env_origins
    return qp

  @property
  def ref_qvel(self) -> torch.Tensor:
    return self.library.qvel[self.frame]

  @property
  def robot_bodies(self) -> BodyState:
    return self._cached(
      "robot_bodies", lambda: self._bodies(self.robot_qpos, self.robot_qvel)
    )

  @property
  def ref_bodies(self) -> BodyState:
    return self._cached(
      "ref_bodies", lambda: self._bodies(self.ref_qpos, self.ref_qvel)
    )

  @property
  def heading_state(self) -> torch.Tensor:
    return heading_state(self.robot_qpos, self.robot_qvel)

  @property
  def root_pose_error(self) -> torch.Tensor:
    rq, fq = self.robot_qpos, self.ref_qpos
    return root_pose_error(rq[:, :3], rq[:, 3:7], fq[:, :3], fq[:, 3:7])

  @property
  def clip_time_left(self) -> torch.Tensor:
    """(E, 1) steps left in the clip, in [0, 1] over ``time_left_cap_s`` (critic input)."""
    cap = self.cfg.time_left_cap_s / self._env.step_dt
    left = (self.library.clip_end[self.clip] - 1 - self.frame).float()
    return (left.clamp(max=cap) / cap)[:, None]

  @property
  def at_clip_end(self) -> torch.Tensor:
    return self.frame >= self.library.clip_end[self.clip] - 1

  def _bodies(self, qpos: torch.Tensor, qvel: torch.Tensor) -> BodyState:
    kin = self._fk_fn(qpos, qvel)
    return BodyState(kin.body_pos, kin.body_quat, kin.body_lin_vel, kin.body_ang_vel)

  def _cached(self, name: str, fn):
    key = (self._env.common_step_counter, self._version)
    if key != self._cache_key:
      self._cache_key, self._cache = key, {}
    if name not in self._cache:
      self._cache[name] = fn()
    return self._cache[name]

  # Evaluation -------------------------------------------------------------------

  def set_eval_clips(self, clip_ids: torch.Tensor) -> None:
    """Make env i restart clip ``clip_ids[i]`` from its first frame at its next reset."""
    self._forced_clip.fill_(-1)
    self._forced_clip[: len(clip_ids)] = clip_ids.to(self.device)

  # CommandTerm hooks ------------------------------------------------------------

  def _update_metrics(self) -> None:
    rq, fq = self.robot_qpos, self.ref_qpos
    self.metrics["error_anchor_pos"] = (rq[:, :3] - fq[:, :3]).norm(dim=-1)
    ids = self._goal_joint_ids
    self.metrics["error_joint_pos"] = (rq[:, 7:][:, ids] - fq[:, 7:][:, ids]).norm(
      dim=-1
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    self._finish_episodes(env_ids)
    n, lib, s = len(env_ids), self.library, self.sampler
    if self.cfg.sampling_mode == "return":
      bins, frames = s.sample(n)
      clips = s.bin_clip[bins]
    elif self.cfg.sampling_mode == "uniform":
      bins = torch.randint(0, s.num_bins, (n,), device=self.device)
      span = (s.bin_stop[bins] - s.bin_start[bins]).clamp_min(1)
      frames = s.bin_start[bins] + (torch.rand(n, device=self.device) * span).long()
      clips = s.bin_clip[bins]
    else:  # "start"
      clips = torch.randint(0, lib.num_clips, (n,), device=self.device)
      frames = lib.clip_start[clips]
      bins = s.clip_bin_offset[clips]
    forced = self._forced_clip[env_ids]
    is_forced = forced >= 0
    clips = torch.where(is_forced, forced, clips)
    frames = torch.where(is_forced, lib.clip_start[forced.clamp_min(0)], frames)
    bins = torch.where(is_forced, s.clip_bin_offset[forced.clamp_min(0)], bins)

    self.clip[env_ids] = clips
    self.frame[env_ids] = frames
    self.start_frame[env_ids] = frames
    self.start_bin[env_ids] = bins
    self._ep_return[env_ids] = 0.0
    self._ep_steps[env_ids] = 0
    self._fresh[env_ids] = True
    self._write_reference_state(env_ids)
    self._version += 1

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    end = self.library.clip_end[self.clip] - 1
    if env_ids is None:
      # Per-step path: accumulate this step's reward for envs that did not just reset.
      live = ~self._fresh
      gamma = self.cfg.return_discount
      reward = self._env.reward_buf
      self._ep_return += torch.where(live, gamma**self._ep_steps * reward, 0.0)
      self._ep_steps += live.long()
      self._fresh.zero_()
      self.frame = torch.minimum(self.frame + 1, end)
    else:
      self._fresh[env_ids] = False
      self.frame[env_ids] = torch.minimum(self.frame[env_ids] + 1, end[env_ids])
    self._version += 1

  # Internals --------------------------------------------------------------------

  def _finish_episodes(self, env_ids: torch.Tensor) -> None:
    """Fold the returns of episodes ending now into the sampler (normalized per start)."""
    done = env_ids[self._ep_steps[env_ids] > 0]
    if len(done) == 0:
      return
    gamma = self.cfg.return_discount
    # The final reward was computed this step, before the reset; add it here.
    ret = (
      self._ep_return[done] + gamma ** self._ep_steps[done] * self._env.reward_buf[done]
    )
    attainable = torch.minimum(
      self.library.clip_end[self.clip[done]] - 1 - self.start_frame[done],
      torch.full_like(done, self._env.max_episode_length),
    ).float()
    best = self._step_reward_max() * (1.0 - gamma**attainable) / (1.0 - gamma)
    normalized = (ret / best).clamp(0.0, 1.0)
    self.sampler.update(self.start_bin[done], normalized)
    self.metrics["sampler_mean_return"][:] = self.sampler.returns.mean()
    self.metrics["sampler_coverage"][:] = (self.sampler.visits > 0).float().mean()

  def _step_reward_max(self) -> float:
    """Largest per-step reward: sum of positive weights (all positive terms are <= 1)."""
    if self._max_step_reward is None:
      rm = self._env.reward_manager
      weights = [rm.get_term_cfg(name).weight for name in rm.active_terms]
      scale = self._env.step_dt if self._env.cfg.scale_rewards_by_dt else 1.0
      self._max_step_reward = sum(max(w, 0.0) for w in weights) * scale
    return self._max_step_reward

  def _write_reference_state(self, env_ids: torch.Tensor) -> None:
    """Reference state at the start frame, with BeyondMimic's reset noise (RSI)."""
    n, cfg = len(env_ids), self.cfg
    qp, qv = (
      self.library.qpos[self.frame[env_ids]],
      self.library.qvel[self.frame[env_ids]],
    )
    pose = self._uniform(cfg.pose_range, n)
    vel = self._uniform(cfg.velocity_range, n)
    root_pos = qp[:, :3] + self._env.scene.env_origins[env_ids] + pose[:, :3]
    root_quat = quat_mul(
      quat_from_euler_xyz(pose[:, 3], pose[:, 4], pose[:, 5]), qp[:, 3:7]
    )
    lin_vel = qv[:, :3] + vel[:, :3]
    ang_vel = quat_rotate(qp[:, 3:7], qv[:, 3:6]) + vel[:, 3:]  # world frame

    joint_pos = qp[:, 7:].clone()
    ids = self._goal_joint_ids
    joint_pos[:, ids] += sample_uniform(
      *cfg.joint_position_range, (n, len(ids)), self.device
    )
    limits = self.robot.data.joint_pos_limits[env_ids]
    joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])

    self.robot.write_joint_state_to_sim(joint_pos, qv[:, 6:], env_ids=env_ids)
    root = torch.cat([root_pos, root_quat, lin_vel, ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _uniform(self, ranges: dict[str, tuple[float, float]], n: int) -> torch.Tensor:
    r = torch.tensor(
      [ranges.get(k, (0.0, 0.0)) for k in _POSE_KEYS], device=self.device
    )
    return sample_uniform(r[:, 0], r[:, 1], (n, 6), device=self.device)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Translucent ghost of the reference at the target frame."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return
    if self._ghost_model is None:
      self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
      m = self._ghost_model
      for g in range(m.ngeom):
        if m.geom_contype[g] != 0 or m.geom_conaffinity[g] != 0:
          m.geom_rgba[g, 3] = 0
        else:
          m.geom_rgba[g] = np.array(self.cfg.viz.ghost_color, dtype=np.float32)
    q_adr = self._q_adr.cpu().numpy()
    ref = self.ref_qpos.cpu().numpy()
    for env in env_indices:
      qpos = np.zeros(self._env.sim.mj_model.nq)
      qpos[q_adr] = ref[env]
      visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{env}")


@dataclass(kw_only=True)
class MotionLibraryCommandCfg(CommandTermCfg):
  dataset: str = ""
  """BONES-SEED retarget ``dataset`` folder (manifest.parquet, motions/, libraries/); may be
  relative to the working directory."""
  subset: str = "dev"
  """Motions to train on: smoke, dev (2.5 h), full, or a number of hours (10, 10h).
  Packed into ``<dataset>/libraries/<subset>`` on first use (``bones_to_library.py``)."""
  library_path: str = ""
  """An already packed library folder (qpos.npy, qvel.npy, clips.json); overrides
  ``dataset`` and ``subset`` when set."""
  clips: tuple[str, ...] = ()
  """Only these clips of the library, by name as in its clips.json (e.g.
  230324/flip_360_001__A304); empty for all. For playing chosen motions of a library."""

  body_names: tuple[str, ...]
  """Tracked bodies; the first one is the anchor (root)."""

  entity_name: str = "robot"
  goal_joint_names: tuple[str, ...] = (".*",)
  """Joints whose reference positions/velocities go into the goal (the actuated ones)."""

  num_future_frames: int = 10
  """K reference frames in the goal (paper Appendix A)."""

  sampling_mode: Literal["return", "uniform", "start"] = "return"
  """'return': paper's return-based sampling (D.1); 'uniform': uniform over 1 s bins;
  'start': first frame of a random clip (evaluation)."""

  bin_seconds: float = 1.0
  sampler_temperature: float = 25.0
  sampler_mix: tuple[float, float, float] = (0.5, 0.4, 0.1)
  sampler_alpha: float = 0.2
  sampler_past_rate: float = 0.01
  return_discount: float = 0.99

  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  """Root pose noise at reset (x, y, z, roll, pitch, yaw)."""
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  """Root velocity noise at reset (world frame)."""
  joint_position_range: tuple[float, float] = (0.0, 0.0)
  """Noise on the goal joints' positions at reset."""

  time_left_cap_s: float = 3.0
  compile_kinematics: bool = True
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)

  @dataclass
  class VizCfg:
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionLibraryCommand:
    return MotionLibraryCommand(self, env)
