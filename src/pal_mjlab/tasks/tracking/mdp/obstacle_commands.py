"""Motion command for a drop-from-obstacle motion with a variable obstacle height.

The reference motion (e.g. ``drop_n_land``) is recorded on an obstacle of height
``reference_obstacle_height`` and has three phases: jump (feet on the obstacle),
flight (no foot in contact) and landing (feet on the ground). For every height in
``obstacle_heights`` a variant of the motion is precomputed:

- Jump phase: the whole robot is shifted vertically by ``dh = h - h_ref`` so it
  starts on top of the new obstacle.
- Flight phase, takeoff -> apex (highest root point): shifted by ``dh`` like the
  jump phase.
- Flight phase, apex -> touchdown: retimed to a new duration and blended so that
  it starts at the shifted apex and ends at the unchanged touchdown. The root
  height over the descent is modelled as ``parabola + residual``: the parabola
  goes through the apex height/vertical velocity and the touchdown height, and
  the residual is what the reference adds on top of it. The new duration is the
  time the same parabola (same initial velocity, same acceleration) takes to cover
  the new drop, i.e. it scales like ``sqrt(drop)`` as in a free fall: a higher
  obstacle gives a longer descent and a faster impact, a lower one a shorter
  descent and a softer impact. Joints, orientations and horizontal motion are
  time-warped over the descent.
- Landing phase: unchanged.

``h == h_ref`` reproduces the reference motion exactly.

Each environment is placed on the terrain row whose box has the sampled height
(one row per height, see ``BoxSingleObstacleTerrainCfg.box_height_range``) and
tracks the matching variant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass
class HeightVariant:
  """One retimed copy of the motion (all tensors have the frame dim first)."""

  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  body_pos_w: torch.Tensor
  body_quat_w: torch.Tensor
  body_lin_vel_w: torch.Tensor
  body_ang_vel_w: torch.Tensor
  flight_frames: tuple[int, int]
  """(takeoff, touchdown) frame indices in this variant."""

  @property
  def num_frames(self) -> int:
    return self.joint_pos.shape[0]


def detect_flight_frames(foot_z: torch.Tensor, tolerance: float) -> tuple[int, int]:
  """Find the (takeoff, touchdown) frames from the foot heights ``[T, num_feet]``.

  Takeoff is the first frame where every foot is ``tolerance`` above its height at
  frame 0 (standing on the obstacle). Touchdown is the first later frame where any
  foot is within ``tolerance`` of its height at the last frame (on the ground).
  """
  off_obstacle = (foot_z > foot_z[0] + tolerance).all(dim=-1)
  if not off_obstacle.any():
    raise ValueError("Could not detect the takeoff: the feet never leave the obstacle.")
  takeoff = int(torch.nonzero(off_obstacle)[0])
  on_ground = (foot_z < foot_z[-1] + tolerance).any(dim=-1)
  on_ground[: takeoff + 1] = False
  if not on_ground.any():
    raise ValueError("Could not detect the touchdown after the takeoff.")
  touchdown = int(torch.nonzero(on_ground)[0])
  return takeoff, touchdown


def _lerp(x: torch.Tensor, i0: torch.Tensor, i1: torch.Tensor, a: torch.Tensor):
  a = a.view(-1, *([1] * (x.dim() - 1)))
  return x[i0] * (1.0 - a) + x[i1] * a


def _nlerp(q: torch.Tensor, i0: torch.Tensor, i1: torch.Tensor, a: torch.Tensor):
  q0, q1 = q[i0], q[i1]
  q1 = torch.where((q0 * q1).sum(-1, keepdim=True) < 0.0, -q1, q1)
  a = a.view(-1, *([1] * (q.dim() - 1)))
  out = q0 * (1.0 - a) + q1 * a
  return out / out.norm(dim=-1, keepdim=True)


def make_height_variant(
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  body_pos_w: torch.Tensor,
  body_quat_w: torch.Tensor,
  body_lin_vel_w: torch.Tensor,
  body_ang_vel_w: torch.Tensor,
  flight_frames: tuple[int, int],
  height_delta: float,
  dt: float,
  root_index: int = 0,
) -> HeightVariant:
  """Retime the motion for an obstacle ``height_delta`` higher than the reference.

  ``flight_frames`` is (takeoff, touchdown) of the reference; the frames before the
  apex are shifted by ``height_delta`` and the descent is retimed (see module doc).

  Bodies are ``[T, num_bodies, ...]``; ``root_index`` selects the body whose
  height defines the flight parabola.
  """
  takeoff, t1 = flight_frames
  n_frames = joint_pos.shape[0]
  if not 0 < takeoff < t1 < n_frames:
    raise ValueError(f"Invalid flight frames {flight_frames} for {n_frames} frames.")
  # Retime the descent only: from the apex of the root to the touchdown.
  t0 = takeoff + int(body_pos_w[takeoff:t1, root_index, 2].argmax())

  z_a = float(body_pos_w[t0, root_index, 2])  # Apex height.
  z_b = float(body_pos_w[t1, root_index, 2])  # Touchdown height.
  v_a = float(body_lin_vel_w[t0, root_index, 2])  # Apex vertical velocity.
  dur0 = (t1 - t0) * dt
  # Reference parabola z_a + v_a t - acc0 t^2 / 2 hitting z_b at dur0.
  acc0 = 2.0 * (z_a - z_b + v_a * dur0) / dur0**2
  if acc0 <= 0.0:
    raise ValueError(
      f"The root does not fall between the apex {t0} and the touchdown {t1} "
      f"(fitted acceleration {acc0:.3f} m/s^2); check the flight frames."
    )

  # Duration for the same parabola to cover the new drop.
  drop = z_a + height_delta - z_b
  disc = max(v_a**2 + 2.0 * acc0 * drop, 0.0)
  dur = (v_a + math.sqrt(disc)) / acc0
  n_flight = max(1, round(dur / dt))
  dur = n_flight * dt
  # Re-fit the acceleration so the rounded duration still lands at z_b.
  acc = 2.0 * (z_a + height_delta - z_b + v_a * dur) / dur**2
  speed = dur0 / dur  # d(source time) / d(new time).

  dev = joint_pos.device
  j = torch.arange(n_flight, device=dev, dtype=torch.float32)
  src = t0 + j * (t1 - t0) / n_flight  # Fractional source frame.
  i0 = src.floor().long().clamp(max=n_frames - 1)
  i1 = (i0 + 1).clamp(max=n_frames - 1)
  a = src - i0.float()

  tau = j * dt  # New time since the apex.
  t_src = (src - t0) * dt  # Source time since the apex.
  dz = (z_a + height_delta + v_a * tau - 0.5 * acc * tau**2) - (
    z_a + v_a * t_src - 0.5 * acc0 * t_src**2
  )
  dvz = (v_a - acc * tau) - (v_a - acc0 * t_src) * speed

  f_pos = _lerp(body_pos_w, i0, i1, a)
  f_pos[..., 2] += dz[:, None]
  f_lin_vel = _lerp(body_lin_vel_w, i0, i1, a) * speed
  f_lin_vel[..., 2] += dvz[:, None]

  pre_pos = body_pos_w[:t0].clone()
  pre_pos[..., 2] += height_delta

  return HeightVariant(
    joint_pos=torch.cat([joint_pos[:t0], _lerp(joint_pos, i0, i1, a), joint_pos[t1:]]),
    joint_vel=torch.cat(
      [joint_vel[:t0], _lerp(joint_vel, i0, i1, a) * speed, joint_vel[t1:]]
    ),
    body_pos_w=torch.cat([pre_pos, f_pos, body_pos_w[t1:]]),
    body_quat_w=torch.cat(
      [body_quat_w[:t0], _nlerp(body_quat_w, i0, i1, a), body_quat_w[t1:]]
    ),
    body_lin_vel_w=torch.cat([body_lin_vel_w[:t0], f_lin_vel, body_lin_vel_w[t1:]]),
    body_ang_vel_w=torch.cat(
      [
        body_ang_vel_w[:t0],
        _lerp(body_ang_vel_w, i0, i1, a) * speed,
        body_ang_vel_w[t1:],
      ]
    ),
    flight_frames=(takeoff, t0 + n_flight),
  )


class _MotionBank:
  """Height variants padded to a common length and flattened to ``[K * T, ...]``.

  Exposes the ``MotionLoader`` attributes so the parent class keeps working;
  index with ``variant * time_step_total + time_step``.
  """

  def __init__(self, variants: list[HeightVariant], device: str) -> None:
    self.time_step_total = max(v.num_frames for v in variants)
    self.lengths = torch.tensor([v.num_frames for v in variants], device=device)
    for name in (
      "joint_pos",
      "joint_vel",
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
    ):
      padded = []
      for v in variants:
        x = getattr(v, name)
        pad = x[-1:].expand(self.time_step_total - x.shape[0], *x.shape[1:])
        padded.append(torch.cat([x, pad]))
      setattr(self, name, torch.cat(padded).to(device))


class ObstacleDropMotionCommand(MotionCommand):
  cfg: ObstacleDropMotionCommandCfg

  def __init__(self, cfg: ObstacleDropMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    ref = self.motion  # MotionLoader of the reference motion.

    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
      raise ValueError("ObstacleDropMotionCommand needs a generator terrain.")
    num_rows = terrain.terrain_origins.shape[0]
    if num_rows != len(cfg.obstacle_heights):
      raise ValueError(
        f"The terrain has {num_rows} rows but {len(cfg.obstacle_heights)} obstacle "
        "heights were given; row i must hold the box of height obstacle_heights[i]."
      )

    if cfg.flight_frames is not None:
      flight = cfg.flight_frames
    else:
      foot_ids = self.robot.find_bodies(cfg.foot_body_names, preserve_order=True)[0]
      foot_z = ref._body_pos_w[:, foot_ids, 2]
      flight = detect_flight_frames(foot_z, cfg.contact_tolerance)
    self.reference_flight_frames = flight

    variants = [
      make_height_variant(
        ref.joint_pos,
        ref.joint_vel,
        ref.body_pos_w,
        ref.body_quat_w,
        ref.body_lin_vel_w,
        ref.body_ang_vel_w,
        flight_frames=flight,
        height_delta=h - cfg.reference_obstacle_height,
        dt=env.step_dt,
        root_index=self.motion_anchor_body_index,
      )
      for h in cfg.obstacle_heights
    ]
    self.variant_flight_frames = [v.flight_frames for v in variants]
    self.motion = _MotionBank(variants, self.device)
    self.obstacle_heights = torch.tensor(cfg.obstacle_heights, device=self.device)
    self.variant_ids = terrain.terrain_levels.clone()
    self.metrics["obstacle_height"] = self.obstacle_heights[self.variant_ids]

  @property
  def motion_lengths(self) -> torch.Tensor:
    """Number of frames of each env's current variant."""
    return self.motion.lengths[self.variant_ids]

  @property
  def _frame_ids(self) -> torch.Tensor:
    return self.variant_ids * self.motion.time_step_total + self.time_steps

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self._frame_ids]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self._frame_ids]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self._frame_ids] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self._frame_ids]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self._frame_ids]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self._frame_ids]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self._frame_ids, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self._frame_ids, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self._frame_ids, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self._frame_ids, self.motion_anchor_body_index]

  def _sample_obstacle_heights(self, env_ids: torch.Tensor) -> None:
    """Pick a height per env and move it to the terrain row with that box."""
    terrain = self._env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    rows = torch.randint(
      0, len(self.cfg.obstacle_heights), (len(env_ids),), device=self.device
    )
    self.variant_ids[env_ids] = rows
    terrain.terrain_levels[env_ids] = rows
    terrain.env_origins[env_ids] = terrain.terrain_origins[
      rows, terrain.terrain_types[env_ids]
    ]
    self.metrics["obstacle_height"][env_ids] = self.obstacle_heights[rows]

  def _resample_command(self, env_ids: torch.Tensor):
    self._sample_obstacle_heights(env_ids)
    super()._resample_command(env_ids)

  # The sampling below is the parent's with per-env motion lengths: the adaptive
  # bins are over the normalized phase of the motion.

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    lengths = self.motion_lengths
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      current_bin_index = torch.clamp(
        (self.time_steps * self.bin_count) // lengths.clamp(min=1),
        0,
        self.bin_count - 1,
      )
      fail_bins = current_bin_index[env_ids][episode_failed]
      self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

    sampling_probabilities = (
      self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
    )
    sampling_probabilities = torch.nn.functional.pad(
      sampling_probabilities.unsqueeze(0).unsqueeze(0),
      (0, self.cfg.adaptive_kernel_size - 1),
      mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
      sampling_probabilities, self.kernel.view(1, 1, -1)
    ).view(-1)
    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    sampled_bins = torch.multinomial(
      sampling_probabilities, len(env_ids), replacement=True
    )
    self.time_steps[env_ids] = (
      (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
      / self.bin_count
      * (lengths[env_ids] - 1)
    ).long()

    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    pmax, imax = sampling_probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

  def _uniform_sampling(self, env_ids: torch.Tensor):
    lengths = self.motion_lengths[env_ids]
    u = torch.rand(len(env_ids), device=self.device)
    self.time_steps[env_ids] = (u * lengths).long().clamp(max=lengths - 1)
    self.metrics["sampling_entropy"][:] = 1.0
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5

  def _update_command(self, env_ids: torch.Tensor | None = None):
    if env_ids is None:
      self.time_steps += 1
    else:
      self.time_steps[env_ids] += 1
    wrap_ids = torch.where(self.time_steps >= self.motion_lengths)[0]
    if wrap_ids.numel() > 0:
      self._resample_command(wrap_ids)

    if self._pending_forward:
      self._pending_forward = False
      self._env.sim.forward()

    self.update_relative_body_poses()

    if env_ids is None and self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    self.time_steps[env_ids] = torch.clamp(
      torch.full_like(env_ids, frame), max=self.motion_lengths[env_ids] - 1
    )
    self._write_reference_state_to_sim(
      env_ids,
      self.body_pos_w[env_ids, 0],
      self.body_quat_w[env_ids, 0],
      self.body_lin_vel_w[env_ids, 0],
      self.body_ang_vel_w[env_ids, 0],
      self.joint_pos[env_ids],
      self.joint_vel[env_ids],
    )


@dataclass(kw_only=True)
class ObstacleDropMotionCommandCfg(MotionCommandCfg):
  obstacle_heights: tuple[float, ...]
  """Box height of each terrain row, in meters (row i <-> obstacle_heights[i])."""
  reference_obstacle_height: float
  """Height of the obstacle the reference motion was recorded on, in meters."""
  flight_frames: tuple[int, int] | None = None
  """(takeoff, touchdown) frames of the reference motion. None detects them from
  the heights of ``foot_body_names``."""
  foot_body_names: tuple[str, ...] = ("leg_left_foot_link", "leg_right_foot_link")
  contact_tolerance: float = 0.02
  """A foot counts as in contact within this height of its initial/final height."""

  def build(self, env: ManagerBasedRlEnv) -> ObstacleDropMotionCommand:
    return ObstacleDropMotionCommand(self, env)
