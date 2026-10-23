from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import os
import itertools
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import mujoco
import mujoco_warp as mjw
import warp as wp
from scipy.spatial.transform import Rotation as R, Slerp

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
  quat_apply_inverse,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from pal_mjlab.scripts.csv_to_npz import ROBOT_CONFIGS
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  KANGAROO_XML,
  KANGAROO_LOWER_BODY_XML,
  KANGAROO_HANDS_XML, 
  KANGAROO_GRIPPERS_XML
)

if TYPE_CHECKING:
  from collections.abc import Callable
  from typing import Any

  import viser

  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))

ROBOT_XML_DICT = {
  "Kangaroo" : KANGAROO_XML,
  "Kangaroo_lower_body" : KANGAROO_LOWER_BODY_XML,
  "Kangaroo_hands" : KANGAROO_HANDS_XML,
  "Kangaroo_grippers" : KANGAROO_GRIPPERS_XML,
}

def _sanity_trim(data, cfg: SanityCheckCfg, name: str = ""):
  """Trim explosive frames at the start/end of a clip. data: tuple of (D, M)."""
  if not cfg.enabled:
    return data

  (root_pos, root_lin_vel, root_lin_acc, root_quat,
   root_ang_vel, root_ang_acc, joint_pos, joint_vel) = data
  M = root_pos.shape[1]

  bad = np.zeros(M, dtype=bool)
  for arr, thr in (
    (joint_vel, cfg.joint_vel),
    (root_lin_vel, cfg.root_lin_vel),
    (root_ang_vel, cfg.root_ang_vel),
    (root_lin_acc, cfg.root_lin_acc),
    (root_ang_acc, cfg.root_ang_acc),
  ):
    bad |= np.abs(arr).max(axis=0) > thr

  n_start = n_end = 0
  head = np.where(bad[: min(cfg.search, M)])[0]
  if len(head):
    n_start = int(head[-1]) + 1 + cfg.margin

  tail_from = max(M - cfg.search, 0)
  tail = np.where(bad[tail_from:])[0]
  if len(tail):
    n_end = (M - (tail_from + int(tail[0]))) + cfg.margin

  if n_start == 0 and n_end == 0:
    return data

  if n_start + n_end >= M:
    if cfg.verbose:
      tqdm.write(f"[sanity_check] {name}: would trim {n_start}+{n_end} of {M} frames, keeping clip as is")
    return data

  if cfg.verbose:
    tqdm.write(f"[sanity_check] {name}: {M} -> {M - n_start - n_end} frames "
          f"(start -{n_start}, end -{n_end})")
  stop = M - n_end
  return tuple(x[:, n_start:stop] for x in data)



# Stored layout per frame: [root_pos 3 | root_quat 4 (xyzw) | root_lin_vel 3 |
#                           root_ang_vel 3 | joint_pos J | joint_vel J]
_ROOT_COLS = 13
# Worker output (D, M): same, plus root_lin_acc 3 | root_ang_acc 3 before the joints.
_RAW_COLS = 19


def _worker_init(nice: int):
  os.environ["OMP_NUM_THREADS"] = "1"
  os.environ["OPENBLAS_NUM_THREADS"] = "1"
  os.environ["MKL_NUM_THREADS"] = "1"
  try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
  except Exception:
    pass
  try:
    os.nice(nice)            # lower priority so the desktop stays responsive
  except Exception:
    pass


def _load_clip(job):
  """Parse + resample one CSV. CPU only. Returns (name, (19+2J, M) float32) or None."""
  path, in_dt, out_dt = job

  raw = np.loadtxt(path, delimiter=",", dtype=np.float32, ndmin=2)   # (N, 7+J)
  if raw.shape[0] < 2:
    return None
  cols = raw.T
  root_pos, root_quat, joint_pos = cols[0:3], cols[3:7], cols[7:]

  n_in = root_pos.shape[1]
  t_in = np.arange(n_in) * in_dt
  t_out = np.arange(0, t_in[-1], out_dt)
  if len(t_out) < 3:
    return None

  def lerp(x):
    return np.stack([np.interp(t_out, t_in, row) for row in x])

  root_pos_o = lerp(root_pos)
  joint_pos_o = lerp(joint_pos)

  r = Slerp(t_in, R.from_quat(root_quat.T))(t_out)
  root_quat_o = r.as_quat().T

  root_lin_vel = np.gradient(root_pos_o, out_dt, axis=1)
  joint_vel = np.gradient(joint_pos_o, out_dt, axis=1)

  w = (r[2:] * r[:-2].inv()).as_rotvec() / (2 * out_dt)
  root_ang_vel = np.concatenate([w[:1], w, w[-1:]]).T

  root_lin_acc = np.gradient(root_lin_vel, out_dt, axis=1)
  root_ang_acc = np.gradient(root_ang_vel, out_dt, axis=1)

  out = np.concatenate(
    [root_pos_o, root_quat_o, root_lin_vel, root_ang_vel,
     root_lin_acc, root_ang_acc, joint_pos_o, joint_vel], axis=0,
  ).astype(np.float32, copy=False)                                   # (19+2J, M)
  return Path(path).stem, out


def _finalize_clip(arr: np.ndarray, cfg: SanityCheckCfg, name: str):
  """Sanity-trim a (19+2J, M) clip and repack to (M, 13+2J)."""
  J = (arr.shape[0] - _RAW_COLS) // 2
  rp, rq = arr[0:3], arr[3:7]
  lv, av = arr[7:10], arr[10:13]
  la, aa = arr[13:16], arr[16:19]
  jp, jv = arr[19:19 + J], arr[19 + J:]
  rp, lv, _, rq, av, _, jp, jv = _sanity_trim(
    (rp, lv, la, rq, av, aa, jp, jv), cfg, name
  )
  packed = np.concatenate([rp, rq, lv, av, jp, jv], axis=0).T
  packed = np.ascontiguousarray(packed, dtype=np.float32)
  return packed if packed.shape[0] > 0 else None


def _bounded_map(ex, fn, jobs, window: int):
  """Like ex.map (ordered) but keeps at most `window` jobs/results in flight."""
  it = iter(jobs)
  pending = deque(ex.submit(fn, j) for j in itertools.islice(it, window))
  while pending:
    res = pending.popleft().result()
    nxt = next(it, None)
    if nxt is not None:
      pending.append(ex.submit(fn, nxt))
    yield res

@dataclass(frozen=True)
class SanityCheckCfg:
  """Thresholds for trimming glitch frames at the ends of CSV clips."""

  enabled: bool = True
  """False skips trimming entirely."""

  joint_vel: float = 10.0
  """Max |joint velocity| (rad/s). Same default as trim_npz.py."""

  root_lin_vel: float = 5.0
  """Max |root linear velocity| (m/s)."""

  root_ang_vel: float = 10.0
  """Max |root angular velocity| (rad/s)."""

  root_lin_acc: float = 150.0
  """Max |root linear acceleration| (m/s^2)."""

  root_ang_acc: float = 600.0
  """Max |root angular acceleration| (rad/s^2)."""

  search: int = 30
  """Only look for spikes within this many frames of either end."""

  margin: int = 2
  """Extra frames removed past the last detected spike."""

  verbose: bool = False
  """Print a line for every clip that gets trimmed."""

class MotionLoader:
    """
    Load multiple motion NPZ files as one contiguous motion library.

    Each NPZ is treated as one trajectory/clip. The individual clips are
    concatenated along the time dimension, while segment_start_idx and
    segment_length keep track of where each trajectory lives.
    """

    def __init__(
        self,
        motion_dir: str,
        body_indexes: torch.Tensor,
        anchor_body_index: int,
        step_dt,
        device: str = "cpu",
    ) -> None:

        source = Path(motion_dir).expanduser().resolve()

        if not source.is_dir():
            raise FileNotFoundError(f"Motion directory not found: {source}")

        motion_paths = sorted(source.glob("*.npz"))

        if not motion_paths:
            raise FileNotFoundError(
                f"No .npz motion files found in {source}"
            )

        # Keep body indices on CPU while loading numpy arrays.
        body_idx = body_indexes.detach().cpu().numpy()

        # Store each clip temporarily as numpy arrays.
        joint_pos_list = []
        joint_vel_list = []
        body_pos_list = []
        body_quat_list = []
        body_lin_vel_list = []
        body_ang_vel_list = []

        segment_start = []
        segment_length = []
        segment_names = []

        current_start = 0

        time_step = []

        for path in motion_paths:

            with np.load(path) as data:

                joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
                joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)

                body_pos_w = np.asarray(
                    data["body_pos_w"][:, body_idx],
                    dtype=np.float32,
                )

                body_quat_w = np.asarray(
                    data["body_quat_w"][:, body_idx],
                    dtype=np.float32,
                )

                body_lin_vel_w = np.asarray(
                    data["body_lin_vel_w"][:, body_idx],
                    dtype=np.float32,
                )

                body_ang_vel_w = np.asarray(
                    data["body_ang_vel_w"][:, body_idx],
                    dtype=np.float32,
                )

            num_frames = joint_pos.shape[0]

            if num_frames == 0:
                continue

            # Save trajectory metadata.
            segment_start.append(current_start)
            segment_length.append(num_frames)
            segment_names.append(path.stem)

            # Add this clip to the library.
            joint_pos_list.append(joint_pos)
            joint_vel_list.append(joint_vel)
            body_pos_list.append(body_pos_w)
            body_quat_list.append(body_quat_w)
            body_lin_vel_list.append(body_lin_vel_w)
            body_ang_vel_list.append(body_ang_vel_w)

            current_start += num_frames

            time_step.append(int(joint_pos.shape[0]))

        if not joint_pos_list:
            raise ValueError(f"No valid motion frames found in {source}")

        # ------------------------------------------------------------
        # Concatenate all trajectories into one motion library.
        # ------------------------------------------------------------

        self.joint_pos = torch.as_tensor(
            np.concatenate(joint_pos_list, axis=0),
            dtype=torch.float32,
            device=device,
        )

        self.joint_vel = torch.as_tensor(
            np.concatenate(joint_vel_list, axis=0),
            dtype=torch.float32,
            device=device,
        )

        self.body_pos_w = torch.as_tensor(
            np.concatenate(body_pos_list, axis=0),
            dtype=torch.float32,
            device=device,
        )

        self.body_quat_w = torch.as_tensor(
            np.concatenate(body_quat_list, axis=0),
            dtype=torch.float32,
            device=device,
        )

        self.body_lin_vel_w = torch.as_tensor(
            np.concatenate(body_lin_vel_list, axis=0),
            dtype=torch.float32,
            device=device,
        )

        self.body_ang_vel_w = torch.as_tensor(
            np.concatenate(body_ang_vel_list, axis=0),
            dtype=torch.float32,
            device=device,
        )

        # ------------------------------------------------------------
        # Trajectory / segment information.
        # ------------------------------------------------------------

        self.segment_start_idx = torch.tensor(
            segment_start,
            dtype=torch.long,
            device=device,
        )

        self.segment_length = torch.tensor(
            segment_length,
            dtype=torch.long,
            device=device,
        )

        self.segment_end_idx = (
            self.segment_start_idx + self.segment_length
        )

        self.num_trajectories = len(segment_start)

        self.segment_names = tuple(segment_names)

        self.time_step_total = max(tsl for tsl in time_step)

        self.trajectory_probs = (
          self.segment_length.float() / self.segment_length.sum().float()
        ).to(device=device)

        anchor_lin_vel = self.body_lin_vel_w[:, anchor_body_index]
        anchor_ang_vel = self.body_ang_vel_w[:, anchor_body_index]
        lin_acc = torch.zeros_like(anchor_lin_vel)
        ang_acc = torch.zeros_like(anchor_ang_vel)
        for s, e in zip(self.segment_start_idx.tolist(), self.segment_end_idx.tolist()):
          if e - s >= 2:
            lin_acc[s:e] = torch.gradient(anchor_lin_vel[s:e], spacing=step_dt, dim=0)[0]
            ang_acc[s:e] = torch.gradient(anchor_ang_vel[s:e], spacing=step_dt, dim=0)[0]
        self.anchor_lin_acc_w, self.anchor_ang_acc_w = lin_acc, ang_acc

class MotionLoaderCsv:
  """Load CSV clips into one contiguous GPU motion library.

  CPU use is capped (few, low-priority workers), clips are streamed straight
  into a preallocated device buffer, and loading stops at `max_frames`.
  """

  def __init__(
    self,
    motion_dir: str,
    input_fps: float = 30.0,
    output_fps: float = 50.0,
    device: str = "cpu",
    sanity: SanityCheckCfg = SanityCheckCfg(),
    num_workers: int | None = None,
    max_frames: int = 3_000_000,
    nice: int = 10,
    store_dtype: torch.dtype = torch.float32,
  ) -> None:
    import random

    self.sanity = sanity
    in_dt, out_dt = 1.0 / input_fps, 1.0 / output_fps

    source = Path(motion_dir).expanduser().resolve()
    if not source.is_dir():
      raise FileNotFoundError(f"Motion directory not found: {source}")

    motion_paths = sorted(source.glob("*.csv"))
    if not motion_paths:
      raise FileNotFoundError(f"No .csv motion files found in {source}")
    # Seeded shuffle so a capped library samples across subjects, reproducibly.
    random.Random(0).shuffle(motion_paths)

    jobs = [(str(p), in_dt, out_dt) for p in motion_paths]
    # Default: at most 4 workers and never more than half the cores.
    n_workers = max(1, min(num_workers or min(4, (os.cpu_count() or 2) // 2), len(jobs)))

    names: list[str] = []
    lengths: list[int] = []
    state = {"lib": None, "i": 0, "full": False}

    pbar_read = tqdm(total=len(jobs), desc="Reading CSVs ", position=0, unit="clip")
    pbar_san = tqdm(total=len(jobs), desc="Sanity check", position=1, unit="clip")

    def handle(res):
      pbar_read.update(1)
      if res is None or state["full"]:
        pbar_san.update(1)
        return
      name, arr = res
      packed = _finalize_clip(arr, sanity, name)       # sanity trim + repack (M, 13+2J)
      pbar_san.update(1)
      if packed is None:
        return
      n = packed.shape[0]
      if state["lib"] is None:
        state["lib"] = torch.empty(
          (max_frames, packed.shape[1]), dtype=store_dtype, device=device
        )
      if state["i"] + n > max_frames:
        state["full"] = True
        return
      i = state["i"]
      state["lib"][i : i + n] = torch.from_numpy(packed).to(
        device=device, dtype=store_dtype
      )
      state["i"] += n
      names.append(name)
      lengths.append(n)

    try:
      if n_workers > 1:
        with ProcessPoolExecutor(
          n_workers,
          mp_context=mp.get_context("fork"),
          initializer=_worker_init,
          initargs=(nice,),
        ) as ex:
          for res in _bounded_map(ex, _load_clip, jobs, window=n_workers * 2):
            handle(res)
            if state["full"]:
              break
      else:
        for j in jobs:
          handle(_load_clip(j))
          if state["full"]:
            break
    finally:
      pbar_read.close()
      pbar_san.close()

    if state["lib"] is None or not lengths:
      raise ValueError(f"No valid motion frames found in {source}")

    lib = state["lib"][: state["i"]]
    total, C = lib.shape
    J = (C - _ROOT_COLS) // 2
    print(
      f"[MotionLoaderCsv] {len(lengths)} clips, {total} frames, "
      f"{lib.numel() * lib.element_size() / 1e6:.0f} MB on {device}"
    )

    # Views into the single library tensor (no extra memory).
    self.root_pos = lib[:, 0:3]
    self.root_quat = lib[:, 3:7]                       # xyzw
    self.root_lin_vel = lib[:, 7:10]
    self.root_ang_vel = lib[:, 10:13]
    self.joint_pos = lib[:, _ROOT_COLS : _ROOT_COLS + J]
    self.joint_vel = lib[:, _ROOT_COLS + J :]
    self._lib = lib

    # ---- segment info ----
    segment_start = np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()
    self.segment_start_idx = torch.tensor(segment_start, dtype=torch.long, device=device)
    self.segment_length = torch.tensor(lengths, dtype=torch.long, device=device)
    self.segment_end_idx = self.segment_start_idx + self.segment_length
    self.num_trajectories = len(lengths)
    self.segment_names = tuple(names)
    self.time_step_total = max(lengths)
    self.trajectory_probs = (
      self.segment_length.float() / self.segment_length.sum().float()
    ).to(device=device)

  def sanity_check(self, data, name: str = ""):
    return _sanity_trim(data, self.sanity, name)

class OnlineCsvToNpz:
  """Computes NPZ-equivalent reference data (joint_pos, joint_vel, body_pos_w,
  body_quat_w, body_lin_vel_w, body_ang_vel_w) on the fly from CSV motions,
  for a batch of global frame indices at once (one MJWarp world per element).

  Quaternions are returned as wxyz (mjlab convention). Body velocities are
  link velocities at the body frame origin, expressed in the world frame.
  """

  def __init__(
    self,
    robot_type: str,
    motion_dir: str,
    body_names: tuple[str, ...],
    batch_size: int,
    input_fps: float = 30.0,
    output_fps: float = 50.0,
    device: str = "cuda:0",
    sanity: SanityCheckCfg = SanityCheckCfg(),
    max_frames: int = 3_000_000,
  ):
    self.device = device
    self.batch_size = batch_size

    spec = mujoco.MjSpec.from_file(str(ROBOT_XML_DICT[robot_type]))
    body_names_in_xml = {b.name for b in spec.bodies}
    for ex in list(spec.excludes):
      if ex.bodyname1 not in body_names_in_xml or ex.bodyname2 not in body_names_in_xml:
        spec.delete(ex)
    self.mjm = spec.compile()
    self.mjd = mujoco.MjData(self.mjm)
    mujoco.mj_forward(self.mjm, self.mjd)

    with wp.ScopedDevice(device):
      self.m = mjw.put_model(self.mjm)
      self.d = mjw.put_data(self.mjm, self.mjd, nworld=batch_size)

    # Zero-copy torch views of the warp buffers (shape: (B, ...)).
    self._qpos = wp.to_torch(self.d.qpos)
    self._qvel = wp.to_torch(self.d.qvel)
    self._xpos = wp.to_torch(self.d.xpos)
    self._xquat = wp.to_torch(self.d.xquat)
    self._cvel = wp.to_torch(self.d.cvel)                 # (B, nbody, 6) [ang, lin]
    self._subtree_com = wp.to_torch(self.d.subtree_com)

    # Resolve requested bodies to MuJoCo body ids.
    ids = []
    for name in body_names:
      bid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, name)
      if bid < 0:
        raise ValueError(f"Body '{name}' not found in {robot_type} model")
      ids.append(bid)
    self.body_ids = torch.tensor(ids, dtype=torch.long, device=device)
    self.body_root_ids = torch.as_tensor(
      self.mjm.body_rootid[ids], dtype=torch.long, device=device
    )

    self.motion_csv = MotionLoaderCsv(
      motion_dir=motion_dir,
      input_fps=input_fps,
      output_fps=output_fps,
      device=device,
      sanity=sanity,
      max_frames=max_frames,
    )

    expected_nq = 7 + self.motion_csv.joint_pos.shape[1]
    if self.mjm.nq != expected_nq:
      raise ValueError(
        f"CSV has {expected_nq} qpos dims (free root + joints) but model has nq={self.mjm.nq}"
      )

  @torch.no_grad()
  def retarget(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
    """t: (N,) global frame indices into the concatenated motion library.
    Returns a dict of tensors with leading dim N."""
    total = self.motion_csv.joint_pos.shape[0]
    t = torch.as_tensor(t, dtype=torch.long, device=self.device).reshape(-1).clamp(0, total - 1)
    n, B = t.numel(), self.batch_size

    chunks = []
    for s in range(0, n, B):
      idx = t[s : s + B]
      k = idx.numel()
      if k < B:  # pad last chunk so the world count stays fixed
        idx = torch.cat([idx, idx[-1:].expand(B - k)])
      out = self._retarget_batch(idx)
      chunks.append({key: v[:k] for key, v in out.items()})

    return {key: torch.cat([c[key] for c in chunks], dim=0) for key in chunks[0]}

  def _retarget_batch(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
    mo = self.motion_csv

    root_pos = mo.root_pos[idx]                              # (B, 3)
    root_quat = mo.root_quat[idx][:, [3, 0, 1, 2]]           # xyzw -> wxyz
    root_lin_vel = mo.root_lin_vel[idx]                      # world frame
    root_ang_vel_w = mo.root_ang_vel[idx]                    # world frame
    joint_pos = mo.joint_pos[idx]
    joint_vel = mo.joint_vel[idx]

    # MuJoCo free joint: linear vel in world frame, angular vel in body frame.
    root_ang_vel_b = quat_apply_inverse(root_quat, root_ang_vel_w)

    self._qpos[:, 0:3] = root_pos
    self._qpos[:, 3:7] = root_quat
    self._qpos[:, 7:] = joint_pos
    self._qvel[:, 0:3] = root_lin_vel
    self._qvel[:, 3:6] = root_ang_vel_b
    self._qvel[:, 6:] = joint_vel

    mjw.forward(self.m, self.d)

    ids = self.body_ids
    body_pos = self._xpos[:, ids]                            # (B, nb, 3)
    body_quat = self._xquat[:, ids]                          # (B, nb, 4) wxyz
    cvel = self._cvel[:, ids]                                # (B, nb, 6)
    com = self._subtree_com[:, self.body_root_ids]           # (B, nb, 3)

    # cvel is referenced at the tree's subtree COM; shift to the body origin.
    ang = cvel[..., :3]
    lin = cvel[..., 3:] + torch.cross(ang, body_pos - com, dim=-1)

    return {
      "joint_pos": joint_pos.clone(),
      "joint_vel": joint_vel.clone(),
      "body_pos_w": body_pos.clone(),
      "body_quat_w": body_quat.clone(),
      "body_lin_vel_w": lin,
      "body_ang_vel_w": ang.clone(),
    }

class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    # if no body name passed, track all
    if self.cfg.tracked_body_names is None :
      self.cfg.tracked_body_names = self.cfg.body_names

    self.motion = self._load_motion()

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.fixed_time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self._build_adaptive_bins()
    self.bin_failed_count = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
        [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
        device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    self._pending_forward = False

    self.rand_motion = torch.multinomial(
      self.motion.trajectory_probs, self.num_envs, replacement=True
    )

    self.moving_envs = torch.ones(self.num_envs, dtype=torch.bool ,device=self.device)

  def _load_motion(self):
    return MotionLoader(
      self.cfg.motion_dir,
      self.body_indexes,
      self.motion_anchor_body_index,
      self._env.step_dt,
      device=self.device,
    )

  def _build_adaptive_bins(self) -> None:
    """One bin per (clip, ~adaptive_bin_seconds chunk), indexed globally."""
    dev = self.device
    m = self.motion
    num_clips = m.num_trajectories
    fpb = max(int(round(self.cfg.adaptive_bin_seconds / self._env.step_dt)), 1)

    bins_per_clip = (m.segment_length + fpb - 1) // fpb             # (C,)
    clip_bin_offset = torch.cumsum(bins_per_clip, 0) - bins_per_clip  # (C,)
    total = int(bins_per_clip.sum())

    bin_clip = torch.repeat_interleave(
      torch.arange(num_clips, device=dev), bins_per_clip
    )                                                                 # (total,)
    local_bin = torch.arange(total, device=dev) - clip_bin_offset[bin_clip]
    bin_start = m.segment_start_idx[bin_clip] + local_bin * fpb
    bin_end = torch.minimum(bin_start + fpb, m.segment_end_idx[bin_clip])

    # Neighbour table for smoothing: replicate-padding applied per clip, so the
    # kernel never reaches into a different clip.
    k = self.cfg.adaptive_kernel_size
    pad = (k - 1) // 2
    first = clip_bin_offset[bin_clip]
    last = first + bins_per_clip[bin_clip] - 1
    idx = torch.arange(total, device=dev)[:, None] + (
      torch.arange(k, device=dev) - pad
    )[None, :]
    self._bin_neighbors = torch.clamp(idx, min=first[:, None], max=last[:, None])

    self._frames_per_bin = fpb
    self._bins_per_clip = bins_per_clip
    self._clip_bin_offset = clip_bin_offset
    self._bin_clip = bin_clip
    self._bin_start = bin_start
    self._bin_end = bin_end
    self.bin_count = total  # `_uniform_sampling` metrics still use this name

    # Length-proportional distribution over bins (matches the old clip lottery).
    uniform_p = (bin_end - bin_start).float()
    self._bin_uniform_p = uniform_p / uniform_p.sum()

  @property
  def ref_base_height(self) -> torch.Tensor:
    """Anchor height relative to env origin (z_I r̂_IB)."""
    return self.anchor_pos_w[:, 2:3] - self._env.scene.env_origins[:, 2:3]

  @property
  def ref_base_lin_vel_b(self) -> torch.Tensor:
    """Reference anchor linear velocity in anchor frame (B v̂_IB)."""
    return quat_apply_inverse(self.anchor_quat_w, self.anchor_lin_vel_w)*self.moving_envs.unsqueeze(-1)

  @property
  def ref_base_ang_vel_b(self) -> torch.Tensor:
    """Reference anchor angular velocity in anchor frame (B ω̂_IB)."""
    return quat_apply_inverse(self.anchor_quat_w, self.anchor_ang_vel_w)*self.moving_envs.unsqueeze(-1)

  @property
  def ref_gravity_b(self) -> torch.Tensor:
    """Reference gravity in anchor frame (B ĝ_I)."""
    return quat_apply_inverse(self.anchor_quat_w, self.robot.data.gravity_vec_w)

  @property
  def ref_base_lin_acc_b(self) -> torch.Tensor:
    return quat_apply_inverse(self.anchor_quat_w, self.anchor_lin_acc_w)*self.moving_envs.unsqueeze(-1)

  @property
  def ref_base_ang_acc_b(self) -> torch.Tensor:
    return quat_apply_inverse(self.anchor_quat_w, self.anchor_ang_acc_w)*self.moving_envs.unsqueeze(-1)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps]*self.moving_envs.unsqueeze(-1)

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps]*self.moving_envs[:, None, None]


  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps]*self.moving_envs[:, None, None]


  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]*self.moving_envs.unsqueeze(-1)

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]*self.moving_envs.unsqueeze(-1)

  @property
  def anchor_lin_acc_w(self) -> torch.Tensor:
    return self.motion.anchor_lin_acc_w[self.time_steps]*self.moving_envs.unsqueeze(-1)

  @property
  def anchor_ang_acc_w(self) -> torch.Tensor:
    return self.motion.anchor_ang_acc_w[self.time_steps]*self.moving_envs.unsqueeze(-1)

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    num_envs = len(env_ids)

    episode_failed = (
      self._env.termination_manager.terminated[env_ids] & self.moving_envs[env_ids]
    )
    if torch.any(episode_failed):
      failed_env_ids = env_ids[episode_failed]
      motion_ids = self.rand_motion[failed_env_ids]

      local_frame = (
        self.time_steps[failed_env_ids] - self.motion.segment_start_idx[motion_ids]
      ).clamp_min(0)
      local_bin = torch.minimum(
        local_frame // self._frames_per_bin, self._bins_per_clip[motion_ids] - 1
      )
      fail_bins = self._clip_bin_offset[motion_ids] + local_bin

      self._current_bin_failed += torch.bincount(
        fail_bins, minlength=self.bin_count
      ).to(dtype=self._current_bin_failed.dtype)

    fail_p = self.bin_failed_count / self.bin_failed_count.sum().clamp_min(1e-12)
    if self.cfg.adaptive_kernel_size > 1:
      fail_p = (fail_p[self._bin_neighbors] * self.kernel).sum(dim=-1)
      fail_p = fail_p / fail_p.sum().clamp_min(1e-12)

    r = self.cfg.adaptive_uniform_ratio
    probs = (1 - r) * fail_p + r * self._bin_uniform_p

    # Optional: cap any bin at N times its uniform share, then renormalize.
    cap = self.cfg.adaptive_max_prob_ratio * self._bin_uniform_p
    probs = torch.minimum(probs, cap)
    probs = probs / probs.sum().clamp_min(1e-12)

    # Sample a global (clip, chunk) bin; the clip follows from the bin.
    sampled_bins = torch.multinomial(probs, num_envs, replacement=True)
    self.rand_motion[env_ids] = self._bin_clip[sampled_bins]

    start = self._bin_start[sampled_bins]
    length = (self._bin_end[sampled_bins] - start).clamp_min(1)
    offset = torch.minimum(
      (torch.rand(num_envs, device=self.device) * length).long(), length - 1
    )
    self.time_steps[env_ids] = start + offset

    H = -(probs * (probs + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    pmax, imax = probs.max(dim=0)
    self.metrics["sampling_entropy"][env_ids] = H_norm
    self.metrics["sampling_top1_prob"][env_ids] = pmax
    self.metrics["sampling_top1_bin"][env_ids] = imax.float() / self.bin_count

  def _random_start_sampling(self, env_ids: torch.Tensor):
    self.rand_motion[env_ids] = torch.randint(0, self.motion.num_trajectories, (len(env_ids),), device=self.device)
    self.time_steps[env_ids] = self.motion.segment_start_idx[self.rand_motion[env_ids]]

  def _uniform_sampling(self, env_ids: torch.Tensor):
    num_envs = len(env_ids)
    
    rand_motion = torch.multinomial(
        self.motion.trajectory_probs, 
        num_envs, 
        replacement=True
    )
    self.rand_motion[env_ids] = rand_motion
    
    start_idxs = self.motion.segment_start_idx[rand_motion]
    end_idxs = self.motion.segment_end_idx[rand_motion]
    
    rand_float = torch.rand(num_envs, device=self.device)
    self.time_steps[env_ids] = (
        start_idxs + (rand_float * (end_idxs - start_idxs)).to(torch.long)
    )
    
    self.metrics["sampling_entropy"][env_ids] = 1.0  # Maximum entropy for uniform.
    self.metrics["sampling_top1_prob"][env_ids] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][env_ids] = 0.5  # No specific bin preference.

  def _sample_moving(self, env_ids: torch.Tensor) -> None:
    fixed = torch.rand(len(env_ids), device=self.device) < self.cfg.rel_fixed_command
    self.moving_envs[env_ids] = ~fixed
    self.fixed_time_steps[env_ids] = 0
    # Fixed envs hold the first frame of their clip.
    start = self.motion.segment_start_idx[self.rand_motion[env_ids]]
    self.time_steps[env_ids] = torch.where(fixed, start, self.time_steps[env_ids])

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Clip joint positions and write root + joint state to sim."""
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      self._random_start_sampling(env_ids)
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)
    self._sample_moving(env_ids)

    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori = quat_mul(orientations_delta, root_ori)
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel += rand_samples[:, :3]
    root_ang_vel += rand_samples[:, 3:]

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )

    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )
    self._pending_forward = True

  def update_relative_body_poses(self) -> None:
    """Recompute ``body_pos_relative_w`` and ``body_quat_relative_w``.

    Called after ``reset_to_frame`` so that termination checks that
    compare relative body positions see the correct state.
    """
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    # Reset-path resamples are followed by the env's own forward(); only
    # compute-path resamples (wraparound or timer expiry) need our refresh.
    self._pending_forward = False
    return extras

  def _update_command(self, env_ids: torch.Tensor | None = None):
    if env_ids is None:
      self.time_steps += self.moving_envs
      self.fixed_time_steps += 1
    else:
      self.time_steps[env_ids] += self.moving_envs[env_ids]
      self.fixed_time_steps[env_ids] += 1
    motion_done = self.time_steps >= self.motion.segment_end_idx[self.rand_motion]
    fixed_done = (
      (self.fixed_time_steps >= self.cfg.max_timesteps_fixed)
      & ~self.moving_envs
    )
    wrap_ids = torch.nonzero(motion_done | fixed_done, as_tuple=False).squeeze(-1)

    if wrap_ids.numel() > 0:
      self._resample_command(wrap_ids)

    # _resample_command writes qpos/qvel but does not refresh derived
    # quantities; forward() so update_relative_body_poses reads the
    # post-teleport robot anchor instead of the stale pre-resample pose.
    # Covers both the wraparound above and a timer-expiry resample that ran
    # in compute before this call.
    if self._pending_forward:
      self._pending_forward = False
      self._env.sim.forward()

    self.update_relative_body_poses()

    # Fold failure counts into the EMA only on the per-step update so
    # manual resets do not decay it faster.
    if env_ids is None and self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        # Build a ghost model with only visual geoms visible. Collision geoms (nonzero
        # contype/conaffinity) get alpha=0 so the viewer's alpha filter excludes them.
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        for gi in range(self._ghost_model.ngeom):
          if (
            self._ghost_model.geom_contype[gi] != 0
            or self._ghost_model.geom_conaffinity[gi] != 0
          ):
            self._ghost_model.geom_rgba[gi, 3] = 0
          else:
            self._ghost_model.geom_rgba[gi] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

        visualizer.add_ghost_mesh(
          qpos,
          model=self._ghost_model,
          label=f"ghost_{batch}",
        )

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create motion scrubber controls in the Viser viewer."""
    max_frame = int(self.motion.time_step_total) - 1

    with server.gui.add_folder(name.capitalize()):
      scrubber = server.gui.add_slider(
        "Frame",
        min=0,
        max=max_frame,
        step=1,
        initial_value=0,
      )

      @scrubber.on_update
      def _(_) -> None:
        idx = get_env_idx()
        motion_id = self.rand_motion[idx]
        start = self.motion.segment_start_idx[motion_id]
        end = self.motion.segment_end_idx[motion_id]
        self.time_steps[idx] = torch.minimum(start + int(scrubber.value), end - 1)
        if on_change is not None:
          on_change()

      all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
      start_btn = server.gui.add_button("Start Here")

      @start_btn.on_click
      def _(_) -> None:
        if request_action is not None:
          request_action(
            "CUSTOM",
            {"type": "gui_reset", "all_envs": all_envs_cb.value},
          )

    self._scrubber_handles = (scrubber, all_envs_cb, start_btn)
    self._set_scrubber_disabled(True)

  def _set_scrubber_disabled(self, disabled: bool) -> None:
    """Enable or disable the motion scrubber GUI controls."""
    for handle in self._scrubber_handles:
      handle.disabled = disabled

  def on_viewer_pause(self, paused: bool) -> None:
    if hasattr(self, "_scrubber_handles"):
      self._set_scrubber_disabled(not paused)

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    if not hasattr(self, "_scrubber_handles"):
      return False
    frame = int(self._scrubber_handles[0].value)
    self.reset_to_frame(env_ids, frame)
    # reset_to_frame writes qpos/qvel; forward so update_relative_body_poses
    # reads the scrubbed pose instead of the stale pre-scrub kinematics.
    self._env.sim.forward()
    self.update_relative_body_poses()
    return True

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    self.moving_envs[env_ids] = True
    self.fixed_time_steps[env_ids] = 0
    start = self.motion.segment_start_idx[self.rand_motion[env_ids]]
    end = self.motion.segment_end_idx[self.rand_motion[env_ids]]
    self.time_steps[env_ids] = torch.clamp(start + frame, min=start, max=end - 1)
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
class MotionCommandCfg(CommandTermCfg):
  motion_dir: str
  """Directory with one .npz per clip; all clips are loaded as one library."""

  anchor_body_name: str
  """Body used as the reference frame (e.g. base_link)."""

  body_names: tuple[str, ...]
  """Bodies loaded from the motion files; must include the anchor."""

  tracked_body_names: tuple[str, ...] | None = None
  """Bodies used for tracking terms. None = all body_names."""

  entity_name: str
  """Name of the robot entity in the scene."""

  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  """Root pose noise added at reset (keys: x, y, z, roll, pitch, yaw)."""

  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  """Root velocity noise added at reset (keys: x, y, z, roll, pitch, yaw)."""

  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  """Uniform noise (rad) added to the reference joint positions at reset."""

  # --- Adaptive sampling (only used when sampling_mode == "adaptive") ---

  adaptive_kernel_size: int = 3
  """Smoothing window (in bins, within a clip) applied to the failure mass.
  1 = no smoothing."""

  adaptive_lambda: float = 0.8
  """Kernel weights are [1, lambda, lambda^2, ...], normalized. With the current
  neighbor-table indexing, the bin right after a failed bin gets the largest share."""

  adaptive_alpha: float = 0.001
  """EMA rate for the failure statistics. Smaller = slower, smoother adaptation."""

  adaptive_uniform_ratio: float = 0.6
  """Fraction of resets drawn uniformly (length-proportional over all bins),
  ignoring failures. Failure-driven sampling gets the remaining (1 - ratio).
  0 = pure failure-driven, 1 = uniform."""

  adaptive_max_prob_ratio: float = 20.0
  """Cap on any bin's probability, as a multiple of its uniform share. Prevents a
  single (clip, chunk) from taking over. Set very large (e.g. 1e9) to disable."""

  adaptive_bin_seconds: float = 1.0
  """Bin (chunk) duration in seconds. Smaller = finer targeting but noisier
  failure statistics."""

  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  """'adaptive': failure-weighted bins mixed with uniform. 'uniform': random clip
  and frame. 'start': first frame of a random clip (used for play/eval)."""

  # --- Fixed (stand-still) commands ---

  rel_fixed_command: float = 0.0
  """Fraction of envs that hold the first frame of their clip with zeroed velocity
  references. 0 = disabled (the moving_envs path becomes a no-op)."""

  max_timesteps_fixed: int = 250
  """Steps a fixed env holds its pose before being resampled
  (250 steps = 5 s at dt 0.02)."""

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    """'ghost': translucent reference robot. 'frames': coordinate frames on bodies."""

    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)
    """RGBA color of the ghost robot."""

  viz: VizCfg = field(default_factory=VizCfg)
  """Debug visualization settings."""

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)


class OnlineMotion:
  """Drop-in for MotionLoader: joint data and segment metadata live in memory,
  while body data is produced on demand by OnlineCsvToNpz."""

  def __init__(self, source: OnlineCsvToNpz, anchor_index: int, step_dt: float, anchor_is_root: float = True):
    self.source = source
    csv_lib = source.motion_csv

    self.joint_pos = csv_lib.joint_pos
    self.joint_vel = csv_lib.joint_vel
    self.segment_start_idx = csv_lib.segment_start_idx
    self.segment_length = csv_lib.segment_length
    self.segment_end_idx = csv_lib.segment_end_idx
    self.num_trajectories = csv_lib.num_trajectories
    self.segment_names = csv_lib.segment_names
    self.time_step_total = csv_lib.time_step_total
    self.trajectory_probs = csv_lib.trajectory_probs

    if anchor_is_root:
        lin_v = csv_lib.root_lin_vel
        ang_v = csv_lib.root_ang_vel
    else:
      # One-time pass to get anchor-body accelerations (same as MotionLoader).
      # Only the anchor is kept, so memory stays small.
      total = self.joint_pos.shape[0]
      B = source.batch_size
      lin_v, ang_v = [], []
      with torch.no_grad():
        for s in range(0, total, B):
          idx = torch.arange(s, min(s + B, total), device=source.device)
          k = idx.numel()
          if k < B:
            idx = torch.cat([idx, idx[-1:].expand(B - k)])
          out = source._retarget_batch(idx)
          lin_v.append(out["body_lin_vel_w"][:k, anchor_index])
          ang_v.append(out["body_ang_vel_w"][:k, anchor_index])
      lin_v = torch.cat(lin_v)
      ang_v = torch.cat(ang_v)

    lin_acc = torch.zeros_like(lin_v)
    ang_acc = torch.zeros_like(ang_v)
    for s, e in zip(self.segment_start_idx.tolist(), self.segment_end_idx.tolist()):
      if e - s >= 2:
        lin_acc[s:e] = torch.gradient(lin_v[s:e], spacing=step_dt, dim=0)[0]
        ang_acc[s:e] = torch.gradient(ang_v[s:e], spacing=step_dt, dim=0)[0]
    self.anchor_lin_acc_w, self.anchor_ang_acc_w = lin_acc, ang_acc


class OnlineMotionCommand(MotionCommand):
  cfg: OnlineMotionCommandCfg

  def _load_motion(self):
    cfg = self.cfg
    # One library frame per env step, so the output rate must be 1 / step_dt.
    output_fps = cfg.output_fps or 1.0 / self._env.step_dt

    self._cache: dict[str, torch.Tensor] | None = None
    self._cache_ts: torch.Tensor | None = None

    source = OnlineCsvToNpz(
      robot_type=cfg.robot_type,
      motion_dir=cfg.motion_dir,
      body_names=cfg.body_names,
      batch_size=self.num_envs,
      input_fps=cfg.input_fps,
      output_fps=output_fps,
      device=self.device,
      sanity=cfg.sanity,
      max_frames=cfg.max_frames,
    )
    return OnlineMotion(source, self.motion_anchor_body_index, self._env.step_dt)

  def _frame(self) -> dict[str, torch.Tensor]:
    """Body data for the current time_steps, recomputed only when they change."""
    ts = self.time_steps
    if self._cache is None or not torch.equal(ts, self._cache_ts):
      self._cache = self.motion.source.retarget(ts)   # one batched mjw.forward
      self._cache_ts = ts.clone()
    return self._cache

  # --- Body quantities now come from the on-the-fly cache ---------------

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._frame()["body_pos_w"] + self._env.scene.env_origins[:, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._frame()["body_quat_w"]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._frame()["body_lin_vel_w"] * self.moving_envs[:, None, None]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._frame()["body_ang_vel_w"] * self.moving_envs[:, None, None]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self._frame()["body_pos_w"][:, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self._frame()["body_quat_w"][:, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return (
      self._frame()["body_lin_vel_w"][:, self.motion_anchor_body_index]
      * self.moving_envs.unsqueeze(-1)
    )

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return (
      self._frame()["body_ang_vel_w"][:, self.motion_anchor_body_index]
      * self.moving_envs.unsqueeze(-1)
    )


@dataclass(kw_only=True)
class OnlineMotionCommandCfg(MotionCommandCfg):
  robot_type: str = "Kangaroo"
  """Key into ROBOT_XML_DICT, used to build the kinematics model."""

  input_fps: float = 30.0
  """Frame rate of the CSV files."""

  output_fps: float | None = None
  """Resampling rate. None = 1 / env.step_dt (required for 1 frame per step)."""

  max_frames: int = 3_000_000
  """Capacity of the preallocated motion library (frames). Loading stops once
  the next clip would not fit."""

  sanity: SanityCheckCfg = SanityCheckCfg()
  # motion_dir now points at a directory of .csv files.

  def build(self, env: ManagerBasedRlEnv) -> OnlineMotionCommand:
    return OnlineMotionCommand(self, env)