"""Pack the Kangaroo BONES-SEED retarget into one motion library for SuperMimic.

Input: the GMR pickles of ``kangaroo_bones_seed`` (``root_pos`` (T, 3), ``root_rot`` (T, 4)
xyzw, ``dof_pos`` (T, 32) in the ``kangaroo_grippers`` qpos order, 50 Hz) and its
``manifest.parquet`` (``final_kept`` selects the clips that passed the paper's filters and a
manual review; ``review_excluded`` marks the clips the review removed).

Output folder (all clips concatenated along time):
  qpos.npy   (N, 39) float32  root pos, root quat (wxyz), 32 joints
  qvel.npy   (N, 38) float32  MuJoCo free-joint convention: root linear velocity (world),
                              root angular velocity (body frame), 32 joint velocities
  clips.json {"fps", "joint_names", "clips": [{"name", "category", "start", "length"}]}

Body poses are not stored: they come from forward kinematics on the fly, which keeps a
frame at ~308 bytes (14 GB for the full 45M frames instead of ~44 GB with body data).

Subsets:
  smoke  ~15 min of walk/jog/run clips (Basic Locomotion Neutral)
  dev    ~2.5 h stratified by category (like LAFAN1's 2.45 h in the paper)
  <h>    ~h hours stratified by category, e.g. "10" or "10h"; nested: a smaller subset is
         contained in a larger one (same per-category clip order), and dev = "2.5h"
  full   every ``final_kept`` clip
Subsets are drawn before the reviewed clips are removed, so a review only takes those clips out.

  python -m pal_mjlab.scripts.bones_to_library --subset 10h --data-dir path/to/data
"""

from __future__ import annotations

import json
import os
import pickle
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import tyro

DATA_DIR = Path("~/Projects/supermimic/kangaroo_bones_seed/dataset").expanduser()
FPS = 50

SUBSET_HOURS = {"smoke": 0.25, "dev": 2.5}


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  aw, ax, ay, az = np.moveaxis(a, -1, 0)
  bw, bx, by, bz = np.moveaxis(b, -1, 0)
  return np.stack(
    [
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ],
    axis=-1,
  )


def _quat_conj(q: np.ndarray) -> np.ndarray:
  return q * np.array([1.0, -1.0, -1.0, -1.0])


def _quat_rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate world vectors ``v`` into the frame of ``q`` (wxyz)."""
  vq = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], axis=-1)
  return _quat_mul(_quat_mul(_quat_conj(q), vq), q)[..., 1:]


def _quat_log(q: np.ndarray) -> np.ndarray:
  """Rotation vector of a unit quaternion (shortest path)."""
  q = np.where(q[..., :1] < 0, -q, q)
  s = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
  angle = 2.0 * np.arctan2(s, q[..., :1])
  scale = np.where(s > 1e-9, angle / np.maximum(s, 1e-12), 2.0)
  return q[..., 1:] * scale


def clip_qpos_qvel(
  root_pos: np.ndarray, root_quat_wxyz: np.ndarray, dof_pos: np.ndarray, fps: float
) -> tuple[np.ndarray, np.ndarray]:
  """qpos (T, 7 + nj) and finite-difference qvel (T, 6 + nj) of one clip, float64."""
  dt = 1.0 / fps
  quat = root_quat_wxyz / np.linalg.norm(root_quat_wxyz, axis=-1, keepdims=True)
  # Hemisphere continuity so neighbouring quaternions are close.
  for t in range(1, len(quat)):
    if np.dot(quat[t], quat[t - 1]) < 0:
      quat[t] = -quat[t]

  lin_vel = np.gradient(root_pos, dt, axis=0)
  joint_vel = np.gradient(dof_pos, dt, axis=0)

  # Angular velocity: central differences of the world-frame rotation increment,
  # one-sided at the ends, expressed in the body frame (MuJoCo free-joint qvel).
  prev = np.concatenate([quat[:1], quat[:-2], quat[-2:-1]])
  nxt = np.concatenate([quat[1:2], quat[2:], quat[-1:]])
  span = np.full((len(quat), 1), 2.0 * dt)
  span[0] = span[-1] = dt
  omega_w = _quat_log(_quat_mul(nxt, _quat_conj(prev))) / span
  omega_b = _quat_rotate_inv(quat, omega_w)

  qpos = np.concatenate([root_pos, quat, dof_pos], axis=-1)
  qvel = np.concatenate([lin_vel, omega_b, joint_vel], axis=-1)
  return qpos, qvel


def load_manifest(data_dir: Path) -> dict[str, list]:
  """The clips that passed the automatic filters; ``excluded`` marks those removed later by
  manual review (``final_kept`` false, ``review_excluded`` true in the manifest).

  Subsets are drawn from all of them and the reviewed clips are dropped afterwards
  (``build_library``), so a review removes clips without reshuffling the others: ``dev``
  stays the same clips minus the reviewed ones.
  """
  path = data_dir / "manifest.parquet"
  cols = ["clip", "category", "move_name", "g1_n_frames", "kangaroo_path", "final_kept"]
  reviewed = "review_excluded" in pq.read_schema(path).names
  table = pq.read_table(path, columns=cols + ["review_excluded"] * reviewed).to_pydict()
  excluded = table.get("review_excluded") or [False] * len(table["clip"])
  keep = [
    i
    for i, (k, x) in enumerate(zip(table["final_kept"], excluded, strict=True))
    if k or x
  ]
  out = {c: [table[c][i] for i in keep] for c in cols if c != "final_kept"}
  out["excluded"] = [bool(excluded[i]) for i in keep]
  return out


def subset_name(subset: str) -> str:
  """Canonical subset name: 'smoke', 'dev', 'full' or hours as '<h>h' ('10' -> '10h')."""
  if subset in ("smoke", "dev", "full"):
    return subset
  try:
    hours = float(subset.removesuffix("h"))
  except ValueError:
    msg = f"subset {subset!r}: use smoke, dev, full or hours (10, 10h)"
    raise ValueError(msg) from None
  if hours <= 0:
    raise ValueError(f"subset {subset!r}: hours must be positive")
  return f"{hours:g}h"


def select_clips(manifest: dict[str, list], subset: str, seed: int = 0) -> list[int]:
  """Row indices of the manifest for a subset (deterministic for a seed)."""
  n = len(manifest["clip"])
  subset = subset_name(subset)
  if subset == "full":
    return list(range(n))
  rng = np.random.default_rng(seed)
  hours = np.asarray(manifest["g1_n_frames"], dtype=np.float64) / FPS / 3600.0
  target = SUBSET_HOURS.get(subset) or float(subset.removesuffix("h"))

  def take(candidates: list[int], budget: float) -> list[int]:
    picked, total = [], 0.0
    for i in rng.permutation(candidates):
      if total >= budget:
        break
      picked.append(int(i))
      total += hours[i]
    return picked

  if subset == "smoke":
    cand = [
      i
      for i in range(n)
      if manifest["category"][i] == "Basic Locomotion Neutral"
      and re.search(r"(^|_)(walk|jog|run)", manifest["move_name"][i])
    ]
    return sorted(take(cand, target))

  # dev and hours: proportional to each category's share of the hours, at least one
  # clip each.
  cats = sorted(set(manifest["category"]))
  total_h = hours.sum()
  picked: list[int] = []
  for c in cats:
    cand = [i for i in range(n) if manifest["category"][i] == c]
    share = target * hours[cand].sum() / total_h
    picked += take(cand, max(share, 1e-9))
  return sorted(picked)


def build_library(
  data_dir: Path,
  subset: str = "dev",
  out_dir: Path | None = None,
  seed: int = 0,
  workers: int = 16,
) -> Path:
  """Pack a subset into ``out_dir`` (default ``<data_dir>/libraries/<subset>``).

  Written to a temporary folder and renamed when complete, so a crash never leaves a
  partial library behind.
  """
  subset = subset_name(subset)
  out = out_dir or data_dir / "libraries" / subset
  final, out = out, out.with_name(f"{out.name}.tmp-{os.getpid()}")
  manifest = load_manifest(data_dir)
  selected = select_clips(manifest, subset, seed)
  rows = [i for i in selected if not manifest["excluded"][i]]
  print(f"[{subset}] {len(rows)} clips ({len(selected) - len(rows)} removed by review)")

  def load(i: int) -> dict:
    with open(data_dir / manifest["kangaroo_path"][i], "rb") as f:
      return pickle.load(f)

  out.mkdir(parents=True, exist_ok=True)
  # The manifest frame count sizes the memmaps; convert() asserts it per clip.
  joint_names = list(load(rows[0])["joint_names"])
  lengths = [int(manifest["g1_n_frames"][i]) for i in rows]
  starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)
  total = int(np.sum(lengths))
  nq, nv = 7 + len(joint_names), 6 + len(joint_names)
  qpos_mm = np.lib.format.open_memmap(out / "qpos.npy", "w+", np.float32, (total, nq))
  qvel_mm = np.lib.format.open_memmap(out / "qvel.npy", "w+", np.float32, (total, nv))

  def convert(k: int) -> None:
    d = load(rows[k])
    assert list(d["joint_names"]) == joint_names, manifest["clip"][rows[k]]
    assert int(d["fps"]) == FPS, manifest["clip"][rows[k]]
    assert len(d["dof_pos"]) == lengths[k], manifest["clip"][rows[k]]
    rot = np.asarray(d["root_rot"], np.float64)[:, [3, 0, 1, 2]]  # xyzw -> wxyz
    qpos, qvel = clip_qpos_qvel(
      np.asarray(d["root_pos"], np.float64),
      rot,
      np.asarray(d["dof_pos"], np.float64),
      FPS,
    )
    s, n = starts[k], lengths[k]
    qpos_mm[s : s + n] = qpos
    qvel_mm[s : s + n] = qvel

  with ThreadPoolExecutor(workers) as pool:
    list(pool.map(convert, range(len(rows))))
  qpos_mm.flush()
  qvel_mm.flush()

  meta = {
    "fps": FPS,
    "joint_names": joint_names,
    "source": str(data_dir),
    "subset": subset,
    "seed": seed,
    "clips": [
      {
        "name": manifest["clip"][i],
        "category": manifest["category"][i],
        "start": int(starts[k]),
        "length": int(lengths[k]),
      }
      for k, i in enumerate(rows)
    ],
  }
  (out / "clips.json").write_text(json.dumps(meta))
  if final.exists():  # built concurrently by another process: keep that one
    for f in out.iterdir():
      f.unlink()
    out.rmdir()
  else:
    out.rename(final)
  hours = total / FPS / 3600
  print(f"[{subset}] {len(rows)} clips, {total} frames, {hours:.2f} h -> {final}")
  return final


def main(
  subset: str = "dev",
  out_dir: Path | None = None,
  data_dir: Path = DATA_DIR,
  seed: int = 0,
  workers: int = 16,
) -> None:
  """``subset``: smoke, dev, full, or a number of hours (10, 10h)."""
  build_library(data_dir.expanduser(), subset, out_dir, seed, workers)


if __name__ == "__main__":
  tyro.cli(main)
