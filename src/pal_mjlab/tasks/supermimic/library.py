"""Motion library: the clips of a retargeted dataset packed in one folder, on the device.

  clips.json  {"fps", "joint_names", "clips": [{"name", "category", "start", "length"}]}
  qpos.npy    (N, 7 + nj) float32: root position [m], root quaternion (w, x, y, z), and the
              ``joint_names`` positions [rad; m for slides]
  qvel.npy    (N, 6 + nj) float32: root linear velocity (world frame), root angular velocity
              (body frame), joint velocities

The clips are concatenated in time; ``start`` and ``length`` are in frames. Body poses are not
stored: callers run forward kinematics on the frames they need.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _load(
  file: Path, device: str | torch.device, rows: np.ndarray | None = None
) -> torch.Tensor:
  """Copy a (frames, dim) array, or its ``rows``, to the device in chunks: a large library
  does not fit in the host RAM of every machine."""
  arr = np.load(file, mmap_mode="r")
  if rows is not None:
    arr = arr[rows]  # reads only those rows of the memmap
  out = torch.empty(arr.shape, dtype=torch.float32, device=device)
  chunk = 1 << 22
  for i in range(0, len(arr), chunk):
    out[i : i + chunk] = torch.from_numpy(np.array(arr[i : i + chunk]))
  return out


class MotionLibrary:
  def __init__(
    self,
    path: str | Path,
    device: str | torch.device = "cpu",
    clips: tuple[str, ...] = (),
  ) -> None:
    """``clips``: only these clips of the library, by name; empty for all of them."""
    path = Path(path).expanduser()
    if not (path / "clips.json").is_file():
      raise FileNotFoundError(
        f"No motion library (clips.json, qpos.npy, qvel.npy) at {path}. Train with "
        "--env.commands.motion.library-path <folder>; play one with "
        "pal_mjlab.tasks.supermimic.scripts.play --library-path <folder>."
      )
    meta = json.loads((path / "clips.json").read_text())
    self.path = path
    self.fps: float = float(meta["fps"])
    self.joint_names: tuple[str, ...] = tuple(meta["joint_names"])
    rows = None
    if clips:
      by_name = {c["name"]: c for c in meta["clips"]}
      missing = [c for c in clips if c not in by_name]
      if missing:
        raise KeyError(f"{len(missing)} clips not in {path}, e.g. {missing[:3]}")
      chosen = [by_name[c] for c in clips]
      rows = np.concatenate(
        [np.arange(c["start"], c["start"] + c["length"]) for c in chosen]
      )
      starts = np.cumsum([0] + [c["length"] for c in chosen[:-1]])
      meta["clips"] = [
        {**c, "start": int(s)} for c, s in zip(chosen, starts, strict=True)
      ]
    clips = meta["clips"]
    self.clip_names: tuple[str, ...] = tuple(c["name"] for c in clips)
    self.categories: tuple[str, ...] = tuple(sorted({c["category"] for c in clips}))
    cat_index = {c: i for i, c in enumerate(self.categories)}

    self.qpos = _load(path / "qpos.npy", device, rows)
    self.qvel = _load(path / "qvel.npy", device, rows)
    self.clip_start = torch.tensor([c["start"] for c in clips], device=device)
    self.clip_length = torch.tensor([c["length"] for c in clips], device=device)
    self.clip_end = self.clip_start + self.clip_length  # exclusive
    self.clip_category = torch.tensor(
      [cat_index[c["category"]] for c in clips], device=device
    )
    assert self.qpos.shape[0] == int(self.clip_length.sum())

  @property
  def num_clips(self) -> int:
    return len(self.clip_names)
