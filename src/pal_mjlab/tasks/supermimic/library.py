"""Packed motion library (see scripts/bones_to_library.py) on the GPU.

A library folder holds ``qpos.npy`` (N, 7 + nj), ``qvel.npy`` (N, 6 + nj) in MuJoCo layout
(free-joint angular velocity in the body frame) and ``clips.json`` with the clip index.
Body poses are not stored; callers run forward kinematics on the frames they need.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _load(
  file: Path,
  device: str | torch.device,
  rows: np.ndarray | None = None,
  columns: np.ndarray | None = None,
) -> torch.Tensor:
  """Copy a (frames, dim) array, or its ``rows`` and ``columns``, to the device in chunks: the
  full library is ~14 GB, more than the free host RAM of some machines."""
  arr = np.load(file, mmap_mode="r")
  if rows is not None:
    arr = arr[rows]  # reads only those rows of the memmap
  dim = arr.shape[1] if columns is None else len(columns)
  out = torch.empty((len(arr), dim), dtype=torch.float32, device=device)
  chunk = 1 << 22
  for i in range(0, len(arr), chunk):
    part = np.array(arr[i : i + chunk])
    out[i : i + chunk] = torch.from_numpy(part if columns is None else part[:, columns])
  return out


class MotionLibrary:
  def __init__(
    self,
    path: str | Path,
    device: str | torch.device = "cpu",
    clips: tuple[str, ...] = (),
    joints: list[int] | None = None,
  ) -> None:
    """``clips``: only these clips of the library, by name (e.g. to play chosen motions of a
    downloaded library); empty for all of them. ``joints``: only these joints in ``qpos`` and
    ``qvel`` (indices into ``joint_names``); ``full_state`` reads the others from the files."""
    path = Path(path).expanduser()
    if not (path / "clips.json").is_file():
      raise FileNotFoundError(
        f"No motion library at {path}: pass --env.commands.motion.library-path <folder>, "
        "a folder with clips.json, qpos.npy and qvel.npy (bones_to_library.py packs one)."
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

    self._rows, self._device = rows, device
    self._full_files = None
    qpos_columns = qvel_columns = None
    if joints is not None:
      kept = np.asarray(joints)
      qpos_columns = np.concatenate([np.arange(7), 7 + kept])
      qvel_columns = np.concatenate([np.arange(6), 6 + kept])
      self._full_files = (
        np.load(path / "qpos.npy", mmap_mode="r"),
        np.load(path / "qvel.npy", mmap_mode="r"),
      )
    self.qpos = _load(path / "qpos.npy", device, rows, qpos_columns)
    self.qvel = _load(path / "qvel.npy", device, rows, qvel_columns)
    self.clip_start = torch.tensor([c["start"] for c in clips], device=device)
    self.clip_length = torch.tensor([c["length"] for c in clips], device=device)
    self.clip_end = self.clip_start + self.clip_length  # exclusive
    self.clip_category = torch.tensor(
      [cat_index[c["category"]] for c in clips], device=device
    )
    assert self.qpos.shape[0] == int(self.clip_length.sum())

  def full_state(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """qpos and qvel of every joint at these frames, also the joints left out of ``qpos``."""
    if self._full_files is None:
      return self.qpos[frames], self.qvel[frames]
    rows = frames.cpu().numpy()
    if self._rows is not None:
      rows = self._rows[rows]
    qpos, qvel = (
      torch.from_numpy(np.array(f[rows])).to(self._device) for f in self._full_files
    )
    return qpos, qvel

  @property
  def num_clips(self) -> int:
    return len(self.clip_names)

  @property
  def num_frames(self) -> int:
    return self.qpos.shape[0]

  @property
  def hours(self) -> float:
    return self.num_frames / self.fps / 3600.0
