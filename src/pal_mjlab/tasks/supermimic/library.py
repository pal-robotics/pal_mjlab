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
  file: Path, device: str | torch.device, rows: np.ndarray | None = None
) -> torch.Tensor:
  """Copy a (frames, dim) array, or its ``rows``, to the device in chunks: the full library
  is ~14 GB, more than the free host RAM of some machines."""
  arr = np.load(file, mmap_mode="r")
  if rows is not None:
    arr = arr[rows]  # reads only those rows of the memmap
  out = torch.empty(arr.shape, dtype=torch.float32, device=device)
  chunk = 1 << 22
  for i in range(0, len(arr), chunk):
    out[i : i + chunk] = torch.from_numpy(np.array(arr[i : i + chunk]))
  return out


def resolve_library(library_path: str, dataset: str, subset: str) -> Path:
  """The packed library to load: ``library_path`` if set, else the ``subset`` of
  ``dataset`` (the BONES-SEED retarget's ``dataset`` folder: manifest.parquet, motions/,
  libraries/) at ``<dataset>/libraries/<subset>``, packed from the motions on first use.

  Paths may be relative (to the working directory) and may start with ``~``.
  """
  if library_path:
    return Path(library_path).expanduser().resolve()
  if not dataset:
    raise ValueError("Set the motion command's library_path or dataset.")
  from pal_mjlab.scripts.bones_to_library import build_library, subset_name

  data_dir = Path(dataset).expanduser().resolve()
  path = data_dir / "libraries" / subset_name(subset)
  if not (path / "clips.json").exists():
    print(f"[INFO] Packing motion library {path} (first use)", flush=True)
    build_library(data_dir, subset)
  return path


class MotionLibrary:
  def __init__(
    self,
    path: str | Path,
    device: str | torch.device = "cpu",
    clips: tuple[str, ...] = (),
  ) -> None:
    """``clips``: only these clips of the library, by name (e.g. to play chosen motions of a
    downloaded library); empty for all of them."""
    path = Path(path).expanduser()
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

  @property
  def num_frames(self) -> int:
    return self.qpos.shape[0]

  @property
  def hours(self) -> float:
    return self.num_frames / self.fps / 3600.0
