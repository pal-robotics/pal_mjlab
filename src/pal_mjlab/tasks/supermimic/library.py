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


class MotionLibrary:
  def __init__(self, path: str | Path, device: str | torch.device = "cpu") -> None:
    path = Path(path).expanduser()
    meta = json.loads((path / "clips.json").read_text())
    self.path = path
    self.fps: float = float(meta["fps"])
    self.joint_names: tuple[str, ...] = tuple(meta["joint_names"])
    clips = meta["clips"]
    self.clip_names: tuple[str, ...] = tuple(c["name"] for c in clips)
    self.categories: tuple[str, ...] = tuple(sorted({c["category"] for c in clips}))
    cat_index = {c: i for i, c in enumerate(self.categories)}

    self.qpos = torch.from_numpy(np.load(path / "qpos.npy")).to(device)
    self.qvel = torch.from_numpy(np.load(path / "qvel.npy")).to(device)
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
