"""Transition data for the world model: k-step windows inside episodes (paper Eq. 1).

Holds rollouts as (T, E, ·) tensors on the device (the offline M2 dataset written by
``scripts/supermimic/collect_rollouts.py``; later, the online replay buffer). A window of
length k starting at (t, e) uses the states s_t .. s_{t+k} of env e and is valid when no
episode ends before its last transition, so windows never mix episodes (D9).
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import torch

from ..library import MotionLibrary
from ..mdp.tracking import projected_gravity_z
from ..state import heading_pose, heading_state, pose_delta


def holdout_clip_mask(names: tuple[str, ...], fraction: float = 0.1) -> np.ndarray:
  """Deterministic clip split by name hash: True = held out (unseen motions)."""
  buckets = 1000
  return np.array(
    [zlib.crc32(n.encode()) % buckets < fraction * buckets for n in names]
  )


class RolloutData:
  def __init__(self, folders: list[Path], library: MotionLibrary, device: str) -> None:
    arrays: dict[str, list[np.ndarray]] = {}
    self.sources: list[str] = []
    for folder in folders:
      for name in (
        "qpos", "qvel", "next_qpos", "next_qvel", "action", "context",
        "clip", "frame", "reward", "failure", "done",
      ):  # fmt: skip
        arrays.setdefault(name, []).append(np.load(folder / f"{name}.npy"))
      meta = json.loads((folder / "meta.json").read_text())
      self.sources += [Path(meta["checkpoint"]).stem] * arrays["qpos"][-1].shape[1]
    # Chunks share T; stack them along the env axis.
    t = {
      k: torch.from_numpy(np.concatenate(v, axis=1)).to(device)
      for k, v in arrays.items()
    }
    self.T, self.E = t["qpos"].shape[:2]
    self.state, self.pose = self._convert(t.pop("qpos"), t.pop("qvel"))
    self.next_state, self.next_pose = self._convert(
      t.pop("next_qpos"), t.pop("next_qvel")
    )
    self.action = t["action"].float()
    self.context = t["context"].float()
    self.clip = t["clip"].long()
    self.frame = t["frame"].long()
    self.reward = t["reward"].float()
    self.failure = t["failure"].bool()
    self.done = t["done"].bool()
    self.source = np.array(self.sources)
    self.library = library
    # Reference root height and gravity-z of each step's target frame (termination input).
    ref = library.qpos[self.frame]
    self.ref_root = torch.stack(
      [ref[..., 2], projected_gravity_z(ref[..., 3:7])], dim=-1
    )

  def _convert(
    self, qpos: torch.Tensor, qvel: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """(T, E, ·) qpos/qvel -> heading state and world pose, in chunks to bound memory."""
    states, poses = [], []
    for qp, qv in zip(qpos.float().split(50), qvel.float().split(50), strict=True):
      n = qp.shape[0] * qp.shape[1]
      states.append(
        heading_state(qp.reshape(n, -1), qv.reshape(n, -1)).reshape(*qp.shape[:2], -1)
      )
      poses.append(heading_pose(qp.reshape(n, -1)).reshape(*qp.shape[:2], -1))
    return torch.cat(states), torch.cat(poses)

  def valid_starts(
    self, k: int, clips: torch.Tensor | None = None, sources: list[str] | None = None
  ) -> torch.Tensor:
    """Flat indices t * E + e of windows of length k (optionally restricted)."""
    done = self.done.float()
    # Episode ends among transitions t .. t+k-2 make the window invalid.
    csum = torch.cat([torch.zeros(1, self.E, device=done.device), done.cumsum(0)], 0)
    starts = torch.arange(self.T - k + 1, device=done.device)
    ends_inside = csum[starts + k - 1] - csum[starts]
    ok = ends_inside == 0  # (T-k+1, E)
    if clips is not None:
      ok &= clips.to(ok.device)[self.clip[: self.T - k + 1]]
    if sources is not None:
      keep = torch.from_numpy(np.isin(self.source, sources)).to(ok.device)
      ok &= keep[None, :]
    t_idx, e_idx = ok.nonzero(as_tuple=True)
    return t_idx * self.E + e_idx

  def windows(self, starts: torch.Tensor, k: int) -> dict[str, torch.Tensor]:
    """Gather windows: states/poses (B, k+1, ·), per-transition tensors (B, k, ·)."""
    t0, e = starts // self.E, starts % self.E
    steps = t0[:, None] + torch.arange(k, device=starts.device)  # (B, k)
    ee = e[:, None].expand_as(steps)
    states = torch.cat(
      [self.state[steps, ee], self.next_state[steps[:, -1:], ee[:, -1:]]], 1
    )
    poses = torch.cat(
      [self.pose[steps, ee], self.next_pose[steps[:, -1:], ee[:, -1:]]], 1
    )
    return {
      "states": states,
      "poses": poses,
      "actions": self.action[steps, ee],
      "context": self.context[t0, e],
      "ref_root": self.ref_root[steps, ee],
      "failure": self.failure[steps, ee],
      "reward": self.reward[steps, ee],
      "frame": self.frame[steps, ee],
      "clip": self.clip[t0, e],
    }

  def transitions(
    self, n: int, generator: torch.Generator | None = None
  ) -> dict[str, torch.Tensor]:
    """Random single transitions (for normalization statistics)."""
    idx = torch.randint(
      0, self.T * self.E, (n,), device=self.state.device, generator=generator
    )
    t, e = idx // self.E, idx % self.E
    return {
      "state": self.state[t, e],
      "action": self.action[t, e],
      "context": self.context[t, e],
      "next_state": self.next_state[t, e],
      "delta_pose": pose_delta(self.pose[t, e], self.next_pose[t, e]),
    }
