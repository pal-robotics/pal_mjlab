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


class ReplayBuffer:
  """SuperMimic's FIFO replay buffer (Alg. A.1, D9): the newest ``capacity`` rows per env.

  A row is a state. A transition row also holds the executed (noisy) action, the
  reference clip and target frame it was scored against and its failure label, and leads
  to the next row. When an episode ends, its terminal state is written as an extra row
  without an action, so the next row starts the new episode. A k-step window is then k
  consecutive transition rows plus the row after them, and never spans two episodes.
  """

  def __init__(
    self,
    num_envs: int,
    capacity: int,
    state_dim: int,
    action_dim: int,
    context_dim: int,
    device: str | torch.device,
  ) -> None:
    c, e = capacity, num_envs
    self.capacity, self.num_envs, self.device = capacity, num_envs, device
    self.state = torch.zeros(c, e, state_dim, device=device)
    self.pose = torch.zeros(c, e, 3, device=device)
    self.action = torch.zeros(c, e, action_dim, device=device)
    self.context = torch.zeros(c, e, context_dim, device=device)
    self.clip = torch.zeros(c, e, dtype=torch.long, device=device)
    self.frame = torch.zeros(c, e, dtype=torch.long, device=device)
    self.failure = torch.zeros(c, e, dtype=torch.bool, device=device)
    self.has_action = torch.zeros(c, e, dtype=torch.bool, device=device)
    self.count = torch.zeros(e, dtype=torch.long, device=device)  # rows written
    self._envs = torch.arange(e, device=device)

  def _write(self, envs: torch.Tensor, **fields: torch.Tensor) -> None:
    slot = self.count[envs] % self.capacity
    for name, value in fields.items():
      getattr(self, name)[slot, envs] = value
    self.count[envs] += 1

  @torch.no_grad()
  def add(
    self,
    state: torch.Tensor,
    pose: torch.Tensor,
    action: torch.Tensor,
    context: torch.Tensor,
    clip: torch.Tensor,
    frame: torch.Tensor,
    failure: torch.Tensor,
    done: torch.Tensor,
    next_state: torch.Tensor,
    next_pose: torch.Tensor,
  ) -> None:
    """One env step of all envs; ``next_*`` is only stored where the episode ended."""
    self._write(
      self._envs, state=state, pose=pose, action=action, context=context,
      clip=clip.long(), frame=frame.long(), failure=failure, has_action=torch.ones_like(done),
    )  # fmt: skip
    ids = done.nonzero().squeeze(-1)
    if len(ids):
      self._write(
        ids, state=next_state[ids], pose=next_pose[ids], context=context[ids],
        clip=clip[ids].long(), frame=frame[ids].long(),
        failure=torch.zeros_like(ids, dtype=torch.bool),
        has_action=torch.zeros_like(ids, dtype=torch.bool),
      )  # fmt: skip

  @classmethod
  def from_rollouts(cls, data: RolloutData) -> ReplayBuffer:
    """The offline M2 rollouts as a replay buffer (nothing overwritten)."""
    capacity = data.T + int(data.done.sum(0).max())
    buf = cls(
      data.E, capacity, data.state.shape[-1], data.action.shape[-1],
      data.context.shape[-1], data.state.device,
    )  # fmt: skip
    for t in range(data.T):
      buf.add(
        data.state[t], data.pose[t], data.action[t], data.context[t], data.clip[t],
        data.frame[t], data.failure[t], data.done[t], data.next_state[t],
        data.next_pose[t],
      )  # fmt: skip
    return buf

  # Sampling -----------------------------------------------------------------------

  def _rows(self, envs: torch.Tensor, rows: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """(slot, env) index pairs of absolute row numbers ``rows`` (any shape) of ``envs``."""
    envs = envs.reshape(envs.shape + (1,) * (rows.dim() - envs.dim())).expand_as(rows)
    return rows % self.capacity, envs

  def _sample_rows(
    self,
    n: int,
    lo_offset: int,
    hi_offset: int,
    generator: torch.Generator | None,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """n uniform (env, row) with oldest + lo_offset <= row < count - hi_offset."""
    envs = torch.randint(
      0, self.num_envs, (n,), device=self.device, generator=generator
    )
    oldest = (self.count[envs] - self.capacity).clamp_min(0)
    lo, hi = oldest + lo_offset, self.count[envs] - hi_offset
    span = (hi - lo).clamp_min(0)
    u = torch.rand(n, device=self.device, generator=generator)
    rows = lo + (u * span).long()
    return envs[span > 0], rows[span > 0]

  def sample_windows(
    self, n: int, k: int, generator: torch.Generator | None = None
  ) -> dict[str, torch.Tensor]:
    """n random k-step windows inside episodes (Eq. 1), the layout of ``RolloutData``."""
    envs, rows = [], []
    for _ in range(100):
      if sum(len(r) for r in rows) >= n:
        break
      e, r = self._sample_rows(2 * n, 0, k, generator)
      steps = r[:, None] + torch.arange(k, device=self.device)
      ok = self.has_action[self._rows(e, steps)].all(-1)
      envs.append(e[ok])
      rows.append(r[ok])
    e, r = torch.cat(envs)[:n], torch.cat(rows)[:n]
    if len(r) < n:
      raise RuntimeError(f"only {len(r)} of {n} valid {k}-step windows in the buffer")
    steps = r[:, None] + torch.arange(k + 1, device=self.device)
    idx = self._rows(e, steps)
    first = self._rows(e, r)
    return {
      "states": self.state[idx],
      "poses": self.pose[idx],
      "actions": self.action[idx][:, :k],
      "context": self.context[first],
      "frame": self.frame[idx][:, :k],
      "failure": self.failure[idx][:, :k],
      "clip": self.clip[first],
    }

  def sample_starts(
    self,
    n: int,
    history: int,
    generator: torch.Generator | None = None,
    clips: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    """n imagination starts (transition rows) with their last ``history`` rows.

    History rows before the episode's first row repeat that first row, as the env
    backfills a reset env's observation history. ``last_action`` of a row is the action
    executed in the row before it (zero at an episode start); ``clips`` optionally
    restricts the starts' clips (a boolean mask over library clips).
    """
    envs, rows = [], []
    for _ in range(100):
      if sum(len(r) for r in rows) >= n:
        break
      # Keep history + 1 rows behind the start in the buffer for the last actions.
      e, r = self._sample_rows(2 * n, history + 1, 0, generator)
      ok = self.has_action[self._rows(e, r)]
      if clips is not None:
        ok &= clips[self.clip[self._rows(e, r)]]
      envs.append(e[ok])
      rows.append(r[ok])
    e, r = torch.cat(envs)[:n], torch.cat(rows)[:n]
    if len(r) < n:
      raise RuntimeError(f"only {len(r)} of {n} valid starts in the buffer")
    back = torch.arange(history - 1, -1, -1, device=self.device)
    hist = r[:, None] - back  # (n, history) row numbers, oldest first
    # The start's episode begins after the latest terminal row before it (if any); older
    # history rows repeat the episode's first row.
    before = hist - 1
    terminal = ~self.has_action[self._rows(e, before)]
    first = torch.where(terminal, before, torch.full_like(before, -1)).amax(1) + 1
    hist = torch.maximum(hist, first[:, None])
    prev = self._rows(e, hist - 1)
    last_action = self.action[prev] * self.has_action[prev][..., None]
    idx = self._rows(e, hist)
    start = self._rows(e, r)
    return {
      "state": self.state[start],
      "pose": self.pose[start],
      "context": self.context[start],
      "clip": self.clip[start],
      "frame": self.frame[start],
      "history_state": self.state[idx],
      "history_pose": self.pose[idx],
      "history_last_action": last_action,
    }
