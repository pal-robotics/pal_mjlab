"""SuperMimic's world model (paper Section 3.2, Table A.4) for the heading-invariant state.

    ŝ_{t+1}, Δpose = f_ω(ŝ_t, a_t, ξ)    dynamics: MLP 2048-2048-2048 on normalized inputs
    d̂_{t+1} = d_ω(ŝ_{t+1}, g_{t+1})      termination head: failure probability (MLP 64-64)

The dynamics predict the normalized change of the state (D4) and the root's planar
displacement and yaw change in its heading frame, integrated into the world pose. Reward
and observations are not learned: they are computed from the predicted state with the
env's own functions (D5). The termination head sees the failure rule's two errors: the
predicted root height and tilt against the reference's (D39).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..kinematics import matrix_from_rot6d, quat_from_matrix
from ..mdp.tracking import projected_gravity_z
from ..state import integrate_pose, pose_delta


def mlp(
  inp: int, hidden: tuple[int, ...], out: int, activation: str = "elu"
) -> nn.Sequential:
  act = {"elu": nn.ELU, "silu": nn.SiLU, "relu": nn.ReLU}[activation]
  layers: list[nn.Module] = []
  for h in hidden:
    layers += [nn.Linear(inp, h), act()]
    inp = h
  layers.append(nn.Linear(inp, out))
  return nn.Sequential(*layers)


@dataclass
class WorldModelCfg:
  state_dim: int = 77
  action_dim: int = 28
  context_dim: int = 32
  hidden: tuple[int, ...] = (2048, 2048, 2048)
  termination_hidden: tuple[int, ...] = (64, 64)
  """The failure head sees the failure rule's two errors: root height and gravity-z
  against the reference (D39, as the SuperMimic-Reproduced code)."""
  activation: str = "elu"
  pose_scale: tuple[float, float] = (0.1, 0.1)
  """Loss scales [m, rad] for the integrated root position and yaw errors."""


class WorldModel(nn.Module):
  def __init__(self, cfg: WorldModelCfg) -> None:
    super().__init__()
    self.cfg = cfg
    s, a, c = cfg.state_dim, cfg.action_dim, cfg.context_dim
    self.dynamics = mlp(s + a + c, cfg.hidden, s + 3, cfg.activation)
    self.termination = mlp(2, cfg.termination_hidden, 1, cfg.activation)
    for name, dim in (("in", s + a + c), ("out", s + 3), ("state", s)):
      self.register_buffer(f"{name}_mean", torch.zeros(dim))
      self.register_buffer(f"{name}_std", torch.ones(dim))
    self.register_buffer("norm_count", torch.zeros((), dtype=torch.long))

  def _norm_data(self, state, action, context, next_state, delta_pose):
    inputs = torch.cat([state, action, context], dim=-1)
    outputs = torch.cat([next_state - state, delta_pose], dim=-1)
    return (("in", inputs), ("out", outputs), ("state", state))

  @torch.no_grad()
  def update_normalization(
    self,
    state: torch.Tensor,
    action: torch.Tensor,
    context: torch.Tensor,
    next_state: torch.Tensor,
    delta_pose: torch.Tensor,
  ) -> None:
    """Running mean/std over every transition seen so far (online, D4)."""
    n, seen = state.shape[0], int(self.norm_count)
    total = seen + n
    for name, x in self._norm_data(state, action, context, next_state, delta_pose):
      mean, std = getattr(self, f"{name}_mean"), getattr(self, f"{name}_std")
      delta = x.mean(0) - mean
      var = (seen * std.square() + n * x.var(0, unbiased=False)) / total
      var += delta.square() * seen * n / total**2
      mean += delta * n / total
      std.copy_(var.sqrt().clamp_min(1e-3))
    self.norm_count.fill_(total)

  def step(
    self,
    state: torch.Tensor,
    pose: torch.Tensor,
    action: torch.Tensor,
    context: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """One control step: (state, world pose (x, y, yaw)) -> next (state, pose)."""
    x = (torch.cat([state, action, context], dim=-1) - self.in_mean) / self.in_std
    y = self.dynamics(x) * self.out_std + self.out_mean
    s = self.cfg.state_dim
    next_state = state + y[:, :s]
    return next_state, integrate_pose(pose, y[:, s:])

  def termination_logit(
    self, state: torch.Tensor, ref_root: torch.Tensor
  ) -> torch.Tensor:
    """Failure logit for a (predicted) state; ``ref_root`` = (reference root height, gz)."""
    gz = projected_gravity_z(quat_from_matrix(matrix_from_rot6d(state[:, 1:7])))
    x = torch.stack([ref_root[:, 0] - state[:, 0], ref_root[:, 1] - gz], dim=-1)
    return self.termination(x).squeeze(-1)


def window_loss(
  model: WorldModel,
  states: torch.Tensor,
  poses: torch.Tensor,
  actions: torch.Tensor,
  context: torch.Tensor,
  ref_root: torch.Tensor,
  failure: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
  """Eq. 1: roll the model out on logged actions from the logged first state.

  states (B, k+1, S), poses (B, k+1, 3), actions (B, k, A), context (B, C),
  ref_root (B, k, 2) for the states 1..k, failure (B, k) labels of the transitions.
  State errors are normalized per dimension by the state's std; the integrated pose by
  ``cfg.pose_scale``.
  """
  k = actions.shape[1]
  pos_scale, yaw_scale = model.cfg.pose_scale
  s, p = states[:, 0], poses[:, 0]
  state_err = pose_err = term = 0.0
  for j in range(k):
    s, p = model.step(s, p, actions[:, j], context)
    state_err = state_err + (((s - states[:, j + 1]) / model.state_std) ** 2).mean()
    d = pose_delta(poses[:, j + 1], p)
    pose_err = (
      pose_err
      + (
        (d[:, :2] / pos_scale).square().sum(-1) + (d[:, 2] / yaw_scale).square()
      ).mean()
    )
    logit = model.termination_logit(s, ref_root[:, j])
    term = term + nn.functional.binary_cross_entropy_with_logits(
      logit, failure[:, j].float()
    )
  state_err, pose_err, term = state_err / k, pose_err / k, term / k
  loss = state_err + pose_err + term
  return loss, {
    "state": state_err.detach(),
    "pose": pose_err.detach(),
    "term": term.detach(),
  }
