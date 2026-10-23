"""SuperMimic's world model (paper Section 3.2, Table A.4) for the heading-invariant state.

    ŝ_{t+1}, Δpose = f_ω(ŝ_t, a_t, ξ)    dynamics: MLP 2048-2048-2048 on normalized inputs
    d̂_{t+1} = d_ω(ŝ_{t+1}, g_{t+1})      termination head: failure probability

The dynamics predict the normalized change of the state (D4) and the root's planar
displacement and yaw change in its heading frame, integrated into the world pose. Reward
and observations are not learned: they are computed from the predicted state with the
env's own functions (D5). The termination head sees the predicted state plus the
reference root height and tilt, which SuperMimic's failure rule compares against.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..kinematics import matrix_from_rot6d
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
  termination_hidden: tuple[int, ...] = (256, 256)
  activation: str = "elu"
  predict_delta: bool = True
  """Predict the state change (True) or the next state itself (False)."""
  pose_scale: tuple[float, float] = (0.1, 0.1)
  """Loss scales [m, rad] for the integrated root position and yaw errors."""


class WorldModel(nn.Module):
  def __init__(self, cfg: WorldModelCfg) -> None:
    super().__init__()
    self.cfg = cfg
    s, a, c = cfg.state_dim, cfg.action_dim, cfg.context_dim
    self.dynamics = mlp(s + a + c, cfg.hidden, s + 3, cfg.activation)
    self.termination = mlp(s + 2, cfg.termination_hidden, 1, cfg.activation)
    for name, dim in (("in", s + a + c), ("out", s + 3), ("state", s)):
      self.register_buffer(f"{name}_mean", torch.zeros(dim))
      self.register_buffer(f"{name}_std", torch.ones(dim))

  @torch.no_grad()
  def set_normalization(
    self,
    state: torch.Tensor,
    action: torch.Tensor,
    context: torch.Tensor,
    next_state: torch.Tensor,
    delta_pose: torch.Tensor,
  ) -> None:
    """Fix input/output statistics from a sample of transitions (frozen afterwards)."""
    inputs = torch.cat([state, action, context], dim=-1)
    target = next_state - state if self.cfg.predict_delta else next_state
    outputs = torch.cat([target, delta_pose], dim=-1)
    for name, x in (("in", inputs), ("out", outputs), ("state", state)):
      getattr(self, f"{name}_mean").copy_(x.mean(0))
      getattr(self, f"{name}_std").copy_(x.std(0).clamp_min(1e-3))

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
    next_state = state + y[:, :s] if self.cfg.predict_delta else y[:, :s]
    return next_state, integrate_pose(pose, y[:, s:])

  def termination_logit(
    self, state: torch.Tensor, ref_root: torch.Tensor
  ) -> torch.Tensor:
    """Failure logit for a (predicted) state; ``ref_root`` = (reference root height, gz)."""
    x = torch.cat([(state - self.state_mean) / self.state_std, ref_root], dim=-1)
    return self.termination(x).squeeze(-1)


def tilt_angle(state_a: torch.Tensor, state_b: torch.Tensor) -> torch.Tensor:
  """Angle between the root tilts (6D, dims 1:7) of two heading states [rad]."""
  ra, rb = matrix_from_rot6d(state_a[..., 1:7]), matrix_from_rot6d(state_b[..., 1:7])
  cos = ((ra * rb).sum((-1, -2)) - 1.0) / 2.0
  return torch.acos(cos.clamp(-1.0, 1.0))


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
