"""SuperMimic policy network as an rsl_rl model (Appendix B/C, Table A.4).

    z = tanh(encoder(goal))          encoder: MLP 2048-1024-512-256, z in R^32
    a = decoder(z, proprio history)  decoder: MLP 2048-1024-512-256

The goal observation group is encoded; every other actor group (the proprioceptive
history) goes straight to the decoder. The policy is deterministic.
"""

from __future__ import annotations

import copy

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization
from tensordict import TensorDict
from torch import nn


class EncoderDecoderModel(MLPModel):
  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (2048, 1024, 512, 256),
    encoder_dims: tuple[int, ...] | list[int] = (2048, 1024, 512, 256),
    latent_dim: int = 32,
    goal_group: str = "goal",
    activation: str = "elu",
    obs_normalization: bool = False,
  ) -> None:
    nn.Module.__init__(self)
    groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
    if goal_group not in groups:
      raise ValueError(
        f"Goal group {goal_group!r} not among the {obs_set} groups {groups}."
      )
    # Goal first, so the encoder input is a prefix of the normalized observation.
    self.obs_groups = [goal_group] + [g for g in groups if g != goal_group]
    self.goal_dim = obs[goal_group].shape[-1]
    self.latent_dim = latent_dim

    self.obs_normalization = obs_normalization
    self.obs_normalizer = (
      EmpiricalNormalization(self.obs_dim) if obs_normalization else nn.Identity()
    )
    self.distribution = None
    self.encoder = MLP(self.goal_dim, latent_dim, encoder_dims, activation)
    self.mlp = MLP(self._get_latent_dim(), output_dim, hidden_dims, activation)

  def _get_latent_dim(self) -> int:
    return self.latent_dim + self.obs_dim - self.goal_dim

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    x = torch.cat([obs[g] for g in self.obs_groups], dim=-1)
    return _encode(self.obs_normalizer(x), self.encoder, self.goal_dim)

  def act(self, x: torch.Tensor) -> torch.Tensor:
    """Action from the concatenated observation (goal first).

    Differentiable in ``x``: SuperMimic's imagination calls it on predicted
    observations.
    """
    return self.mlp(_encode(self.obs_normalizer(x), self.encoder, self.goal_dim))

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _ExportModel(self)


def _encode(x: torch.Tensor, encoder: nn.Module, goal_dim: int) -> torch.Tensor:
  z = torch.tanh(encoder(x[:, :goal_dim]))
  return torch.cat([z, x[:, goal_dim:]], dim=-1)


class _ExportModel(nn.Module):
  """The policy on the concatenated observation (goal first)."""

  def __init__(self, model: EncoderDecoderModel) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.encoder = copy.deepcopy(model.encoder)
    self.mlp = copy.deepcopy(model.mlp)
    self.goal_dim = model.goal_dim
    self.input_size = model.obs_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.mlp(_encode(self.obs_normalizer(x), self.encoder, self.goal_dim))

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
