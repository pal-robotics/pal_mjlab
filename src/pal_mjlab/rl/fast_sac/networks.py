"""FastSAC neural network architectures.

Actor (tanh-squashed Gaussian policy) and distributional critic (C51)
networks ported from holosoma's FastSAC implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from rsl_rl.models import MLPModel
from tensordict import TensorDict
from copy import deepcopy


class _OnnxMLPModel(nn.Module):
  """Exportable MLP model for ONNX."""

  is_recurrent: bool = False

  def __init__(self, model: MLPModel, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = deepcopy(model.obs_normalizer)
    self.mlp = deepcopy(model.mlp)
    self.state_dependent_std = model.state_dependent_std
    self.input_size = model.obs_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.obs_normalizer(x)
    out = self.mlp(x)
    if self.state_dependent_std:
      return out[..., 0, :]
    return out

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
  
class Actor(MLPModel):
  """Tanh-squashed Gaussian policy with halving hidden dimensions."""

  def __init__(
    self,
    n_obs: int,
    n_act: int,
    obs_g : dict[str, list[str]], 
    obs_d : TensorDict,
    hidden_dim: int = 512,
    log_std_min: float = -5.0,
    log_std_max: float = 0.0,
    use_tanh: bool = True,
    use_layer_norm: bool = True,
    action_scale: torch.Tensor | None = None,
    action_bias: torch.Tensor | None = None,
    device: torch.device | str | None = None,
  ):
    
    #def to_MLPModel(self,):

    super().__init__(
      obs = obs_d,
      obs_groups = obs_g,
      obs_set = "actor",
      output_dim = n_act,
      hidden_dims = (hidden_dim,) * 3,
      activation =  "swish",
      obs_normalization = True,
    )
    self.n_act = n_act
    self.n_obs = n_obs
    self.log_std_min = log_std_min
    self.log_std_max = log_std_max
    self.use_tanh = use_tanh
    self.hidden_dim = hidden_dim

    def _ln(dim: int) -> nn.Module:
      return nn.LayerNorm(dim, device=device) if use_layer_norm else nn.Identity()

    self.net = nn.Sequential(
      nn.Linear(n_obs, hidden_dim, device=device),
      _ln(hidden_dim),
      nn.SiLU(),
      nn.Linear(hidden_dim, hidden_dim // 2, device=device),
      _ln(hidden_dim // 2),
      nn.SiLU(),
      nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
      _ln(hidden_dim // 4),
      nn.SiLU(),
    )
    self.fc_mu = nn.Linear(hidden_dim // 4, n_act, device=device)
    self.fc_logstd = nn.Linear(hidden_dim // 4, n_act, device=device)

    # Zero-init output heads for stable initial policy
    nn.init.constant_(self.fc_mu.weight, 0.0)
    nn.init.constant_(self.fc_mu.bias, 0.0)
    nn.init.constant_(self.fc_logstd.weight, 0.0)
    nn.init.constant_(self.fc_logstd.bias, 0.0)

    if action_scale is not None:
      self.register_buffer("action_scale", action_scale.to(device))
    else:
      self.register_buffer("action_scale", torch.ones(n_act, device=device))

    if action_bias is not None:
      self.register_buffer("action_bias", action_bias.to(device))
    else:
      self.register_buffer("action_bias", torch.zeros(n_act, device=device))

  # -- public property expected by the runner logger --
  @property
  def output_std(self) -> torch.Tensor:
    """Return a scalar tensor representing 'typical' exploration noise.

    The runner logger calls ``self.alg.get_policy().output_std``.
    For SAC the noise is state-dependent, so we just return a
    reasonable proxy (the midpoint of the allowed log-std range
    exponentiated).
    """
    mid = 0.5 * (self.log_std_min + self.log_std_max)
    return torch.tensor(mid).exp()

  def forward(
    self, obs: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = self.net(obs)
    mean = self.fc_mu(x)
    log_std = self.fc_logstd(x)
    # Squash log_std into [log_std_min, log_std_max]
    log_std = torch.tanh(log_std)
    log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
      log_std + 1
    )

    if self.use_tanh:
      action = torch.tanh(mean) * self.action_scale + self.action_bias
    else:
      action = mean

    return action, mean, log_std

  def get_actions_and_log_probs(
    self, obs: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    _, mean, log_std = self(obs)
    std = log_std.exp()
    dist = torch.distributions.Normal(mean, std)
    raw_action = dist.rsample()

    if self.use_tanh:
      tanh_action = torch.tanh(raw_action)
      action = tanh_action * self.action_scale + self.action_bias
      log_prob = dist.log_prob(raw_action)
      log_prob -= torch.log(1 - tanh_action.pow(2) + 1e-6)
      log_prob -= torch.log(self.action_scale + 1e-6)
    else:
      action = raw_action
      log_prob = dist.log_prob(raw_action)

    log_prob = log_prob.sum(1)
    return action, log_prob

  @torch.no_grad()
  def explore(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
    _, mean, log_std = self(obs)
    if deterministic:
      if self.use_tanh:
        return torch.tanh(mean) * self.action_scale + self.action_bias
      return mean

    std = log_std.exp()
    dist = torch.distributions.Normal(mean, std)
    raw_action = dist.rsample()

    if self.use_tanh:
      return torch.tanh(raw_action) * self.action_scale + self.action_bias
    return raw_action
  
  def as_onnx(self, verbose: bool) -> nn.Module:
      """Return a version of the model compatible with ONNX export."""
      return _OnnxMLPModel(self, verbose)


class DistributionalQNetwork(nn.Module):
  """Single distributional (C51) Q-network."""

  def __init__(
    self,
    n_obs: int,
    n_act: int,
    num_atoms: int = 101,
    v_min: float = -20.0,
    v_max: float = 20.0,
    hidden_dim: int = 768,
    use_layer_norm: bool = True,
    device: torch.device | str | None = None,
  ):
    super().__init__()
    self.v_min = v_min
    self.v_max = v_max
    self.num_atoms = num_atoms

    def _ln(dim: int) -> nn.Module:
      return nn.LayerNorm(dim, device=device) if use_layer_norm else nn.Identity()

    self.net = nn.Sequential(
      nn.Linear(n_obs + n_act, hidden_dim, device=device),
      _ln(hidden_dim),
      nn.SiLU(),
      nn.Linear(hidden_dim, hidden_dim // 2, device=device),
      _ln(hidden_dim // 2),
      nn.SiLU(),
      nn.Linear(hidden_dim // 2, hidden_dim // 4, device=device),
      _ln(hidden_dim // 4),
      nn.SiLU(),
      nn.Linear(hidden_dim // 4, num_atoms, device=device),
    )

  def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return self.net(torch.cat([obs, actions], dim=1))

  def projection(
    self,
    obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    bootstrap: torch.Tensor,
    discount: torch.Tensor,
    q_support: torch.Tensor,
    device: torch.device,
  ) -> torch.Tensor:
    """C51 categorical projection of the target distribution."""
    delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
    batch_size = rewards.shape[0]

    target_z = (
      rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount.unsqueeze(1) * q_support
    )
    target_z = target_z.clamp(self.v_min, self.v_max)
    b = (target_z - self.v_min) / delta_z
    lower = torch.floor(b).long()
    upper = torch.ceil(b).long()

    # Handle the case where b is exactly an integer
    is_integer = upper == lower
    lower_mask = torch.logical_and(lower > 0, is_integer)
    upper_mask = torch.logical_and(lower == 0, is_integer)
    lower = torch.where(lower_mask, lower - 1, lower)
    upper = torch.where(upper_mask, upper + 1, upper)

    next_dist = F.softmax(self(obs, actions), dim=1)
    proj_dist = torch.zeros_like(next_dist)
    offset = (
      torch.linspace(
        0,
        (batch_size - 1) * self.num_atoms,
        batch_size,
        device=device,
      )
      .unsqueeze(1)
      .expand(batch_size, self.num_atoms)
      .long()
    )

    lower_indices = (lower + offset).view(-1)
    upper_indices = (upper + offset).view(-1)
    max_index = proj_dist.numel() - 1
    lower_indices = torch.clamp(lower_indices, 0, max_index)
    upper_indices = torch.clamp(upper_indices, 0, max_index)

    proj_dist.view(-1).index_add_(
      0,
      lower_indices,
      (next_dist * (upper.float() - b)).view(-1),
    )
    proj_dist.view(-1).index_add_(
      0,
      upper_indices,
      (next_dist * (b - lower.float())).view(-1),
    )
    return proj_dist


class Critic(nn.Module):
  """Ensemble of distributional Q-networks."""

  def __init__(
    self,
    n_obs: int,
    n_act: int,
    num_atoms: int = 101,
    v_min: float = -20.0,
    v_max: float = 20.0,
    hidden_dim: int = 768,
    use_layer_norm: bool = True,
    num_q_networks: int = 2,
    device: torch.device | str | None = None,
  ):
    super().__init__()
    self.num_atoms = num_atoms
    self.qnets = nn.ModuleList(
      [
        DistributionalQNetwork(
          n_obs=n_obs,
          n_act=n_act,
          num_atoms=num_atoms,
          v_min=v_min,
          v_max=v_max,
          hidden_dim=hidden_dim,
          use_layer_norm=use_layer_norm,
          device=device,
        )
        for _ in range(num_q_networks)
      ]
    )
    self.register_buffer(
      "q_support",
      torch.linspace(v_min, v_max, num_atoms, device=device),
    )

  def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Return stacked logits: ``(num_q, batch, num_atoms)``."""
    return torch.stack([qnet(obs, actions) for qnet in self.qnets], dim=0)

  def projection(
    self,
    obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    bootstrap: torch.Tensor,
    discount: torch.Tensor,
  ) -> torch.Tensor:
    """Stacked projected distributions: ``(num_q, batch, atoms)``."""
    device = self.q_support.device
    return torch.stack(
      [
        qnet.projection(
          obs,
          actions,
          rewards,
          bootstrap,
          discount,
          self.q_support,
          device,
        )
        for qnet in self.qnets
      ],
      dim=0,
    )

  def get_value(self, probs: torch.Tensor) -> torch.Tensor:
    """Expected value from probability distribution over support."""
    return torch.sum(probs * self.q_support, dim=-1)