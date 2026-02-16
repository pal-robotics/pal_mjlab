"""Per-environment replay buffer with n-step return support.

Ported from holosoma's SimpleReplayBuffer.
"""

from __future__ import annotations

import torch
from torch import nn


class ReplayBuffer(nn.Module):
  """Circular per-environment replay buffer with optional n-step returns."""

  def __init__(
    self,
    n_env: int,
    buffer_size: int,
    n_obs: int,
    n_critic_obs: int | None,
    n_act: int,
    n_steps: int = 1,
    gamma: float = 0.99,
    device: torch.device | str | None = None,
  ):
    super().__init__()
    self.n_env = n_env
    self.buffer_size = buffer_size
    self.n_obs = n_obs
    self.n_critic_obs = n_obs if n_critic_obs is None else n_critic_obs
    self.n_act = n_act
    self.gamma = gamma
    self.n_steps = n_steps
    self.device = device

    self.observations = torch.zeros((n_env, buffer_size, n_obs), device=device)
    self.actions = torch.zeros((n_env, buffer_size, n_act), device=device)
    self.rewards = torch.zeros((n_env, buffer_size), device=device)
    self.dones = torch.zeros((n_env, buffer_size), device=device, dtype=torch.long)
    self.truncations = torch.zeros(
      (n_env, buffer_size), device=device, dtype=torch.long
    )
    self.next_observations = torch.zeros((n_env, buffer_size, n_obs), device=device)
    self.critic_observations = torch.zeros(
      (n_env, buffer_size, self.n_critic_obs), device=device
    )
    self.next_critic_observations = torch.zeros(
      (n_env, buffer_size, self.n_critic_obs), device=device
    )
    self.ptr = 0

  def extend(
    self,
    observations: torch.Tensor,
    critic_observations: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    truncations: torch.Tensor,
    next_observations: torch.Tensor,
    next_critic_observations: torch.Tensor,
  ) -> None:
    """Add a batch of transitions (one per env) to the buffer."""
    ptr = self.ptr % self.buffer_size
    self.observations[:, ptr] = observations
    self.critic_observations[:, ptr] = critic_observations
    self.actions[:, ptr] = actions
    self.rewards[:, ptr] = rewards
    self.dones[:, ptr] = dones
    self.truncations[:, ptr] = truncations
    self.next_observations[:, ptr] = next_observations
    self.next_critic_observations[:, ptr] = next_critic_observations
    self.ptr += 1

  @torch.no_grad()
  def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
    """Sample ``n_env * batch_size`` transitions.

    Returns a flat dict with keys: observations, actions, rewards,
    dones, truncations, next_observations, effective_n_steps.
    """
    valid = min(self.buffer_size, self.ptr)

    if self.n_steps == 1:
      return self._sample_1step(batch_size, valid)
    return self._sample_nstep(batch_size, valid)

  def _sample_1step(self, batch_size: int, valid: int) -> dict[str, torch.Tensor]:
    indices = torch.randint(0, valid, (self.n_env, batch_size), device=self.device)
    flat = self.n_env * batch_size

    obs_idx = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
    critic_obs_idx = indices.unsqueeze(-1).expand(-1, -1, self.n_critic_obs)
    act_idx = indices.unsqueeze(-1).expand(-1, -1, self.n_act)

    return {
      "observations": torch.gather(self.observations, 1, obs_idx).reshape(
        flat, self.n_obs
      ),
      "critic_observations": torch.gather(
        self.critic_observations, 1, critic_obs_idx
      ).reshape(flat, self.n_critic_obs),
      "next_observations": torch.gather(self.next_observations, 1, obs_idx).reshape(
        flat, self.n_obs
      ),
      "next_critic_observations": torch.gather(
        self.next_critic_observations, 1, critic_obs_idx
      ).reshape(flat, self.n_critic_obs),
      "actions": torch.gather(self.actions, 1, act_idx).reshape(flat, self.n_act),
      "rewards": torch.gather(self.rewards, 1, indices).reshape(flat),
      "dones": torch.gather(self.dones, 1, indices).reshape(flat),
      "truncations": torch.gather(self.truncations, 1, indices).reshape(flat),
      "effective_n_steps": torch.ones(flat, device=self.device),
    }

  def _sample_nstep(self, batch_size: int, valid: int) -> dict[str, torch.Tensor]:
    flat = self.n_env * batch_size

    if self.ptr >= self.buffer_size:
      current_pos = self.ptr % self.buffer_size
      saved_truncations = self.truncations[:, current_pos - 1].clone()
      self.truncations[:, current_pos - 1] = torch.logical_not(
        self.dones[:, current_pos - 1]
      ).long()
      indices = torch.randint(
        0,
        self.buffer_size,
        (self.n_env, batch_size),
        device=self.device,
      )
    else:
      saved_truncations = None
      max_start_idx = max(1, self.ptr - self.n_steps + 1)
      indices = torch.randint(
        0,
        max_start_idx,
        (self.n_env, batch_size),
        device=self.device,
      )

    obs_idx = indices.unsqueeze(-1).expand(-1, -1, self.n_obs)
    critic_obs_idx = indices.unsqueeze(-1).expand(-1, -1, self.n_critic_obs)
    act_idx = indices.unsqueeze(-1).expand(-1, -1, self.n_act)

    observations = torch.gather(self.observations, 1, obs_idx).reshape(flat, self.n_obs)
    critic_observations = torch.gather(
      self.critic_observations, 1, critic_obs_idx
    ).reshape(flat, self.n_critic_obs)
    actions = torch.gather(self.actions, 1, act_idx).reshape(flat, self.n_act)

    seq_offsets = torch.arange(self.n_steps, device=self.device).view(1, 1, -1)
    all_indices = (indices.unsqueeze(-1) + seq_offsets) % self.buffer_size

    all_rewards = torch.gather(
      self.rewards.unsqueeze(-1).expand(-1, -1, self.n_steps),
      1,
      all_indices,
    )
    all_dones = torch.gather(
      self.dones.unsqueeze(-1).expand(-1, -1, self.n_steps),
      1,
      all_indices,
    )
    all_truncations = torch.gather(
      self.truncations.unsqueeze(-1).expand(-1, -1, self.n_steps),
      1,
      all_indices,
    )

    # Mask out rewards after the first done
    all_dones_shifted = torch.cat(
      [
        torch.zeros_like(all_dones[:, :, :1]),
        all_dones[:, :, :-1],
      ],
      dim=2,
    )
    done_masks = torch.cumprod(1.0 - all_dones_shifted.float(), dim=2)
    effective_n_steps = done_masks.sum(2)

    discounts = torch.pow(
      self.gamma,
      torch.arange(self.n_steps, device=self.device),
    )
    n_step_rewards = (all_rewards * done_masks * discounts.view(1, 1, -1)).sum(dim=2)

    # Find the terminal index for next_obs
    first_done = torch.argmax((all_dones > 0).float(), dim=2)
    first_trunc = torch.argmax((all_truncations > 0).float(), dim=2)
    no_dones = all_dones.sum(dim=2) == 0
    no_truncs = all_truncations.sum(dim=2) == 0
    first_done = torch.where(no_dones, self.n_steps - 1, first_done)
    first_trunc = torch.where(no_truncs, self.n_steps - 1, first_trunc)
    final_indices = torch.minimum(first_done, first_trunc)

    final_next_obs_indices = torch.gather(
      all_indices, 2, final_indices.unsqueeze(-1)
    ).squeeze(-1)

    result = {
      "observations": observations,
      "critic_observations": critic_observations,
      "actions": actions,
      "rewards": n_step_rewards.reshape(flat),
      "dones": self.dones.gather(1, final_next_obs_indices).reshape(flat),
      "truncations": self.truncations.gather(1, final_next_obs_indices).reshape(flat),
      "next_observations": self.next_observations.gather(
        1,
        final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_obs),
      ).reshape(flat, self.n_obs),
      "next_critic_observations": self.next_critic_observations.gather(
        1,
        final_next_obs_indices.unsqueeze(-1).expand(-1, -1, self.n_critic_obs),
      ).reshape(flat, self.n_critic_obs),
      "effective_n_steps": effective_n_steps.reshape(flat),
    }

    if saved_truncations is not None:
      current_pos = self.ptr % self.buffer_size
      self.truncations[:, current_pos - 1] = saved_truncations

    return result