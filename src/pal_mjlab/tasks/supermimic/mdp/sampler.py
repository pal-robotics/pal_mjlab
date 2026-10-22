"""Return-based reset sampling (SuperMimic Appendix D.1).

The motion timeline is split into 1 s bins per clip. Each bin keeps a running estimate of
the (normalized) discounted return of episodes started in it. Resets mix
  50% uniform over bins,
  40% softmax(-25 * G_b),
  10% in proportion to each bin's past softmax weight (hard motions keep some practice).

Our reading of the details the paper leaves open (D21):
  - G_b is normalized per bin by the largest return attainable from that start (the clip may
    end before the discount horizon), not by one global maximum; otherwise every clip's
    last second would look like a failure and dominate the 40% branch.
  - G_b starts at 0 (unvisited bins count as hard), and is an EMA with rate ``alpha``.
  - The past weight is an EMA (rate ``past_rate``) of the softmax distribution.
"""

from __future__ import annotations

import torch


class ReturnSampler:
  def __init__(
    self,
    clip_start: torch.Tensor,
    clip_length: torch.Tensor,
    frames_per_bin: int,
    min_frames_left: int = 2,
    temperature: float = 25.0,
    mix: tuple[float, float, float] = (0.5, 0.4, 0.1),
    alpha: float = 0.2,
    past_rate: float = 0.01,
  ) -> None:
    device = clip_start.device
    bins_per_clip = (clip_length + frames_per_bin - 1) // frames_per_bin
    offset = torch.cumsum(bins_per_clip, 0) - bins_per_clip
    self.num_bins = int(bins_per_clip.sum())
    self.bin_clip = torch.repeat_interleave(
      torch.arange(len(clip_start), device=device), bins_per_clip
    )
    local = torch.arange(self.num_bins, device=device) - offset[self.bin_clip]
    self.bin_start = clip_start[self.bin_clip] + local * frames_per_bin
    clip_end = clip_start + clip_length
    # Starts must leave at least ``min_frames_left`` frames (a target for the first step).
    last_start = clip_end[self.bin_clip] - min_frames_left
    self.bin_start = torch.minimum(self.bin_start, last_start)
    self.bin_stop = torch.minimum(self.bin_start + frames_per_bin, last_start + 1)
    self.clip_bin_offset = offset
    self.frames_per_bin = frames_per_bin
    self.temperature = temperature
    self.mix = mix
    self.alpha = alpha
    self.past_rate = past_rate
    self.returns = torch.zeros(self.num_bins, device=device)
    self.visits = torch.zeros(self.num_bins, device=device)
    self.past_weight = torch.full((self.num_bins,), 1.0 / self.num_bins, device=device)

  def probabilities(self) -> torch.Tensor:
    hard = torch.softmax(-self.temperature * self.returns, dim=0)
    uniform = torch.full_like(hard, 1.0 / self.num_bins)
    past = self.past_weight / self.past_weight.sum()
    u, h, p = self.mix
    return u * uniform + h * hard + p * past, hard

  def sample(
    self, n: int, generator: torch.Generator | None = None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample n start frames; returns (bin index, global frame index)."""
    probs, hard = self.probabilities()
    self.past_weight.mul_(1.0 - self.past_rate).add_(self.past_rate * hard)
    bins = torch.multinomial(probs, n, replacement=True, generator=generator)
    span = (self.bin_stop[bins] - self.bin_start[bins]).clamp_min(1)
    u = torch.rand(n, device=bins.device, generator=generator)
    frames = self.bin_start[bins] + (u * span).long().clamp_max(span - 1)
    return bins, frames

  def update(self, bins: torch.Tensor, normalized_returns: torch.Tensor) -> None:
    """Fold finished episodes (started in ``bins``) into the running estimates."""
    if bins.numel() == 0:
      return
    # Average duplicates within the batch, then one EMA step per touched bin.
    total = torch.zeros_like(self.returns).index_add_(0, bins, normalized_returns)
    count = torch.zeros_like(self.returns).index_add_(
      0, bins, torch.ones_like(normalized_returns)
    )
    touched = count > 0
    mean = total[touched] / count[touched]
    first = self.visits[touched] == 0
    current = self.returns[touched]
    self.returns[touched] = torch.where(
      first, mean, (1.0 - self.alpha) * current + self.alpha * mean
    )
    self.visits[touched] += count[touched]

  def state_dict(self) -> dict[str, torch.Tensor]:
    return {
      "returns": self.returns,
      "visits": self.visits,
      "past_weight": self.past_weight,
    }

  def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
    self.returns.copy_(state["returns"])
    self.visits.copy_(state["visits"])
    self.past_weight.copy_(state["past_weight"])
