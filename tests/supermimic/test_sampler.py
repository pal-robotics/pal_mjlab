"""Return-based reset sampler (paper Appendix D.1)."""

import pytest
import torch
from pal_mjlab.tasks.supermimic.mdp.sampler import ReturnSampler


def _clips(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
  g = torch.Generator().manual_seed(seed)
  lengths = torch.randint(2, 400, (30,), generator=g)
  return torch.cumsum(lengths, 0) - lengths, lengths


def _sampler(seed: int = 0) -> ReturnSampler:
  return ReturnSampler(*_clips(seed), frames_per_bin=50)


@pytest.mark.parametrize("seed", range(5))
def test_samples_leave_a_target_frame(seed):
  starts, lengths = _clips(seed)
  s = ReturnSampler(starts, lengths, frames_per_bin=50)
  bins, frames = s.sample(5000, generator=torch.Generator().manual_seed(seed))
  clip = s.bin_clip[bins]
  # Inside the sampled bin and clip, with at least one frame after it (the first target).
  assert (frames >= s.bin_start[bins]).all() and (frames < s.bin_stop[bins]).all()
  assert (frames >= starts[clip]).all()
  assert (frames <= starts[clip] + lengths[clip] - 2).all()
  probs, _ = s.probabilities()
  assert torch.isclose(probs.sum(), torch.tensor(1.0))


def test_hard_bins_get_more_mass():
  s = _sampler()
  bins = torch.arange(s.num_bins)
  returns = torch.ones(s.num_bins)
  returns[:5] = 0.1  # five bins where the policy does badly
  s.update(bins, returns)
  probs, hard = s.probabilities()
  assert hard[:5].sum() > 0.99
  # Mixture: 50% uniform + 40% hard + 10% past weight.
  assert probs[:5].sum() > 0.4
  assert probs[5:].min() > 0.4 / s.num_bins


def test_update_is_an_ema_after_the_first_visit():
  s = _sampler()
  b = torch.tensor([3, 3, 7])
  s.update(b, torch.tensor([0.2, 0.4, 1.0]))  # first visit: batch mean
  assert torch.isclose(s.returns[3], torch.tensor(0.3)) and s.returns[7] == 1.0
  s.update(torch.tensor([3]), torch.tensor([1.0]))
  assert torch.isclose(s.returns[3], torch.tensor(0.8 * 0.3 + 0.2 * 1.0))
  assert s.visits[3] == 3


def test_short_clips_still_sample():
  s = ReturnSampler(torch.tensor([0, 2]), torch.tensor([2, 3]), frames_per_bin=50)
  _, frames = s.sample(100, generator=torch.Generator().manual_seed(0))
  assert set(frames.tolist()) <= {0, 2, 3}
