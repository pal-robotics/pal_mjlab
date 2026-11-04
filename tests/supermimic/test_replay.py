"""Online replay buffer: windows stay inside episodes, starts carry the right history."""

import pytest
import torch
from pal_mjlab.tasks.supermimic.rl.replay import ReplayBuffer

E, CAPACITY, STEPS = 5, 40, 130  # wraps around about three times


def _fill(seed: int) -> ReplayBuffer:
  """State = (env step t, episode id, env, terminal flag, step in episode); action = t + 1."""
  g = torch.Generator().manual_seed(seed)
  buf = ReplayBuffer(E, CAPACITY, 5, 2, 1, "cpu")
  episode, in_episode = torch.zeros(E), torch.zeros(E)
  envs = torch.arange(E, dtype=torch.float)
  for t in range(STEPS):
    state = torch.stack(
      [torch.full((E,), float(t)), episode, envs, torch.zeros(E), in_episode], 1
    )
    done = torch.rand(E, generator=g) < 0.08
    nxt = state.clone()
    nxt[:, 0] += 1
    nxt[:, 3] = 1.0
    action = torch.full((E, 2), t + 1.0)
    zero = torch.zeros(E, dtype=torch.long)
    buf.add(
      state, torch.zeros(E, 3), action, torch.zeros(E, 1), zero, zero, done, done,
      nxt, torch.zeros(E, 3),
    )  # fmt: skip
    episode = episode + done.float()
    in_episode = torch.where(done, 0.0, in_episode + 1)
  return buf


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("k", [1, 8])
def test_windows_are_consecutive_steps_of_one_episode(seed, k):
  buf = _fill(seed)
  w = buf.sample_windows(500, k, torch.Generator().manual_seed(seed))
  s = w["states"]
  assert (s[:, :, 1] == s[:, :1, 1]).all() and (s[:, :, 2] == s[:, :1, 2]).all()
  torch.testing.assert_close(s[:, 1:, 0] - s[:, :-1, 0], torch.ones(500, k))
  # Only the last row may be a terminal state, and then the last action ended it.
  assert (s[:, :-1, 3] == 0).all()
  assert torch.equal(s[:, -1, 3] == 1, w["failure"][:, -1])
  torch.testing.assert_close(w["actions"][..., 0], s[:, :-1, 0] + 1)
  # Only the newest CAPACITY rows of each env are used.
  assert (s[:, 0, 0] >= STEPS - CAPACITY).all()


@pytest.mark.parametrize("seed", range(3))
def test_starts_carry_their_episode_history(seed):
  buf = _fill(seed)
  h = 10
  st = buf.sample_starts(500, h, torch.Generator().manual_seed(seed))
  hist, t0 = st["history_state"], st["state"][:, 0]
  torch.testing.assert_close(hist[:, -1], st["state"])
  assert (st["state"][:, 3] == 0).all()  # starts are not terminal states
  assert (hist[:, :, 1] == st["state"][:, None, 1]).all()  # same episode
  assert (hist[:, :, 3] == 0).all()
  # Steps count up to the start; before the episode's first step they repeat it.
  first = t0 - st["state"][:, 4]
  expected = torch.maximum(t0[:, None] - torch.arange(h - 1, -1, -1), first[:, None])
  torch.testing.assert_close(hist[:, :, 0], expected)
  # Last action: the previous step's action (= its step + 1), zero at the episode start.
  last = st["history_last_action"][..., 0]
  torch.testing.assert_close(last, torch.where(hist[:, :, 4] == 0, 0.0, hist[:, :, 0]))
