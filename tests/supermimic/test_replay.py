"""Online replay buffer: windows stay inside episodes, starts carry the right history."""

import pytest
import torch
from pal_mjlab.rl.supermimic.replay import ReplayBuffer

E, CAPACITY, STEPS = 5, 40, 130  # wraps around about three times


def _fill(seed: int) -> ReplayBuffer:
  """State = (env step t, episode id, env, terminal flag, step in episode); action = t + 1;
  command state c = {"t": t (long), "vec": (t, env) (float, 2-d)}."""
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
    command = {
      "t": torch.full((E,), t, dtype=torch.long),
      "vec": torch.stack([torch.full((E,), float(t)), envs], 1),
    }
    buf.add(
      state, torch.zeros(E, 3), action, torch.zeros(E, 1), command, done, done,
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
  # Each transition's command state, stored as given (dtype and shape kept).
  c = w["command"]
  assert c["t"].dtype == torch.long and c["t"].shape == (500, k)
  assert torch.equal(c["t"], s[:, :-1, 0].long())
  torch.testing.assert_close(c["vec"], s[:, :-1, [0, 2]])


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
  # And the action before it, zero unless both rows before are in the episode.
  prev2 = st["history_prev_action"][..., 0]
  torch.testing.assert_close(prev2, torch.where(hist[:, :, 4] <= 1, 0.0, hist[:, :, 0] - 1))
  # The history rows' command states.
  assert torch.equal(st["history_command"]["t"], hist[:, :, 0].long())
  # The start row's command state.
  assert torch.equal(st["command"]["t"], t0.long())
  torch.testing.assert_close(st["command"]["vec"], st["state"][:, [0, 2]])


def test_command_state_layout_must_not_change():
  buf = ReplayBuffer(E, CAPACITY, 5, 2, 1, "cpu")
  args = (torch.zeros(E, 5), torch.zeros(E, 3), torch.zeros(E, 2), torch.zeros(E, 1))
  no = torch.zeros(E, dtype=torch.bool)
  tail = (no, no, torch.zeros(E, 5), torch.zeros(E, 3))
  buf.add(*args, {"t": torch.zeros(E, dtype=torch.long)}, *tail)
  with pytest.raises(ValueError, match="layout changed"):
    buf.add(*args, {"t": torch.zeros(E)}, *tail)  # float instead of long
