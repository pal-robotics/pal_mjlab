"""Tests for the reset offset of a get-on-box motion on boxes of other heights."""

import pytest
import torch
from pal_mjlab.tasks.tracking.mdp.obstacle_climb_commands import (
  climb_spawn_offset,
  feet_over_obstacle,
)

FOOTPRINT = (-0.5, 0.5, -1.0, -0.2)  # Box ahead of the robot, along -y.


def _feet():
  """Two feet: both on the ground, one on the box, both on the box (3 frames)."""
  xy = torch.tensor(
    [
      [[0.1, 0.1], [-0.1, 0.1]],
      [[0.1, 0.1], [-0.1, -0.5]],
      [[0.1, -0.5], [-0.1, -0.5]],
    ]
  )
  sole = torch.tensor([[0.0, 0.0], [0.0, 0.42], [0.47, 0.47]])
  return xy, sole


def test_feet_over_obstacle():
  xy, _ = _feet()
  over = feet_over_obstacle(xy, FOOTPRINT, margin=0.12)
  assert over.tolist() == [[False, False], [False, True], [True, True]]
  # Within the margin of the near edge counts, beyond it does not.
  edge = torch.tensor([[[0.0, -0.1]], [[0.0, -0.05]]])
  assert feet_over_obstacle(edge, FOOTPRINT, margin=0.12).flatten().tolist() == [
    True,
    False,
  ]


@pytest.mark.parametrize("height", [0.3, 0.47, 0.55])
def test_climb_spawn_offset(height):
  xy, sole = _feet()
  over = feet_over_obstacle(xy, FOOTPRINT, margin=0.12)
  steps = torch.arange(3)
  dz = climb_spawn_offset(over, sole, steps, torch.full((3,), height), clearance=0.01)
  # On the ground: unchanged.
  assert dz[0] == 0.0
  # One foot on the box: lifted so it clears the box, never lowered.
  torch.testing.assert_close(dz[1], torch.tensor(max(height + 0.01 - 0.42, 0.0)))
  # Both on the box: the soles are put on top of it, either way.
  torch.testing.assert_close(dz[2], torch.tensor(height + 0.01 - 0.47))


def test_nearest_true():
  from pal_mjlab.tasks.tracking.mdp.obstacle_climb_commands import nearest_true

  mask = torch.tensor(
    [[True, False, False, False, True], [False, True, True, True, True]]
  )
  # Row 0: frames 1 -> 0, 2 -> 4 (tie goes later), 3 -> 4; row 1: frame 0 -> 1.
  assert nearest_true(mask).tolist() == [[0, 0, 4, 4, 4], [1, 1, 2, 3, 4]]
  with pytest.raises(ValueError):
    nearest_true(torch.zeros(1, 3, dtype=torch.bool))
