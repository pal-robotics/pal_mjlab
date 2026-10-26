"""Tests for the terrain-dependent yaw reset event."""

import math
from types import SimpleNamespace

import pytest
import torch
from pal_mjlab.tasks.velocity.mdp import events

SUB_TERRAINS = ("flat", "stairs_up", "stairs_down")


def _make_env(curriculum=True, terrain_types=None):
  generator = SimpleNamespace(
    sub_terrains={name: None for name in SUB_TERRAINS}, curriculum=curriculum
  )
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(terrain_generator=generator), terrain_types=terrain_types
  )
  return SimpleNamespace(scene=SimpleNamespace(terrain=terrain))


@pytest.fixture
def uniform_calls(monkeypatch):
  """Records (env_ids, pose_range) of every reset_root_state_uniform call."""
  calls = []
  monkeypatch.setattr(
    events,
    "reset_root_state_uniform",
    lambda env, env_ids, pose_range, *_: calls.append((env_ids.tolist(), pose_range)),
  )
  return calls


@pytest.mark.parametrize("curriculum", [True, False])
def test_unknown_name_fails_early(curriculum):
  with pytest.raises(ValueError, match="stairs_upp"):
    events.reset_root_state_terrain_yaw(
      _make_env(curriculum=curriculum),
      None,
      pose_range={},
      terrain_yaw={"stairs_upp": (0.0,)},
    )


def test_terrain_yaw_without_generator_fails_early():
  env = SimpleNamespace(scene=SimpleNamespace(terrain=None))
  with pytest.raises(ValueError, match="stairs_up"):
    events.reset_root_state_terrain_yaw(
      env, None, pose_range={}, terrain_yaw={"stairs_up": (0.0,)}
    )


def test_yaw_is_overridden_only_on_named_terrains(uniform_calls):
  # Env 5 is on stairs but not being reset, so it must not be touched.
  terrain_types = torch.tensor([0, 1, 2, 1, 2, 1])
  env_ids = torch.tensor([0, 1, 2, 3, 4])
  pose_range = {"x": (-0.5, 0.5), "yaw": (-3.14, 3.14)}
  terrain_yaw = {"stairs_up": (0.0,), "stairs_down": (0.0, math.pi)}

  events.reset_root_state_terrain_yaw(
    _make_env(terrain_types=terrain_types),
    env_ids,
    pose_range,
    terrain_yaw,
    yaw_half_range=0.3,
  )

  # Every env being reset first gets the plain pose_range.
  assert uniform_calls[0] == ([0, 1, 2, 3, 4], pose_range)

  # Then only the stair envs are reset again, facing one of their terrain's yaws.
  overridden = []
  for ids, sub_pose_range in uniform_calls[1:]:
    yaw_low, yaw_high = sub_pose_range.pop("yaw")
    assert sub_pose_range == {"x": (-0.5, 0.5)}
    assert yaw_high - yaw_low == pytest.approx(0.6)
    base = (yaw_low + yaw_high) / 2
    for i in ids:
      yaws = terrain_yaw[SUB_TERRAINS[terrain_types[i]]]
      assert any(base == pytest.approx(yaw) for yaw in yaws)
    overridden += ids
  assert sorted(overridden) == [1, 2, 3, 4]


def test_yaw_is_not_overridden_without_curriculum(uniform_calls):
  events.reset_root_state_terrain_yaw(
    _make_env(curriculum=False, terrain_types=torch.tensor([1, 2])),
    torch.tensor([0, 1]),
    pose_range={},
    terrain_yaw={"stairs_up": (0.0,)},
  )
  assert len(uniform_calls) == 1
