"""Tests for the straight staircase sub-terrain."""

import mujoco
import numpy as np
import pytest
from pal_mjlab.terrains import BoxStraightStairsTerrainCfg

SIZE = (8.0, 8.0)  # Patch size of mjlab's rough terrain generator.
STEP_HEIGHT_RANGE = (0.02, 0.2)


def _generate(cfg, difficulty, seed):
  """Returns the spawn origin, the floor geom and the stair boxes.

  Each box is a row ``(x0, x1, y0, y1, top)``, ordered as generated: up flight,
  landing, down flight.
  """
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  out = cfg.function(difficulty, spec, np.random.default_rng(seed))
  spec.compile()  # The generated geoms must make a valid model.
  floor, *steps = (g.geom for g in out.geometries)
  boxes = np.array(
    [
      [
        g.pos[0] - g.size[0],
        g.pos[0] + g.size[0],
        g.pos[1] - g.size[1],
        g.pos[1] + g.size[1],
        g.pos[2] + g.size[2],
      ]
      for g in steps
    ]
  )
  return out.origin, floor, boxes


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("difficulty", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("spawn", ["top", "bottom"])
def test_staircase_layout(spawn, difficulty, seed):
  cfg = BoxStraightStairsTerrainCfg(
    size=SIZE, step_height_range=STEP_HEIGHT_RANGE, spawn=spawn
  )
  origin, floor, boxes = _generate(cfg, difficulty, seed)
  x0, x1, y0, y1, top = boxes.T
  step_height = np.interp(difficulty, [0.0, 1.0], STEP_HEIGHT_RANGE)
  num_steps = (len(boxes) - 1) // 2

  # Floor covers the whole patch, with its top at z=0.
  np.testing.assert_allclose(2 * floor.size[:2], SIZE)
  np.testing.assert_allclose(floor.pos[:2], np.array(SIZE) / 2)
  assert floor.pos[2] + floor.size[2] == pytest.approx(0.0)

  # Up flight, landing and down flight, contiguous along x within the margins.
  assert len(boxes) == 2 * num_steps + 1
  assert 1 <= num_steps <= cfg.max_steps
  np.testing.assert_allclose(x0[1:], x1[:-1])
  x_start = cfg.bottom_spawn_margin if spawn == "bottom" else cfg.end_margin
  assert x0[0] == pytest.approx(x_start)
  assert x1[-1] == pytest.approx(SIZE[0] - cfg.end_margin)

  # One riser per step up to the landing and back down.
  risers = np.r_[np.arange(1, num_steps + 2), np.arange(num_steps, 0, -1)]
  np.testing.assert_allclose(top, step_height * risers)

  # Each flight has a constant tread within range, and the landing is long enough.
  treads = x1 - x0
  for flight in (treads[:num_steps], treads[num_steps + 1 :]):
    np.testing.assert_allclose(flight, flight[0])
    assert cfg.tread_range[0] <= flight[0] <= cfg.tread_range[1]
  assert treads[num_steps] >= cfg.min_landing_length

  # Constant width within range, shifted laterally by at most max_lateral_offset.
  np.testing.assert_allclose(y0, y0[0])
  np.testing.assert_allclose(y1, y1[0])
  assert cfg.stair_width_range[0] <= y1[0] - y0[0] <= cfg.stair_width_range[1]
  y_center = (y0[0] + y1[0]) / 2
  assert abs(y_center - SIZE[1] / 2) <= cfg.max_lateral_offset + 1e-9

  # Spawn on the landing center or on the floor before the up flight.
  if spawn == "top":
    expected_origin = [(x0[num_steps] + x1[num_steps]) / 2, y_center, top[num_steps]]
  else:
    expected_origin = [x_start / 2, y_center, 0.0]
  np.testing.assert_allclose(origin, expected_origin)


def test_short_patch_has_only_landing():
  # 2 m patch with bottom spawn: no room for any step, only the landing.
  cfg = BoxStraightStairsTerrainCfg(
    size=(2.0, 2.0), step_height_range=STEP_HEIGHT_RANGE, spawn="bottom"
  )
  _, _, boxes = _generate(cfg, difficulty=1.0, seed=0)
  assert len(boxes) == 1
  assert boxes[0, 4] == pytest.approx(STEP_HEIGHT_RANGE[1])
