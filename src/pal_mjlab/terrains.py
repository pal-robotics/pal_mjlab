"""Custom PAL sub-terrains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import (
  SubTerrainCfg,
  TerrainGeometry,
  TerrainOutput,
)
from mjlab.terrains.utils import make_plane
from mjlab.utils.color import brand_ramp

_ORANGE = (0.95, 0.55, 0.15)
_MUJOCO_BLUE = (0.20, 0.45, 0.95)

@dataclass(kw_only=True)
class BoxStraightStairsTerrainCfg(SubTerrainCfg):
  """Straight, limited-width staircase over a flat floor (no railings).

  A single flight goes up along +x, crosses a landing and goes back down. Its
  sides are open drop-offs, so stepping off laterally means falling to the floor,
  like real stairs without railings. Width, lateral offset and up/down tread are
  sampled per patch, so the staircase is neither centered nor symmetric.
  """

  step_height_range: tuple[float, float]
  """Min and max riser height, in meters. Interpolated by difficulty."""
  tread_range: tuple[float, float] = (0.28, 0.4)
  """Tread depth range, sampled independently for the up and down flights."""
  stair_width_range: tuple[float, float] = (1.0, 1.5)
  """Lateral width range of the staircase, in meters."""
  max_lateral_offset: float = 0.5
  """Max lateral shift of the staircase from the patch center line, in meters."""
  max_steps: int = 4
  """Max number of steps per flight (reduced if they don't fit the patch)."""
  min_landing_length: float = 0.8
  """Min length of the top landing along x, in meters."""
  spawn: Literal["top", "bottom"] = "top"
  """``top``: spawn on the landing (walk down / risk falling off the sides).
  ``bottom``: spawn on the floor before the up flight (walk up)."""
  bottom_spawn_margin: float = 0.9
  """Floor length before the up flight when ``spawn="bottom"``, in meters."""
  end_margin: float = 0.2
  """Floor length kept at the patch ends, in meters."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries = []

    step_height = self.step_height_range[0] + difficulty * (
      self.step_height_range[1] - self.step_height_range[0]
    )
    tread_up = rng.uniform(*self.tread_range)
    tread_down = rng.uniform(*self.tread_range)
    width = rng.uniform(*self.stair_width_range)
    max_offset = min(self.max_lateral_offset, (self.size[1] - width) / 2.0)
    y_center = self.size[1] / 2.0 + rng.uniform(-max_offset, max_offset)

    # Flat floor everywhere, so falling off the sides lands on z=0.
    floor = make_plane(body, self.size, 0.0, center_zero=False)[0]
    geometries.append(TerrainGeometry(geom=floor, color=(0.5, 0.5, 0.5, 1.0)))

    x_start = self.bottom_spawn_margin if self.spawn == "bottom" else self.end_margin
    span = self.size[0] - x_start - self.end_margin
    num_steps = int((span - self.min_landing_length) / (tread_up + tread_down))
    num_steps = max(0, min(self.max_steps, num_steps))
    landing_length = span - num_steps * (tread_up + tread_down)
    landing_height = (num_steps + 1) * step_height

    def add_box(x0: float, x1: float, height: float, t: float) -> None:
      box = body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=((x1 - x0) / 2.0, width / 2.0, max(height, 1e-3) / 2.0),
        pos=((x0 + x1) / 2.0, y_center, max(height, 1e-3) / 2.0),
      )
      geometries.append(TerrainGeometry(geom=box, color=brand_ramp(_ORANGE, t)))

    # Up flight, landing, down flight. Solid blocks resting on the floor.
    x = x_start
    for k in range(num_steps):
      add_box(x, x + tread_up, (k + 1) * step_height, k / max(num_steps, 1))
      x += tread_up
    landing_x0 = x
    add_box(x, x + landing_length, landing_height, 1.0)
    x += landing_length
    for k in reversed(range(num_steps)):
      add_box(x, x + tread_down, (k + 1) * step_height, k / max(num_steps, 1))
      x += tread_down

    if self.spawn == "top":
      origin = np.array([landing_x0 + landing_length / 2.0, y_center, landing_height])
    else:
      origin = np.array([x_start / 2.0, y_center, 0.0])
    return TerrainOutput(origin=origin, geometries=geometries)

@dataclass(kw_only=True)
class BoxSingleObstacleTerrainCfg(SubTerrainCfg):
  """Flat ground with a single box obstacle."""

  box_size: tuple[float, float] = (0.5, 0.5)
  """Box footprint (x, y), in meters."""
  box_height: float = 0.2
  """Height of the box top above the ground, in meters."""
  box_offset: tuple[float, float] = (0.0, 0.0)
  """Box center offset from the tile center (x, y), in meters."""
  spawn_on_box: bool = False
  """If True, the origin is on top of the box; otherwise on the ground at the tile center."""

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del difficulty, rng  # Unused.
    body = spec.body("terrain")
    cx = self.size[0] / 2 + self.box_offset[0]
    cy = self.size[1] / 2 + self.box_offset[1]

    ground = make_plane(body, self.size, 0.0, center_zero=False)[0]
    geometries = [TerrainGeometry(geom=ground, color=(0.5, 0.5, 0.5, 1.0))]

    # Sink the box slightly below z=0 so its bottom face isn't coplanar with the ground.
    sink = 0.01
    box = body.add_geom(
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=(self.box_size[0] / 2, self.box_size[1] / 2, (self.box_height + sink) / 2),
      pos=(cx, cy, (self.box_height - sink) / 2),
    )
    geometries.append(TerrainGeometry(geom=box, color=brand_ramp(_MUJOCO_BLUE, 0.5)))

    if self.spawn_on_box:
      origin = np.array([cx, cy, self.box_height])
    else:
      origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
    return TerrainOutput(origin=origin, geometries=geometries)