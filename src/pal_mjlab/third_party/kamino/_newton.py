# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The Newton definitions Kamino uses, copied from Newton v1.7.0 so that nothing imports it.

``Model``, ``State`` and friends are empty stand-ins: they only appear in type hints and in
``from_newton`` paths that the mjlab bridge never calls.
"""

from enum import IntEnum
from typing import Literal, override

import warp as wp

__all__ = ["MAXVAL", "Axis", "AxisType", "BodyFlags", "Contacts", "Control", "GeoType", "Heightfield", "JointTargetMode"]
__all__ += ["JointType", "Mesh", "Model", "ModelFlags", "ModelBuilder", "ModelView", "ShapeFlags", "State", "Vec2", "Vec3"]
__all__ += ["override", "transform_3d_rotational_axes"]

MAXVAL = 1e10

Vec2 = list[float] | tuple[float, float] | wp.vec2
Vec3 = list[float] | tuple[float, float, float] | wp.vec3


class Axis(IntEnum):
  X = 0
  Y = 1
  Z = 2

  @classmethod
  def from_any(cls, value) -> "Axis":
    if isinstance(value, str):
      return cls["XYZ".index(value.upper())]
    return cls(value)


AxisType = Axis | Literal["X", "Y", "Z"] | Literal[0, 1, 2] | int | str


class BodyFlags(IntEnum):
  DYNAMIC = 1 << 0
  KINEMATIC = 1 << 1
  PROXY = 1 << 2


class JointType(IntEnum):
  PRISMATIC = 0
  REVOLUTE = 1
  BALL = 2
  FIXED = 3
  FREE = 4
  DISTANCE = 5
  D6 = 6
  CABLE = 7


class JointTargetMode(IntEnum):
  NONE = 0
  POSITION = 1
  VELOCITY = 2
  POSITION_VELOCITY = 3
  EFFORT = 4


class GeoType(IntEnum):
  NONE = 0
  PLANE = 1
  HFIELD = 2
  SPHERE = 3
  CAPSULE = 4
  ELLIPSOID = 5
  CYLINDER = 6
  BOX = 7
  MESH = 8
  CONE = 9
  CONVEX_MESH = 10
  GAUSSIAN = 11


class ShapeFlags(IntEnum):
  VISIBLE = 1 << 0
  COLLIDE_SHAPES = 1 << 1
  COLLIDE_PARTICLES = 1 << 2
  SITE = 1 << 3
  HYDROELASTIC = 1 << 4


class Mesh:
  MAX_HULL_VERTICES = 64


class Heightfield: ...


class Model: ...


class ModelBuilder: ...


class ModelView: ...


class ModelFlags: ...


class State: ...


class Control: ...


class Contacts: ...


@wp.func
def transform_3d_rotational_axes(axis_0: wp.vec3, axis_1: wp.vec3, axis_2: wp.vec3, q0: float, q1: float):
  q_0 = wp.quat_from_axis_angle(axis_0, q0)
  axis_1_w = wp.quat_rotate(q_0, axis_1)
  q_1 = wp.quat_from_axis_angle(axis_1_w, q1)
  axis_2_w = wp.quat_rotate(q_1 * q_0, axis_2)
  return axis_0, axis_1_w, axis_2_w
