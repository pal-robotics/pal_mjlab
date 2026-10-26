"""Replay clips of a motion library on the Kangaroo as retargeted: kinematic, no physics and no
policy. MuJoCo's viewer (needs a display).

  python -m pal_mjlab.scripts.supermimic.view_motion <library folder> \
    [--clips 230324/flip_360_001__A304 ...]

The robot (grippers or hands) is the one whose joints are the library's. The clips play one
after another at the library's rate, in a loop; the camera follows the base.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import mujoco
import mujoco.viewer
import pal_mjlab.tasks  # noqa: F401  (before pal_mjlab.robots, which it imports: a cycle otherwise)
import tyro
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  get_kangaroo_grippers_spec,
  get_kangaroo_hands_spec,
)
from pal_mjlab.tasks.supermimic.kinematics import compile_kinematic_model
from pal_mjlab.tasks.supermimic.library import MotionLibrary


def robot_model(joint_names: Sequence[str]) -> mujoco.MjModel:
  """The Kangaroo variant with these joints (after the free joint), on a floor."""
  for spec_fn in (get_kangaroo_grippers_spec, get_kangaroo_hands_spec):
    spec = spec_fn()
    spec.worldbody.add_geom(
      type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.05), rgba=(0.6, 0.6, 0.6, 1)
    )
    model = compile_kinematic_model(spec)
    if [model.joint(i).name for i in range(1, model.njnt)] == list(joint_names):
      return model
  raise ValueError("No Kangaroo variant has the library's joints.")


def main(library: tyro.conf.Positional[Path], clips: tuple[str, ...] = ()) -> None:
  """``clips``: only these clips of the library, by name; empty for all."""
  lib = MotionLibrary(library, clips=clips)
  model = robot_model(lib.joint_names)
  data = mujoco.MjData(model)
  qpos = lib.qpos.numpy()
  spans = list(
    zip(lib.clip_names, lib.clip_start.tolist(), lib.clip_length.tolist(), strict=True)
  )
  with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = model.body("base_link").id
    while viewer.is_running():
      for name, start, length in spans:
        print(f"{name} ({length / lib.fps:.1f} s)", flush=True)
        for t in range(start, start + length):
          tic = time.perf_counter()
          data.qpos[:] = qpos[t]
          mujoco.mj_forward(model, data)
          viewer.sync()
          if not viewer.is_running():
            return
          time.sleep(max(0.0, 1.0 / lib.fps - (time.perf_counter() - tic)))


if __name__ == "__main__":
  tyro.cli(main)
