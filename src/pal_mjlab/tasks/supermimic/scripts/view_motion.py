"""View the motions of a motion library as retargeted: kinematic, no physics and no policy.

  uv run -m pal_mjlab.tasks.supermimic.scripts.view_motion Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers \
    [--library-path <library>] [--clips <name> ...]

The clips play one after another at the library's rate, in a loop, on the task's robot; the
camera follows its base. MuJoCo's viewer (needs a display).
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import mjlab
import mjlab.tasks  # noqa: F401  (populates the task registry)
import mujoco
import mujoco.viewer
import numpy as np
import tyro
from mjlab.scene import Scene
from mjlab.tasks.registry import list_tasks, load_env_cfg
from pal_mjlab.tasks.supermimic.library import MotionLibrary
from pal_mjlab.tasks.supermimic.mdp import MotionLibraryCommandCfg


@dataclass(frozen=True)
class ViewMotionConfig:
  library_path: str | None = None
  """Motion library folder (clips.json, qpos.npy, qvel.npy); the task's default if not given."""
  clips: tuple[str, ...] = ()
  """Only these motions, by name as in clips.json; empty for all."""


def model_qpos(
  model: mujoco.MjModel, library: MotionLibrary, motion: MotionLibraryCommandCfg
) -> np.ndarray:
  """(frames, nq) the library's frames as qpos of the scene's model, passive joints closed."""
  prefix = f"{motion.entity_name}/"

  def address(joint_name: str) -> int:
    return int(model.joint(prefix + joint_name).qposadr[0])

  root = model.body(prefix + motion.body_names[0])
  free = int(model.jnt_qposadr[root.jntadr[0]])
  qpos = np.tile(model.qpos0, (library.qpos.shape[0], 1))
  qpos[:, free : free + 7] = library.qpos[:, :7].numpy()
  joints = dict(zip(library.joint_names, library.qpos[:, 7:].unbind(-1), strict=True))
  for name, value in joints.items():
    qpos[:, address(name)] = value.numpy()
  if motion.passive_joint_state is not None:
    rates = dict(zip(library.joint_names, library.qvel[:, 6:].unbind(-1), strict=True))
    for name, (value, _) in motion.passive_joint_state(joints, rates).items():
      qpos[:, address(name)] = value.numpy()
  return qpos


def run_view_motion(task_id: str, cfg: ViewMotionConfig) -> None:
  env_cfg = load_env_cfg(task_id, play=True)
  motion = env_cfg.commands["motion"]
  assert isinstance(motion, MotionLibraryCommandCfg)
  library = MotionLibrary(cfg.library_path or motion.library_path, clips=cfg.clips)
  env_cfg.scene.num_envs = 1
  model = Scene(env_cfg.scene, "cpu").compile()
  data = mujoco.MjData(model)
  qpos = model_qpos(model, library, motion)
  spans = list(
    zip(
      library.clip_names,
      library.clip_start.tolist(),
      library.clip_length.tolist(),
      strict=True,
    )
  )
  with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = model.body(
      f"{motion.entity_name}/{motion.body_names[0]}"
    ).id
    while viewer.is_running():
      for name, start, length in spans:
        print(f"{name} ({length / library.fps:.1f} s)", flush=True)
        for frame in range(start, start + length):
          tic = time.perf_counter()
          data.qpos[:] = qpos[frame]
          mujoco.mj_forward(model, data)
          viewer.sync()
          if not viewer.is_running():
            return
          time.sleep(max(0.0, 1.0 / library.fps - (time.perf_counter() - tic)))


def main() -> None:
  tasks = [t for t in list_tasks() if "SuperMimic" in t]
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  # mjlab's flags, but clips as plain names (--clips a b), not a Python tuple literal.
  flags = tuple(
    f
    for f in mjlab.TYRO_FLAGS
    if f is not tyro.conf.UsePythonSyntaxForLiteralCollections
  )
  args = tyro.cli(
    ViewMotionConfig,
    args=remaining_args,
    default=ViewMotionConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=flags,
  )
  run_view_motion(chosen_task, args)


if __name__ == "__main__":
  main()
