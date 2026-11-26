"""Replay motions of a motion library with a trained SuperMimic policy, in MuJoCo physics.

  uv run -m pal_mjlab.tasks.supermimic.scripts.play Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers \
    --checkpoint-file <run>/model_N.pt [--library-path <library>] [--clips <name> ...]

mjlab's ``play`` replays the task's default library with the evaluation settings; this
script also chooses the library and its clips, and shows falls. Without ``--clips`` the
library's motions play in random order, one after another; with it only those, so a single
one repeats. Each starts from its first frame. There is no reset when the robot fails: the
physics goes on, so a fall is seen as it happens, and the next motion starts after
``FALL_HOLD_S`` seconds in that state, or at the clip's end.

``--viewer``: MuJoCo's window with a display, else viser in the browser (``auto``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from typing import Literal

import mjlab
import mjlab.tasks  # noqa: F401  (populates the task registry)
import tyro
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.tasks.registry import (
  list_tasks,
  load_env_cfg,
  load_rl_cfg,
  load_runner_cls,
  register_mjlab_task,
)
from pal_mjlab.tasks.supermimic import mdp

FALL_HOLD_S = 3.0  # seconds a failure lasts on screen before the next motion


@dataclass(frozen=True)
class ReplayConfig:
  checkpoint_file: str
  """Policy checkpoint (model_N.pt)."""
  library_path: str | None = None
  """Motion library folder (clips.json, qpos.npy, qvel.npy); the task's default if not given."""
  clips: tuple[str, ...] = ()
  """Only these motions, by name as in clips.json; empty for all, in random order."""
  viewer: Literal["auto", "native", "viser"] = "auto"


def replay_env_cfg(task_id: str, cfg: ReplayConfig) -> ManagerBasedRlEnvCfg:
  """The task's play configuration with the library, the clips and the held failure."""
  env_cfg = load_env_cfg(task_id, play=True)
  motion = env_cfg.commands["motion"]
  assert isinstance(motion, mdp.MotionLibraryCommandCfg)
  if cfg.library_path is not None:
    motion.library_path = cfg.library_path
  motion.clips = cfg.clips
  failure = env_cfg.terminations["anchor_failure"]
  env_cfg.terminations["anchor_failure"] = replace(
    failure,
    func=mdp.anchor_failure_held,
    params={**failure.params, "seconds": FALL_HOLD_S},
  )
  return env_cfg


def run_replay(task_id: str, cfg: ReplayConfig) -> None:
  # mjlab's run_play loads its configuration from the registry, by task.
  replay_task = f"{task_id}-Replay"
  env_cfg = replay_env_cfg(task_id, cfg)
  register_mjlab_task(
    task_id=replay_task,
    env_cfg=env_cfg,
    play_env_cfg=env_cfg,
    rl_cfg=load_rl_cfg(task_id),
    runner_cls=load_runner_cls(task_id),
  )
  run_play(
    replay_task,
    PlayConfig(checkpoint_file=cfg.checkpoint_file, viewer=cfg.viewer, num_envs=1),
  )


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
    ReplayConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=flags,
  )
  run_replay(chosen_task, args)


if __name__ == "__main__":
  main()
