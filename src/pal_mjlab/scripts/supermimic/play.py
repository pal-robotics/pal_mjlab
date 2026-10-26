"""Replay motions of a motion library with a trained SuperMimic policy, in MuJoCo physics (D43, D45):

  python -m pal_mjlab.scripts.supermimic.play Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers-ArmWaist1Hz \
    --checkpoint-file <run>/model_N.pt --library <library folder> [--clips 230324/flip_360_001__A304 ...]

The library is a folder with clips.json, qpos.npy and qvel.npy: what training reads and the motion
browser downloads. Without ``--clips`` its motions play in random order, one after another; with it
only those, so a single one repeats. Each starts from its first frame with the evaluation settings
(no randomization, no noise). There is no reset when the robot fails: it leaves the reference (the
benchmark's rule) and the physics goes on, so a fall is seen as it happens; the next motion starts
after ``FALL_HOLD_S`` seconds in that state, or at the clip's end.

``--viewer``: MuJoCo's window with a display, else viser in the browser (``auto``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from typing import Literal

import mjlab
import mjlab.tasks  # noqa: F401  (populates the task registry)
import tyro
from mjlab.managers import TerminationTermCfg
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.tasks import registry

from pal_mjlab.tasks.supermimic import mdp

FALL_HOLD_S = 3.0  # seconds a failure lasts on screen before the next motion


@dataclass(frozen=True)
class PlayArgs:
  checkpoint_file: str
  """Policy checkpoint (model_N.pt)."""
  library: str
  """Motion library folder (clips.json, qpos.npy, qvel.npy)."""
  clips: tuple[str, ...] = ()
  """Only these motions, by name (e.g. 230324/flip_360_001__A304); empty for all, in random order."""
  viewer: Literal["auto", "native", "viser"] = "auto"


def configure(task: str, args: PlayArgs) -> PlayConfig:
  """Set the library, the clips and the failure handling on the task's registered play config (mjlab's
  run_play loads a copy of it), and return run_play's config."""
  env_cfg = registry._REGISTRY[task].play_env_cfg
  motion = env_cfg.commands["motion"]
  motion.library_path = args.library
  motion.clips = args.clips
  failure = env_cfg.terminations["anchor_failure"]
  if failure.func is mdp.anchor_failure:
    env_cfg.terminations["anchor_failure"] = replace(
      failure,
      func=mdp.anchor_failure_held,
      params={**failure.params, "seconds": FALL_HOLD_S},
    )
  return PlayConfig(checkpoint_file=args.checkpoint_file, viewer=args.viewer, num_envs=1)


def main() -> None:
  task, remaining = tyro.cli(
    tyro.extras.literal_type_from_choices(registry.list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  # mjlab's flags but clips as plain names (--clips a/b c/d), not a Python tuple literal
  flags = tuple(
    f
    for f in mjlab.TYRO_FLAGS
    if f is not tyro.conf.UsePythonSyntaxForLiteralCollections
  )
  args = tyro.cli(PlayArgs, args=remaining, prog=f"{sys.argv[0]} {task}", config=flags)
  run_play(task, configure(task, args))


if __name__ == "__main__":
  main()
