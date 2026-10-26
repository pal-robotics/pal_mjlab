"""mjlab's ``play`` on a chosen motion library and clips (D43): a downloaded library, a subset
of a dataset, or one motion of either. Same CLI as ``mjlab play`` plus ``--library`` and
``--clips``:

  python -m pal_mjlab.scripts.supermimic.play Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers-ArmWaist1Hz \
    --checkpoint-file <run>/model_N.pt --library <library folder> --clips 230324/flip_360_001__A304

Each episode plays one of the clips from its first frame (the paper's evaluation setting), so
a single clip repeats. Without ``--library`` the task's default library plays (its dataset's
``dev`` subset). ``mjlab play`` itself has no way to set them: it loads the registered play
config as is.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import mjlab
import mjlab.tasks  # noqa: F401  (populates the task registry)
import tyro
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.tasks import registry


@dataclass(frozen=True)
class SuperMimicPlayConfig(PlayConfig):
  library: str = ""
  """Motion library folder (qpos.npy, qvel.npy, clips.json); empty for the task's default."""
  clips: tuple[str, ...] = ()
  """Only these clips of the library, by name (e.g. 230324/flip_360_001__A304); empty for
  all."""


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
  args = tyro.cli(
    SuperMimicPlayConfig,
    args=remaining,
    default=SuperMimicPlayConfig(),
    prog=f"{sys.argv[0]} {task}",
    config=flags,
  )
  # run_play loads a copy of the registered play config: set the motions on that one.
  motion = registry._REGISTRY[task].play_env_cfg.commands["motion"]
  if args.library:
    motion.library_path = args.library
  motion.clips = args.clips
  run_play(task, args)


if __name__ == "__main__":
  main()
