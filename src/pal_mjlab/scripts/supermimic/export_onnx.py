"""Export SuperMimic checkpoints as ONNX policies for the robot (D48). Training writes model_N.onnx at
every save; this covers checkpoints saved before that, e.g. a run's best one:

  python -m pal_mjlab.scripts.supermimic.export_onnx Mjlab-SuperMimic-Flat-Pal-Kangaroo-Hands-ArmWaist1Hz \
    --library <library folder> <run>/model_N.pt [<run>/model_M.pt ...]

Writes model_N.onnx next to each checkpoint (format: ``tasks/supermimic/rl/export.py``). The library
only sets up the env: any library of the task's robot, e.g. ``smoke``.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mjlab
import mjlab.tasks  # noqa: F401  (populates the task registry)
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

from pal_mjlab.tasks.supermimic.rl.export import export_policy
from pal_mjlab.tasks.supermimic.rl.runner import SuperMimicRunner


@dataclass(frozen=True)
class ExportArgs:
  checkpoints: tyro.conf.Positional[tuple[str, ...]]
  """Checkpoints (model_N.pt) of runs of the task."""
  library: str
  """Motion library folder of the task's robot (clips.json, qpos.npy, qvel.npy)."""
  device: str = "cpu"


def main() -> None:
  task, remaining = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  flags = tuple(
    f
    for f in mjlab.TYRO_FLAGS
    if f is not tyro.conf.UsePythonSyntaxForLiteralCollections
  )
  args = tyro.cli(
    ExportArgs, args=remaining, prog=f"{sys.argv[0]} {task}", config=flags
  )
  env_cfg = load_env_cfg(task, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.commands["motion"].library_path = args.library
  env_cfg.commands["motion"].compile_kinematics = False
  agent_cfg = load_rl_cfg(task)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  runner = SuperMimicRunner(
    RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions),
    {**asdict(agent_cfg), "compile": False},
    device=args.device,
  )
  for checkpoint in args.checkpoints:
    runner.load(checkpoint, load_cfg={"actor": True}, map_location=args.device)
    out = Path(checkpoint).with_suffix(".onnx")
    export_policy(env, runner.alg.actor, out, checkpoint)
    print(f"wrote {out}")
  env.close()


if __name__ == "__main__":
  main()
