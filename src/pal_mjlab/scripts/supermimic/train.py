"""Resumable training launcher: mjlab's ``train`` that continues from the latest checkpoint.

Local GPU jobs can be preempted and rerun from the same command, so this picks the
highest-iteration ``model_*.pt`` among the runs of ``<log_root>/<experiment_name>/`` that have
the same ``--agent.run-name`` and resumes from it; with no checkpoint it starts fresh. Same CLI
as ``mjlab train``:

  python -m pal_mjlab.scripts.supermimic.train Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Grippers
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import mjlab
import mjlab.tasks  # noqa: F401  (populates the task registry)
import tyro
from mjlab.scripts.train import TrainConfig, launch_training
from mjlab.tasks.registry import list_tasks


def latest_checkpoint(experiment_dir: Path, run_name: str = "") -> Path | None:
  runs = f"*_{run_name}" if run_name else "*"
  checkpoints = [
    (int(p.stem.removeprefix("model_")), p)
    for p in experiment_dir.glob(f"{runs}/model_*.pt")
  ]
  return max(checkpoints)[1] if checkpoints else None


def main() -> None:
  task, remaining = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    TrainConfig,
    args=remaining,
    default=TrainConfig.from_task(task),
    prog=f"{sys.argv[0]} {task}",
    config=mjlab.TYRO_FLAGS,
  )
  checkpoint = latest_checkpoint(
    Path(args.log_root) / args.agent.experiment_name, args.agent.run_name
  )
  if checkpoint is not None:
    print(f"[INFO] Resuming from {checkpoint}")
    agent = replace(
      args.agent,
      resume=True,
      load_run=checkpoint.parent.name,
      load_checkpoint=checkpoint.name,
    )
    args = replace(args, agent=agent)
  launch_training(task_id=task, args=args)


if __name__ == "__main__":
  main()
