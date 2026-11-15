"""Measure contact and constraint counts of the safe-fall task to size buffers.

MuJoCo Warp preallocates its contact (``nconmax``) and constraint (``njmax``)
buffers. Too small and contacts are silently dropped during a fall; too large
and the simulation does not fit on the GPU. This script runs the task with
oversized buffers under the hardest kicks, records the peaks actually reached,
and reports GPU memory and throughput for a candidate configuration.

Measure the peaks:

  uv run python -m pal_mjlab.scripts.measure_fall_buffers --num-envs 1024

Check that a candidate configuration fits and how fast it runs:

  uv run python -m pal_mjlab.scripts.measure_fall_buffers --num-envs 4096 \
      --nconmax 64 --njmax 300 --steps 200
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


@dataclass
class Args:
  task: str = "Mjlab-SafeFall-Flat-Pal-Kangaroo"
  num_envs: int = 1024
  steps: int = 600
  """Control steps to run (an episode is 200)."""
  nconmax: int = 256
  """Contacts per world to allocate. Oversized by default, to measure peaks."""
  njmax: int = 1000
  """Constraint rows per world to allocate. Oversized by default."""
  max_kick_speed: float = 5.0
  """Kick range upper bound; use the end of the curriculum for the worst case."""
  action_std: float = 1.0
  """Std of the random actions; 0 holds the default pose."""
  device: str = "cuda:0"


def main(args: Args) -> None:
  cfg = load_env_cfg(args.task)
  cfg.scene.num_envs = args.num_envs
  cfg.sim.nconmax = args.nconmax
  cfg.sim.njmax = args.njmax
  cfg.curriculum = {}
  cfg.events["reset_fall_start"].params["max_kick_speed"] = args.max_kick_speed

  free_before, total = torch.cuda.mem_get_info(args.device)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  env.reset()
  wp_data = env.sim.wp_data
  action_dim = env.action_manager.total_action_dim

  peak_nefc = 0
  peak_ncon_world = 0
  peak_ncon_mean = 0.0
  nefc_samples = []
  ncon_samples = []
  start = time.time()
  for _ in range(args.steps):
    action = args.action_std * torch.randn(args.num_envs, action_dim, device=env.device)
    env.step(action)
    nefc = wp_data.nefc.numpy()
    nacon = int(wp_data.nacon.numpy()[0])
    world_ids = wp_data.contact.worldid.numpy()[:nacon]
    ncon = np.bincount(world_ids, minlength=args.num_envs)
    peak_nefc = max(peak_nefc, int(nefc.max()))
    peak_ncon_world = max(peak_ncon_world, int(ncon.max()))
    peak_ncon_mean = max(peak_ncon_mean, nacon / args.num_envs)
    nefc_samples.append(nefc)
    ncon_samples.append(ncon)
  elapsed = time.time() - start
  torch.cuda.synchronize()
  free_after, _ = torch.cuda.mem_get_info(args.device)

  nefc_all = np.concatenate(nefc_samples)
  ncon_all = np.concatenate(ncon_samples)
  gib = 2**30
  print()
  print(f"task {args.task}, {args.num_envs} envs, {args.steps} control steps")
  print(f"allocated: nconmax={args.nconmax} njmax={args.njmax}")
  print(
    f"constraint rows per world: peak {peak_nefc}, "
    f"p99.9 {np.percentile(nefc_all, 99.9):.0f}, mean {nefc_all.mean():.1f}"
  )
  print(
    f"contacts per world:        peak {peak_ncon_world}, "
    f"p99.9 {np.percentile(ncon_all, 99.9):.0f}, mean {ncon_all.mean():.1f}"
  )
  print(f"contacts averaged over worlds, worst step: {peak_ncon_mean:.1f}")
  print(
    f"  -> njmax must exceed {peak_nefc} (hard per-world limit); "
    f"nconmax must exceed {peak_ncon_mean:.1f} (shared pool, per-world average)"
  )
  if peak_nefc >= args.njmax or peak_ncon_mean >= args.nconmax:
    print("  !! a buffer was saturated: the peaks above are a lower bound")
  print(f"throughput: {args.steps * args.num_envs / elapsed:,.0f} env-steps/s")
  print(
    f"GPU memory: this process uses {(free_before - free_after) / gib:.2f} GiB, "
    f"{free_after / gib:.2f} GiB free of {total / gib:.2f} GiB"
  )
  env.close()


if __name__ == "__main__":
  main(tyro.cli(Args))
