"""Train the SuperMimic world model offline on collected rollouts (M2).

Paper settings (Section 3.2, Table A.5): k-step autoregressive windows (k = 8), Adam with
lr 2e-4, batches of 2048 windows, global gradient-norm clipping 1.0, TF32. Normalization
statistics are computed once from training transitions and frozen. Held-out clips are
never trained on; their 8- and 32-step losses are logged every ``val_every`` steps and
the best 8-step checkpoint is kept.

  python -m pal_mjlab.scripts.supermimic.train_world_model \\
    --data logs/supermimic_m2/data/model_1000 ... --out-dir logs/supermimic_m2/wm/main
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
import tyro
from mjlab.utils.torch import configure_torch_backends
from pal_mjlab.tasks.supermimic.library import MotionLibrary
from pal_mjlab.tasks.supermimic.rl.replay import RolloutData, holdout_clip_mask
from pal_mjlab.tasks.supermimic.rl.world_model import (
  WorldModel,
  WorldModelCfg,
  window_loss,
)
from pal_mjlab.tasks.supermimic.state import pose_delta
from torch.utils.tensorboard import SummaryWriter

DEFAULT_LIBRARY = "~/Projects/supermimic/kangaroo_bones_seed/data/library/dev"


def batch_loss(model: WorldModel, w: dict[str, torch.Tensor]):
  return window_loss(
    model,
    w["states"],
    w["poses"],
    w["actions"],
    w["context"],
    w["ref_root"],
    w["failure"],
  )


def main(
  data: list[Path],
  out_dir: Path,
  library: str = DEFAULT_LIBRARY,
  steps: int = 50_000,
  batch: int = 2048,
  k: int = 8,
  hidden: tuple[int, ...] = (2048, 2048, 2048),
  activation: str = "elu",
  predict_delta: bool = True,
  lr: float = 2e-4,
  holdout_fraction: float = 0.1,
  train_sources: list[str] | None = None,
  val_every: int = 2000,
  val_windows: int = 8192,
  seed: int = 0,
  device: str = "cuda",
) -> None:
  configure_torch_backends()
  torch.manual_seed(seed)
  gen = torch.Generator(device=device).manual_seed(seed)
  lib = MotionLibrary(library, device)
  rollouts = RolloutData(data, lib, device)
  held_out = torch.from_numpy(holdout_clip_mask(lib.clip_names, holdout_fraction)).to(
    device
  )
  train_starts = rollouts.valid_starts(k, clips=~held_out, sources=train_sources)
  val = {}
  for h in (8, 32):
    starts = rollouts.valid_starts(h, clips=held_out)
    val[h] = starts[
      torch.randperm(len(starts), generator=gen, device=device)[:val_windows]
    ]
  print(
    f"{rollouts.T}x{rollouts.E} transitions; train windows {len(train_starts)}, "
    f"held-out clips {int(held_out.sum())}/{len(held_out)}",
    flush=True,
  )

  cfg = WorldModelCfg(
    state_dim=rollouts.state.shape[-1],
    action_dim=rollouts.action.shape[-1],
    context_dim=rollouts.context.shape[-1],
    hidden=hidden,
    activation=activation,
    predict_delta=predict_delta,
  )
  model = WorldModel(cfg).to(device)
  sample = train_starts[
    torch.randint(0, len(train_starts), (200_000,), device=device, generator=gen)
  ]
  w = rollouts.windows(sample, 1)
  model.set_normalization(
    w["states"][:, 0],
    w["actions"][:, 0],
    w["context"],
    w["states"][:, 1],
    pose_delta(w["poses"][:, 0], w["poses"][:, 1]),
  )
  opt = torch.optim.Adam(model.parameters(), lr=lr)

  out_dir.mkdir(parents=True, exist_ok=True)
  run = {
    "data": [str(d) for d in data],
    "library": library,
    "steps": steps,
    "batch": batch,
    "k": k,
    "lr": lr,
    "holdout_fraction": holdout_fraction,
    "train_sources": train_sources,
    "seed": seed,
    "model": asdict(cfg),
  }
  (out_dir / "run.json").write_text(json.dumps(run, indent=1))
  writer = SummaryWriter(out_dir / "tb")

  def save(path: Path, step: int, val_losses: dict) -> None:
    torch.save(
      {
        "cfg": asdict(cfg),
        "state_dict": model.state_dict(),
        "step": step,
        "val": val_losses,
      },
      path,
    )

  @torch.no_grad()
  def validate() -> dict[str, float]:
    out = {}
    for h, starts in val.items():
      parts = []
      for chunk in starts.split(2048):
        _, info = batch_loss(model, rollouts.windows(chunk, h))
        parts.append(info)
      for key in parts[0]:
        out[f"val{h}/{key}"] = float(torch.stack([p[key] for p in parts]).mean())
    return out

  best = float("inf")
  t0 = time.perf_counter()
  for step in range(1, steps + 1):
    idx = train_starts[
      torch.randint(0, len(train_starts), (batch,), device=device, generator=gen)
    ]
    loss, info = batch_loss(model, rollouts.windows(idx, k))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 100 == 0:
      writer.add_scalar("train/loss", loss.item(), step)
      writer.add_scalar("train/grad_norm", grad_norm.item(), step)
      for key, v in info.items():
        writer.add_scalar(f"train/{key}", v.item(), step)
    if step % val_every == 0 or step == steps:
      v = validate()
      for key, value in v.items():
        writer.add_scalar(key, value, step)
      save(out_dir / "wm_last.pt", step, v)
      if v["val8/state"] + v["val8/pose"] < best:
        best = v["val8/state"] + v["val8/pose"]
        save(out_dir / "wm_best.pt", step, v)
      elapsed = (time.perf_counter() - t0) / 60
      print(
        f"step {step} ({elapsed:.1f} min): train {loss.item():.4f} | "
        + " ".join(f"{key} {value:.4f}" for key, value in v.items()),
        flush=True,
      )
  writer.close()


if __name__ == "__main__":
  tyro.cli(main)
