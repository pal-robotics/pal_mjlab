"""Figures for the M5 runs (docs/supermimic/m5/runs.md, report.tex):

- benchmark_dev: the benchmark of the 24 h `dev` run and of the `full` run (first job up to 44,000, then the job
  resumed from it, D44) on the reviewed 1,294-clip `dev` (paper F.1: every clip once from its first frame), every
  2000 iterations and the last checkpoint;
- heldout: the final policies on a second stratified 2.5 h sample of `full` (seed 1, 1,242 clips, 14 of them in
  `dev`), next to their `dev` benchmark;
- training: training signals of both runs from their TensorBoard logs.

Run from the repository root:

  .venv/bin/python docs/supermimic/m5/make_figures.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "m4"))
from make_figures import ALT, INK2, SM, benchmarks, scalars, smooth  # noqa: E402  (also sets the plot style)

EVAL = Path("logs/supermimic_eval")
LOGS = Path("logs/rsl_rl/supermimic_d39")
OUT = Path("docs/supermimic/m5/figures")
DEV_RUN, FULL1, FULL2 = "2026-10-26_01-55-17_dev-24h", "2026-10-26_01-55-21_full-24h", "2026-10-26_19-30-13_full-24h"
RESUMED = 44_000  # full resumed from this checkpoint with njmax 1024 (D44)
RUNS = {  # label: (benchmark folder, TensorBoard folders in order, color)
  "dev (1,294 clips)": (EVAL / "m5_dev-24h", [DEV_RUN], SM),
  "full (124,077 clips)": (EVAL / "m5_full-24h", [FULL1, FULL2], ALT),
}
# Earlier results on the 1,296-clip dev (it also holds the 2 clips removed in D40, which every policy fails):
REFERENCES = [  # label, tracking return, label above (+1) or below (-1) the line, at the left or right end
  ("PPO-Sim, 24 h", 0.770, 1, "right"),
  ("old SuperMimic baseline, best of 24 h", 0.872, -1, "right"),
  ("stability runs, best (A2′)", 0.894, 1, "left"),
]
HELDOUT = [  # label, run folder, iteration
  ("dev, 71k", DEV_RUN, 71_000),
  ("full, 44k", FULL1, 44_000),
  ("full, 70k", FULL2, 70_000),
  ("full, 121k", FULL2, 121_000),
]
FAIL_TOP = 12  # failure axis cut here, larger counts are marked at the top
RETURN_BOTTOM = 0.76  # return axis cut here, lower values are marked at the bottom


def save(fig, name: str) -> None:
  fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
  fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=160)
  plt.close(fig)


def summary(folder: Path, run: str, it: int) -> dict:
  return json.loads((folder / f"{run}_model_{it}.json").read_text())["summary"]


def fig_benchmark() -> None:
  every = {k: benchmarks(f) for k, (f, _, _) in RUNS.items()}  # every 2000 iterations and the last checkpoint
  data = {k: {it: v for it, v in b.items() if it % 2000 == 0 or it == max(b)} for k, b in every.items()}
  fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.8))
  for label, (_, _, color) in RUNS.items():
    its = [it / 1000 for it in data[label]]
    fails = [s["failed"] for s in data[label].values()]
    ret = [s["tracking_return"] for s in data[label].values()]
    a.plot(its, [max(r, RETURN_BOTTOM) for r in ret], "o-", color=color, label=label, markersize=2.5)
    for x, r in zip(its, ret, strict=True):
      if r < RETURN_BOTTOM:
        a.annotate(f"{r:.2f}", (x, RETURN_BOTTOM), textcoords="offset points", xytext=(4, 1), fontsize=7, color=color)
    b.plot(its, [min(f, FAIL_TOP) for f in fails], "o-", color=color, label=label, markersize=2.5)
    for x, f in zip(its, fails, strict=True):
      if f > FAIL_TOP:
        b.annotate(str(f), (x, FAIL_TOP), textcoords="offset points", xytext=(4, -3), fontsize=7, color=color)
  right = max(max(d) for d in data.values()) / 1000
  for text, y, side, end in REFERENCES:
    a.axhline(y, color=INK2, linewidth=0.8, linestyle="--")
    a.text(right if end == "right" else 0, y + side * 0.002, text, fontsize=6.5, color=INK2, ha=end,
           va="bottom" if side > 0 else "top")
  for ax in (a, b):
    ax.axvline(RESUMED / 1000, color=ALT, linewidth=0.8, linestyle=":")
    ax.set_xlabel("iteration (thousands)")
  b.text(RESUMED / 1000 + 1, FAIL_TOP - 1, "full resumed\n(njmax 1024)", fontsize=6.5, color=ALT, va="top")
  a.set_ylabel("tracking return")
  a.set_ylim(RETURN_BOTTOM - 0.005, 0.92)
  a.set_title("Benchmark on dev: tracking return")
  b.set_ylabel("failed clips (of 1,294)")
  b.set_ylim(-0.5, FAIL_TOP + 0.5)
  b.set_title("Benchmark on dev: failed clips")
  a.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1))
  fig.tight_layout()
  save(fig, "benchmark_dev")
  for label in RUNS:
    best = max(data[label].items(), key=lambda kv: kv[1]["tracking_return"])
    print(f"{label}: {len(data[label])} checkpoints, best {best[0]}: {best[1]['failed']} fails, "
          f"{best[1]['tracking_return']:.3f}")


def fig_heldout() -> None:
  rows = []
  for label, run, it in HELDOUT:
    dev = summary(EVAL / ("m5_dev-24h" if run == DEV_RUN else "m5_full-24h"), run, it)
    held = summary(EVAL / "m5_seed1", run, it)
    rows.append((label, dev, held))
  fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.6))
  x = np.arange(len(rows))
  for ax, key, title in ((a, "tracking_return", "Tracking return"), (b, "failed", "Failed clips")):
    for dx, (name, idx, color) in zip((-0.12, 0.12), (("dev (1,294 clips)", 1, SM), ("second sample of full (1,242 clips)", 2, ALT)),
                                      strict=True):
      vals = [r[idx][key] for r in rows]
      ax.plot(x + dx, vals, "o", color=color, label=name, markersize=6)
      for xi, v in zip(x + dx, vals, strict=True):
        ax.annotate(f"{v:.3f}" if key == "tracking_return" else str(v), (xi, v), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=6.5, color=INK2)
    ax.set_xticks(x, [r[0] for r in rows])
    ax.set_title(title)
  a.set_ylabel("tracking return")
  b.set_ylabel("clips")
  b.set_ylim(bottom=-1)
  a.legend(loc="lower right")
  fig.tight_layout()
  save(fig, "heldout")
  for label, dev, held in rows:
    print(f"{label}: dev {dev['failed']} / {dev['tracking_return']:.3f}, held-out {held['failed']} / "
          f"{held['tracking_return']:.3f}")


def series(folders: list[str], tag: str) -> tuple[np.ndarray, np.ndarray]:
  """(iteration, value) of a scalar, each step from the latest folder that has it (the resumed job overrides)."""
  points: dict[int, float] = {}
  for f in folders:
    for e in scalars(LOGS / f).get(tag, []):
      points[e.step] = e.value
  steps = np.array(sorted(points))
  return steps, np.array([points[s] for s in steps])


def fig_training() -> None:
  panels = (
    ("world_model/loss", "World-model loss (Eq. 1)", 25, "log"),
    ("episode/failure", "Sim episodes ending in failure", 101, None),
    ("env/Metrics/motion/sampler_mean_return", "Sampler: mean normalized bin return", 1, None),
    ("env/Metrics/motion/sampler_coverage", "Sampler: share of bins visited", 1, None),
  )
  fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.6))
  for ax, (tag, title, n, scale) in zip(axes.flat, panels, strict=True):
    for label, (_, folders, color) in RUNS.items():
      x, y = series(folders, tag)
      if len(x):
        ax.plot(x / 1000, smooth(y, n) if n > 1 else y, color=color, lw=0.9, label=label)
    ax.axvline(RESUMED / 1000, color=ALT, linewidth=0.8, linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("iteration (thousands)")
    if scale:
      ax.set_yscale(scale)
  axes[1, 1].legend(loc="lower right")
  fig.tight_layout()
  save(fig, "training")


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  fig_benchmark()
  if all((EVAL / "m5_seed1" / f"{run}_model_{it}.json").exists() for _, run, it in HELDOUT):
    fig_heldout()
  fig_training()


if __name__ == "__main__":
  main()
