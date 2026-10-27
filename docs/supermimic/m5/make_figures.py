"""Figures for the M5 runs (docs/supermimic/m5/runs.md): the benchmark of the 24 h `dev` run and of the
`full` run's first job on the reviewed 1,294-clip `dev` (paper F.1: every clip once from its first frame),
every 2000 iterations (the odd thousands are benchmarked after these and left out, so both runs match).
Run from the repository root:

  .venv/bin/python docs/supermimic/m5/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[1] / "m4"))
from make_figures import ALT, INK2, SM, benchmarks  # noqa: E402  (also sets the plot style)

EVAL = Path("logs/supermimic_eval")
OUT = Path("docs/supermimic/m5/figures")
RUNS = {  # label: (benchmark folder, color)
  "dev (1,294 clips)": (EVAL / "m5_dev-24h", SM),
  "full (124,077 clips), first job": (EVAL / "m5_full-24h", ALT),
}
# Earlier results on the 1,296-clip dev (it also holds the 2 clips removed in D40, which every policy fails):
REFERENCES = [  # label, tracking return, label above (+1) or below (-1) the line
  ("PPO-Sim, 24 h", 0.770, 1),
  ("old SuperMimic baseline, best of 24 h", 0.872, -1),
  ("stability runs, best (A2′)", 0.894, -1),
]
FAIL_TOP = 12  # failure axis cut here, larger counts are marked at the top


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  data = {k: {it: v for it, v in benchmarks(folder).items() if it % 2000 == 0} for k, (folder, _) in RUNS.items()}
  fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 2.7))
  for label, (_, color) in RUNS.items():
    its = [it / 1000 for it in data[label]]
    ret = [s["tracking_return"] for s in data[label].values()]
    fails = [s["failed"] for s in data[label].values()]
    a.plot(its, ret, "o-", color=color, label=label, markersize=3)
    b.plot(its, [min(f, FAIL_TOP) for f in fails], "o-", color=color, label=label, markersize=3)
    for x, f in zip(its, fails):
      if f > FAIL_TOP:
        b.annotate(str(f), (x, FAIL_TOP), textcoords="offset points", xytext=(4, -3), fontsize=7, color=color)
  right = max(max(d) for d in data.values()) / 1000
  for text, y, side in REFERENCES:
    a.axhline(y, color=INK2, linewidth=0.8, linestyle="--")
    a.text(right, y + side * 0.003, text, fontsize=6.5, color=INK2, ha="right", va="bottom" if side > 0 else "top")
  a.set_xlabel("iteration (thousands)")
  a.set_ylabel("tracking return")
  a.set_ylim(0.76, 0.92)
  a.set_title("Benchmark on dev: tracking return")
  b.set_xlabel("iteration (thousands)")
  b.set_ylabel("failed clips (of 1,294)")
  b.set_ylim(-0.5, FAIL_TOP + 0.5)
  b.set_title("Benchmark on dev: failed clips")
  b.legend(loc="upper right")
  fig.tight_layout()
  fig.savefig(OUT / "benchmark_dev.pdf", bbox_inches="tight")
  fig.savefig(OUT / "benchmark_dev.png", bbox_inches="tight", dpi=160)
  plt.close(fig)
  for label in RUNS:
    best = max(data[label].items(), key=lambda kv: kv[1]["tracking_return"])
    print(f"{label}: {len(data[label])} checkpoints, best {best[0]}: "
          f"{best[1]['failed']} fails, {best[1]['tracking_return']:.3f}")


if __name__ == "__main__":
  main()
