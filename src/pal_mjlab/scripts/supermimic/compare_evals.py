"""Compare benchmark results (eval.py JSONs) of two runs: curves and a paired per-clip test.

  python -m pal_mjlab.scripts.supermimic.compare_evals --a results/a --b results/b \\
    --label-a "3.5 Hz" --label-b "1 Hz"

Prints markdown: summary metrics per checkpoint for each run, then the final checkpoints
compared clip by clip (fraction of clips where each run is better, median difference).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import tyro

CURVE = (
  "failed",
  "tracking_return",
  "rmpkpe_legs",
  "rmpkpe_arms",
  "rmpkpe_torso",
  "joint_err_arms",
  "joint_err_waist",
  "torque_rate_arms",
  "saturation_arms",
)
# Paired metrics and whether lower is better.
PAIRED = {
  "rmpkpe_arms": True,
  "rmpkpe_legs": True,
  "rmpkpe_torso": True,
  "joint_err_arms": True,
  "tracking_return": False,
}


def _load(folder: Path) -> list[tuple[int, dict]]:
  runs = []
  for f in folder.glob("*.json"):
    it = int(re.search(r"model_(\d+)", f.stem).group(1))
    runs.append((it, json.loads(f.read_text())))
  return sorted(runs, key=lambda r: r[0])


def _curve(label: str, runs: list[tuple[int, dict]]) -> None:
  print(f"\n**{label}**\n")
  print("| iteration | " + " | ".join(CURVE) + " |")
  print("|" + "---|" * (len(CURVE) + 1))
  for it, r in runs:
    s = r["summary"]
    cells = [f"{s[k]:.0f}" if k == "failed" else f"{s[k]:.4f}" for k in CURVE]
    print(f"| {it} | " + " | ".join(cells) + " |")


def main(a: Path, b: Path, label_a: str = "A", label_b: str = "B") -> None:
  runs_a, runs_b = _load(a), _load(b)
  _curve(label_a, runs_a)
  _curve(label_b, runs_b)
  (it_a, ra), (it_b, rb) = runs_a[-1], runs_b[-1]
  ca, cb = ra["clips"], rb["clips"]
  assert ca["name"] == cb["name"], "results are on different clip sets"
  n = len(ca["name"])
  print(
    f"\n**Paired, final checkpoints ({label_a} it {it_a} vs {label_b} it {it_b}, {n} clips)**\n"
  )
  print(
    f"| metric | {label_a} mean | {label_b} mean | {label_b} better on | median {label_b} - {label_a} |"
  )
  print("|---|---|---|---|---|")
  for k, lower in PAIRED.items():
    va, vb = np.asarray(ca[k]), np.asarray(cb[k])
    better = (vb < va) if lower else (vb > va)
    print(
      f"| {k} | {va.mean():.4f} | {vb.mean():.4f} | {better.mean():.1%} | "
      f"{np.median(vb - va):+.4f} |"
    )
  fa, fb = np.asarray(ca["failed"]) > 0, np.asarray(cb["failed"]) > 0
  print(
    f"\nFailures: {label_a} {fa.sum()}, {label_b} {fb.sum()} "
    f"(both {(fa & fb).sum()}, only {label_a} {(fa & ~fb).sum()}, only {label_b} {(fb & ~fa).sum()})"
  )


if __name__ == "__main__":
  tyro.cli(main)
