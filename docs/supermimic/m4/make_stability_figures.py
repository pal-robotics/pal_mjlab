"""Figures and tables for the M4 stability study (docs/supermimic/m4/stability.md).

Reads the TensorBoard logs and benchmark JSONs of the stability runs (12 h each on Leonardo) and
of the old cluster baseline, and writes ``stability_*`` PDFs, PNGs and LaTeX tables to
``docs/supermimic/m4/figures/``. Run from the repository root:

  .venv/bin/python docs/supermimic/m4/make_stability_figures.py
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from make_figures import OUT, benchmarks, save, scalars, smooth  # noqa: E402

LOGS, EVAL = Path("logs/rsl_rl"), Path("logs/supermimic_eval")
TARGET = (
  0.742  # PPO-Sim's local 12 h tracking return: "take-off" for the collapse count
)
COLLAPSED = 0.6


def stability_run(name: str) -> Path:
  return Path(sorted(glob.glob(str(LOGS / f"supermimic_stability/*_{name}")))[-1])


# name: (log folder, benchmark folder, color, line style)
RUNS = {
  "baseline (seed 42)": (
    LOGS / "supermimic_wm_1hz/2026-10-24_11-21-34_dev-paper",
    EVAL / "wm_1hz_dev_paper_cluster", "#8d8b85", "--",
  ),
  "B (seed 43)": (stability_run("B-seed43"), EVAL / "stability_B-seed43", "#2a78d6", "-"),
  "A1 no critic": (stability_run("A1-nocritic"), EVAL / "stability_A1-nocritic", "#eb6834", "-"),
  "A2' rule head": (stability_run("A2r-rulehead"), EVAL / "stability_A2r-rulehead", "#1f9d6b", "-"),
  "A5 survival critic": (stability_run("A5-survival"), EVAL / "stability_A5-survival", "#8a56c2", "-"),
  "A6 no privileged": (stability_run("A6-noprivileged"), EVAL / "stability_A6-noprivileged", "#c23b4a", ":"),
}  # fmt: skip


def load() -> dict[str, dict]:
  runs = {}
  for name, (log, ev, color, ls) in RUNS.items():
    s = scalars(log)
    last = max(e.step for e in s["world_model/loss"])
    b = {it: v for it, v in benchmarks(ev).items() if it <= 49_300}
    runs[name] = dict(scalars=s, bench=b, color=color, ls=ls, last=last)
  return runs


def series(run: dict, tag: str, lo: int = 0, hi: int = 10**9):
  pts = [(e.step, e.value) for e in run["scalars"].get(tag, []) if lo <= e.step <= hi]
  x, y = (
    (np.array(v) for v in zip(*pts, strict=True)) if pts else (np.zeros(0), np.zeros(0))
  )
  return x, y


def spikes(
  run: dict, hi: int, start: int = 5000, gap: int = 300, level: float = 0.5
) -> int:
  """Onsets of failure spikes in training (share of sim episodes ending in a failure)."""
  x, y = series(run, "episode/failure", start, hi)
  n, last = 0, -(10**9)
  for k, v in zip(x, y, strict=True):
    if v > level:
      if k - last > gap:
        n += 1
      last = k
  return n


def fig_benchmark(runs: dict) -> None:
  fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.7))
  for name, r in runs.items():
    b = r["bench"]
    if not b:
      continue
    kw = dict(color=r["color"], ls=r["ls"], lw=1.2, marker="o", ms=2.0, label=name)
    ax[0].plot(list(b), [v["tracking_return"] for v in b.values()], **kw)
    ax[1].plot(list(b), [max(v["failed"], 1) for v in b.values()], **kw)
  ax[0].set_title("Benchmark tracking return")
  ax[0].axhline(COLLAPSED, color="#8d8b85", lw=0.6, ls=":")
  ax[1].set_title("Clips failed (of 1296)")
  ax[1].set_yscale("log")
  for a in ax:
    a.set_xlabel("iteration")
  ax[0].legend(fontsize=6.3, loc="lower right")
  fig.tight_layout()
  save(fig, "stability_benchmark")


def fig_training(runs: dict) -> None:
  panels = (
    ("policy/value", "Critic value V (smoothed)", 25, None),
    ("episode/failure", "Sim episodes ending in failure (smoothed)", 25, None),
    ("policy/critic_loss", "Critic loss", 1, "log"),
    ("policy/actor_grad_norm", "Policy gradient norm, before clipping", 1, "log"),
  )
  fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.4))
  for a, (tag, title, n, scale) in zip(axes.flat, panels, strict=True):
    for name, r in runs.items():
      x, y = series(r, tag)
      if len(x) == 0:
        continue
      a.plot(
        x,
        smooth(y, n) if n > 1 else y,
        color=r["color"],
        ls=r["ls"],
        lw=0.8,
        label=name,
      )
    a.set_title(title)
    a.set_xlabel("iteration")
    if scale:
      a.set_yscale(scale)
  axes[0, 0].set_ylim(-20, 20)
  axes[0, 0].legend(fontsize=6, ncol=2)
  fig.tight_layout()
  save(fig, "stability_training")


def fig_early(runs: dict) -> None:
  """The first 4000 iterations: the early critic blow-up and the take-off."""
  fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.5))
  for name, r in runs.items():
    for a, tag in zip(ax, ("policy/value", "episode/failure"), strict=True):
      x, y = series(r, tag, 0, 4000)
      a.plot(x, y, color=r["color"], ls=r["ls"], lw=0.9, label=name)
  ax[0].set_title("Critic value, first 4000 iterations")
  ax[0].set_yscale("symlog", linthresh=10)
  ax[1].set_title("Sim episodes ending in failure")
  for a in ax:
    a.set_xlabel("iteration")
  handles, labels = ax[1].get_legend_handles_labels()
  fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=6.3, frameon=False)
  fig.tight_layout(rect=(0, 0.08, 1, 1))
  save(fig, "stability_early")


def summary(runs: dict) -> list[dict]:
  rows = []
  cut = min(r["last"] for n, r in runs.items() if not n.startswith(("baseline", "A6")))
  for name, r in runs.items():
    b = {it: v for it, v in r["bench"].items() if it <= cut}
    hi = min(cut, r["last"])
    _, value = series(r, "policy/value", 0, 3000)
    _, closs = series(r, "policy/critic_loss", 3000, hi)
    first = next((it for it, v in b.items() if v["tracking_return"] >= TARGET), None)
    late = [it for it in b if first is not None and it >= first]
    best = max(b, key=lambda it: b[it]["tracking_return"]) if b else None
    rows.append(
      dict(
        name=name,
        up_to=hi,
        benchmarked=len(b),
        early_value_min=float(value.min()) if len(value) else float("nan"),
        take_off=first,
        best=(best, b[best]["failed"], b[best]["tracking_return"]) if best else None,
        last=(max(b), b[max(b)]["failed"], b[max(b)]["tracking_return"]) if b else None,
        collapses=sum(b[it]["tracking_return"] < COLLAPSED for it in late),
        late=len(late),
        over_50=sum(b[it]["failed"] > 50 for it in late),
        spikes_per_10k=spikes(r, hi) / max(hi - 5000, 1) * 1e4,
        critic_bursts=int((closs > 1e3).sum()),
      )  # fmt: skip
    )
  lines = [
    r"\begin{tabular}{lrrrrrrrr}\toprule",
    r"run & up to & early min V & take-off & best & last & collapses & spikes/10k & critic bursts \\\midrule",
  ]
  for x in rows:
    best = f"{x['best'][1]} / {x['best'][2]:.3f}" if x["best"] else "--"
    last = f"{x['last'][1]} / {x['last'][2]:.3f}" if x["last"] else "--"
    take = f"{x['take_off']:,}" if x["take_off"] else "--"
    lines.append(
      f"{x['name'].replace('_', ' ')} & {x['up_to']:,} & {x['early_value_min']:.1f} & {take} & {best} & "
      f"{last} & {x['collapses']} of {x['late']} & {x['spikes_per_10k']:.1f} & {x['critic_bursts']} \\\\"
    )
  lines.append(r"\bottomrule\end{tabular}")
  (OUT / "tab_stability_summary.tex").write_text("\n".join(lines) + "\n")
  return rows


def main() -> None:
  runs = load()
  fig_benchmark(runs)
  fig_training(runs)
  fig_early(runs)
  for row in summary(runs):
    print(row)


if __name__ == "__main__":
  main()
