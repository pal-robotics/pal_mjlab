"""Figures and tables for the M4 cluster report: SuperMimic and PPO-Sim on ``dev``, local vs Leonardo.

Reuses M4's helpers (``make_figures.py``: Clock, scalars, benchmarks, smooth, style) and reads the
TensorBoard logs and benchmark JSONs of four runs. Writes ``cluster_*`` PDFs, PNGs and LaTeX tables
to ``docs/supermimic/m4/figures/``. Run from the repository root:

  .venv/bin/python docs/supermimic/m4/make_cluster_figures.py
"""

from __future__ import annotations

import sys
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from make_figures import (  # noqa: E402
  INK2,
  OUT,
  PPO,
  SM,
  Clock,
  benchmarks,
  save,
  scalars,
  smooth,
)

LOGS, EVAL = Path("logs/rsl_rl"), Path("logs/supermimic_eval")
# name: (log folder, benchmark folder, clock tag, sim steps per iteration, color, line style)
RUNS = {
  "SuperMimic, cluster": (
    LOGS / "supermimic_wm_1hz/2026-10-24_11-21-34_dev-paper",
    EVAL / "wm_1hz_dev_paper_cluster", "world_model/loss", 2.5 * 4096, SM, "-",
  ),
  "SuperMimic, local": (
    LOGS / "supermimic_wm_1hz/2026-10-23_20-13-19_dev-local",
    EVAL / "wm_1hz_dev_local", "world_model/loss", 2.5 * 4096, SM, "--",
  ),
  "PPO-Sim, cluster": (
    LOGS / "supermimic_ppo_sim_1hz/2026-10-24_11-21-37_dev-cluster",
    EVAL / "ppo_sim_1hz_dev_cluster", "Perf/total_fps", 24 * 8192, PPO, "-",
  ),
  "PPO-Sim, local": (
    LOGS / "supermimic_ppo_sim_1hz/2026-10-22_21-11-56",
    EVAL / "ppo_sim_1hz", "Perf/total_fps", 24 * 8192, PPO, "--",
  ),
}  # fmt: skip
SM_RUNS = ("SuperMimic, cluster", "SuperMimic, local")
SM_LOCAL = "#9dc1ec"  # lighter tint of SM for the local run where lines overlap
COLLAPSED = (
  0.6  # tracking return below this, after first exceeding PPO-Sim's local 12 h level
)


def load() -> dict[str, dict]:
  runs = {}
  for name, (log, ev, tag, steps, color, ls) in RUNS.items():
    s = scalars(log)
    b = benchmarks(ev)
    if name.startswith("PPO"):  # every 1000 iterations, plus the last
      b = {it: v for it, v in b.items() if it % 1000 == 0 or it == max(b)}
    runs[name] = dict(
      scalars=s, clock=Clock([s], tag), bench=b, steps=steps, color=color, ls=ls
    )
  return runs


def fig_benchmark(runs: dict) -> None:
  fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
  for name, r in runs.items():
    b, clock = r["bench"], r["clock"]
    if not b:
      continue
    h = [clock.hours(it) for it in b]
    kw = dict(color=r["color"], ls=r["ls"], lw=1.3, label=name, marker="o", ms=2.2)
    if r["ls"] == "--":
      kw.update(mfc="white")
    ax[0].plot(h, [v["tracking_return"] for v in b.values()], **kw)
    ax[1].plot(h, [max(v["failed"], 1) for v in b.values()], **kw)
    ax[2].plot(h, [100 * v["rmpkpe_arms"] for v in b.values()], **kw)
  ax[0].set_title("Tracking return")
  ax[1].set_title("Clips failed (of 1296)")
  ax[1].set_yscale("log")
  ax[2].set_title("Arm position error [cm]")
  ax[2].set_ylim(5, 30)
  for a in ax:
    a.set_xlabel("training hours")
  ax[0].legend(loc="lower right", fontsize=6.5)
  fig.tight_layout()
  save(fig, "cluster_benchmark")


def fig_stability(runs: dict) -> None:
  """SuperMimic cluster vs local by iteration: critic, policy gradient, sim failures, benchmark."""
  fig, axes = plt.subplots(2, 2, figsize=(7.4, 4.2))
  for name in reversed(
    SM_RUNS
  ):  # local underneath, in a lighter tint: dense lines hide dashes
    r = runs[name]
    s = r["scalars"]
    color = SM_LOCAL if name.endswith("local") else r["color"]
    kw = dict(color=color, ls=r["ls"], lw=1.0, label=name)
    for a, tag, n in (
      (axes[0, 0], "policy/value", 25),
      (axes[0, 1], "policy/actor_grad_norm", 1),
      (axes[1, 0], "episode/failure", 25),
    ):
      x = np.array([e.step for e in s[tag]])
      y = np.array([e.value for e in s[tag]])
      a.plot(x, smooth(y, n) if n > 1 else y, **kw)
    b = r["bench"]
    axes[1, 1].plot(
      list(b), [v["tracking_return"] for v in b.values()], marker="o", ms=2.2, **kw
    )
  axes[0, 0].set_title("Critic value V (smoothed)")
  axes[0, 0].set_ylim(-20, 20)
  axes[0, 1].set_title("Policy gradient norm, before clipping")
  axes[0, 1].set_yscale("log")
  axes[1, 0].set_title("Sim episodes ending in failure (smoothed)")
  axes[1, 1].set_title("Benchmark tracking return")
  axes[1, 1].axhline(COLLAPSED, color=INK2, lw=0.6, ls=":")
  for a in axes.flat:
    a.set_xlabel("iteration")
  axes[0, 0].legend(fontsize=6.5)
  fig.tight_layout()
  save(fig, "cluster_stability")


def row(name: str, r: dict, it: int) -> str:
  v = r["bench"][it]
  return (
    f"{name} & {r['clock'].hours(it):.1f} & {it:,} & {v['failed']} & {v['tracking_return']:.3f} & "
    f"{100 * v['rmpkpe_legs']:.1f} & {100 * v['rmpkpe_arms']:.1f} & {100 * v['rmpkpe_torso']:.1f} \\\\"
  )


def tex_summary(runs: dict) -> dict:
  """Best and last checkpoint of each run, plus SuperMimic's collapse count."""
  lines = [
    r"\begin{tabular}{lrrrrrrr}\toprule",
    r"checkpoint & hours & iteration & failed & tracking return & legs [cm] & arms [cm] & torso [cm] \\\midrule",
  ]
  stats = {}
  target = runs["PPO-Sim, local"]["bench"]
  target = target[max(target)]["tracking_return"] if target else float("nan")
  for name, r in runs.items():
    b = r["bench"]
    if not b:
      continue
    best = max(b, key=lambda it: b[it]["tracking_return"])
    last = max(b)
    lines += [row(f"{name}, best", r, best), row(f"{name}, last", r, last), r"\midrule"]
    first = next((it for it, v in b.items() if v["tracking_return"] >= target), None)
    late = [it for it in b if first is not None and it >= first]
    stats[name] = dict(
      best=best,
      last=last,
      first_above_ppo_local=first,
      first_hours=r["clock"].hours(first) if first else None,
      late=len(late),
      collapsed=sum(b[it]["tracking_return"] < COLLAPSED for it in late),
    )
  lines[-1] = r"\bottomrule\end{tabular}"
  (OUT / "tab_cluster_summary.tex").write_text("\n".join(lines) + "\n")
  return stats


def vs_ppo(
  runs: dict, sm: str = "SuperMimic, cluster", ppo: str = "PPO-Sim, cluster"
) -> dict:
  """SuperMimic's checkpoints against PPO-Sim's final policy on the same machine."""
  b, p = runs[sm]["bench"], runs[ppo]["bench"]
  final = p[max(p)]
  first = next(
    it for it, v in b.items() if v["tracking_return"] >= final["tracking_return"]
  )
  later = [it for it in b if it >= first]
  return dict(
    ppo_final=(final["failed"], round(final["tracking_return"], 3)),
    first=first,
    first_hours=round(runs[sm]["clock"].hours(first), 2),
    later=len(later),
    above_return=sum(
      b[it]["tracking_return"] >= final["tracking_return"] for it in later
    ),
    fewer_failures=sum(b[it]["failed"] < final["failed"] for it in later),
  )


def spikes(r: dict, start: int = 5000, gap: int = 300, level: float = 0.5) -> dict:
  """Onsets of failure spikes in training: logged share of sim episodes ending in failure above
  ``level``, with no spike in the previous ``gap`` iterations; and how many follow a critic jump
  (value in the 100 iterations before > 1.5x its median over the 200 before that)."""
  s = r["scalars"]
  f = {e.step: e.value for e in s["episode/failure"]}
  v = {e.step: e.value for e in s["policy/value"]}
  onsets, last = [], -(10**9)
  for it in sorted(f):
    if it >= start and f[it] > level:
      if it - last > gap:
        onsets.append(it)
      last = it

  def critic_first(it: int) -> bool:
    base = [x for k, x in v.items() if it - 300 <= k < it - 100]
    pre = [x for k, x in v.items() if it - 100 <= k < it]
    return bool(base and pre) and max(pre) > 1.5 * float(np.median(base))

  edges = (5000, 16000, 27000, 38000, 50000)
  return dict(
    onsets=onsets,
    by_range={f"{a}-{b}": sum(a <= o < b for o in onsets) for a, b in pairwise(edges)},
    critic_first=sum(critic_first(o) for o in onsets),
  )


def tex_all(runs: dict) -> None:
  """Every benchmarked checkpoint of the two cluster runs (appendix), one table per run."""
  for name, file in (("PPO-Sim, cluster", "ppo"), ("SuperMimic, cluster", "sm")):
    r = runs[name]
    lines = [
      r"\begin{tabular}{lrrrrrrr}\toprule",
      r"run & hours & iteration & failed & tracking return & legs [cm] & arms [cm] & torso [cm] \\\midrule",
      *[row(name.split(",")[0], r, it) for it in r["bench"]],
      r"\bottomrule\end{tabular}",
    ]
    (OUT / f"tab_cluster_all_{file}.tex").write_text("\n".join(lines) + "\n")


def tex_throughput(runs: dict) -> None:
  lines = [
    r"\begin{tabular}{lrrrr}\toprule",
    r"run & iterations & hours & iterations/h & sim steps [B] \\\midrule",
  ]
  for name, r in runs.items():
    seg = r["clock"].segments[-1]
    it, hours = seg[1][-1], seg[2][-1]
    lines.append(
      f"{name} & {it:,.0f} & {hours:.1f} & {it / hours:,.0f} & {it * r['steps'] / 1e9:.2f} \\\\"
    )
  lines.append(r"\bottomrule\end{tabular}")
  (OUT / "tab_cluster_throughput.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
  runs = load()
  fig_benchmark(runs)
  fig_stability(runs)
  tex_throughput(runs)
  tex_all(runs)
  for name, st in tex_summary(runs).items():
    print(name, st)
  print("vs PPO-Sim cluster final:", vs_ppo(runs))
  for name in SM_RUNS:
    sp = spikes(runs[name])
    print(
      f"{name}: {len(sp['onsets'])} failure spikes after iteration 5000 {sp['by_range']}, "
      f"{sp['critic_first']} after a critic jump"
    )
  for name, r in runs.items():
    print(f"{name}: {len(r['bench'])} benchmarks")


if __name__ == "__main__":
  main()
