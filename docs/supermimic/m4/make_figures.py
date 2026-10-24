"""Figures and tables for the M4 report (docs/supermimic/m4/): SuperMimic vs PPO-Sim on ``dev``.

Reads the TensorBoard logs of both runs and the paper's benchmark of their checkpoints
(``scripts/supermimic/eval.py``), and writes PDFs, PNGs and LaTeX tables to
``docs/supermimic/m4/figures/``. Run from the repository root:

  .venv/bin/python docs/supermimic/m4/make_figures.py

Wall-clock: training hours at each iteration. A resumed SuperMimic run continues in a new log
folder from its last checkpoint; the hours of work lost after that checkpoint are not counted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SM_RUNS = sorted(Path("logs/rsl_rl/supermimic_wm_1hz").glob("*_dev-local"))
PPO_RUN = Path("logs/rsl_rl/supermimic_ppo_sim_1hz/2026-10-22_21-11-56")
SM_EVAL = Path("logs/supermimic_eval/wm_1hz_dev_local")
PPO_EVAL = Path("logs/supermimic_eval/ppo_sim_1hz")
OUT = Path("docs/supermimic/m4/figures")
SM, PPO, ALT, GRAY = "#2a78d6", "#8d8b85", "#eb6834", "#8d8b85"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
  {
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "axes.titleweight": "semibold",
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,
  }
)


def save(fig, name: str) -> None:
  fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
  fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=160)
  plt.close(fig)


def scalars(run: Path) -> dict[str, list]:
  ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
  ea.Reload()
  return {t: ea.Scalars(t) for t in ea.Tags()["scalars"]}


class Clock:
  """Training hours at an iteration, across resumed log folders."""

  def __init__(self, runs: list[dict[str, list]], tag: str) -> None:
    self.segments = []  # (first step, steps, hours) per folder, hours continuing
    offset = 0.0
    for run in runs:
      ev = run[tag]
      steps = np.array([e.step for e in ev], dtype=float)
      wall = np.array([e.wall_time for e in ev])
      if self.segments:  # resumed: continue from the previous folder at this step
        prev_steps, prev_hours = self.segments[-1][1], self.segments[-1][2]
        offset = float(np.interp(steps[0], prev_steps, prev_hours))
      hours = offset + (wall - wall[0]) / 3600
      self.segments.append((steps[0], steps, hours))

  def hours(self, step: float) -> float:
    for first, steps, hours in reversed(self.segments):
      if step >= first:
        return float(np.interp(step, steps, hours))
    return 0.0

  def series(
    self, runs: list[dict[str, list]], tag: str
  ) -> tuple[np.ndarray, np.ndarray]:
    """(hours, value) of a scalar, keeping each step from the latest folder that has it."""
    points: dict[int, float] = {}
    for run in runs:
      for e in run.get(tag, []):
        points[e.step] = e.value
    steps = np.array(sorted(points))
    return np.array([self.hours(s) for s in steps]), np.array(
      [points[s] for s in steps]
    )


def benchmarks(folder: Path) -> dict[int, dict]:
  out = {}
  for f in folder.glob("*.json"):
    m = re.search(r"model_(\d+)\.json$", f.name)
    if m:
      d = json.loads(f.read_text())
      out[int(m.group(1))] = d.get("summary", d)
  return dict(sorted(out.items()))


def smooth(y: np.ndarray, n: int = 25) -> np.ndarray:
  """Centered moving average; shorter windows at the ends (no zero padding)."""
  c = np.concatenate([[0.0], np.cumsum(y)])
  i = np.arange(len(y))
  lo, hi = np.maximum(i - n // 2, 0), np.minimum(i + n // 2 + 1, len(y))
  return (c[hi] - c[lo]) / (hi - lo)


def fig_benchmark(sm_b, sm_clock, ppo_b, ppo_clock) -> None:
  fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
  for b, clock, color, label, marker in (
    (ppo_b, ppo_clock, PPO, "PPO-Sim", "s"),
    (sm_b, sm_clock, SM, "SuperMimic", "o"),
  ):
    if not b:
      continue
    h = [clock.hours(it) for it in b]
    ax[0].plot(
      h,
      [v["tracking_return"] for v in b.values()],
      marker=marker,
      color=color,
      label=label,
    )
    ax[1].plot(
      h, [v["failed"] for v in b.values()], marker=marker, color=color, label=label
    )
  ax[0].set_title("Tracking return (benchmark)")
  ax[1].set_title("Clips failed (of 1296)")
  ax[1].set_yscale("log")
  for a in ax:
    a.set_xlabel("training hours (local GPU)")
  ax[0].legend(loc="lower right")
  fig.tight_layout()
  save(fig, "benchmark")


def fig_training(sm_runs, clock) -> None:
  """SuperMimic's own signals during training."""
  panels = [
    ("Sim episodes", [("episode/length", "length [steps]", SM)], None),
    ("Imagined vs simulated 32-step return",
     [("eval/sim_return", "simulator", SM), ("eval/imagined_return_head", "imagination", ALT)], None),
    ("Failures within 32 steps",
     [("eval/sim_failure", "simulator", SM), ("eval/imagined_failure_head", "learned head", ALT),
      ("eval/imagined_failure_rule", "env's rule", GRAY)], None),
    ("Ranking fresh failures (AUC)",
     [("termination/head_auc", "learned head", ALT), ("termination/rule_auc", "env's rule", GRAY)], None),
    ("World-model loss (Eq. 1)", [("world_model/loss", "", SM)], "log"),
    ("Iteration time", [("time/collect", "collect", GRAY), ("time/world_model", "world model", ALT),
                        ("time/policy", "policy + critic", SM)], None),
  ]  # fmt: skip
  fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.4))
  for a, (title, lines, scale) in zip(axes.flat, panels, strict=True):
    for tag, label, color in lines:
      h, y = clock.series(sm_runs, tag)
      if len(h) == 0:
        continue
      if tag.startswith(("eval/", "termination/")):
        a.plot(h, y, color=color, marker="o", ms=2.5, lw=1.0, label=label)
      else:
        a.plot(h, smooth(y), color=color, lw=1.2, label=label)
    a.set_title(title)
    a.set_xlabel("training hours")
    if scale:
      a.set_yscale(scale)
    if a.get_legend_handles_labels()[0]:
      a.legend(fontsize=6.5)
  fig.tight_layout()
  save(fig, "training")


def fig_vs_ppo(sm_runs, sm_clock, ppo, ppo_clock) -> None:
  """Training signals both runs log, at matched hours (training starts, with exploration)."""
  panels = [
    ("Sampler mean return", "Metrics/motion/sampler_mean_return"),
    ("Joint error, episode end [rad]", "Metrics/motion/error_joint_pos"),
    ("Root error, episode end [m]", "Metrics/motion/error_anchor_pos"),
  ]
  fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.4))
  for a, (title, tag) in zip(ax, panels, strict=True):
    h, y = ppo_clock.series([ppo], tag)
    a.plot(h, smooth(y, 51), color=PPO, label="PPO-Sim")
    h, y = sm_clock.series(sm_runs, f"env/{tag}")
    a.plot(h, smooth(y, 51), color=SM, label="SuperMimic")
    a.set_title(title)
    a.set_xlabel("training hours")
  ax[0].legend(loc="lower right")
  fig.tight_layout()
  save(fig, "vs_ppo_training")


def fig_collapse(sm_runs, lo: int = 8800, hi: int = 9020) -> None:
  """One policy collapse at full resolution: what moves first."""

  def get(tag):
    pts = {
      e.step: e.value for run in sm_runs for e in run.get(tag, []) if lo <= e.step <= hi
    }
    x = np.array(sorted(pts))
    return x, np.array([pts[i] for i in x])

  fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.4))
  x, y = get("policy/value")
  ax[0].plot(x, y, color=SM, label="critic value V")
  x, y = get("policy/lambda_return")
  ax[0].plot(x, y, color=SM, ls=":", label="its target")
  x, y = get("policy/imagined_return")
  ax[0].plot(x, y, color=ALT, label="imagined reward")
  ax[0].set_title("Critic vs imagined reward")
  ax[0].legend(fontsize=6.5, loc="upper left")
  x, y = get("policy/actor_grad_norm")
  ax[1].plot(x, y, color=SM)
  ax[1].set_yscale("log")
  ax[1].set_title("Policy gradient norm")
  x, y = get("sim/reward")
  ax[2].plot(x, y / 0.02, color=SM, label="reward per step / dt")
  x, y = get("episode/failure")
  ax[2].plot(x, y, color=ALT, label="episodes failing")
  ax[2].set_title("Simulator")
  ax[2].legend(fontsize=6.5, loc="center left")
  for a in ax:
    a.set_xlabel("iteration")
  fig.tight_layout()
  save(fig, "collapse")


def tex_tables(sm_b, sm_clock, ppo_b, ppo_clock) -> None:
  lines = [
    r"\begin{tabular}{lrrrrrrr}\toprule",
    r"run & iteration & hours & failed & tracking return & legs [cm] & arms [cm] & torso [cm] \\\midrule",
  ]
  for name, b, clock in (("PPO-Sim", ppo_b, ppo_clock), ("SuperMimic", sm_b, sm_clock)):
    for it, v in b.items():
      lines.append(
        f"{name} & {it:,} & {clock.hours(it):.1f} & {v['failed']} & {v['tracking_return']:.3f} & "
        f"{100 * v['rmpkpe_legs']:.1f} & {100 * v['rmpkpe_arms']:.1f} & {100 * v['rmpkpe_torso']:.1f} \\\\"
      )
    lines.append(r"\midrule")
  lines[-1] = r"\bottomrule\end{tabular}"
  (OUT / "tab_benchmark.tex").write_text("\n".join(lines) + "\n")


def tex_summary(sm_b, sm_clock, ppo_b, ppo_clock) -> dict:
  """Key comparisons: PPO-Sim's 12 h policy vs SuperMimic's checkpoints."""
  ppo_last = max(ppo_b)
  target = ppo_b[ppo_last]["tracking_return"]
  first = next(it for it, v in sm_b.items() if v["tracking_return"] >= target)
  best = max(sm_b, key=lambda it: sm_b[it]["tracking_return"])
  last = max(sm_b)
  late = [it for it in sm_b if sm_clock.hours(it) >= sm_clock.hours(first)]
  beat = [it for it in late if sm_b[it]["tracking_return"] >= target]
  collapsed = [it for it in late if sm_b[it]["tracking_return"] < 0.6]
  rows = [
    ("PPO-Sim, final", ppo_clock.hours(ppo_last), ppo_last, ppo_b[ppo_last]),
    (
      "SuperMimic, first checkpoint above it",
      sm_clock.hours(first),
      first,
      sm_b[first],
    ),
    ("SuperMimic, best checkpoint", sm_clock.hours(best), best, sm_b[best]),
    ("SuperMimic, last checkpoint", sm_clock.hours(last), last, sm_b[last]),
  ]
  lines = [
    r"\begin{tabular}{lrrrrrrr}\toprule",
    r"policy & hours & iteration & failed & tracking return & legs [cm] & arms [cm] & torso [cm] \\\midrule",
  ]
  for name, h, it, v in rows:
    lines.append(
      f"{name} & {h:.1f} & {it:,} & {v['failed']} & {v['tracking_return']:.3f} & "
      f"{100 * v['rmpkpe_legs']:.1f} & {100 * v['rmpkpe_arms']:.1f} & {100 * v['rmpkpe_torso']:.1f} \\\\"
    )
  lines.append(r"\bottomrule\end{tabular}")
  (OUT / "tab_summary.tex").write_text("\n".join(lines) + "\n")
  return {
    "target": target,
    "first": first,
    "first_hours": sm_clock.hours(first),
    "best": best,
    "late": len(late),
    "beat": len(beat),
    "collapsed": len(collapsed),
  }


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  sm_runs = [scalars(r) for r in SM_RUNS]
  sm_clock = Clock(sm_runs, "world_model/loss")
  ppo = scalars(PPO_RUN)
  ppo_clock = Clock([ppo], "Perf/total_fps")
  sm_b, ppo_b = benchmarks(SM_EVAL), benchmarks(PPO_EVAL)
  fig_benchmark(sm_b, sm_clock, ppo_b, ppo_clock)
  fig_training(sm_runs, sm_clock)
  fig_vs_ppo(sm_runs, sm_clock, ppo, ppo_clock)
  fig_collapse(sm_runs)
  tex_tables(sm_b, sm_clock, ppo_b, ppo_clock)
  if sm_b:
    print("summary:", tex_summary(sm_b, sm_clock, ppo_b, ppo_clock))
  last = max(s[1][-1] for s in sm_clock.segments)
  print(
    f"SuperMimic: {len(SM_RUNS)} log folder(s), last iteration {last:.0f} at "
    f"{sm_clock.hours(last):.2f} h; {len(sm_b)} benchmarks; PPO-Sim {len(ppo_b)} benchmarks"
  )


if __name__ == "__main__":
  main()
