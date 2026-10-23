"""Figures and tables for the M3 report (docs/supermimic/m3/).

Reads the ``scripts/supermimic/m3_check.py`` results under ``logs/supermimic_m3/`` and the
local smoke run of the M4 task, and writes PDFs, PNGs and LaTeX tables to
``docs/supermimic/m3/figures/``. Run from the repository root:

  .venv/bin/python docs/supermimic/m3/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("logs/supermimic_m3")
SMOKE = sorted(Path("logs/supermimic_m3/smoke_runs").glob("*smoke-local-bf16"))
OUT = Path("docs/supermimic/m3/figures")
# Update runs of check 2 (one change each from the paper's settings).
RUNS = {
  "paper": ("m3.json", "paper settings"),
  "no_critic": ("m3_no_critic.json", "no critic"),
  "lr": ("m3_lr2e-5.json", "policy lr 2e-5"),
  "rule": ("m3_rule.json", "rule termination"),
}
TEX = {
  "paper": "paper settings (learned head, critic, lr $2\\cdot10^{-4}$)",
  "no_critic": "without the critic",
  "lr": "policy learning rate $2\\cdot10^{-5}$",
  "rule": "termination by the env's rule",
}
# Validated categorical palette (fixed slot per run) and marker as secondary encoding.
COLOR = {"paper": "#2a78d6", "no_critic": "#1baf7a", "lr": "#eda100", "rule": "#e87ba4"}
MARKER = {"paper": "o", "no_critic": "s", "lr": "^", "rule": "D"}
SIM, IMAG, GRAY = "#2a78d6", "#eb6834", "#8d8b85"
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


def load() -> dict[str, dict]:
  return {
    name: json.loads((ROOT / file).read_text())
    for name, (file, _) in RUNS.items()
    if (ROOT / file).exists()
  }


def series(run: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
  its = sorted(run["closed_loop"], key=int)
  return np.array([int(i) for i in its]), np.array(
    [run["closed_loop"][i][key] for i in its]
  )


def fig_check1(run: dict) -> None:
  """Check 1: the PPO-Sim actor in imagination and in the simulator, same starts."""
  c = run["closed_loop"]["0"]
  sim, imag = (
    np.array(c["reward_curve"]["sim"]),
    np.array(c["reward_curve"]["imagined"]),
  )
  t = np.arange(1, len(sim) + 1)
  fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
  ax[0].plot(t, sim, color=SIM, label="simulator")
  ax[0].plot(t, imag, color=IMAG, ls="--", label="imagination (learned head)")
  ax[0].set_xlabel("step of the rollout")
  ax[0].set_ylabel("mean reward per step")
  ax[0].set_title("Reward along 32 closed-loop steps")
  ax[0].legend(loc="lower left")
  steps = sorted(c["body_error_cm"], key=int)
  ax[1].plot(
    [int(s) for s in steps], [c["body_error_cm"][s] for s in steps], marker="o",
    color=IMAG,
  )  # fmt: skip
  ax[1].set_xlabel("step of the rollout")
  ax[1].set_ylabel("body error [cm]")
  ax[1].set_ylim(bottom=0)
  ax[1].set_title("Imagined vs simulated body positions")
  fig.tight_layout()
  save(fig, "check1")


def fig_update(runs: dict[str, dict]) -> None:
  """Check 2: the policy in closed loop during the update runs."""
  fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
  for name, run in runs.items():
    style = dict(color=COLOR[name], marker=MARKER[name], label=RUNS[name][1])
    x, sim = series(run, "sim_return")
    ax[0].plot(x, sim, **style)
    x, fail = series(run, "sim_failure")
    ax[1].plot(x, 100 * fail, **style)
  start = runs["paper"]["closed_loop"]["0"]["sim_return"]
  ax[0].axhline(start, color=GRAY, ls=":", lw=1.0)
  ax[0].set_title("32-step return in the sim")
  ax[0].set_xlabel("policy update")
  ax[1].set_title("Failures within 32 steps")
  ax[1].set_xlabel("policy update")
  ax[1].set_ylabel("% of starts")
  ax[1].set_ylim(bottom=0)
  # The paper run: what the learned head and the rule predicted vs the sim.
  run = runs["paper"]
  x, sim = series(run, "sim_failure")
  _, head = series(run, "imagined_failure_head")
  _, rule = series(run, "imagined_failure_rule")
  ax[2].plot(x, 100 * sim, color=SIM, marker="o", label="simulator")
  ax[2].plot(x, 100 * head, color=IMAG, marker="s", ls="--", label="learned head")
  ax[2].plot(x, 100 * rule, color=GRAY, marker="^", ls="--", label="env's rule")
  ax[2].set_title("Failures: sim vs predicted (paper)")
  ax[2].set_xlabel("policy update")
  ax[2].set_ylim(bottom=0)
  ax[2].legend(loc="upper right", fontsize=6.5)
  handles, labels = ax[0].get_legend_handles_labels()
  fig.legend(
    handles, labels, loc="lower center", ncol=len(runs), bbox_to_anchor=(0.36, -0.08)
  )
  fig.tight_layout(rect=(0, 0.07, 1, 1))
  save(fig, "update")


def fig_critic(run: dict) -> None:
  """The critic fit of the paper run: first with the actor fixed, then during updates."""
  log = run["log"]
  n_critic = sum(r["phase"] == "critic" for r in log)
  x = np.arange(len(log)) - n_critic + 1
  fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.4))
  ax[0].plot(x, [r["critic_loss"] for r in log], color=SIM, lw=1.0)
  ax[0].set_yscale("log")
  ax[0].set_title("Critic loss")
  ax[0].set_ylabel("MSE")
  ax[1].plot(x, [r["value"] for r in log], color=SIM, lw=1.0, label="V at the start")
  ax[1].plot(
    x, [r["lambda_return"] for r in log], color=IMAG, ls="--", lw=1.0,
    label="its λ-return target",
  )  # fmt: skip
  ax[1].set_title("Critic value")
  ax[1].legend(loc="lower right")
  for a in ax:
    a.axvline(0, color=GRAY, ls=":", lw=1.0)
    a.set_xlabel("iteration (< 0: critic only, actor fixed)")
  fig.tight_layout()
  save(fig, "critic")


def fig_benchmark(runs: dict[str, dict]) -> None:
  """The paper's benchmark (every dev clip from its first frame), start vs after 300 updates."""
  b0 = runs["paper"]["benchmark"]["0"]
  names = ["start"] + list(runs)
  bench = [b0] + [r["benchmark"][max(r["benchmark"], key=int)] for r in runs.values()]
  labels = ["PPO-Sim actor"] + [RUNS[n][1] for n in runs]
  colors = [GRAY] + [COLOR[n] for n in runs]
  y = np.arange(len(names))[::-1]
  fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
  for a, key, title, fmt in (
    (ax[0], "failed", "Clips failed (of 1296)", "{:.0f}"),
    (ax[1], "tracking_return", "Tracking return", "{:.3f}"),
  ):
    vals = [b[key] for b in bench]
    a.barh(y, vals, color=colors, height=0.6)
    for yi, v in zip(y, vals, strict=True):
      a.text(v, yi, " " + fmt.format(v), va="center", fontsize=7.5, color=INK2)
    a.set_title(title)
    a.grid(axis="y", visible=False)
    a.set_axisbelow(True)
    a.set_xlim(0, max(vals) * 1.22)
  ax[0].set_yticks(y, labels)
  fig.tight_layout()
  save(fig, "benchmark")


def smoke_scalars() -> dict[str, list[float]]:
  if not SMOKE:
    return {}
  ea = EventAccumulator(str(SMOKE[-1]))
  ea.Reload()
  return {t: [e.value for e in ea.Scalars(t)] for t in ea.Tags()["scalars"]}


def tex_tables(runs: dict[str, dict], smoke: dict[str, list[float]]) -> None:
  c = runs["paper"]["closed_loop"]["0"]
  rows = [
    ("simulator", c["sim_return"], None, c["sim_failure"]),
    ("imagination, learned head", c["imagined_return_head"], c["return_gap_head"],
     c["imagined_failure_head"]),
    ("imagination, env's rule", c["imagined_return_rule"], c["return_gap_rule"],
     c["imagined_failure_rule"]),
  ]  # fmt: skip
  lines = [
    r"\begin{tabular}{lrrr}\toprule",
    r" & 32-step return & gap & failures \\\midrule",
  ]
  for name, ret, gap, fail in rows:
    g = "" if gap is None else f"${100 * gap:+.1f}$\\%"
    lines.append(f"{name} & {ret:.3f} & {g} & {100 * fail:.2f}\\% \\\\")
  lines.append(r"\bottomrule\end{tabular}")
  (OUT / "tab_check1.tex").write_text("\n".join(lines) + "\n")

  lines = [
    r"\begin{tabular}{lrrrrrr}\toprule",
    r" & \multicolumn{3}{c}{benchmark (1296 clips)} & \multicolumn{3}{c}{closed loop, 32 steps} \\",
    r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
    r"policy & failed & tracking return & arms [cm] & return & failures & worst gap \\\midrule",
  ]
  b0 = runs["paper"]["benchmark"]["0"]
  lines.append(
    f"PPO-Sim actor (start) & {b0['failed']} & {b0['tracking_return']:.3f} & "
    f"{100 * b0['rmpkpe_arms']:.1f} & "
    f"{c['sim_return']:.3f} & {100 * c['sim_failure']:.1f}\\% & \\\\\\midrule"
  )
  for name, run in runs.items():
    end = max(run["benchmark"], key=int)
    b = run["benchmark"][end]
    last = run["closed_loop"][max(run["closed_loop"], key=int)]
    gaps = [abs(v["return_gap_head" if name != "rule" else "return_gap_rule"])
            for k, v in run["closed_loop"].items() if k != "0"]  # fmt: skip
    lines.append(
      f"after {end} updates, {TEX[name]} & {b['failed']} & {b['tracking_return']:.3f} & "
      f"{100 * b['rmpkpe_arms']:.1f} & "
      f"{last['sim_return']:.3f} & {100 * last['sim_failure']:.1f}\\% & "
      f"{100 * max(gaps):.1f}\\% \\\\"
    )
  lines.append(r"\bottomrule\end{tabular}")
  (OUT / "tab_update.tex").write_text("\n".join(lines) + "\n")

  if smoke:
    t = {
      k: float(np.median(smoke[f"time/{k}"][1:]))
      for k in ("collect", "world_model", "policy")
    }
    total = sum(t.values())
    lines = [
      r"\begin{tabular}{lr}\toprule",
      f"collect (2.5 env steps) & {t['collect']:.2f} s \\\\",
      f"world model (10 $\\times$ 2048 windows) & {t['world_model']:.2f} s \\\\",
      f"policy and critic (4096 rollouts $\\times$ 32 steps) & {t['policy']:.2f} s \\\\\\midrule",
      f"iteration & {total:.2f} s ({3600 / total:,.0f} per hour) \\\\",
      f"peak GPU memory & {max(smoke['memory/max_allocated_gb']):.1f} GB \\\\",
      r"\bottomrule\end{tabular}",
    ]
    (OUT / "tab_smoke.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  runs = load()
  fig_check1(runs["paper"])
  fig_update(runs)
  fig_critic(runs["paper"])
  fig_benchmark(runs)
  tex_tables(runs, smoke_scalars())
  print(f"figures and tables -> {OUT} (runs: {', '.join(runs)})")


if __name__ == "__main__":
  main()
