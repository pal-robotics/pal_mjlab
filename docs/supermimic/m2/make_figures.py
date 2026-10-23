"""Figures for the M2 world-model report (docs/supermimic/m2/).

Reads the evaluation JSONs and TensorBoard logs under ``logs/supermimic_m2/`` and writes
vector PDFs to ``docs/supermimic/m2/figures/``. Run from the repository root:

  .venv/bin/python docs/supermimic/m2/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("logs/supermimic_m2")
OUT = Path("docs/supermimic/m2/figures")

# Reference categorical palette (light), fixed slot per entity; baseline in neutral gray.
COLOR = {
  "main": "#2a78d6",  # slot 1 blue
  "k1": "#eb6834",  # slot 2 orange
  "absolute": "#1baf7a",  # slot 3 aqua
  "w1024": "#eda100",  # slot 4 yellow (low contrast: always direct-labelled)
  "no4000": "#e87ba4",  # slot 5 magenta
  "no_change": "#8d8b85",  # neutral baseline
}
LABEL = {
  "main": "main: k=8, 2048×3, Δ",
  "k1": "k=1 training",
  "absolute": "absolute targets",
  "w1024": "width 1024",
  "no4000": "unseen policy",
  "no_change": "no change",
}
TEX_LABEL = {
  "main": r"main ($k{=}8$, $2048{\times}3$, $\Delta$)",
  "k1": r"$k{=}1$ training",
  "absolute": "absolute targets",
  "w1024": "width 1024",
  "no4000": "unseen policy",
  "no_change": "no change",
}
DIVERGED_CM = 1000.0  # errors above 10 m mean the rollout blew up
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def cm(v: float, fmt: str = ".2f") -> str:
  return "diverged" if v > DIVERGED_CM else format(v, fmt)


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
  """Vector PDF for the LaTeX report, PNG for the markdown report."""
  fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
  fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=160)


def load(name: str) -> dict:
  return json.loads((ROOT / "eval" / name).read_text())


def direct_label(ax, x, y, text, color, dy=0.0):
  ax.annotate(
    text, (x, y), xytext=(4, dy), textcoords="offset points", va="center",
    fontsize=7.5, color=INK,
  )  # fmt: skip
  ax.plot([x], [y], "o", color=color, ms=4.5, mec="white", mew=0.8, zorder=5)


def fig_horizon(res: dict) -> None:
  """Open-loop prediction error vs horizon: body positions and imagined return."""
  fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
  order = ["no_change", "k1", "absolute", "w1024", "main"]
  for ax, key, ylabel, scale in (
    (axes[0], "body_world_cm", "body position error [cm]", 1.0),
    (axes[1], "return_rel", "imagined return error [%]", 100.0),
  ):
    for name in order:
      hs = sorted(int(h) for h in res[name])
      ys = np.array([res[name][str(h)][key] * scale for h in hs])
      style = "--" if name == "no_change" else "-"
      # A diverged rollout (k=1) would flatten the log axis: stop its line and mark it.
      shown = ys <= (DIVERGED_CM if key == "body_world_cm" else np.inf)
      ax.plot(
        np.array(hs)[shown],
        ys[shown],
        style,
        color=COLOR[name],
        marker="o",
        mec="white",
        mew=0.6,
        label=LABEL[name],
      )
      if not shown.all():
        last = int(np.nonzero(shown)[0][-1])
        ax.annotate(
          "k=1 diverges",
          (hs[last], ys[last]),
          xytext=(hs[last] * 1.25, 45),
          fontsize=7,
          color=INK,
          arrowprops={"arrowstyle": "->", "color": COLOR[name], "lw": 1},
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 8, 16, 32, 64], ["1", "8", "16", "32", "64"])
    ax.set_xlabel("prediction horizon [control steps of 20 ms]")
    ax.set_ylabel(ylabel)
  axes[0].set_title("Body position error")
  axes[1].set_title("Imagined return error")
  axes[0].set_ylim(top=80)
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
  fig.tight_layout(rect=(0, 0.08, 1, 1))
  save(fig, "horizon")
  plt.close(fig)


def fig_groups(res: dict) -> None:
  """Where the main model's error is: root-relative body error by group vs horizon."""
  fig, ax = plt.subplots(figsize=(3.3, 2.4))
  groups = [("arms", "#2a78d6", 5), ("legs", "#eb6834", -5), ("torso", "#1baf7a", 0)]
  hs = sorted(int(h) for h in res["main"])
  for g, color, dy in groups:
    ys = [res["main"][str(h)][f"body_local_{g}_cm"] for h in hs]
    ax.plot(hs, ys, color=color, marker="o", mec="white", mew=0.6)
    direct_label(ax, hs[-1], ys[-1], g, color, dy=dy)
  ax.set_xscale("log", base=2)
  ax.set_yscale("log")
  ax.set_xticks([1, 8, 16, 32, 64], ["1", "8", "16", "32", "64"])
  ax.set_xlim(0.8, 110)
  ax.set_xlabel("prediction horizon [steps]")
  ax.set_ylabel("error relative to the root [cm]")
  ax.set_title("Main model: body error by group")
  fig.tight_layout()
  save(fig, "groups")
  plt.close(fig)


def fig_gradients(grad: dict) -> None:
  """Gradient check: improvement rate, predicted vs simulated gain, gradient profile."""
  fig, axes = plt.subplots(
    1, 3, figsize=(6.8, 2.4), gridspec_kw={"width_ratios": [1.0, 1.1, 1.1]}
  )
  hs = sorted(int(h) for h in grad)
  # (a) Fraction of start states whose simulated return improved.
  ax = axes[0]
  width = 0.26
  x = np.arange(len(hs))
  series = [
    ("gradient, δ=0.05", "#2a78d6", lambda r: r["grad@0.05"]["improved"]),
    ("gradient, δ=0.1", "#eb6834", lambda r: r["grad@0.1"]["improved"]),
    ("random, δ=0.05", "#8d8b85",
     lambda r: np.mean([r[k]["improved"] for k in r if k.startswith("rand") and k.endswith("0.05")])),
  ]  # fmt: skip
  for i, (label, color, f) in enumerate(series):
    vals = [100 * f(grad[str(h)]) for h in hs]
    ax.bar(x + (i - 1) * width, vals, width * 0.9, color=color, label=label)
  ax.axhline(50, color=INK2, lw=0.8, ls=":")
  ax.set_xticks(x, [f"H={h}" for h in hs])
  ax.set_ylim(0, 105)
  ax.set_ylabel("starts improved [%]")
  ax.set_title("(a) Does the step help?")
  ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=6.5, ncol=1)
  # (b) Model-predicted vs simulated gain per start (δ=0.05).
  ax = axes[1]
  for h, color in zip(hs, ["#86b6ef", "#2a78d6", "#104281"], strict=True):
    ps = grad[str(h)]["per_start"]
    wm = np.array(ps["wm_gain"]["grad@0.05"])
    sim = np.array(ps["sim_gain"]["grad@0.05"])
    ax.scatter(wm, sim, s=6, color=color, alpha=0.7, lw=0, label=f"H={h}")
  lim = ax.get_xlim()[1]
  ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls=":")
  ax.set_xlabel("gain predicted by the world model")
  ax.set_ylabel("gain in simulation")
  ax.set_title("(b) Predicted vs real gain (δ=0.05)")
  ax.legend(loc="lower right", fontsize=6.5, markerscale=2)
  # (c) Gradient magnitude along the horizon (H=32), normalized.
  ax = axes[2]
  prof = np.array(grad[str(max(hs))]["grad_norm_profile"])
  ax.plot(np.arange(len(prof)), prof / prof[0], color="#2a78d6")
  ax.set_xlabel(r"action step $t$ within the horizon")
  ax.set_ylabel(r"$\|\partial J/\partial a_t\|$ / $\|\partial J/\partial a_0\|$")
  ax.set_ylim(0, 1.05)
  ax.set_title(f"(c) Gradient profile, H={max(hs)}")
  fig.tight_layout()
  save(fig, "gradients")
  plt.close(fig)


def roc(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  order = np.argsort(-scores)
  lab = labels[order]
  tpr = np.concatenate([[0], np.cumsum(lab) / lab.sum()])
  fpr = np.concatenate([[0], np.cumsum(1 - lab) / (1 - lab).sum()])
  return fpr, tpr


def fig_termination(res: dict) -> None:
  term = res["termination"]["main"]
  sc = term["scores"]
  labels = np.array(sc["label"])
  fig, ax = plt.subplots(figsize=(3.0, 2.5))
  for key, name, color in (
    ("rule", "analytic rule", "#2a78d6"),
    ("head", "learned head", "#eb6834"),
  ):
    fpr, tpr = roc(np.array(sc[key]), labels)
    auc_key = "rule_auc" if key == "rule" else "auc"
    ax.plot(fpr, tpr, color=color, label=f"{name} (AUC {term[auc_key]:.3f})")
  ax.set_xscale("log")
  ax.set_xlim(1e-4, 1)
  ax.set_xlabel("false-alarm rate")
  ax.set_ylabel("failures detected")
  ax.set_title("Failure detection on predicted states")
  ax.legend(loc="lower right", fontsize=6.5)
  fig.tight_layout()
  save(fig, "termination")
  plt.close(fig)


def fig_training() -> None:
  """Held-out validation loss during training (state + pose terms) for each model."""
  fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
  for name in ["k1", "absolute", "w1024", "no4000", "main"]:
    acc = EventAccumulator(
      str(ROOT / "wm" / name / "tb"), size_guidance={"scalars": 0}
    ).Reload()
    for ax, h in zip(axes, (8, 32), strict=True):
      st = acc.Scalars(f"val{h}/state")
      po = acc.Scalars(f"val{h}/pose")
      steps = np.array([e.step for e in st]) / 1000
      vals = np.array([a.value + b.value for a, b in zip(st, po, strict=True)])
      ax.plot(steps, vals, color=COLOR[name], label=LABEL[name])
  for ax, h in zip(axes, (8, 32), strict=True):
    ax.set_yscale("log")
    ax.set_xlabel("training step [thousands]")
    ax.set_ylabel(f"{h}-step held-out loss")
    ax.set_title(f"Validation, {h}-step windows")
  axes[1].set_ylim(top=10)
  axes[1].text(31, 3.0, "k=1: rollouts diverge", fontsize=7, color=INK)
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.06))
  fig.tight_layout(rect=(0, 0.08, 1, 1))
  save(fig, "training")
  plt.close(fig)


def tex_tables(res: dict, shift: dict, grad: dict) -> None:
  """LaTeX tables generated from the results, so the report never hand-copies numbers."""
  H = res["horizons"]
  rows = []
  for name in ["no_change", "main"]:
    for h in ["1", "8", "16", "32", "64"]:
      m = H[name][h]
      label = TEX_LABEL[name] if h == "1" else ""  # name once per block
      rows.append(
        f"{label} & {h} & {m['body_world_cm']:.2f} & {m['body_world_cm_p95']:.2f} & "
        f"{m['body_local_arms_cm']:.2f} & {m['body_local_legs_cm']:.2f} & {m['root_z_cm']:.2f} & "
        f"{m['tilt_deg']:.2f} & {m['root_xy_cm']:.2f} & {100 * m['return_rel']:.2f} \\\\"
      )
    rows.append("\\midrule")
  (OUT / "tab_prediction.tex").write_text(
    "\\begin{tabular}{lrrrrrrrrr}\\toprule\n"
    "predictor & $h$ & body & body p95 & arms (local) & legs (local) & root $z$ & tilt & root $xy$ & return err \\\\\n"
    " & steps & cm & cm & cm & cm & cm & deg & cm & \\% \\\\\\midrule\n"
    + "\n".join(rows[:-1])
    + "\n\\bottomrule\\end{tabular}\n"
  )
  rows = []
  for name in ["main", "w1024", "absolute", "k1", "no_change"]:
    cells = []
    for h in ["8", "32"]:
      m = H[name][h]
      cells += [cm(m["body_world_cm"]), f"{100 * m['return_rel']:.2f}"]
    cells.append(cm(H[name]["64"]["body_world_cm"], ".1f"))
    rows.append(f"{TEX_LABEL[name]} & " + " & ".join(cells) + " \\\\")
  (OUT / "tab_ablation.tex").write_text(
    "\\begin{tabular}{lrrrrr}\\toprule\n"
    " & \\multicolumn{2}{c}{8 steps} & \\multicolumn{2}{c}{32 steps} & 64 steps \\\\\n"
    "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-6}\n"
    "model & body [cm] & return [\\%] & body [cm] & return [\\%] & body [cm] \\\\\\midrule\n"
    + "\n".join(rows)
    + "\n\\bottomrule\\end{tabular}\n"
  )
  rows = []
  for name in ["main", "no4000"]:
    m8, m32 = shift["horizons"][name]["8"], shift["horizons"][name]["32"]
    rows.append(
      f"{TEX_LABEL[name]} & {m8['body_world_cm']:.2f} & {m32['body_world_cm']:.2f} & "
      f"{100 * m32['return_rel']:.2f} \\\\"
    )
  (OUT / "tab_shift.tex").write_text(
    "\\begin{tabular}{lrrr}\\toprule\n"
    "model (data of the 4000-iteration policy) & body, 8 steps [cm] & body, 32 steps [cm] & return err, 32 [\\%] \\\\\\midrule\n"
    + "\n".join(rows)
    + "\n\\bottomrule\\end{tabular}\n"
  )
  rows = []
  for h in sorted(grad, key=int):
    r = grad[h]
    rnd = [r[k]["improved"] for k in r if k.startswith("rand") and k.endswith("@0.05")]
    rows.append(
      f"{h} & {100 * r['grad@0.05']['improved']:.0f} & {100 * r['grad@0.1']['improved']:.0f} & "
      f"{100 * np.mean(rnd):.0f} & {r['grad@0.05']['wm_predicted_gain']:.2f} & "
      f"{r['grad@0.05']['mean_gain']:.2f} & {r['grad@0.05']['wm_vs_sim_gain_corr']:.2f} & "
      f"{r['grad@0.1']['wm_predicted_gain']:.2f} & {r['grad@0.1']['mean_gain']:.2f} \\\\"
    )
  (OUT / "tab_gradients.tex").write_text(
    "\\begin{tabular}{rrrrrrrrr}\\toprule\n"
    " & \\multicolumn{3}{c}{starts improved [\\%]} & \\multicolumn{3}{c}{$\\delta=0.05$} & \\multicolumn{2}{c}{$\\delta=0.1$} \\\\\n"
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\n"
    "$H$ & grad $0.05$ & grad $0.1$ & random & predicted & real & corr & predicted & real \\\\\\midrule\n"
    + "\n".join(rows)
    + "\n\\bottomrule\\end{tabular}\n"
  )


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  res = load("all.json")
  fig_horizon(res["horizons"])
  fig_groups(res["horizons"])
  fig_termination(res)
  fig_gradients(load("grad_main.json"))
  fig_training()
  tex_tables(res, load("policy_shift.json"), load("grad_main.json"))
  print("figures and tables ->", OUT)


if __name__ == "__main__":
  main()
