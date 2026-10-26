"""Figures for the action-scale note (docs/supermimic/m1/).

Two sources:
  - the `dev` reference motions alone: the action each joint needs to hold its reference
    pose, the action-rate penalty of following it, and how far one action unit moves the
    tracked bodies (lever arm), at 3.5 Hz and 1 Hz arm/waist gains;
  - rollouts of the D27 A/B policies (``scripts/supermimic/action_stats.py`` outputs in
    ``logs/action_stats/``): measured actions, penalty, and how much of each joint's
    reference motion the policy reproduces.

Run from the repository root: .venv/bin/python docs/supermimic/m1/make_action_scale_figures.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import mjlab  # noqa: F401  (loads the task registry before pal_mjlab.robots)
import numpy as np
import torch
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  INIT_STATE,
  get_kangaroo_grippers_spec,
)
from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import DEFAULT_LIBRARY, TRACKED_BODIES
from pal_mjlab.tasks.supermimic.kinematics import (
  TorchKinematics,
  compile_kinematic_model,
)

STATS = Path("logs/action_stats")
OUT = Path("docs/supermimic/m1/figures")
DT = 0.02  # control period
RATE_W = 0.1  # action-rate weight (Table A.2)
TRACK_MAX = (
  5.0  # sum of the tracking weights: the largest per-step tracking reward (both x dt)
)

# Fixed slot per entity (validated pair); G1 reference in neutral gray.
C35, C1, GRAY = "#2a78d6", "#eb6834", "#8d8b85"
INK, MUTED, GRID = "#1f1f1f", "#5c5c5c", "#e4e4e1"

plt.rcParams.update(
  {
    "font.size": 8,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,
  }
)

# Joints shown: left arm 1-7, waist, and two leg joints for scale.
SHOW = [f"arm_left_{i}_joint" for i in range(1, 8)] + [
  "pelvis_1_joint",
  "pelvis_2_joint",
  "leg_left_2_joint",
  "leg_left_4_joint",
]
LABEL = {
  **{f"arm_left_{i}_joint": f"arm {i}" for i in range(1, 8)},
  "pelvis_1_joint": "waist 1",
  "pelvis_2_joint": "waist 2",
  "leg_left_2_joint": "hip pitch",
  "leg_left_4_joint": "ankle",
}


def load_runs() -> dict[str, dict]:
  runs = {}
  for tag in ("ab_3p5hz", "ab_1hz"):
    d = np.load(STATS / f"{tag}.npz")
    runs[tag] = {k: d[k] for k in d.files}
  assert list(runs["ab_3p5hz"]["joint_names"]) == list(runs["ab_1hz"]["joint_names"])
  return runs


def default_pose(names: list[str]) -> np.ndarray:
  q0 = np.zeros(len(names))
  for pattern, value in INIT_STATE.joint_pos.items():
    for i, n in enumerate(names):
      if re.fullmatch(pattern, n):
        q0[i] = value
  return q0


def reference_demand(names, alpha35, alpha1):
  """Per joint, from the dev references: |hold action| quantiles and action-rate penalty."""
  lib = Path(DEFAULT_LIBRARY).expanduser()
  lib_names = json.loads((lib / "clips.json").read_text())["joint_names"]
  idx = [lib_names.index(n) for n in names]
  qpos = np.load(lib / "qpos.npy", mmap_mode="r")[:, 7:][:, idx]
  qvel = np.load(lib / "qvel.npy", mmap_mode="r")[:, 6:][:, idx]
  offset = np.abs(qpos - default_pose(names))
  step = np.abs(qvel) * DT  # setpoint travel per control step
  out = {}
  for tag, alpha in (("3.5", alpha35), ("1", alpha1)):
    hold = offset / alpha
    pen = RATE_W * (step / alpha) ** 2 / TRACK_MAX  # share of the max tracking reward
    out[tag] = {
      "hold50": np.median(hold, 0),
      "hold95": np.percentile(hold, 95, 0),
      "pen95": np.percentile(pen, 95, 0),
    }
  return out


def lever_arms(names: list[str], n_frames: int = 2000) -> np.ndarray:
  """Per joint: RMS over dev frames of the largest tracked-body displacement per radian."""
  model = compile_kinematic_model(get_kangaroo_grippers_spec())
  fk = TorchKinematics(model, TRACKED_BODIES, dtype=torch.float64)
  lib = Path(DEFAULT_LIBRARY).expanduser()
  qpos = np.load(lib / "qpos.npy", mmap_mode="r")
  rng = np.random.default_rng(0)
  rows = np.sort(rng.choice(len(qpos), n_frames, replace=False))
  q = torch.from_numpy(np.asarray(qpos[rows], dtype=np.float64))
  zero = torch.zeros(len(rows), model.nv, dtype=torch.float64)
  base = fk(q, zero).body_pos
  lib_names = json.loads((lib / "clips.json").read_text())["joint_names"]
  eps, out = 1e-4, []
  for n in names:
    dq = q.clone()
    dq[:, 7 + lib_names.index(n)] += eps
    disp = (fk(dq, zero).body_pos - base).norm(dim=-1) / eps  # (frames, bodies) m/rad
    out.append(float(disp.max(dim=-1).values.square().mean().sqrt()))
  return np.array(out)


def bars(ax, x, values, colors, log=False, width=0.36, zero_note=None):
  for k, (v, c) in enumerate(zip(values, colors, strict=True)):
    ax.bar(x + (k - (len(values) - 1) / 2) * width, v, width * 0.92, color=c)
  if log:
    ax.set_yscale("log")
  if zero_note is not None:
    for i in np.flatnonzero(np.asarray(values[0]) <= 0):
      ax.text(x[i], zero_note, "0", ha="center", va="bottom", color=MUTED, fontsize=7)


def per_env_steps(r: dict, key: str) -> np.ndarray:
  """Per-step change of ``key`` within each env's sequence (rows are in step order)."""
  env = r["env"]
  order = np.argsort(env, kind="stable")
  v, e = r[key][order], env[order]
  d = np.diff(v, axis=0)
  return d[e[1:] == e[:-1]]


def legend(fig, labels, colors) -> None:
  handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
  fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=len(labels),
    fontsize=7.5,
    bbox_to_anchor=(0.5, 1.12),
  )


def finish(fig, axs, ticks, x, name) -> None:
  for ax in np.ravel(axs):
    ax.set_xticks(x, ticks, rotation=60, ha="right")
  fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
  fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight")


def main() -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  runs = load_runs()
  names = list(runs["ab_3p5hz"]["joint_names"])
  sel = [names.index(n) for n in SHOW]
  a35, a1 = runs["ab_3p5hz"]["alpha"], runs["ab_1hz"]["alpha"]
  demand = reference_demand(names, a35, a1)
  lever = lever_arms([names[i] for i in sel])
  x = np.arange(len(SHOW))
  ticks = [LABEL[n] for n in SHOW]
  gains = ["3.5 Hz", "1 Hz on arms and waist"]

  # Figure 1: what the dev references demand, per joint (no policy involved).
  fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.3), constrained_layout=True)
  ax = axs[0]
  bars(
    ax,
    x,
    [demand["3.5"]["hold95"][sel], demand["1"]["hold95"][sel]],
    [C35, C1],
    log=True,
  )
  ax.set_title(
    "Action needed to hold the reference\np95 of $|q_{ref}-q_0|/\\alpha$", fontsize=8
  )
  ax.axhline(4, color=GRAY, lw=1)
  ax.text(len(x) - 0.4, 4.6, "$|a|=4$", color=MUTED, fontsize=7, ha="right")
  ax = axs[1]
  bars(
    ax,
    x,
    [100 * demand["3.5"]["pen95"][sel], 100 * demand["1"]["pen95"][sel]],
    [C35, C1],
    log=True,
  )
  ax.set_title(
    "Action-rate penalty of following it\np95, % of max tracking reward", fontsize=8
  )
  ax = axs[2]
  per_unit = [100 * lever * a35[sel], 100 * lever * a1[sel]]
  bars(ax, x, per_unit, [C35, C1], log=True, zero_note=0.12)
  ax.set_ylim(0.1, 100)
  ax.set_title(
    "Tracked-body motion per action unit\ncm (largest body; 0 = rotation only)",
    fontsize=8,
  )
  legend(fig, gains, [C35, C1])
  finish(fig, axs, ticks, x, "reference_demand")

  # Figure 2: what the trained A/B policies do (96 dev clips, first 10 s each).
  measured = {}
  for tag in ("ab_3p5hz", "ab_1hz"):
    r = runs[tag]
    alpha = r["alpha"]
    a, ap, q, qr = r["action"], r["action_prev"], r["q"], r["q_ref"]
    slope = ((q - q.mean(0)) * (qr - qr.mean(0))).mean(0) / np.maximum(qr.var(0), 1e-8)
    dref = per_env_steps(r, "q_ref")
    measured[tag] = {
      "a95": np.percentile(np.abs(a), 95, 0),
      "reach95": np.percentile(np.abs(a), 95, 0) * alpha,
      "need95": np.percentile(np.abs(qr), 95, 0),
      "pen_paid": 100 * RATE_W * ((a - ap) ** 2).mean(0) / TRACK_MAX,
      "pen_follow": 100 * RATE_W * ((dref / alpha) ** 2).mean(0) / TRACK_MAX,
      "follow": slope,
      "err": np.abs(q - qr).mean(0),
    }
  m35, m1 = measured["ab_3p5hz"], measured["ab_1hz"]
  fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.6), constrained_layout=True)
  ax = axs[0, 0]
  bars(ax, x, [m35["a95"][sel], m1["a95"][sel]], [C35, C1])
  ax.set_title("Policy action, p95 of $|a|$ (units)", fontsize=8)
  ax = axs[0, 1]
  bars(ax, x, [m35["reach95"][sel], m1["reach95"][sel]], [C35, C1])
  ax.plot(
    x,
    m35["need95"][sel],
    ls="none",
    marker="_",
    ms=14,
    mew=2,
    color=INK,
    label="reference needs, p95 $|q_{ref}-q_0|$",
  )
  ax.legend(loc="upper left", fontsize=7)
  ax.set_title("What those actions reach: p95 $|a|\\cdot\\alpha$ (rad)", fontsize=8)
  ax = axs[1, 0]
  bars(
    ax, x, [m35["pen_follow"][sel], m35["pen_paid"][sel]], [C35, "#9ec2ef"], log=True
  )
  ax.set_title(
    "3.5 Hz: action-rate penalty, % of max tracking reward\n"
    "if it followed the reference (dark) vs paid (light)",
    fontsize=8,
  )
  ax = axs[1, 1]
  bars(ax, x, [m35["follow"][sel], m1["follow"][sel]], [C35, C1])
  ax.axhline(1, color=GRAY, lw=1)
  ax.set_ylim(0, 1.15)
  ax.set_title(
    "Reference motion reproduced: slope of $q$ on $q_{ref}$ (1 = follows)", fontsize=8
  )
  legend(fig, ["3.5 Hz policy", "1 Hz policy (arms, waist)"], [C35, C1])
  finish(fig, axs, ticks, x, "policy_actions")

  summary = {
    "joints": [LABEL[n] for n in SHOW],
    "alpha_3.5": a35[sel].round(4).tolist(),
    "alpha_1": a1[sel].round(4).tolist(),
    "lever_cm_per_rad": (100 * lever).round(1).tolist(),
    "reference": {
      k: {kk: vv[sel].round(4).tolist() for kk, vv in v.items()}
      for k, v in demand.items()
    },
    "measured": {
      k: {kk: vv[sel].round(4).tolist() for kk, vv in v.items()}
      for k, v in measured.items()
    },
    "steps": {k: int(len(runs[k]["action"])) for k in runs},
  }
  (OUT / "action_scale_numbers.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
  main()
