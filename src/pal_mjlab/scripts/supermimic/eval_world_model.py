"""Evaluate world models on held-out clips: open-loop prediction in physical units (M2).

For each horizon h, windows of h transitions from held-out clips are rolled out from the
logged first state on the logged actions. At step h:
  root_z [cm], tilt [deg], yaw [deg], root_xy [cm]   root errors
  joint_<group> [rad], joint_vel [rad/s]              joint errors by group
  body_world [cm], body_local_<group> [cm]            tracked bodies after FK: world error,
                                                      and error relative to the root
                                                      (heading frame), by group
Along the window:
  reward_abs        mean |r̂ - r| per step of the env's tracking reward (term values)
  return_rel        |Σ r̂ - Σ r| / Σ r over the window
and the termination head's AUC on held-out failures (windows up to 8 steps).
A "no change" predictor (state and pose kept) is the floor.

  python -m pal_mjlab.scripts.supermimic.eval_world_model --models main=wm/main/wm_best.pt \\
    k1=wm/k1/wm_best.pt --data ... --out-json results.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import mjlab.tasks  # noqa: F401  (populates the task registry)
import torch
import tyro
from mjlab.tasks.registry import load_env_cfg
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import get_kangaroo_grippers_spec
from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import TRACKED_BODIES
from pal_mjlab.tasks.supermimic.kinematics import (
  TorchKinematics,
  compile_kinematic_model,
  matrix_from_rot6d,
  quat_rotate_inv,
  yaw_quat,
)
from pal_mjlab.tasks.supermimic.library import MotionLibrary
from pal_mjlab.tasks.supermimic.mdp.rewards import tracking_reward_fn
from pal_mjlab.tasks.supermimic.mdp.tracking import BodyState
from pal_mjlab.tasks.supermimic.rl.replay import RolloutData, holdout_clip_mask
from pal_mjlab.tasks.supermimic.rl.world_model import (
  WorldModel,
  WorldModelCfg,
  tilt_angle,
)
from pal_mjlab.tasks.supermimic.state import pose_delta, state_to_qpos_qvel

DEFAULT_LIBRARY = "~/Projects/supermimic/kangaroo_bones_seed/data/library/dev"
DEFAULT_TASK = "Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Grippers-ArmWaist1Hz"
JOINT_GROUPS = {
  "legs": r"leg_.*_(\d|length)_joint",
  "passive": r"leg_.*_(femur|knee)_joint",
  "waist": r"pelvis_.*",
  "arms": r"arm_.*",
}
BODY_GROUPS = {
  "torso": r"base_link|pelvis_\d_link",
  "legs": r"leg_.*",
  "arms": r"arm_.*",
}


def load_model(path: Path, device: str) -> WorldModel:
  ckpt = torch.load(path, map_location=device, weights_only=False)
  cfg = ckpt["cfg"]
  cfg["hidden"] = tuple(cfg["hidden"])
  cfg["termination_hidden"] = tuple(cfg["termination_hidden"])
  cfg["pose_scale"] = tuple(cfg["pose_scale"])
  model = WorldModel(WorldModelCfg(**cfg)).to(device)
  model.load_state_dict(ckpt["state_dict"])
  return model.eval()


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
  pos, neg = scores[labels], scores[~labels]
  if len(pos) == 0 or len(neg) == 0:
    return float("nan")
  ranks = torch.cat([pos, neg]).argsort().argsort().float() + 1
  return float(
    (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
  )


def failure_rule_score(state: torch.Tensor, ref_root: torch.Tensor) -> torch.Tensor:
  """SuperMimic's failure rule as a margin (> 1 = failure): root height error / 0.25 m or
  gravity-z error / 0.8, whichever is larger."""
  gz = -matrix_from_rot6d(state[:, 1:7])[:, 2, 2]
  return torch.maximum(
    (state[:, 0] - ref_root[:, 0]).abs() / 0.25, (gz - ref_root[:, 1]).abs() / 0.8
  )


class Evaluator:
  def __init__(self, library: MotionLibrary, task: str, device: str) -> None:
    self.lib = library
    model = compile_kinematic_model(get_kangaroo_grippers_spec())
    self.fk = TorchKinematics(model, TRACKED_BODIES, device=device)
    self.reward = tracking_reward_fn(load_env_cfg(task).rewards)
    names = library.joint_names
    self.joint_groups = {
      g: torch.tensor(
        [i for i, n in enumerate(names) if re.fullmatch(p, n)], device=device
      )
      for g, p in JOINT_GROUPS.items()
    }
    self.body_groups = {
      g: torch.tensor(
        [i for i, n in enumerate(TRACKED_BODIES) if re.fullmatch(p, n)], device=device
      )
      for g, p in BODY_GROUPS.items()
    }
    self.nj = len(names)

  def bodies(self, state: torch.Tensor, pose: torch.Tensor) -> BodyState:
    kin = self.fk(*state_to_qpos_qvel(state, pose))
    return BodyState(kin.body_pos, kin.body_quat, kin.body_lin_vel, kin.body_ang_vel)

  def reference(self, frame: torch.Tensor) -> BodyState:
    kin = self.fk(self.lib.qpos[frame], self.lib.qvel[frame])
    return BodyState(kin.body_pos, kin.body_quat, kin.body_lin_vel, kin.body_ang_vel)

  def local(self, b: BodyState) -> torch.Tensor:
    heading = yaw_quat(b.quat[:, :1]).expand_as(b.quat)
    return quat_rotate_inv(heading, b.pos - b.pos[:, :1])

  def errors(
    self, s: torch.Tensor, p: torch.Tensor, s_true: torch.Tensor, p_true: torch.Tensor
  ):
    m: dict[str, torch.Tensor] = {}
    m["root_z_cm"] = (s[:, 0] - s_true[:, 0]).abs() * 100
    m["tilt_deg"] = torch.rad2deg(tilt_angle(s, s_true))
    d = pose_delta(p_true, p)
    m["yaw_deg"] = torch.rad2deg(d[:, 2].abs())
    m["root_xy_cm"] = d[:, :2].norm(dim=-1) * 100
    q, qt = s[:, 13 : 13 + self.nj], s_true[:, 13 : 13 + self.nj]
    for g, ids in self.joint_groups.items():
      m[f"joint_{g}_rad"] = (q[:, ids] - qt[:, ids]).abs().mean(-1)
    m["joint_vel_rads"] = (
      (s[:, 13 + self.nj :] - s_true[:, 13 + self.nj :]).abs().mean(-1)
    )
    b, bt = self.bodies(s, p), self.bodies(s_true, p_true)
    m["body_world_cm"] = (b.pos - bt.pos).norm(dim=-1).mean(-1) * 100
    local_err = (self.local(b) - self.local(bt)).norm(dim=-1) * 100
    for g, ids in self.body_groups.items():
      m[f"body_local_{g}_cm"] = local_err[:, ids].mean(-1)
    return m

  @torch.no_grad()
  def rollout(self, model: WorldModel | None, w: dict[str, torch.Tensor], h: int):
    s, p = w["states"][:, 0], w["poses"][:, 0]  # model None: "no change" predictor
    rewards, logits, rules = [], [], []
    for j in range(h):
      if model is not None:
        s, p = model.step(s, p, w["actions"][:, j], w["context"])
        logits.append(model.termination_logit(s, w["ref_root"][:, j]))
        rules.append(failure_rule_score(s, w["ref_root"][:, j]))
      ref = self.reference(w["frame"][:, j])
      rewards.append(self.reward(self.bodies(s, p), ref))
    if logits:
      self.last_rule_scores = torch.stack(rules, 1)
    return s, p, torch.stack(rewards, 1), (torch.stack(logits, 1) if logits else None)


def main(
  models: list[str],
  data: list[Path],
  out_json: Path,
  library: str = DEFAULT_LIBRARY,
  task: str = DEFAULT_TASK,
  horizons: tuple[int, ...] = (1, 8, 16, 32, 64),
  windows: int = 4096,
  holdout_fraction: float = 0.1,
  sources: list[str] | None = None,
  seed: int = 0,
  device: str = "cuda",
) -> None:
  """``models``: name=path pairs."""
  gen = torch.Generator(device=device).manual_seed(seed)
  lib = MotionLibrary(library, device)
  rollouts = RolloutData(data, lib, device)
  ev = Evaluator(lib, task, device)
  held_out = torch.from_numpy(holdout_clip_mask(lib.clip_names, holdout_fraction)).to(
    device
  )
  predictors: dict[str, WorldModel | None] = {"no_change": None}
  for spec in models:
    name, path = spec.split("=", 1)
    predictors[name] = load_model(Path(path), device)

  results: dict[str, dict] = {name: {} for name in predictors}
  parity = None
  for h in horizons:
    starts = rollouts.valid_starts(h, clips=held_out, sources=sources)
    starts = starts[torch.randperm(len(starts), generator=gen, device=device)[:windows]]
    w = rollouts.windows(starts, h)
    s_true, p_true = w["states"][:, -1], w["poses"][:, -1]
    # True per-step rewards: the analytic reward on the logged next states.
    r_true = torch.stack(
      [
        ev.reward(
          ev.bodies(w["states"][:, j + 1], w["poses"][:, j + 1]),
          ev.reference(w["frame"][:, j]),
        )
        for j in range(h)
      ],
      1,
    )
    if parity is None:
      parity = float((r_true - w["reward"]).abs().max())
    for name, model in predictors.items():
      s, p, r_pred, _ = ev.rollout(model, w, h)
      errs = ev.errors(s, p, s_true, p_true)
      m = {k: float(v.mean()) for k, v in errs.items()}
      m["body_world_cm_p95"] = float(torch.quantile(errs["body_world_cm"], 0.95))
      m["reward_abs"] = float((r_pred - r_true).abs().mean())
      m["return_rel"] = float(
        ((r_pred.sum(1) - r_true.sum(1)).abs() / r_true.sum(1).clamp_min(1e-6)).mean()
      )
      m["windows"] = len(starts)
      results[name][h] = m

  # Termination head: held-out windows of up to 8 steps (failures are short-window events).
  term = {}
  # Every held-out window ending in a failure, plus 4x as many random ones.
  starts = rollouts.valid_starts(8, clips=held_out, sources=sources)
  ends_in_failure = rollouts.failure[starts // rollouts.E + 7, starts % rollouts.E]
  positives = starts[ends_in_failure]
  pick = torch.randperm(len(starts), generator=gen, device=device)[: 4 * len(positives)]
  w = rollouts.windows(torch.cat([positives, starts[pick]]), 8)
  for name, model in predictors.items():
    if model is None:
      continue
    *_, logits = ev.rollout(model, w, 8)
    labels = w["failure"].flatten()
    rule = ev.last_rule_scores.flatten()
    term[name] = {
      "auc": auc(logits.flatten(), labels),
      "rule_auc": auc(rule, labels),
      "positives": int(labels.sum()),
      # Scores for ROC plots: the learned head (logit) and the analytic rule margin.
      "scores": {
        "label": labels.int().tolist(),
        "head": logits.flatten().tolist(),
        "rule": rule.tolist(),
      },
    }

  report = {"reward_parity_max_abs": parity, "horizons": results, "termination": term}
  out_json.parent.mkdir(parents=True, exist_ok=True)
  out_json.write_text(json.dumps(report, indent=1))
  print(f"reward parity (analytic on logged states vs env): max |diff| {parity:.2e}")
  keys = [
    "body_world_cm", "body_world_cm_p95", "body_local_arms_cm", "body_local_legs_cm",
    "root_z_cm", "tilt_deg", "root_xy_cm", "joint_arms_rad", "joint_legs_rad",
    "reward_abs", "return_rel",
  ]  # fmt: skip
  for h in horizons:
    print(f"\n**horizon {h}** ({results['no_change'][h]['windows']} windows)\n")
    print("| predictor | " + " | ".join(keys) + " |")
    print("|---|" + "---|" * len(keys))
    for name in predictors:
      print(
        f"| {name} | " + " | ".join(f"{results[name][h][k]:.4g}" for k in keys) + " |"
      )
  print(
    "\ntermination AUC (head / analytic rule on predicted states):",
    {k: (round(v["auc"], 4), round(v["rule_auc"], 4)) for k, v in term.items()},
  )


if __name__ == "__main__":
  tyro.cli(main)
