"""Benchmark SuperMimic policies the paper's way (Appendix F.1), with tracking metrics.

Every clip of the motion library plays once from its first frame in the task's play
config: foot friction 1.0, no randomization, no observation or reset noise, deterministic
policy. Per clip:

  failed           the paper's termination fired (root height or tilt error)
  steps            steps tracked before the clip ended or the robot failed
  tracking_return  normalized return of the six tracking terms only; the regularizers are
                   left out because they are measured in action units, which differ between
                   action parameterizations
  rmpkpe_<group>   root-relative mean per-body position error [m] (reference re-anchored on
                   the robot's root xy and heading, as in the reward), per body group
  joint_err_<g>    mean |q - q_ref| [rad] of the actuated joints, per joint group
  torque_rate_<g>  mean |Δτ| / τ_max per step, per joint group
  saturation_<g>   fraction of (step, actuator) samples at the torque limit, per joint group
  foot_slide       mean horizontal speed [m/s] of the robot's feet while the reference foot is planted
                   (reference foot below FOOT_UP and slower than STILL horizontally and vertically)
  foot_slip        share of those foot-steps above SLIP
  root_drift       mean horizontal speed [m/s] of the robot's root while the reference root stands still
                   (slower than STILL, turning slower than 0.1 rad/s): standing still

Usage:
  python -m pal_mjlab.scripts.supermimic.eval <task> --checkpoints run/model_200.pt ... \\
    --out-dir results/
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import mjlab.tasks  # noqa: F401  (populates the task registry)
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from pal_mjlab.tasks.supermimic.library import resolve_library
from pal_mjlab.tasks.supermimic.mdp import MotionLibraryCommand
from pal_mjlab.tasks.supermimic.mdp.tracking import relative_reference

BODY_GROUPS = {
  "torso": r"base_link|pelvis_\d_link",
  "legs": r"leg_.*",
  "arms": r"arm_.*",
}
JOINT_GROUPS = {"legs": r"leg_.*", "waist": r"pelvis_.*", "arms": r"arm_.*"}
FEET = r"leg_.*_5_link"  # the tracked foot bodies
FOOT_UP, STILL, SLIP = 0.15, 0.02, 0.05  # m, m/s, m/s


def _group_index(names, patterns: dict[str, str], device) -> dict[str, torch.Tensor]:
  return {
    g: torch.tensor(
      [i for i, n in enumerate(names) if re.fullmatch(p, n)],
      device=device,
      dtype=torch.long,
    )
    for g, p in patterns.items()
  }


class Benchmark:
  def __init__(self, task: str, library: str | None, device: str) -> None:
    cfg = load_env_cfg(task, play=True)
    motion = cfg.commands["motion"]
    if library is not None:
      motion.library_path = library
    # One env per clip, so every clip plays exactly once.
    clips = json.loads(
      (
        resolve_library(motion.library_path, motion.dataset, motion.subset)
        / "clips.json"
      ).read_text()
    )
    cfg.scene.num_envs = len(clips["clips"])
    self.env = ManagerBasedRlEnv(cfg=cfg, device=device)
    self.vec = RslRlVecEnvWrapper(self.env)
    self.task, self.device = task, device
    self.command: MotionLibraryCommand = self.env.command_manager.get_term("motion")
    lib = self.command.library
    self.robot = self.env.scene["robot"]

    self.body_groups = _group_index(self.command.cfg.body_names, BODY_GROUPS, device)
    self.feet = _group_index(self.command.cfg.body_names, {"feet": FEET}, device)[
      "feet"
    ]
    goal_names = self.command.goal_joint_names
    self.joint_ids = self.command._goal_joint_ids
    self.joint_groups = _group_index(goal_names, JOINT_GROUPS, device)
    # Actuators: joint name and force limit, from the compiled scene model.
    mj = self.env.sim.mj_model
    ctrl = self.robot.indexing.ctrl_ids.cpu().numpy()
    act_joint = [
      mj.joint(int(mj.actuator_trnid[c, 0])).name.split("/", 1)[-1] for c in ctrl
    ]
    self.act_groups = _group_index(act_joint, JOINT_GROUPS, device)
    self.tau_max = torch.tensor(mj.actuator_forcerange[ctrl, 1], device=device).float()
    rm = self.env.reward_manager
    self.tracking_terms = [
      i for i, n in enumerate(rm.active_terms) if n.startswith("motion_")
    ]
    self.tracking_max = sum(
      rm.get_term_cfg(rm.active_terms[i]).weight for i in self.tracking_terms
    )
    self.clip_lengths = lib.clip_length
    self.clip_names, self.categories = lib.clip_names, lib.categories
    self.clip_category = lib.clip_category

  def load_policy(self, checkpoint: Path):
    runner_cls = load_runner_cls(self.task) or MjlabOnPolicyRunner
    runner = runner_cls(self.vec, asdict(load_rl_cfg(self.task)), None, self.device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, map_location=self.device)
    return runner.get_inference_policy(self.device)

  @torch.no_grad()
  def run(self, policy) -> dict:
    env, cmd, n = self.env, self.command, self.env.num_envs
    dev = self.device
    cmd.set_eval_clips(torch.arange(n, device=dev))
    obs, _ = self.vec.reset()
    assert (cmd.clip == torch.arange(n, device=dev)).all()

    finished = torch.zeros(n, dtype=torch.bool, device=dev)
    failed = torch.zeros(n, dtype=torch.bool, device=dev)
    steps = torch.zeros(n, device=dev)
    sums: dict[str, torch.Tensor] = {}
    counts = torch.zeros(n, device=dev)
    planted_steps = torch.zeros(n, device=dev)
    still_steps = torch.zeros(n, device=dev)

    def add(name: str, value: torch.Tensor, mask: torch.Tensor) -> None:
      sums[name] = sums.get(name, torch.zeros(n, device=dev)) + torch.where(
        mask, value, 0.0
      )

    lib = cmd.library
    prev_tau = self.robot.data.actuator_force.clone()
    max_steps = int(self.clip_lengths.max()) + 2
    for _ in range(max_steps):
      target = cmd.frame.clone()  # the frame this step is scored against
      obs, _, dones, _ = self.vec.step(policy(obs))
      active = ~finished
      done = dones.bool()
      step_reward = env.reward_manager._step_reward
      tracking = step_reward[:, self.tracking_terms].sum(-1) * env.step_dt
      add("tracking_return", tracking, active)
      failed |= active & env.termination_manager.get_term("anchor_failure")
      steps += active.float()

      # Kinematic and effort metrics on envs that are still in their episode (the state of
      # envs that just ended has been reset).
      live = active & ~done
      qp = lib.qpos[target].clone()
      qp[:, :3] += env.scene.env_origins
      ref = cmd._bodies(qp, lib.qvel[target])
      robot = cmd._bodies(cmd.robot_qpos, cmd.robot_qvel)
      rel_pos, _ = relative_reference(ref, robot)
      body_err = (rel_pos - robot.pos).norm(dim=-1)
      for g, ids in self.body_groups.items():
        add(f"rmpkpe_{g}", body_err[:, ids].mean(-1), live)
      q_err = (cmd.robot_qpos[:, 7:] - qp[:, 7:])[:, self.joint_ids].abs()
      for g, ids in self.joint_groups.items():
        add(f"joint_err_{g}", q_err[:, ids].mean(-1), live)
      tau = self.robot.data.actuator_force
      rate = (tau - prev_tau).abs() / self.tau_max
      sat = (tau.abs() >= 0.999 * self.tau_max).float()
      for g, ids in self.act_groups.items():
        add(f"torque_rate_{g}", rate[:, ids].mean(-1), live)
        add(f"saturation_{g}", sat[:, ids].mean(-1), live)
      prev_tau = tau.clone()
      counts += live.float()

      # Feet that slide where the reference foot is planted; a root that drifts where it stands still.
      f = self.feet
      planted = (
        (ref.pos[:, f, 2] < FOOT_UP)
        & (ref.lin_vel[:, f, :2].norm(dim=-1) < STILL)
        & (ref.lin_vel[:, f, 2].abs() < STILL)
        & live[:, None]
      )
      speed = robot.lin_vel[:, f, :2].norm(dim=-1)
      add("foot_slide", (speed * planted).sum(-1), live)
      add("foot_slip", ((speed > SLIP) & planted).float().sum(-1), live)
      planted_steps += planted.float().sum(-1)
      still = (
        (ref.lin_vel[:, 0, :2].norm(dim=-1) < STILL)
        & (ref.ang_vel[:, 0].norm(dim=-1) < 0.1)
        & live
      )
      add("root_drift", robot.lin_vel[:, 0, :2].norm(dim=-1), still)
      still_steps += still.float()

      finished |= done
      if finished.all():
        break

    per_clip = {"failed": failed.float(), "steps": steps}
    reward_steps = (self.clip_lengths - 1).float().clamp_min(1)
    per_clip["tracking_return"] = sums.pop("tracking_return") / (
      self.tracking_max * env.step_dt * reward_steps
    )
    for k in ("foot_slide", "foot_slip"):
      per_clip[k] = torch.where(
        planted_steps > 0, sums.pop(k) / planted_steps, torch.nan
      )
    per_clip["root_drift"] = torch.where(
      still_steps > 0, sums.pop("root_drift") / still_steps, torch.nan
    )
    for k, v in sums.items():
      per_clip[k] = v / counts.clamp_min(1)
    return {k: v.cpu().numpy() for k, v in per_clip.items()}

  def report(self, per_clip: dict[str, np.ndarray], checkpoint: Path) -> dict:
    summary = {k: float(np.nanmean(v)) for k, v in per_clip.items()}
    summary["failed"] = int(per_clip["failed"].sum())
    cats = self.clip_category.cpu().numpy()
    by_category = {
      c: {
        "clips": int((cats == i).sum()),
        "failed": int(per_clip["failed"][cats == i].sum()),
        "tracking_return": float(per_clip["tracking_return"][cats == i].mean()),
        "rmpkpe_arms": float(per_clip["rmpkpe_arms"][cats == i].mean()),
      }
      for i, c in enumerate(self.categories)
      if (cats == i).any()
    }
    return {
      "task": self.task,
      "checkpoint": str(checkpoint),
      "summary": summary,
      "by_category": by_category,
      "clips": {
        "name": list(self.clip_names),
        **{k: v.tolist() for k, v in per_clip.items()},
      },
    }


def main(
  task: str,
  checkpoints: list[Path],
  out_dir: Path,
  library: str | None = None,
  device: str = "cuda",
) -> None:
  configure_torch_backends()
  bench = Benchmark(task, library, device)
  out_dir.mkdir(parents=True, exist_ok=True)
  for ckpt in checkpoints:
    result = bench.report(bench.run(bench.load_policy(ckpt)), ckpt)
    out = out_dir / f"{ckpt.parent.name}_{ckpt.stem}.json"
    out.write_text(json.dumps(result))
    s = result["summary"]
    print(
      f"{ckpt}: failed {s['failed']}/{len(result['clips']['name'])}, "
      f"tracking_return {s['tracking_return']:.3f}, "
      f"rmpkpe legs/arms/torso {s['rmpkpe_legs']:.3f}/{s['rmpkpe_arms']:.3f}/"
      f"{s['rmpkpe_torso']:.3f} m -> {out}",
      flush=True,
    )


if __name__ == "__main__":
  tyro.cli(main)
