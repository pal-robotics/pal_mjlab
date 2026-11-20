"""Benchmark SuperMimic policies the paper's way (Appendix F.1), with tracking metrics.

  uv run -m pal_mjlab.tasks.supermimic.scripts.evaluate Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers \
    --checkpoint-files <run>/model_N.pt ... --output-dir <folder> [--library-path <library>]

Every clip of the motion library plays once from its first frame in the task's play
configuration: foot friction 1.0, no randomization, no observation or reset noise. One env
per clip, so the library must fit in the simulator at once. Per clip:

  failed           the paper's termination fired (root height or tilt error)
  steps            steps tracked before the clip ended or the robot failed
  tracking_return  normalized return of the tracking terms only; the regularizers are left
                   out because they are in action units, which differ between action scales
  rmpkpe_<group>   root-relative mean per-body position error [m] (reference re-anchored on
                   the robot's root xy and heading, as in the reward), per body group
  joint_err_<g>    mean |q - q_ref| [rad; m for slides], per joint group
  torque_rate_<g>  mean |Δτ| / τ_max per step, per joint group
  saturation_<g>   fraction of (step, actuator) samples at the torque limit, per joint group
  foot_slide       mean horizontal speed [m/s] of the robot's feet while the reference foot is
                   planted (below FOOT_UP and slower than STILL horizontally and vertically)
  foot_slip        share of those foot-steps above SLIP
  root_drift       mean horizontal speed [m/s] of the robot's root while the reference root
                   stands still (slower than STILL, turning slower than 0.1 rad/s)

One JSON per checkpoint is written to the output folder, with the mean of every metric, the
failures per motion category and the per-clip values.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import mjlab
import mjlab.tasks  # noqa: F401  (populates the task registry)
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from pal_mjlab.tasks.supermimic.mdp import MotionLibraryCommand, MotionLibraryCommandCfg
from pal_mjlab.tasks.supermimic.mdp.rewards import TRACKING_ERRORS
from pal_mjlab.tasks.supermimic.mdp.tracking import relative_reference

# Groups by PAL's body and joint names.
BODY_GROUPS = {
  "torso": r"base_link|pelvis_\d_link",
  "legs": r"leg_.*",
  "arms": r"arm_.*",
}
JOINT_GROUPS = {"legs": r"leg_.*", "waist": r"pelvis_.*", "arms": r"arm_.*"}
FEET = r"leg_.*_5_link"
FOOT_UP, STILL, SLIP = 0.15, 0.02, 0.05  # m, m/s, m/s


@dataclass(frozen=True)
class EvaluateConfig:
  checkpoint_files: tuple[str, ...]
  """Policy checkpoints (model_N.pt) to benchmark."""
  output_dir: str
  """Folder for the results, one JSON per checkpoint."""
  library_path: str | None = None
  """Motion library folder; the task's default if not given."""
  device: str | None = None
  """Device to run on. Defaults to CUDA if available."""


def _group_index(names, patterns: dict[str, str], device) -> dict[str, torch.Tensor]:
  return {
    group: torch.tensor(
      [i for i, name in enumerate(names) if re.fullmatch(pattern, name)],
      device=device,
      dtype=torch.long,
    )
    for group, pattern in patterns.items()
  }


class Benchmark:
  def __init__(self, task_id: str, library_path: str | None, device: str) -> None:
    cfg = load_env_cfg(task_id, play=True)
    motion = cfg.commands["motion"]
    assert isinstance(motion, MotionLibraryCommandCfg)
    if library_path is not None:
      motion.library_path = library_path
    # One env per clip, so every clip plays exactly once. The envs are reset here: the
    # metrics of a clip's last step are read from its terminal state.
    clips = json.loads(
      (Path(motion.library_path).expanduser() / "clips.json").read_text()
    )
    cfg.scene.num_envs = len(clips["clips"])
    cfg.auto_reset = False
    self.env = ManagerBasedRlEnv(cfg=cfg, device=device)
    self.vec = RslRlVecEnvWrapper(self.env)
    self.task_id, self.device = task_id, device
    self.command = cast(
      MotionLibraryCommand, self.env.command_manager.get_term("motion")
    )
    self.robot = self.env.scene["robot"]

    body_names = self.command.cfg.body_names
    self.body_groups = _group_index(body_names, BODY_GROUPS, device)
    self.feet = _group_index(body_names, {"feet": FEET}, device)["feet"]
    joint_names = self.command.library.joint_names
    self.joint_groups = _group_index(joint_names, JOINT_GROUPS, device)
    # Actuators: joint name and force limit, from the compiled scene model.
    mj = self.env.sim.mj_model
    ctrl = self.robot.indexing.ctrl_ids.cpu().numpy()
    actuated = [
      mj.joint(int(mj.actuator_trnid[c, 0])).name.split("/", 1)[-1] for c in ctrl
    ]
    self.actuator_groups = _group_index(actuated, JOINT_GROUPS, device)
    self.tau_max = torch.tensor(mj.actuator_forcerange[ctrl, 1], device=device).float()
    rm = self.env.reward_manager
    terms = [rm.get_term_cfg(name) for name in rm.active_terms]
    self.tracking_terms = [
      (t.weight, t.params["std"], TRACKING_ERRORS[t.func])
      for t in terms
      if t.func in TRACKING_ERRORS
    ]

  def load_policy(self, checkpoint: Path):
    runner_cls = load_runner_cls(self.task_id)
    assert runner_cls is not None
    runner = runner_cls(self.vec, asdict(load_rl_cfg(self.task_id)), None, self.device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, map_location=self.device)
    return runner.get_inference_policy(self.device)

  @torch.no_grad()
  def run(self, policy) -> dict[str, np.ndarray]:
    env, cmd, n, dev = self.env, self.command, self.env.num_envs, self.device
    lib = cmd.library
    cmd.set_eval_clips(torch.arange(n, device=dev))
    obs, _ = self.vec.reset()
    assert (cmd.clip == torch.arange(n, device=dev)).all()

    finished = torch.zeros(n, dtype=torch.bool, device=dev)
    failed = torch.zeros(n, dtype=torch.bool, device=dev)
    steps = torch.zeros(n, device=dev)
    planted_steps = torch.zeros(n, device=dev)
    still_steps = torch.zeros(n, device=dev)
    sums: dict[str, torch.Tensor] = {}

    def add(name: str, value: torch.Tensor, mask: torch.Tensor) -> None:
      sums[name] = sums.get(name, torch.zeros(n, device=dev)) + torch.where(
        mask, value, 0.0
      )

    prev_tau = self.robot.data.actuator_force.clone()
    for _ in range(int(lib.clip_length.max()) + 2):
      target = cmd.frame.clone()  # the frame this step is scored against
      obs, _, dones, _ = self.vec.step(policy(obs))
      active = ~finished
      steps += active.float()
      failed |= active & env.termination_manager.get_term("anchor_failure")

      ref_qpos = lib.qpos[target].clone()
      ref_qpos[:, :3] += env.scene.env_origins
      ref = cmd.bodies(ref_qpos, lib.qvel[target])
      robot = cmd.bodies(cmd.robot_qpos, cmd.robot_qvel)
      tracking = sum(
        weight * torch.exp(-error(robot, ref) / std**2)
        for weight, std, error in self.tracking_terms
      )
      add("tracking_return", tracking, active)
      rel_pos, _ = relative_reference(ref, robot)
      body_err = (rel_pos - robot.pos).norm(dim=-1)
      for group, ids in self.body_groups.items():
        add(f"rmpkpe_{group}", body_err[:, ids].mean(-1), active)
      joint_err = (cmd.robot_qpos[:, 7:] - ref_qpos[:, 7:]).abs()
      for group, ids in self.joint_groups.items():
        add(f"joint_err_{group}", joint_err[:, ids].mean(-1), active)
      tau = self.robot.data.actuator_force
      rate = (tau - prev_tau).abs() / self.tau_max
      saturated = (tau.abs() >= 0.999 * self.tau_max).float()
      for group, ids in self.actuator_groups.items():
        add(f"torque_rate_{group}", rate[:, ids].mean(-1), active)
        add(f"saturation_{group}", saturated[:, ids].mean(-1), active)
      prev_tau = tau.clone()

      # Feet that slide where the reference foot is planted; a root that drifts where the
      # reference stands still.
      feet = self.feet
      planted = (
        (ref.pos[:, feet, 2] < FOOT_UP)
        & (ref.lin_vel[:, feet, :2].norm(dim=-1) < STILL)
        & (ref.lin_vel[:, feet, 2].abs() < STILL)
        & active[:, None]
      )
      speed = robot.lin_vel[:, feet, :2].norm(dim=-1)
      add("foot_slide", (speed * planted).sum(-1), active)
      add("foot_slip", ((speed > SLIP) & planted).float().sum(-1), active)
      planted_steps += planted.float().sum(-1)
      still = (
        (ref.lin_vel[:, 0, :2].norm(dim=-1) < STILL)
        & (ref.ang_vel[:, 0].norm(dim=-1) < 0.1)
        & active
      )
      add("root_drift", robot.lin_vel[:, 0, :2].norm(dim=-1), still)
      still_steps += still.float()

      done = dones.bool()
      finished |= done
      if finished.all():
        break
      if done.any():
        # Finished envs restart their clip and are no longer counted.
        env.reset(env_ids=done.nonzero().squeeze(-1))
        obs = self.vec.get_observations()

    per_clip = {"failed": failed.float(), "steps": steps}
    tracking_max = sum(weight for weight, _, _ in self.tracking_terms)
    reward_steps = (lib.clip_length - 1).float().clamp_min(1)
    per_clip["tracking_return"] = sums.pop("tracking_return") / (
      tracking_max * reward_steps
    )
    for name in ("foot_slide", "foot_slip"):
      per_clip[name] = torch.where(
        planted_steps > 0, sums.pop(name) / planted_steps, torch.nan
      )
    per_clip["root_drift"] = torch.where(
      still_steps > 0, sums.pop("root_drift") / still_steps, torch.nan
    )
    for name, value in sums.items():
      per_clip[name] = value / steps.clamp_min(1)
    return {name: value.cpu().numpy() for name, value in per_clip.items()}

  def report(self, per_clip: dict[str, np.ndarray], checkpoint: Path) -> dict:
    lib = self.command.library
    summary = {name: float(np.nanmean(value)) for name, value in per_clip.items()}
    summary["failed"] = int(per_clip["failed"].sum())
    categories = lib.clip_category.cpu().numpy()
    by_category = {
      category: {
        "clips": int((categories == i).sum()),
        "failed": int(per_clip["failed"][categories == i].sum()),
        "tracking_return": float(per_clip["tracking_return"][categories == i].mean()),
      }
      for i, category in enumerate(lib.categories)
    }
    return {
      "task": self.task_id,
      "checkpoint": str(checkpoint),
      "library": str(lib.path),
      "summary": summary,
      "by_category": by_category,
      "clips": {
        "name": list(lib.clip_names),
        **{name: value.tolist() for name, value in per_clip.items()},
      },
    }


def run_evaluate(task_id: str, cfg: EvaluateConfig) -> list[dict]:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  benchmark = Benchmark(task_id, cfg.library_path, device)
  output_dir = Path(cfg.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  results = []
  for checkpoint in map(Path, cfg.checkpoint_files):
    result = benchmark.report(
      benchmark.run(benchmark.load_policy(checkpoint)), checkpoint
    )
    output = output_dir / f"{checkpoint.parent.name}_{checkpoint.stem}.json"
    output.write_text(json.dumps(result))
    summary = result["summary"]
    print(
      f"{checkpoint}: failed {summary['failed']}/{len(result['clips']['name'])}, "
      f"tracking_return {summary['tracking_return']:.3f} -> {output}",
      flush=True,
    )
    results.append(result)
  benchmark.env.close()
  return results


def main() -> None:
  tasks = [t for t in list_tasks() if "SuperMimic" in t]
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  # mjlab's flags, but checkpoints as plain paths (--checkpoint-files a b).
  flags = tuple(
    f
    for f in mjlab.TYRO_FLAGS
    if f is not tyro.conf.UsePythonSyntaxForLiteralCollections
  )
  args = tyro.cli(
    EvaluateConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=flags,
  )
  run_evaluate(chosen_task, args)


if __name__ == "__main__":
  main()
