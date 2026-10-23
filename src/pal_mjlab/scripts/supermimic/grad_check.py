"""Do world-model gradients improve the real return? (M2b)

From held-out logged start states (state, sim context ξ, reference clip/frame), take the
logged H-step action sequence a and the world model's imagined return

    J(a) = Σ_t γ^t ĉ_t r̂_{t+1},   ĉ the survival weight from the termination head

(paper Eq. 2 without the critic). Perturb a along the normalized gradient ∂J/∂a and,
as a control, along random directions of the same size, then replay every sequence open
loop in the simulator from the same start state and ξ. Useful gradients improve the
simulated return clearly more often than random directions.

Also reports the gradient-norm profile ||∂J/∂a_t|| along the horizon and how well the
model's predicted improvement correlates with the simulated one.
"""

from __future__ import annotations

import json
from pathlib import Path

import mjlab.tasks  # noqa: F401  (populates the task registry)
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from pal_mjlab.scripts.supermimic.eval_world_model import (
  DEFAULT_LIBRARY,
  DEFAULT_TASK,
  Evaluator,
  load_model,
)
from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import FOOT_GEOMS, TORSO_BODY
from pal_mjlab.tasks.supermimic.kinematics import quat_rotate
from pal_mjlab.tasks.supermimic.library import MotionLibrary
from pal_mjlab.tasks.supermimic.mdp import MotionLibraryCommand
from pal_mjlab.tasks.supermimic.rl.replay import RolloutData, holdout_clip_mask
from pal_mjlab.tasks.supermimic.state import state_to_qpos_qvel


def imagined_return(model, ev: Evaluator, w: dict, actions: torch.Tensor, gamma: float):
  s, p = w["states"][:, 0], w["poses"][:, 0]
  alive = torch.ones(s.shape[0], device=s.device)
  total = torch.zeros_like(alive)
  for j in range(actions.shape[1]):
    s, p = model.step(s, p, actions[:, j], w["context"])
    r = ev.reward(ev.bodies(s, p), ev.reference(w["frame"][:, j]))
    total = total + gamma**j * alive * r
    alive = alive * (
      1.0 - torch.sigmoid(model.termination_logit(s, w["ref_root"][:, j]))
    )
  return total


class SimReplay:
  """Replays action sequences open loop from given start states in the training env."""

  def __init__(self, task: str, library: str, num_envs: int, device: str) -> None:
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = num_envs
    cfg.commands["motion"].library_path = library
    self.env = ManagerBasedRlEnv(cfg=cfg, device=device)
    self.env.reset()
    self.cmd: MotionLibraryCommand = self.env.command_manager.get_term("motion")
    self.robot = self.env.scene["robot"]
    idx = self.robot.indexing
    self.foot_geoms = idx.geom_ids[self.robot.find_geoms(FOOT_GEOMS)[0]].long()
    self.torso = idx.body_ids[self.robot.find_bodies((TORSO_BODY,))[0]].long()
    self.bias_joints = torch.tensor(self.cmd._goal_joint_ids.tolist(), device=device)
    rm = self.env.reward_manager
    self.tracking = [
      i for i, n in enumerate(rm.active_terms) if n.startswith("motion_")
    ]

  def run(
    self, w: dict, actions: torch.Tensor, gamma: float
  ) -> tuple[torch.Tensor, torch.Tensor]:
    env, robot, n = self.env, self.robot, actions.shape[0]
    assert n == env.num_envs
    ids = torch.arange(n, device=actions.device)
    qpos, qvel = state_to_qpos_qvel(w["states"][:, 0], w["poses"][:, 0])
    root = torch.cat(
      [
        qpos[:, :3] + env.scene.env_origins,
        qpos[:, 3:7],
        qvel[:, :3],
        quat_rotate(qpos[:, 3:7], qvel[:, 3:6]),
      ],
      dim=-1,
    )
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(qpos[:, 7:], qvel[:, 6:])
    robot.reset(env_ids=ids)
    xi = w["context"]
    model = env.sim.model
    model.geom_friction[:, self.foot_geoms, 0] = xi[:, :1]
    robot.data.encoder_bias[:, self.bias_joints] = xi[:, 1 : 1 + len(self.bias_joints)]
    default_ipos = env.sim.get_default_field("body_ipos")[self.torso]
    model.body_ipos[:, self.torso] = default_ipos + xi[:, None, -3:]
    env.sim.recompute_constants(RecomputeLevel.set_const)
    self.cmd.clip[:] = w["clip"]
    self.cmd.frame[:] = w["frame"][:, 0]
    self.cmd._version += 1
    env.sim.forward()

    total = torch.zeros(n, device=actions.device)
    alive = torch.ones(n, dtype=torch.bool, device=actions.device)
    for j in range(actions.shape[1]):
      _, _, terminated, truncated, _ = env.step(actions[:, j])
      r = env.reward_manager._step_reward[:, self.tracking].sum(-1)
      total += torch.where(alive, gamma**j * r, 0.0)
      alive &= ~(terminated | truncated)
    return total, ~alive


def main(
  model: Path,
  data: list[Path],
  out_json: Path,
  library: str = DEFAULT_LIBRARY,
  task: str = DEFAULT_TASK,
  horizons: tuple[int, ...] = (8, 16, 32),
  starts: int = 256,
  step_sizes: tuple[float, ...] = (0.05, 0.1),
  random_dirs: int = 2,
  gamma: float = 0.99,
  holdout_fraction: float = 0.1,
  seed: int = 0,
  device: str = "cuda",
) -> None:
  configure_torch_backends()
  gen = torch.Generator(device=device).manual_seed(seed)
  lib = MotionLibrary(library, device)
  rollouts = RolloutData(data, lib, device)
  ev = Evaluator(lib, task, device)
  wm = load_model(model, device)
  for prm in wm.parameters():
    prm.requires_grad_(False)
  held_out = torch.from_numpy(holdout_clip_mask(lib.clip_names, holdout_fraction)).to(
    device
  )
  variants = ["original"] + [f"grad@{d}" for d in step_sizes]
  variants += [f"rand{i}@{d}" for d in step_sizes for i in range(random_dirs)]
  sim = SimReplay(task, str(Path(library).expanduser()), starts * len(variants), device)

  results = {}
  for h in horizons:
    pool = rollouts.valid_starts(h, clips=held_out)
    idx = pool[torch.randperm(len(pool), generator=gen, device=device)[:starts]]
    w = rollouts.windows(idx, h)
    a = w["actions"].clone().requires_grad_()
    j0 = imagined_return(wm, ev, w, a, gamma)
    (grad,) = torch.autograd.grad(j0.sum(), a)
    profile = grad.norm(dim=-1).mean(0)  # ||∂J/∂a_t|| averaged over starts
    unit = grad / grad.flatten(1).norm(dim=-1).clamp_min(1e-12)[:, None, None]
    seqs, wm_gain = [a.detach()], {}
    for d in step_sizes:
      size = d * (h * a.shape[-1]) ** 0.5  # RMS change of d action units per entry
      seqs.append(a.detach() + size * unit)
      with torch.no_grad():
        wm_gain[d] = imagined_return(wm, ev, w, seqs[-1], gamma) - j0.detach()
    for d in step_sizes:
      size = d * (h * a.shape[-1]) ** 0.5
      for _ in range(random_dirs):
        r = torch.randn(a.shape, device=device, generator=gen)
        seqs.append(a.detach() + size * r / r.flatten(1).norm(dim=-1)[:, None, None])
    stacked = {
      k: torch.cat([v] * len(variants), 0) for k, v in w.items() if k != "actions"
    }
    returns, failed = sim.run(stacked, torch.cat(seqs, 0), gamma)
    returns = returns.reshape(len(variants), starts)
    failed = failed.reshape(len(variants), starts)
    logged = (gamma ** torch.arange(h, device=device) * w["reward"]).sum(-1)
    gain = returns[1:] - returns[:1]
    res = {
      "replay_vs_logged_abs": float((returns[0] - logged).abs().mean()),
      "return_mean": float(returns[0].mean()),
      "grad_norm_profile": profile.tolist(),
      "failed_original": int(failed[0].sum()),
    }
    for i, name in enumerate(variants[1:]):
      res[name] = {
        "improved": float((gain[i] > 0).float().mean()),
        "mean_gain": float(gain[i].mean()),
        "failed": int(failed[i + 1].sum()),
      }
    for i, d in enumerate(step_sizes):
      c = np.corrcoef(wm_gain[d].cpu().numpy(), gain[i].cpu().numpy())[0, 1]
      res[f"grad@{d}"]["wm_vs_sim_gain_corr"] = float(c)
      res[f"grad@{d}"]["wm_predicted_gain"] = float(wm_gain[d].mean())
    results[h] = res
    print(
      f"\nH={h}: replay vs logged |ΔR| {res['replay_vs_logged_abs']:.4f} (return {res['return_mean']:.3f})"
    )
    for name in variants[1:]:
      r = res[name]
      extra = (
        f", wm-predicted gain {r['wm_predicted_gain']:+.4f}, corr {r['wm_vs_sim_gain_corr']:.2f}"
        if name.startswith("grad")
        else ""
      )
      print(
        f"  {name:10s} improved {r['improved']:.0%}, mean gain {r['mean_gain']:+.4f}, failed {r['failed']}{extra}"
      )
    print(
      "  grad-norm profile (t=0..H-1):", np.round(profile.cpu().numpy(), 4).tolist()
    )
  out_json.parent.mkdir(parents=True, exist_ok=True)
  out_json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
  tyro.cli(main)
