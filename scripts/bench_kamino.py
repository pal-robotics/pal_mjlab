"""mujoco_warp against Kamino on kangaroo_full: simulation speed and constraint violation.

The one benchmark for the Kamino bridge. Runs each case in its own process and keeps
its JSON, so an interrupted run resumes, then writes the table and the plot:

  python scripts/bench_kamino.py --out logs/bench_kamino --mjw-policy A.pt --kamino-policy B.pt
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

TASKS = {
  "mjwarp": "Mjlab-Velocity-Flat-Pal-Kangaroo-Full",
  "kamino": "Mjlab-Velocity-Flat-Pal-Kangaroo-Full-Kamino",
}
SPEED_WORLDS = (256, 1024, 4096)
KAMINO_BUDGETS = (30, 120)  # Besides the default of 60.
DENSE_LLT = dict(linear_solver="LLTB", max_iterations=200, rho_0=0.01)
RL_WARMSTART = dict(contact_warmstart="geom_pair_net_force")
STAND_S, WALK_S, ROLLOUT_WORLDS = 5.0, 20.0, 256
SPEED_STEPS, WARMUP_STEPS, REPEATS = 20, 10, 3
GROUPS = {"hip": "hip", "ankle": "ankle", "knee": ("knee", "femur_rod")}


def _make_env(case: dict):
  """The flat task, with no noise, randomization or pushes, and a fixed command."""
  import mjlab.tasks  # noqa: F401
  import pal_mjlab.tasks  # noqa: F401
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.envs.mdp.terminations import nan_detection
  from mjlab.managers.termination_manager import TerminationTermCfg
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg(TASKS[case["sim"]])
  cfg.scene.num_envs = case["worlds"]
  cfg.seed = 1
  cfg.events = {k: v for k, v in cfg.events.items() if v.mode == "reset"}
  cfg.terminations["nan"] = TerminationTermCfg(func=nan_detection)
  cfg.observations["actor"].enable_corruption = False
  cfg.curriculum = {}
  twist = cfg.commands["twist"]
  twist.ranges.lin_vel_x = (case["speed"], case["speed"])
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.heading_command = False
  twist.ranges.heading = None
  twist.rel_standing_envs = 0.0
  twist.rel_heading_envs = 0.0
  twist.resampling_time_range = (1e6, 1e6)
  for field, value in case.get("solver", {}).items():
    setattr(cfg.sim, field, value)
  return ManagerBasedRlEnv(cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")


def _make_policy(env, case: dict):
  """Actions from ``case["policy"]``, or zeros (hold the home pose) without one."""
  import torch

  if case.get("policy") is None:
    actions = torch.zeros(
      env.num_envs, env.action_manager.total_action_dim, device=env.device
    )
    return lambda: actions

  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

  agent = load_rl_cfg(TASKS[case["sim"]])
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
  runner_cls = load_runner_cls(TASKS[case["sim"]]) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent), device=env.device)
  runner.load(case["policy"], map_location=env.device)
  policy = runner.get_inference_policy(device=env.device)

  def act():
    with torch.inference_mode():
      return policy(wrapped.get_observations())

  return act


def _kamino_loop_gap_mm(sim):
  """Loop closure measured on Kamino's own bodies, not on the qpos projected from them."""
  import warp as wp

  joints, nbody = sim.kamino.joints, len(sim.layout.body_mj)
  loops = sim.layout.loops
  njoint = joints.num_joints // sim.num_envs
  to = lambda a: wp.to_torch(a)  # noqa: E731
  bodies = to(sim.state_n.q_i).view(sim.num_envs, nbody, 7)
  world = lambda field: to(field).view(sim.num_envs, njoint, -1)[:, loops]  # noqa: E731
  parent = (to(joints.bid_B).view(sim.num_envs, njoint)[:, loops]) % nbody
  child = (to(joints.bid_F).view(sim.num_envs, njoint)[:, loops]) % nbody

  def anchor(index, offset):
    gather = bodies.gather(1, index[..., None].expand(-1, -1, 7))
    q = gather[..., 3:7]
    u = q[..., :3]
    t = 2.0 * u.cross(offset, dim=-1)
    return gather[..., :3] + offset + q[..., 3:] * t + u.cross(t, dim=-1)

  gap = anchor(parent, world(joints.B_r_Bj)) - anchor(child, world(joints.F_r_Fj))
  return gap.norm(dim=-1) * 1e3


def _tree_residual(sim):
  """Largest kinematic residual per tree-joint row, in mm or mrad (zero on mjwarp)."""
  import warp as wp

  r_j = wp.to_torch(sim._solver.data.joints.r_j).view(sim.num_envs, -1).abs()
  return r_j[:, : -3 * len(sim.layout.loops)] * 1e3


def _rollout(case: dict) -> dict:
  """Run ``case["seconds"]`` of simulation, measuring every env step."""
  import numpy as np
  import torch
  import warp as wp

  env = _make_env(case)
  env.reset()
  act = _make_policy(env, case)
  sim, m = env.sim, env.sim.mj_model
  site1, site2 = m.eq_obj1id, m.eq_obj2id
  names = [m.site(int(s)).name for s in site1]
  masks = {
    group: [
      any(k in n for k in ((key,) if isinstance(key, str) else key)) for n in names
    ]
    for group, key in GROUPS.items()
  }
  body1, body2 = m.site_bodyid[site1], m.site_bodyid[site2]
  rootid = m.body_rootid[body1]
  gaps, rel_vels, tree, own = [], [], [], []
  iters, conv = [], []
  ends = torch.zeros(env.num_envs, device=env.device)
  nans = torch.zeros(env.num_envs, device=env.device)
  steps = int(case["seconds"] / env.step_dt)
  with torch.no_grad():
    for _ in range(steps):
      _, _, terminated, _, _ = env.step(act())
      ends += terminated.float()
      nans += env.termination_manager.get_term("nan").float()
      d = sim.wp_data
      x = wp.to_torch(d.site_xpos)
      gaps.append(((x[:, site1] - x[:, site2]).norm(dim=-1) * 1e3).cpu())
      cvel, com = wp.to_torch(d.cvel), wp.to_torch(d.subtree_com)[:, rootid]
      v1 = cvel[:, body1, 3:] + cvel[:, body1, :3].cross(x[:, site1] - com, dim=-1)
      v2 = cvel[:, body2, 3:] + cvel[:, body2, :3].cross(x[:, site2] - com, dim=-1)
      rel_vels.append(((v1 - v2).norm(dim=-1) * 1e3).cpu())
      if case["sim"] == "kamino":
        own.append(_kamino_loop_gap_mm(sim).cpu())
        tree.append(_tree_residual(sim).cpu())
        status = sim._solver.solver_status.numpy()
        iters.append(status["iterations"])
        conv.append(status["converged"])

  def stats(samples: list, mask=None) -> dict:
    values = torch.stack(samples)
    values = values[..., mask] if mask is not None else values
    values = values[~values.isnan()].reshape(-1)
    return {
      "mean": values.mean().item(),
      "p95": torch.quantile(values[:: max(1, len(values) // 200_000)], 0.95).item(),
      "max": values.max().item(),
    }

  result = {
    "loop_mm": stats(gaps),
    "loop_mm_by_group": {g: stats(gaps, torch.tensor(k)) for g, k in masks.items()},
    "loop_vel_mm_s": stats(rel_vels),
    "terminated_per_world": ends.mean().item(),
    "nan_per_world": nans.mean().item(),
  }
  if iters:
    result["kamino_loop_mm"] = stats(own)
    result["tree_residual"] = stats(tree)
    result["iterations_mean"] = float(np.mean(iters))
    result["converged_frac"] = float(np.mean(conv))
  return result


def _speed(case: dict) -> dict:
  """Env-steps/s through the full env, then physics-only substeps/s, which continues
  the same trajectory but handles no resets."""
  import torch
  import warp as wp

  env = _make_env(case)
  env.reset()
  act = _make_policy(env, case)
  rates, physics = [], []
  with torch.no_grad():
    for _ in range(WARMUP_STEPS):
      env.step(act())
    for _ in range(REPEATS):
      wp.synchronize()
      start = time.perf_counter()
      for _ in range(SPEED_STEPS):
        env.step(act())
      wp.synchronize()
      rates.append(env.num_envs * SPEED_STEPS / (time.perf_counter() - start))
      start = time.perf_counter()
      for _ in range(SPEED_STEPS):
        env.sim.step()
      wp.synchronize()
      physics.append(env.num_envs * SPEED_STEPS / (time.perf_counter() - start))
  return {
    "env_steps_per_s": statistics.median(rates),
    "substeps_per_s": statistics.median(physics),
    "all_env_steps_per_s": rates,
  }


def _cases(args) -> dict:
  policies = {"mjw": args.mjw_policy, "kam": args.kamino_policy}
  cases = {}
  for worlds in SPEED_WORLDS:
    cases[f"speed_mjwarp_{worlds}"] = dict(
      sim="mjwarp", worlds=worlds, speed=0.0, kind="speed"
    )
    cases[f"speed_kamino_{worlds}"] = dict(
      sim="kamino", worlds=worlds, speed=0.0, kind="speed"
    )
  rollout = dict(
    sim="kamino", worlds=ROLLOUT_WORLDS, speed=0.0, seconds=STAND_S, kind="rollout"
  )
  speed = dict(sim="kamino", worlds=1024, speed=0.0, kind="speed")
  for budget in KAMINO_BUDGETS:
    solver = {"max_iterations": budget}
    cases[f"speed_kamino_1024_b{budget}"] = dict(speed, solver=solver)
    cases[f"stand_kamino_b{budget}"] = dict(rollout, solver=solver)
  cases["speed_kamino_1024_dense"] = dict(speed, solver=DENSE_LLT)
  cases["stand_kamino_dense"] = dict(rollout, solver=DENSE_LLT)
  cases["speed_kamino_1024_rl_warmstart"] = dict(speed, solver=RL_WARMSTART)
  cases["stand_kamino_rl_warmstart"] = dict(rollout, solver=RL_WARMSTART)
  if args.mjw_policy:
    for sim in TASKS:
      cases[f"speed_{sim}_1024_walk"] = dict(
        sim=sim, worlds=1024, speed=1.0, kind="speed", policy=args.mjw_policy
      )
  for sim in TASKS:
    cases[f"stand_{sim}"] = dict(
      sim=sim, worlds=ROLLOUT_WORLDS, speed=0.0, seconds=STAND_S, kind="rollout"
    )
    for tag, policy in policies.items():
      if policy:
        cases[f"walk_{sim}_policy_{tag}"] = dict(
          sim=sim,
          worlds=ROLLOUT_WORLDS,
          speed=1.0,
          seconds=WALK_S,
          kind="rollout",
          policy=policy,
        )
  return cases


def _training(run_dir: Path) -> dict:
  """Reward against wall-clock from a run's tensorboard log."""
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  events = EventAccumulator(str(run_dir))
  events.Reload()
  reward = events.Scalars("Train/mean_reward")
  length = events.Scalars("Train/mean_episode_length")
  return {
    "iterations": reward[-1].step + 1,
    "hours": (reward[-1].wall_time - reward[0].wall_time) / 3600,
    "mean_reward": reward[-1].value,
    "mean_episode_length": length[-1].value,
  }


def _plot(out: Path, results: dict) -> None:
  """Speed against violation: each setting is one dot, its bar runs from mean to max."""
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
  series = {
    "Current (mujoco_warp)": (
      "#2a78d6",
      {"current": ("speed_mjwarp_1024", "stand_mjwarp")},
    ),
    "Kamino": (
      "#eb6834",
      {
        "CRF, 30 iterations": ("speed_kamino_1024_b30", "stand_kamino_b30"),
        "CRF, 60 (default)": ("speed_kamino_1024", "stand_kamino"),
        "CRF, 120": ("speed_kamino_1024_b120", "stand_kamino_b120"),
        "dense LLT": ("speed_kamino_1024_dense", "stand_kamino_dense"),
      },
    ),
  }
  fig, ax = plt.subplots(figsize=(8, 5), facecolor=surface)
  ax.set_facecolor(surface)
  for name, (color, points) in series.items():
    for i, (label, (speed, stand)) in enumerate(points.items()):
      if "loop_mm" not in results.get(stand, {}) or speed not in results:
        continue
      # Kamino's own bodies, since projecting to qpos hides part of its error.
      stats = results[stand].get("kamino_loop_mm", results[stand]["loop_mm"])
      x, loop = results[speed]["env_steps_per_s"], stats
      ax.plot(
        [x, x], [loop["mean"], loop["max"]], color=color, lw=2, solid_capstyle="round"
      )
      ax.scatter(
        x,
        loop["p95"],
        s=90,
        color=color,
        edgecolor=surface,
        lw=2,
        zorder=3,
        label=None if i else name,
      )
      ax.annotate(
        label,
        (x, loop["p95"]),
        xytext=(9, 0),
        textcoords="offset points",
        va="center",
        color=ink,
        fontsize=9,
      )
  ax.set(xscale="log", yscale="log")
  ax.set_xlabel("Simulation speed at 1024 worlds (env-steps/s)", color=muted)
  ax.set_ylabel("Loop closure error under zero action (mm)", color=muted)
  ax.set_title(
    "Dot is p95, bar runs from mean to max. Further right and lower is better.\n"
    "Zero action holds the home pose for 5 s: the robot settles, tips over near 3 s, resets.",
    loc="left",
    color=muted,
    fontsize=10,
  )
  ax.grid(color="#e8e7e3", lw=0.8)
  ax.set_axisbelow(True)
  ax.tick_params(colors=muted, which="both")
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color("#c3c2b7")
  ax.legend(frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.13))
  fig.tight_layout()
  fig.savefig(out / "speed_vs_violation.png", dpi=160, facecolor=surface)


def _fmt(value, digits: int = 3) -> str:
  return "-" if value is None else f"{value:.{digits}f}"


def _report(out: Path, results: dict, runs: dict, failed: list) -> None:
  speed = ["| Case | Env-steps/s | Physics substeps/s |", "|---|---|---|"]
  loops = [
    "| Case | Loop mm mean | p95 | max | Kamino bodies mm max | Loop mm/s mean | Tree mm or mrad max | PADMM iterations | Converged | Falls per world | NaN |",
    "|" + "---|" * 11,
  ]
  for name, r in results.items():
    if "env_steps_per_s" in r:
      speed.append(
        f"| {name} | {r['env_steps_per_s']:.0f} | {r['substeps_per_s']:.0f} |"
      )
    else:
      loops.append(
        f"| {name} | {_fmt(r['loop_mm']['mean'])} | {_fmt(r['loop_mm']['p95'])} | {_fmt(r['loop_mm']['max'])}"
        f" | {_fmt(r.get('kamino_loop_mm', {}).get('max'))} | {_fmt(r['loop_vel_mm_s']['mean'], 2)}"
        f" | {_fmt(r.get('tree_residual', {}).get('max'))} | {_fmt(r.get('iterations_mean'), 1)}"
        f" | {_fmt(r.get('converged_frac'), 2)} | {_fmt(r['terminated_per_world'], 2)} | {_fmt(r['nan_per_world'], 3)} |"
      )
  training = [
    "| Training run | Iterations | Hours | Mean reward | Mean episode length |",
    "|---|---|---|---|---|",
  ]
  for name, r in runs.items():
    training.append(
      f"| {name} | {r['iterations']} | {_fmt(r['hours'], 2)} | {_fmt(r['mean_reward'], 2)} | {_fmt(r['mean_episode_length'], 0)} |"
    )
  text = "\n\n".join(["\n".join(t) for t in (speed, loops, training)]) + "\n"
  if failed:
    text += f"\nFailed cases: {', '.join(failed)}\n"
  (out / "results.md").write_text(text)
  _plot(out, results)
  print(text)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--mjw-policy", default=None)
  parser.add_argument("--kamino-policy", default=None)
  parser.add_argument("--case", default=None, help=argparse.SUPPRESS)
  args = parser.parse_args()

  if args.case:
    case = json.loads(args.case)
    result = (_speed if case["kind"] == "speed" else _rollout)(case)
    args.out.write_text(json.dumps(result))  # A file, as kernels printf to stdout.
    return

  args.out.mkdir(parents=True, exist_ok=True)
  results, failed = {}, []
  for name, case in _cases(args).items():
    path = args.out / f"{name}.json"
    if not path.exists():
      print(f"[bench] {name}", flush=True)
      run = subprocess.run(
        [sys.executable, __file__, "--out", str(path), "--case", json.dumps(case)],
        capture_output=True,
        text=True,
      )
      if run.returncode != 0:
        failed.append(name)
        print(f"[bench] {name} failed:\n{run.stderr[-800:]}", flush=True)
        continue
    results[name] = json.loads(path.read_text())
  runs = {
    name: _training(Path(policy).parent)
    for name, policy in (("mjwarp", args.mjw_policy), ("kamino", args.kamino_policy))
    if policy
  }
  _report(args.out, results, runs, failed)


if __name__ == "__main__":
  main()
