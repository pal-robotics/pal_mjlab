"""M3: check the imagination code with the PPO-Sim actor on the frozen M2 world model.

The paper never trains a policy on a frozen world model; these are checks of the code
the online loop (M4) uses, run where the world model knows the data: the PPO-Sim actor
(same network, D17) and replay states of that same policy.

1. Closed loop, no training. From held-out starts, the actor acts H steps in imagination
   and in the simulator: same state, ξ, reference and observation history, no sensor or
   action noise. Compared: H-step returns (learned termination head and failure rule),
   failures, and root-relative body positions along the way.
2. A short update run. The critic is fit with the actor held fixed, then SuperMimic's
   policy step (Eqs. 2, 3) runs on starts from the training clips. The policy is
   evaluated in the simulator along the way: check 1 and the paper's benchmark (every
   clip from its first frame, ``eval.py``).

  python -m pal_mjlab.scripts.supermimic.m3_check --out-json logs/supermimic_m3/m3.json
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import mjlab.tasks  # noqa: F401  (populates the task registry)
import numpy as np
import torch
import tyro
from mjlab.utils.torch import configure_torch_backends
from pal_mjlab.scripts.supermimic.eval import Benchmark
from pal_mjlab.scripts.supermimic.grad_check import SimReplay
from pal_mjlab.tasks.supermimic.library import MotionLibrary
from pal_mjlab.tasks.supermimic.rl.heads import AnalyticHeads
from pal_mjlab.tasks.supermimic.rl.replay import (
  ReplayBuffer,
  RolloutData,
  holdout_clip_mask,
)
from pal_mjlab.tasks.supermimic.rl.supermimic import (
  SuperMimic,
  SuperMimicCfg,
  horizon_return,
  imagine,
)
from pal_mjlab.tasks.supermimic.rl.world_model import load_world_model
from pal_mjlab.tasks.supermimic.state import heading_state
from tensordict import TensorDict

TASK = "Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Grippers-ArmWaist1Hz"
RUN = "logs/rsl_rl/supermimic_ppo_sim_1hz/2026-10-22_21-11-56"
LIBRARY = "~/Projects/supermimic/kangaroo_bones_seed/dataset/libraries/dev"


@torch.no_grad()
def closed_loop(
  sim: SimReplay, heads: AnalyticHeads, alg: SuperMimic, start: dict, horizon: int
) -> dict:
  """Check 1: the actor in imagination and in the simulator from the same starts."""
  env, n = sim.env, start["state"].shape[0]
  pc = replace(alg.cfg.policy, horizon=horizon, sensor_noise=False)
  last_action = start["history_last_action"][:, -1]
  sim.set_start(
    start["state"], start["pose"], start["context"], start["clip"], start["frame"]
  )
  # Same observation history on both sides: the start frame, backfilled.
  env.action_manager._action[:] = last_action
  env.observation_manager.reset()
  obs = TensorDict(env.observation_manager.compute(update_history=True), [n])
  imagined_start = {
    **start,
    "history": heads.unflatten_history(obs["proprio"]),
    "last_action": last_action,
  }
  with alg._autocast():
    roll = imagine(heads, alg.world_model, alg.actor, alg.critic, imagined_start, pc)

  cmd = sim.cmd
  rewards, failures, alive_steps, states = [], [], [], []
  alive = torch.ones(n, dtype=torch.bool, device=env.device)
  failed = torch.zeros_like(alive)
  for _ in range(horizon):
    alive_steps.append(alive.clone())
    obs_dict, reward, terminated, truncated, _ = env.step(alg.actor(obs))
    obs = TensorDict(obs_dict, [n])
    failure = env.termination_manager.get_term("anchor_failure")
    rewards.append(torch.where(alive, reward, 0.0))
    failed |= alive & failure
    failures.append(failed.clone())
    qpos = cmd.robot_qpos.clone()
    qpos[:, :3] -= env.scene.env_origins
    states.append(heading_state(qpos, cmd.robot_qvel))
    alive &= ~(terminated | truncated)
  sim_reward = torch.stack(rewards, 1)
  discount = pc.gamma ** torch.arange(horizon, device=env.device)
  sim_return = (discount * sim_reward).sum(1)

  out = {"sim_return": float(sim_return.mean())}
  out["sim_failure"] = float(failed.float().mean())
  for name, prob in (("head", roll.head_prob), ("rule", roll.rule_prob)):
    imagined = horizon_return(roll, prob, pc.gamma)
    out[f"imagined_return_{name}"] = float(imagined.mean())
    out[f"return_gap_{name}"] = float(
      (imagined - sim_return).mean() / sim_return.mean()
    )
    out[f"return_corr_{name}"] = float(
      np.corrcoef(imagined.cpu().numpy(), sim_return.cpu().numpy())[0, 1]
    )
    out[f"imagined_failure_{name}"] = float(1 - (1 - prob * roll.alive).prod(1).mean())

  # Root-relative body positions of imagined vs simulated states, while the sim lives.
  zero = torch.zeros(n, 3, device=env.device)
  errors = {}
  for t in (8, 16, horizon):
    sim_body = heads.kinematics(states[t - 1], zero).bodies.pos
    imag_body = heads.kinematics(roll.critic_input[:, t, : states[0].shape[1]], zero)
    err = (imag_body.bodies.pos - sim_body).norm(dim=-1).mean(-1) * 100
    live = alive_steps[t - 1]
    errors[t] = float(err[live].mean())
  out["body_error_cm"] = errors
  cont = (1 - roll.head_prob) * (1 - roll.clip_end)
  survival = torch.cat([torch.ones_like(cont[:, :1]), cont.cumprod(1)[:, :-1]], 1)
  out["reward_curve"] = {
    "sim": sim_reward.mean(0).tolist(),
    "imagined": (survival * roll.reward).mean(0).tolist(),
  }
  return out


def benchmark(bench: Benchmark, alg: SuperMimic) -> dict:
  """The paper's evaluation (every dev clip from its first frame), ``eval.py``."""

  def policy(obs):
    with torch.no_grad():
      return alg.actor(obs)

  per_clip = bench.run(policy)
  summary = {k: float(np.mean(v)) for k, v in per_clip.items()}
  summary["failed"] = int(per_clip["failed"].sum())
  return summary


def main(
  out_json: Path,
  task: str = TASK,
  checkpoint: Path = Path(RUN) / "model_8800.pt",
  world_model: Path = Path("logs/supermimic_m2/wm/main/wm_best.pt"),
  data: Path = Path("logs/supermimic_m2/data/model_8800"),
  library: str = LIBRARY,
  starts: int = 1024,
  horizon: int = 32,
  critic_iterations: int = 200,
  policy_iterations: int = 300,
  eval_every: int = 50,
  run_benchmark: bool = True,
  use_critic: bool = True,
  actor_lr: float = 2.0e-4,
  termination: str = "learned",
  precision: str = "bf16",
  seed: int = 0,
  device: str = "cuda",
) -> None:
  configure_torch_backends()
  torch.manual_seed(seed)
  gen = torch.Generator(device=device).manual_seed(seed)
  lib_path = str(Path(library).expanduser())
  sim = SimReplay(task, lib_path, starts, device, obs_noise=False)
  heads = AnalyticHeads(sim.env)
  obs = TensorDict(sim.env.observation_manager.compute(), [starts])
  cfg = SuperMimicCfg(precision=precision)  # the paper's settings (Table A.5)
  cfg.policy.use_critic = use_critic  # False: the paper's no-critic ablation
  cfg.policy.learning_rate = actor_lr  # paper 2e-4; PPO-Sim ended at 2.25e-5
  cfg.policy.termination = termination  # learned head (paper) or the env's rule (D31)
  context_dim = sim.env.observation_manager.get_term_cfg("critic", "sim_context")
  ctx = context_dim.func(sim.env, **context_dim.params).shape[-1]
  alg = SuperMimic(
    obs, heads, sim.env.action_manager.total_action_dim, ctx, cfg, device
  )
  alg.world_model = load_world_model(world_model, device)
  alg.load_ppo_actor(str(checkpoint))
  alg.set_iteration(0)

  library_ = MotionLibrary(lib_path, device)
  held_out = torch.from_numpy(holdout_clip_mask(library_.clip_names)).to(device)
  replay = ReplayBuffer.from_rollouts(RolloutData([data], library_, device))
  torch.cuda.empty_cache()
  check_starts = replay.sample_starts(starts, heads.history, gen, clips=held_out)
  bench = Benchmark(task, lib_path, device) if run_benchmark else None
  config = dict(
    task=task, checkpoint=str(checkpoint), world_model=str(world_model),
    data=str(data), starts=starts, horizon=horizon, precision=precision,
    critic_iterations=critic_iterations, policy_iterations=policy_iterations,
    use_critic=use_critic, actor_lr=actor_lr, termination=termination,
  )  # fmt: skip
  result: dict = {"config": config}

  # Check 1 with the PPO-Sim actor as is.
  t0 = time.perf_counter()
  result["closed_loop"] = {0: closed_loop(sim, heads, alg, check_starts, horizon)}
  result["benchmark"] = {0: benchmark(bench, alg)} if bench else {}
  print("check 1:", json.dumps(
    {k: v for k, v in result["closed_loop"][0].items() if k != "reward_curve"}
  ), flush=True)  # fmt: skip
  print("benchmark (PPO-Sim actor):", result["benchmark"].get(0), flush=True)

  # Check 2: critic first, with the actor held fixed; then policy and critic.
  def critic_inputs_normalizer():
    sample = replay.sample_starts(cfg.policy.num_rollouts, heads.history, gen)
    with torch.no_grad(), alg._autocast():
      roll = imagine(heads, alg.world_model, alg.actor, alg.critic, sample, cfg.policy)
    alg.critic.obs_normalizer.update(roll.critic_input.flatten(0, 1))

  for _ in range(5):
    critic_inputs_normalizer()
  train = ~held_out
  log = []
  for it in range(critic_iterations):
    stats = alg.update_policy(replay, gen, update_actor=False, clips=train)
    log.append({"phase": "critic", "iteration": it, **stats})
  if log:
    print(f"critic fit: loss {log[-1]['critic_loss']:.4g}, value {log[-1]['value']:.4g}, "
          f"lambda return {log[-1]['lambda_return']:.4g}", flush=True)  # fmt: skip
  for it in range(1, policy_iterations + 1):
    t1 = time.perf_counter()
    stats = alg.update_policy(replay, gen, clips=train)
    torch.cuda.synchronize()
    log.append(
      {"phase": "policy", "iteration": it, "seconds": time.perf_counter() - t1, **stats}
    )
    if it % eval_every == 0:
      result["closed_loop"][it] = closed_loop(sim, heads, alg, check_starts, horizon)
      c = result["closed_loop"][it]
      print(
        f"it {it}: objective {stats['objective']:.4g} | closed loop sim "
        f"{c['sim_return']:.4g} imagined {c['imagined_return_head']:.4g} "
        f"(gap {100 * c['return_gap_head']:+.1f}%) body err {c['body_error_cm']}",
        flush=True,
      )
  if bench:
    result["benchmark"][policy_iterations] = benchmark(bench, alg)
    print("benchmark (after the update run):", result["benchmark"][policy_iterations])
  result["log"] = log
  result["minutes"] = (time.perf_counter() - t0) / 60
  result["max_memory_gb"] = torch.cuda.max_memory_allocated() / 1e9
  policy_time = [r["seconds"] for r in log if r["phase"] == "policy"]
  print(f"{result['minutes']:.1f} min, peak memory {result['max_memory_gb']:.1f} GB, "
        f"policy step {np.median(policy_time):.2f} s", flush=True)  # fmt: skip
  out_json.parent.mkdir(parents=True, exist_ok=True)
  out_json.write_text(json.dumps(result, indent=1))
  torch.save(
    {"actor": alg.actor.state_dict(), "critic": alg.critic.state_dict()},
    out_json.with_suffix(".pt"),
  )


if __name__ == "__main__":
  tyro.cli(main)
