"""Find imagined rollouts whose reward explodes, and which penalty does it.

Loads a SuperMimic training checkpoint (actor, critic, world model, sampler) and fills a
replay buffer with its own policy plus collection noise in the training env, as training
does. Then it imagines rollouts from replay starts exactly as the policy update does, with
no update. For every step whose reward is below ``EXTREME_REWARD`` it records the term
that dominates, the step, the survival weight ĉ_t before it, the failure probabilities of
the learned head and of the env's rule, and the largest action component. It also splits
the critic's Eq. 3 error into the steps before and after the predicted fall (ĉ_t < 0.5).

Usage:
  python -m pal_mjlab.scripts.supermimic.reward_extremes --task <task> \\
    --checkpoints run/model_46000.pt run/model_47000.pt --out-dir results/
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import mjlab.tasks  # noqa: F401  (populates the task registry)
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from pal_mjlab.tasks.supermimic.rl.supermimic import (
  EXTREME_REWARD,
  lambda_returns,
)


@torch.no_grad()
def analyse(runner, batches: int, collect_steps: int, seed: int) -> dict:
  alg, heads = runner.alg, runner.heads
  pc = alg.cfg.policy
  alg.set_matmul_precision()
  runner.generator = torch.Generator(device=runner.device).manual_seed(seed)
  obs = runner.start_collection()
  for _ in range(collect_steps):
    obs, _ = runner._collect(obs)

  rows, rollouts, extreme_rollouts = [], 0, 0
  critic = {
    "steps": 0.0,
    "post_fall_steps": 0.0,
    "sq_error": 0.0,
    "post_fall_sq_error": 0.0,
  }
  for _ in range(batches):
    start = runner.replay.sample_starts(
      pc.num_rollouts, heads.history, runner.generator
    )
    with alg._autocast():
      roll = alg.imagine(start, runner.generator)
    inside = roll.alive > 0
    extreme = inside & (roll.reward < EXTREME_REWARD)
    rollouts += len(roll.reward)
    extreme_rollouts += int(extreme.any(1).sum())
    c = torch.cat([torch.ones_like(roll.cont[:, :1]), roll.cont.cumprod(1)[:, :-1]], 1)
    per_step = {
      "reward": roll.reward,
      "action_rate": roll.action_rate,
      "joint_limit": roll.joint_limit,
      "tracking": roll.reward - roll.action_rate - roll.joint_limit,
      "survival_before": c,
      "head_prob": roll.head_prob,
      "head_prob_max_before": roll.head_prob.cummax(1).values,
      "rule_prob": roll.rule_prob,
      "rule_prob_max_before": roll.rule_prob.cummax(1).values,
      "action_max": roll.action_max,
    }
    typical = float(roll.action_max[inside].median())
    # The critic's Eq. 3 errors before its update, split at the predicted fall (ĉ < 0.5).
    target = lambda_returns(
      roll.reward, roll.cont, roll.value, pc.gamma, alg.cfg.critic.lam
    )
    err = (roll.value[:, :-1] - target).square()[inside]
    post = (c < 0.5)[inside]
    critic["steps"] += float(inside.sum())
    critic["post_fall_steps"] += float(post.sum())
    critic["sq_error"] += float(err.sum())
    critic["post_fall_sq_error"] += float(err[post].sum())
    first = extreme & (extreme.cumsum(1) == 1)  # first extreme step of each rollout
    for name, mask in (("all", extreme), ("first", first)):
      bb, tt = mask.nonzero(as_tuple=True)
      cols = {k: v[bb, tt].tolist() for k, v in per_step.items()}
      cols["step"] = tt.tolist()
      for i in range(len(cols["step"])):
        rows.append(
          {"which": name, "action_max_typical": typical}
          | {k: v[i] for k, v in cols.items()}
        )
  out = summarize(rows, rollouts, extreme_rollouts)
  out["critic"] = {
    "post_fall_step_share": critic["post_fall_steps"] / critic["steps"],
    "post_fall_error_share": critic["post_fall_sq_error"] / critic["sq_error"],
    "mse": critic["sq_error"] / critic["steps"],
    "mse_before_fall": (critic["sq_error"] - critic["post_fall_sq_error"])
    / (critic["steps"] - critic["post_fall_steps"]),
  }
  return out


def summarize(rows: list[dict], rollouts: int, extreme_rollouts: int) -> dict:
  out = {"rollouts": rollouts, "extreme_rollouts": extreme_rollouts}
  for which in ("all", "first"):
    r = [x for x in rows if x["which"] == which]
    if not r:
      continue

    def med(key, rr=r):
      return float(torch.tensor([x[key] for x in rr]).median())

    dominant = [min(("action_rate", "joint_limit"), key=lambda k, x=x: x[k]) for x in r]
    out[which] = {
      "steps": len(r),
      "dominant_action_rate": dominant.count("action_rate"),
      "dominant_joint_limit": dominant.count("joint_limit"),
      "reward_min": min(x["reward"] for x in r),
      "median": {
        k: med(k)
        for k in (
          "step",
          "reward",
          "action_rate",
          "joint_limit",
          "survival_before",
          "head_prob_max_before",
          "rule_prob_max_before",
          "action_max",
          "action_max_typical",
        )
      },  # fmt: skip
      "dead_before": {  # predicted dead before the extreme step, by each criterion
        "survival_below_0.5": sum(x["survival_before"] < 0.5 for x in r),
        "rule_prob_above_0.5": sum(x["rule_prob_max_before"] > 0.5 for x in r),
      },
    }
  return out


def main(
  task: str,
  checkpoints: list[Path],
  out_dir: Path,
  num_envs: int = 4096,
  collect_steps: int = 300,
  batches: int = 20,
  precision: str = "tf32",
  seed: int = 0,
  device: str = "cuda",
) -> None:
  configure_torch_backends()
  env_cfg = load_env_cfg(task)
  env_cfg.scene.num_envs = num_envs
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device))
  agent = asdict(load_rl_cfg(task))
  agent["replay_steps"] = collect_steps
  agent["algorithm"]["precision"] = precision
  out_dir.mkdir(parents=True, exist_ok=True)
  for ckpt in checkpoints:
    runner = load_runner_cls(task)(env, agent, None, device)
    runner.load(str(ckpt), map_location=device)
    result = analyse(runner, batches, collect_steps, seed)
    out = out_dir / f"{ckpt.parent.name}_{ckpt.stem}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"{ckpt}: {json.dumps(result)}", flush=True)


if __name__ == "__main__":
  tyro.cli(main)
