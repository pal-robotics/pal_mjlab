"""Time one SuperMimic iteration's learning updates and check them against eager (D37, D41).

Builds the training runner, fills a small replay with the policy, then:
  1. computes the imagined objective and the actor gradient on fixed starts, eagerly and with
     the runner's (compiled) step, and prints their relative difference;
  2. times ``update_policy`` (imagination, backward, actor step, critic fit) and
     ``update_world_model``.

  python -m pal_mjlab.scripts.supermimic.bench_update --dataset <dataset> \\
    --precision tf32 --num-envs 4096 [--fused-adam]
"""

from __future__ import annotations

import time
from dataclasses import asdict

import mjlab.tasks  # noqa: F401  (populates the task registry)
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from pal_mjlab.tasks.supermimic.rl import supermimic as sm


def timed(fn, n: int) -> float:
  torch.cuda.synchronize()
  t = time.perf_counter()
  for _ in range(n):
    fn()
  torch.cuda.synchronize()
  return (time.perf_counter() - t) / n


def main(
  dataset: str,
  task: str = "Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers-ArmWaist1Hz",
  subset: str = "dev",
  precision: str = "tf32",
  num_envs: int = 4096,
  compile: bool = True,
  fused_adam: bool = False,
) -> None:
  configure_torch_backends()
  env_cfg = load_env_cfg(task)
  env_cfg.scene.num_envs = num_envs
  env_cfg.commands["motion"].dataset = dataset
  env_cfg.commands["motion"].subset = subset
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device="cuda"))
  agent = asdict(load_rl_cfg(task))
  agent["replay_steps"] = 200
  agent["compile"] = compile
  agent["algorithm"]["precision"] = precision
  runner = load_runner_cls(task)(env, agent, None, "cuda")
  alg, heads, pc = runner.alg, runner.heads, runner.alg.cfg.policy
  alg.set_matmul_precision()
  runner.generator = torch.Generator(device="cuda").manual_seed(0)
  obs = runner.start_collection()
  for _ in range(120):
    obs, _ = runner._collect(obs)
  if fused_adam:
    for name in ("wm_optimizer", "actor_optimizer", "critic_optimizer"):
      old = getattr(alg, name).param_groups[0]
      opt = torch.optim.Adam(
        old["params"], lr=old["lr"], betas=old["betas"], fused=True
      )
      setattr(alg, name, opt)

  # Same starts and noise: eager reference vs the runner's step.
  start = runner.replay.sample_starts(
    pc.num_rollouts, heads.history, torch.Generator(device="cuda").manual_seed(3)
  )
  actor = list(alg.actor.parameters())
  frozen = [*alg.world_model.parameters(), *alg.critic.parameters()]
  for p in frozen:
    p.requires_grad_(False)
  out = {}
  for name, step, repeats in (
    ("eager", sm.make_step(heads, pc), 1),
    ("run", alg._step, 3),
  ):
    for _ in range(repeats):  # repeats warm up the compiled step / record its graphs
      with alg._autocast():
        roll = sm.imagine(
          heads, alg.world_model, alg.actor, alg.critic, start, pc,
          torch.Generator(device="cuda").manual_seed(7), step,
        )  # fmt: skip
      grad = torch.autograd.grad(-roll.objective.mean(), actor)
    out[name] = (
      roll.objective.detach().clone(),
      torch.cat([g.flatten() for g in grad]),
    )
  for p in frozen:
    p.requires_grad_(True)
  (oe, ge), (o, g) = out["eager"], out["run"]
  print(
    f"vs eager: objective {float((o - oe).norm() / oe.norm()):.2e}, "
    f"actor gradient {float((g - ge).norm() / ge.norm()):.2e} (relative)"
  )

  for _ in range(3):
    alg.update_policy(runner.replay, runner.generator)
    alg.update_world_model(runner.replay, runner.generator)
  policy = timed(lambda: alg.update_policy(runner.replay, runner.generator), 10)
  wm = timed(lambda: alg.update_world_model(runner.replay, runner.generator), 5)
  print(
    f"policy update {policy:.3f} s, world model update {wm:.3f} s, "
    f"peak memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
  )


if __name__ == "__main__":
  tyro.cli(main)
