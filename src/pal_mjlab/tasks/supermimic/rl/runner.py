"""SuperMimic training loop as an mjlab runner (Algorithm A.1, Table A.5).

Registered as the task's ``runner_cls``, so mjlab's ``train`` and ``play`` drive it:

  for each iteration:
    collect   ~2.5 env steps (update-to-data ratio 0.4) with the policy plus action noise
              N(0, 0.1²) into the FIFO replay buffer (1000 steps per env, D9)
    fit       the world model on 10 x 2048 replay windows (Eq. 1)
    improve   the policy by BPTT through the frozen world model, then the critic (Eqs. 2, 3)

The env must run with ``auto_reset=False``: terminal states are stored, then the done envs
are reset here, with starts drawn by the return-based sampler (Appendix D.1).

Logged besides the losses: the sim's episode returns, the imagined-vs-sim return gap of
the current policy from the same states (every ``eval_interval`` iterations), and how the
learned termination head and the env's failure rule rank fresh sim failures (D31).
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from ..mdp.commands import MotionLibraryCommand
from ..state import heading_pose, heading_state, pose_delta
from .heads import AnalyticHeads
from .replay import ReplayBuffer
from .supermimic import (
  CriticUpdateCfg,
  PolicyUpdateCfg,
  SuperMimic,
  SuperMimicCfg,
  WorldModelUpdateCfg,
  horizon_return,
  imagine,
)


@dataclass
class SuperMimicRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "SuperMimicRunner"
  algorithm: SuperMimicCfg = field(default_factory=SuperMimicCfg)
  action_noise: float = 0.1
  """Std of the Gaussian collection noise on each action dimension."""
  replay_steps: int = 1000
  """Replay buffer size per env."""
  env_steps_per_iteration: float = 2.5
  """Vector env steps per iteration: update-to-data ratio 0.4 (D10)."""
  warmup_steps: int = 32
  """Env steps collected before the first update."""
  init_actor_checkpoint: str = ""
  """Optional PPO-Sim checkpoint to initialize the actor (not used by the paper)."""
  compile_kinematics: bool = True
  log_interval: int = 10
  eval_interval: int = 500
  """Iterations between imagined-vs-sim checks of the current policy (0: never)."""
  eval_envs: int = 512


def _from_dict(cfg: dict) -> SuperMimicCfg:
  alg = dict(cfg["algorithm"])
  return SuperMimicCfg(
    world_model=WorldModelUpdateCfg(**alg.pop("world_model")),
    policy=PolicyUpdateCfg(**alg.pop("policy")),
    critic=CriticUpdateCfg(**alg.pop("critic")),
    **alg,
  )


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
  pos, neg = scores[labels], scores[~labels]
  if len(pos) == 0 or len(neg) == 0:
    return float("nan")
  ranks = torch.cat([pos, neg]).argsort().argsort().float() + 1
  return float(
    (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
  )


class SimCollector:
  """Steps the env and returns world-model transitions (states in the local frame)."""

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self.env = env
    self.cmd = cast(MotionLibraryCommand, env.command_manager.get_term("motion"))
    self._context = env.observation_manager.get_term_cfg("critic", "sim_context")

  def state(self) -> tuple[torch.Tensor, torch.Tensor]:
    qpos = self.cmd.robot_qpos.clone()
    qpos[:, :3] -= self.env.scene.env_origins
    return heading_state(qpos, self.cmd.robot_qvel), heading_pose(qpos)

  def context(self) -> torch.Tensor:
    return self._context.func(self.env, **self._context.params)

  def observations(self) -> TensorDict:
    obs = self.env.observation_manager.compute()
    return TensorDict(obs, batch_size=[self.env.num_envs])

  def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor], TensorDict]:
    env, cmd = self.env, self.cmd
    state, pose = self.state()
    t = {
      "state": state,
      "pose": pose,
      "action": action,
      "context": self.context(),
      "clip": cmd.clip.clone(),
      "frame": cmd.frame.clone(),
    }
    _, reward, terminated, truncated, _ = env.step(action)
    t["next_state"], t["next_pose"] = self.state()  # terminal state where done
    t["failure"] = env.termination_manager.get_term("anchor_failure").clone()
    t["done"] = terminated | truncated
    t["truncated"] = truncated & ~terminated
    t["reward"] = reward.clone()
    t["log"] = {}
    if t["done"].any():
      _, extras = env.reset(env_ids=t["done"].nonzero().squeeze(-1))
      t["log"] = extras.get("log", {})
    return t, self.observations()


class SuperMimicRunner:
  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    self.env, self.cfg, self.log_dir, self.device = env, train_cfg, log_dir, device
    raw = env.unwrapped
    self.collector = SimCollector(raw)
    self.heads = AnalyticHeads(raw, compile_fk=train_cfg["compile_kinematics"])
    obs = self.collector.observations()
    self.alg = SuperMimic(
      obs,
      self.heads,
      raw.action_manager.total_action_dim,
      self.collector.context().shape[-1],
      _from_dict(train_cfg),
      device,
    )
    self.current_learning_iteration = 0
    self._git_paths: list[str] = []

  # mjlab runner interface -----------------------------------------------------------

  def add_git_repo_to_log(self, path: str) -> None:
    self._git_paths.append(path)

  def get_inference_policy(self, device: str | None = None):
    actor = self.alg.actor
    actor.eval()
    if device is not None:
      actor.to(device)

    def policy(obs: TensorDict) -> torch.Tensor:
      with torch.no_grad():
        return actor(obs)

    return policy

  def save(self, path: str) -> None:
    torch.save(
      {
        "alg": self.alg.state_dict(),
        "iter": self.current_learning_iteration,
        "sampler": self.collector.cmd.sampler.state_dict(),
        "common_step_counter": self.env.unwrapped.common_step_counter,
      },
      path,
    )

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Resume training, or (``load_cfg={"actor": True}``, as ``play`` does) the actor."""
    ckpt = torch.load(
      path, map_location=map_location or self.device, weights_only=False
    )
    if load_cfg is not None and {k for k, v in load_cfg.items() if v} == {"actor"}:
      self.alg.actor.load_state_dict(ckpt["alg"]["actor"])
      return {}
    self.alg.load_state_dict(ckpt["alg"])
    self.current_learning_iteration = ckpt["iter"]
    self.collector.cmd.sampler.load_state_dict(ckpt["sampler"])
    self.env.unwrapped.common_step_counter = ckpt["common_step_counter"]
    return {}

  # Training ---------------------------------------------------------------------------

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    """Train until ``num_learning_iterations`` iterations in total (resumes count)."""
    del init_at_random_ep_len  # episodes start at sampled frames anyway
    cfg, alg, raw = self.cfg, self.alg, self.env.unwrapped
    if raw.cfg.auto_reset:
      raise ValueError("SuperMimic stores terminal states: set env auto_reset=False.")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
      raise NotImplementedError("SuperMimicRunner trains on one GPU.")
    torch.manual_seed(cfg["seed"])
    alg.set_matmul_precision()
    self.generator = torch.Generator(device=self.device).manual_seed(cfg["seed"])
    self.writer = SummaryWriter(self.log_dir) if self.log_dir else None
    self._log_git()
    if cfg["init_actor_checkpoint"] and self.current_learning_iteration == 0:
      alg.load_ppo_actor(cfg["init_actor_checkpoint"])

    obs = TensorDict(raw.reset()[0], batch_size=[raw.num_envs])
    state = self.collector.state()[0]
    self.replay = ReplayBuffer(
      raw.num_envs, cfg["replay_steps"], state.shape[-1],
      raw.action_manager.total_action_dim, self.collector.context().shape[-1],
      self.device,
    )  # fmt: skip
    self._stats: dict[str, list[float]] = {}
    self._episode_return = torch.zeros(raw.num_envs, device=self.device)
    self._episode_length = torch.zeros(raw.num_envs, device=self.device)
    self._term_scores: list[torch.Tensor] = []
    for _ in range(cfg["warmup_steps"]):
      obs, _ = self._collect(obs)

    budget = 0.0
    start = self.current_learning_iteration
    for it in range(start, num_learning_iterations):
      self.current_learning_iteration = it
      t0 = time.perf_counter()
      budget += cfg["env_steps_per_iteration"]
      while budget >= 1.0:
        obs, _ = self._collect(obs)
        budget -= 1.0
      t1 = time.perf_counter()
      alg.set_iteration(it)
      self._record(alg.update_world_model(self.replay, self.generator), "world_model")
      t2 = time.perf_counter()
      self._record(alg.update_policy(self.replay, self.generator), "policy")
      t3 = time.perf_counter()
      self._record(
        {"collect": t1 - t0, "world_model": t2 - t1, "policy": t3 - t2}, "time"
      )
      if cfg["eval_interval"] and (it + 1) % cfg["eval_interval"] == 0:
        obs, steps = self._evaluate(obs)
        budget -= steps
      if (it + 1) % cfg["log_interval"] == 0:
        self._log(it + 1)
      if self.log_dir and (it + 1) % cfg["save_interval"] == 0:
        self.current_learning_iteration = it + 1
        self.save(os.path.join(self.log_dir, f"model_{it + 1}.pt"))
    self.current_learning_iteration = num_learning_iterations
    if self.log_dir:
      self.save(os.path.join(self.log_dir, f"model_{num_learning_iterations}.pt"))

  @torch.no_grad()
  def _collect(self, obs: TensorDict, deterministic: torch.Tensor | None = None):
    """One env step with the policy plus collection noise, into the replay buffer."""
    alg = self.alg
    alg.actor.update_normalization(obs)
    alg.critic.update_normalization(obs)
    action = alg.actor(obs)
    noise = self.cfg["action_noise"] * torch.randn(
      action.shape, device=action.device, generator=self.generator
    )
    if deterministic is not None:
      noise[deterministic] = 0.0
    t, obs = self.collector.step(action + noise)
    wm = alg.world_model
    wm.update_normalization(
      t["state"], t["action"], t["context"], t["next_state"],
      pose_delta(t["pose"], t["next_pose"]),
    )  # fmt: skip
    self.replay.add(
      t["state"], t["pose"], t["action"], t["context"], t["clip"], t["frame"],
      t["failure"], t["done"], t["next_state"], t["next_pose"],
    )  # fmt: skip

    # Termination head vs the env's rule on fresh sim states (D31): all failures and a
    # 2% sample of the rest (an unbiased AUC). Scored on all envs: a fixed batch size
    # for the compiled kinematics.
    s, p, f = t["next_state"], t["next_pose"], t["frame"]
    head = wm.termination_logit(s, self.heads.ref_root(f)).float()
    kin = self.heads.kinematics(s, p)
    rule = self.heads.failure_margin(kin, self.heads.reference(f))
    label = t["failure"]
    sample = torch.rand(label.shape, device=self.device, generator=self.generator)
    keep = label | (sample < 0.02)
    self._term_scores.append(torch.stack([head, rule, label.float()], 1)[keep])

    self._episode_return += t["reward"]
    self._episode_length += 1
    done = t["done"]
    if done.any():
      self._add("episode/return", self._episode_return[done])
      self._add("episode/length", self._episode_length[done])
      self._add("episode/failure", t["failure"][done].float())
      self._episode_return[done] = 0.0
      self._episode_length[done] = 0.0
      for k, v in t["log"].items():
        self._record({k: float(v)}, "env")
    self._record({"reward": t["reward"].mean()}, "sim")
    return obs, t

  @torch.no_grad()
  def _evaluate(self, obs: TensorDict) -> tuple[TensorDict, int]:
    """Imagined vs simulated H-step return of the current policy from the same states.

    The first ``eval_envs`` envs act without collection noise for H steps in the sim,
    and imagination starts from their current state and observation history. Envs that
    hit the episode time cap are left out (truncation is not modeled).
    """
    heads, alg, raw = self.heads, self.alg, self.env.unwrapped
    pc = alg.cfg.policy
    n = min(self.cfg["eval_envs"], raw.num_envs)
    state, pose = self.collector.state()
    cmd = self.collector.cmd
    start = {
      "state": state[:n],
      "pose": pose[:n],
      "context": self.collector.context()[:n],
      "clip": cmd.clip[:n].clone(),
      "frame": cmd.frame[:n].clone(),
      "history": heads.unflatten_history(obs["proprio"][:n]),
      "last_action": raw.action_manager.action[:n].clone(),
    }
    with alg._autocast():
      roll = imagine(
        heads, alg.world_model, alg.actor, alg.critic, start, pc, self.generator
      )

    deterministic = torch.zeros(raw.num_envs, dtype=torch.bool, device=self.device)
    deterministic[:n] = True
    sim_return = torch.zeros(n, device=self.device)
    alive = torch.ones(n, dtype=torch.bool, device=self.device)
    failed, truncated = torch.zeros_like(alive), torch.zeros_like(alive)
    for step in range(pc.horizon):
      obs, t = self._collect(obs, deterministic)
      sim_return += torch.where(alive, pc.gamma**step * t["reward"][:n], 0.0)
      failed |= alive & t["failure"][:n]
      truncated |= alive & t["truncated"][:n]
      alive &= ~t["done"][:n]

    keep = ~truncated
    sim, stats = sim_return[keep], {"sim_return": float(sim_return[keep].mean())}
    stats["sim_failure"] = float(failed[keep].float().mean())
    for name, prob in (("head", roll.head_prob), ("rule", roll.rule_prob)):
      imag = horizon_return(roll, prob, pc.gamma)[keep]
      stats[f"imagined_return_{name}"] = float(imag.mean())
      stats[f"return_gap_{name}"] = float((imag - sim).mean() / sim.abs().mean())
      stats[f"return_corr_{name}"] = float(
        torch.corrcoef(torch.stack([imag, sim]))[0, 1]
      )
      survival = (1 - prob * roll.alive).prod(1)[keep]
      stats[f"imagined_failure_{name}"] = float(1 - survival.mean())
    self._record(stats, "eval")
    return obs, pc.horizon

  # Logging ------------------------------------------------------------------------------

  def _add(self, key: str, values: torch.Tensor) -> None:
    self._stats.setdefault(key, []).extend(values.float().cpu().tolist())

  def _record(self, stats: dict[str, float], prefix: str) -> None:
    for k, v in stats.items():
      self._stats.setdefault(f"{prefix}/{k}", []).append(float(v))

  def _log(self, it: int) -> None:
    if self._term_scores:
      scores = torch.cat(self._term_scores)
      labels = scores[:, 2] > 0.5
      if labels.any():
        self._record(
          {
            "head_auc": auc(scores[:, 0], labels),
            "rule_auc": auc(scores[:, 1], labels),
            "failures": float(labels.sum()),
          },
          "termination",
        )
      self._term_scores = []
    means = {k: sum(v) / len(v) for k, v in self._stats.items() if v}
    self._stats = {}
    if torch.cuda.is_available():
      means["memory/max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    if self.writer is not None:
      for k, v in means.items():
        self.writer.add_scalar(k, v, it)
    keys = ("episode/return", "episode/length", "policy/objective", "world_model/loss",
            "termination/head_auc", "termination/rule_auc")  # fmt: skip
    shown = " ".join(f"{k.split('/')[-1]} {means[k]:.4g}" for k in keys if k in means)
    print(f"[SuperMimic] it {it}: {shown}", flush=True)

  def _log_git(self) -> None:
    if not self.log_dir or not self._git_paths:
      return
    root = Path(self._git_paths[0]).parent
    out = Path(self.log_dir) / "git"
    out.mkdir(parents=True, exist_ok=True)
    for name, args in (("commit.txt", ["rev-parse", "HEAD"]), ("diff.patch", ["diff"])):
      result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True
      )
      (out / name).write_text(result.stdout)
