"""SuperMimic training loop as an mjlab runner (Algorithm A.1, Table A.5).

Registered as the task's ``runner_cls``, so mjlab's ``train`` and ``play`` drive it:

  for each iteration:
    collect   ~2.5 env steps (update-to-data ratio 0.4) with the policy plus action noise
              N(0, 0.1²) into the FIFO replay buffer (1000 steps per env)
    fit       the world model on 10 x 2048 replay windows (Eq. 1)
    improve   the policy by BPTT through the frozen world model, then the critic (Eqs. 2, 3)

The env must run with ``auto_reset=False``: terminal states are stored, then the done envs
are reset here, with starts drawn by the return-based sampler (Appendix D.1).

Logs to TensorBoard: the losses, the imagined rollouts' statistics and the sim's episode
returns.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter
from mjlab.entity import Entity

from pal_mjlab.rl.command_state import command_state_dicts, load_command_state_dicts

from .command_model import CommandModel, command_model_of
from .export import export_policy
from .heads import AnalyticHeads
from .names import ImaginationCfg, critic_group, imagination_cfg
from .replay import ReplayBuffer
from .state import heading_pose, heading_state, pose_delta
from .supermimic import (
  CriticUpdateCfg,
  PolicyUpdateCfg,
  SuperMimic,
  SuperMimicCfg,
  WorldModelUpdateCfg,
)


def _from_dict(cfg: dict) -> SuperMimicCfg:
  alg = dict(cfg["algorithm"])
  return SuperMimicCfg(
    world_model=WorldModelUpdateCfg(**alg.pop("world_model")),
    policy=PolicyUpdateCfg(**alg.pop("policy")),
    critic=CriticUpdateCfg(**alg.pop("critic")),
    **alg,
  )

class SimCollector:
  """Steps the env and returns world-model transitions (states in the local frame)."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    names: ImaginationCfg,
    context_cfg,
    command_model: CommandModel,
  ) -> None:
    """``context_cfg``: the sim context's observation term (``AnalyticHeads.context_cfg``);
    ``command_model``: gives the command state c each transition is scored against."""
    self.env = env
    self.command_model = command_model
    self._context = context_cfg
    self._failure_terms = names.failure_terms
    # The state's joints are the action joints, on the action term's entity (the robot
    # contract: state joints = action joints; passive joints of a closed chain are out).
    # Raw qpos/qvel, not the entity's root_link_* accessors: the heading state expects
    # MuJoCo's free-joint layout (angular velocity in the body frame, as in qvel).
    action = env.action_manager.get_term(names.action_term)
    robot: Entity = env.scene[action.cfg.entity_name]
    idx, joints = robot.indexing, action.target_ids
    self._q_adr = torch.cat([idx.free_joint_q_adr, idx.joint_q_adr[joints]])
    self._v_adr = torch.cat([idx.free_joint_v_adr, idx.joint_v_adr[joints]])

  def state(self) -> tuple[torch.Tensor, torch.Tensor]:
    qpos = self.env.sim.data.qpos[:, self._q_adr]  # advanced indexing: a copy
    qvel = self.env.sim.data.qvel[:, self._v_adr]
    qpos[:, :3] -= self.env.scene.env_origins  # root position in the env's local frame
    return heading_state(qpos, qvel), heading_pose(qpos)

  def context(self) -> torch.Tensor:
    return self._context.func(self.env, **self._context.params)

  def observations(self) -> TensorDict:
    obs = self.env.observation_manager.compute()
    return TensorDict(obs, batch_size=[self.env.num_envs])

  def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor], TensorDict]:
    env = self.env
    state, pose = self.state()
    t = {
      "state": state,
      "pose": pose,
      "action": action,
      "context": self.context(),
      "command": self.command_model.state(),
    }
    _, reward, terminated, truncated, _ = env.step(action)
    t["next_state"], t["next_pose"] = self.state()  # terminal state where done
    tm = env.termination_manager
    failure = tm.get_term(self._failure_terms[0]).clone()
    for name in self._failure_terms[1:]:
      failure |= tm.get_term(name)
    t["failure"] = failure
    t["done"] = terminated | truncated
    t["reward"] = reward.clone()
    t["log"] = {}
    if t["done"].any():
      _, extras = env.reset(env_ids=t["done"].nonzero().squeeze(-1))
      t["log"] = extras.get("log", {})
    return t, self.observations()

class SuperMimicRunner:
  """Runner for reinforcement learning algorithms using the supermimic training loop."""

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    self.env, self.cfg, self.log_dir, self.device = env, train_cfg, log_dir, device
    raw = env.unwrapped
    self.names = imagination_cfg(raw)
    critic = critic_group(train_cfg, self.names)
    self.command_model = command_model_of(raw, self.names.command)
    self.heads = AnalyticHeads(
      raw, self.command_model, compile_fk=train_cfg["compile"], names=self.names,
      critic_group=critic,
    )
    self.collector = SimCollector(
      raw, self.names, self.heads.context_cfg, self.command_model
    )
    obs = self.collector.observations()
    self.alg = SuperMimic(
      obs,
      self.heads,
      self.command_model,
      raw.action_manager.total_action_dim,
      self.collector.context().shape[-1],
      self.collector.state()[0].shape[-1],
      _from_dict(train_cfg),
      device,
      compile=train_cfg["compile"],
      actor_groups=self.names.actor_groups,
      critic_group=critic,
    )
    self.current_learning_iteration = 0


  def add_git_repo_to_log(self, path: str) -> None:
    """Called by mjlab's ``train``; the run's configuration is logged by ``train`` itself."""

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
        "commands": command_state_dicts(self.env.unwrapped),
        "common_step_counter": self.env.unwrapped.common_step_counter,
      },
      path,
    )
    # The policy for the robot next to each checkpoint (model_N.onnx).
    try:
      export_policy(
        self.env.unwrapped, self.alg.actor, Path(path).with_suffix(".onnx"), path,
        names=self.names,
      )
    except Exception as e:  # never stop a training run over the export
      print(f"[WARN] ONNX export failed (training continues): {e}")

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
    if "commands" in ckpt:
      commands = ckpt["commands"]
    else:  # checkpoints before command-term checkpointing held the motion sampler alone
      commands = {self.names.command: {"sampler": ckpt["sampler"]}}
    load_command_state_dicts(self.env.unwrapped, commands, strict=strict)
    self.env.unwrapped.common_step_counter = ckpt["common_step_counter"]
    return {}


  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    """Train until ``num_learning_iterations`` iterations in total (resumes count)."""
    del init_at_random_ep_len  # episodes start where the command's reset puts them
    cfg, alg, raw = self.cfg, self.alg, self.env.unwrapped
    if raw.cfg.auto_reset:
      raise ValueError("SuperMimic stores terminal states: set env auto_reset=False.")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
      raise NotImplementedError("SuperMimicRunner trains on one GPU.")
    torch.manual_seed(cfg["seed"])
    self.generator = torch.Generator(device=self.device).manual_seed(cfg["seed"])
    self.writer = SummaryWriter(self.log_dir) if self.log_dir else None

    obs = self.start_collection()
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
      if (it + 1) % cfg["log_interval"] == 0:
        self._log(it + 1)
      if self.log_dir and (it + 1) % cfg["save_interval"] == 0:
        self.current_learning_iteration = it + 1
        self.save(os.path.join(self.log_dir, f"model_{it + 1}.pt"))
    self.current_learning_iteration = num_learning_iterations
    if self.log_dir:
      self.save(os.path.join(self.log_dir, f"model_{num_learning_iterations}.pt"))

  def start_collection(self) -> TensorDict:
    """Reset the envs; create the replay buffer and the episode statistics."""
    raw = self.env.unwrapped
    obs = TensorDict(raw.reset()[0], batch_size=[raw.num_envs])
    state = self.collector.state()[0]
    self.replay = ReplayBuffer(
      raw.num_envs, self.cfg["replay_steps"], state.shape[-1],
      raw.action_manager.total_action_dim, self.collector.context().shape[-1],
      self.device,
    )  # fmt: skip
    self._stats: dict[str, list[float]] = {}
    self._episode_return = torch.zeros(raw.num_envs, device=self.device)
    self._episode_length = torch.zeros(raw.num_envs, device=self.device)
    return obs

  @torch.no_grad()
  def _collect(self, obs: TensorDict) -> tuple[TensorDict, dict]:
    """One env step with the policy plus collection noise, into the replay buffer."""
    alg = self.alg
    alg.actor.update_normalization(obs)
    alg.critic.update_normalization(obs)
    action = alg.actor(obs)
    noise = self.cfg["action_noise"] * torch.randn(
      action.shape, device=action.device, generator=self.generator
    )
    t, obs = self.collector.step(action + noise)
    wm = alg.world_model
    wm.update_normalization(
      t["state"], t["action"], t["context"], t["next_state"],
      pose_delta(t["pose"], t["next_pose"]),
    )  # fmt: skip
    self.replay.add(
      t["state"], t["pose"], t["action"], t["context"], t["command"],
      t["failure"], t["done"], t["next_state"], t["next_pose"],
    )  # fmt: skip

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

  # Logging ------------------------------------------------------------------------------

  def _add(self, key: str, values: torch.Tensor) -> None:
    self._stats.setdefault(key, []).extend(values.float().cpu().tolist())

  def _record(self, stats: dict[str, float], prefix: str) -> None:
    for k, v in stats.items():
      if not math.isnan(v):
        self._stats.setdefault(f"{prefix}/{k}", []).append(float(v))

  def _log(self, it: int) -> None:
    means = {k: sum(v) / len(v) for k, v in self._stats.items() if v}
    self._stats = {}
    if torch.cuda.is_available():
      means["memory/max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
    if self.writer is not None:
      for k, v in means.items():
        self.writer.add_scalar(k, v, it)
    keys = ("episode/return", "episode/length", "policy/objective", "world_model/loss")
    shown = " ".join(f"{k.split('/')[-1]} {means[k]:.4g}" for k in keys if k in means)
    print(f"[SuperMimic] it {it}: {shown}", flush=True)