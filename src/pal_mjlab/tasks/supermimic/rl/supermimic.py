"""SuperMimic's learning updates (paper Section 3, Algorithm A.1, Table A.5).

Each iteration, after collecting simulator transitions into the replay buffer:

  update_world_model   Eq. 1 on k-step replay windows (``world_model.window_loss``)
  update_policy        Eq. 2: imagine H steps through the frozen world model from replay
                       states and step the policy along the analytic gradient of the
                       imagined return; then Eq. 3: fit the critic to the λ-returns of
                       the same rollouts

The world model holds the learned dynamics and termination head; reward, observation and
goal are the analytic heads of ``heads.py`` (D5). The policy is deterministic and sees the
predicted noise-free observation plus re-sampled sensor noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from ..state import state_to_qpos_qvel
from .heads import AnalyticHeads
from .models import EncoderDecoderModel
from .replay import ReplayBuffer
from .world_model import WorldModel, WorldModelCfg, window_loss


@dataclass
class WorldModelUpdateCfg:
  window: int = 8
  """k: autoregressive window length of Eq. 1."""
  batches: int = 10
  """Optimizer steps per iteration ("10 batches of 2048 windows per update", D10)."""
  batch_size: int = 2048
  learning_rate: float = 2.0e-4
  max_grad_norm: float = 1.0
  hidden: tuple[int, ...] = (2048, 2048, 2048)
  termination_hidden: tuple[int, ...] = (256, 256)
  termination_inputs: Literal["state", "rule"] = "state"
  """The learned failure head's input: the predicted state (D31), or only the failure
  rule's two errors (stability ablation A2', with ``termination_hidden=(64, 64)``)."""


@dataclass
class PolicyUpdateCfg:
  horizon: int = 32
  num_rollouts: int = 4096
  gamma: float = 0.99
  learning_rate: float = 2.0e-4
  betas: tuple[float, float] = (0.9, 0.95)
  max_grad_norm: float = 1.0
  """Global gradient-norm clip, applied first."""
  max_grad_value: float = 0.1
  """Element-wise gradient clip, applied second."""
  use_critic: bool = True
  """Bootstrap the imagined return with the critic (Eq. 2); False is the paper's
  no-critic ablation (the H-step reward sum only)."""
  sensor_noise: bool = True
  termination: Literal["learned", "rule"] = "learned"
  """Failure probability in the survival weight: the learned head (paper) or the env's
  failure rule, smoothed (D31). The other one is computed alongside and logged."""
  rule_temperature: float = 0.05
  hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
  encoder_dims: tuple[int, ...] = (2048, 1024, 512, 256)
  latent_dim: int = 32


@dataclass
class CriticUpdateCfg:
  hidden_dims: tuple[int, ...] = (256, 128)
  lam: float = 0.95
  learning_rate: float = 2.0e-4
  max_grad_norm: float = 1.0
  epochs: int = 5
  minibatches: int = 4
  survival_weighted: bool = False
  """Stability ablation A5: weight each step's error by the survival ĉ_t before it, so
  imagined steps after a predicted fall barely train the critic. The paper's Eq. 3 is
  unweighted."""
  privileged_inputs: bool = True
  """The root-pose error and time left in the critic's input (D7, D22); False is
  stability ablation A6."""


@dataclass
class SuperMimicCfg:
  world_model: WorldModelUpdateCfg = field(default_factory=WorldModelUpdateCfg)
  policy: PolicyUpdateCfg = field(default_factory=PolicyUpdateCfg)
  critic: CriticUpdateCfg = field(default_factory=CriticUpdateCfg)
  lr_decay_iterations: int = 250_000
  """Cosine decay of every learning rate to ``lr_final_ratio`` over this many
  iterations, then constant."""
  lr_final_ratio: float = 0.03
  precision: Literal["tf32", "bf16"] = "tf32"
  """Matmul precision of the learning updates: the paper's TF32, or bf16 autocast for
  16 GB GPUs (D16)."""


EXTREME_REWARD = -1.0
"""A per-step reward below this is logged as extreme: about 10x the largest positive
per-step reward (the tracking weights sum to 5, times dt)."""


@dataclass
class Rollout:
  """An imagined rollout. Only ``objective`` carries gradients."""

  objective: torch.Tensor  # (B,) Eq. 2
  reward: torch.Tensor  # (B, H) r̂_{t+1}
  cont: torch.Tensor  # (B, H) continuation after step t: (1 - d̂_{t+1})(1 - clip end)
  alive: torch.Tensor  # (B, H) step t is inside the episode (before any clip end)
  clip_end: torch.Tensor  # (B, H) step t reaches the clip's last frame
  head_prob: torch.Tensor  # (B, H) learned failure probability
  rule_prob: torch.Tensor  # (B, H) smoothed failure rule
  action_rate: torch.Tensor  # (B, H) the reward's action-rate penalty (unbounded)
  joint_limit: torch.Tensor  # (B, H) the reward's joint-limit penalty (unbounded)
  action_max: torch.Tensor  # (B, H) largest |action| component
  critic_input: torch.Tensor  # (B, H + 1, D)
  value: torch.Tensor  # (B, H + 1) critic before this update


def critic_value(critic: MLPModel, x: torch.Tensor) -> torch.Tensor:
  return critic.mlp(critic.obs_normalizer(x)).squeeze(-1)


def make_step(
  heads: AnalyticHeads,
  world_model: WorldModel,
  actor: EncoderDecoderModel,
  cfg: PolicyUpdateCfg,
):
  """One imagined step from ŝ_t, as a function of tensors only (so it can be compiled).

  The policy acts on the goal and its observation history; the world model predicts
  ŝ_{t+1}; the analytic heads give the reward terms, both failure probabilities and the
  next observation, to which the sensor noise η (sampled outside) is added.
  """

  def step(s, p, prev_action, history, ctx, clip, frame, noise):
    root_quat = state_to_qpos_qvel(s, p)[0][:, 3:7]
    goal = heads.goal(root_quat, clip, frame)
    x = heads.critic_input(s, p, goal, ctx, clip, frame)
    action = actor.act(torch.cat([goal, heads.flatten_history(history)], dim=-1))
    action = action.float()  # bf16 autocast: keep the state and reward math in fp32
    s_next, p_next = world_model.step(s, p, action, ctx)
    kin = heads.kinematics(s_next, p_next)
    ref = heads.reference(frame)
    terms = heads.reward_terms(s_next, kin, ref, action, prev_action)
    logit = world_model.termination_logit(s_next, heads.ref_root(frame)).float()
    rule = torch.sigmoid((heads.failure_margin(kin, ref) - 1.0) / cfg.rule_temperature)
    obs = heads.observation(s_next, kin, ctx, action)
    obs = {n: o + noise[n] if n in noise else o for n, o in obs.items()}
    history = {
      n: torch.cat([history[n][:, 1:], obs[n][:, None]], dim=1) for n in history
    }
    return x, action, s_next, p_next, terms, torch.sigmoid(logit), rule, history

  return step


def imagine(
  heads: AnalyticHeads,
  world_model: WorldModel,
  actor: EncoderDecoderModel,
  critic: MLPModel,
  start: dict[str, torch.Tensor],
  cfg: PolicyUpdateCfg,
  generator: torch.Generator | None = None,
  step=None,
) -> Rollout:
  """Roll the policy out for H steps in the world model from replay states (Eq. 2).

  ``start`` comes from ``ReplayBuffer.sample_starts``: the start state, pose, ξ and
  reference, and the previous rows whose observations fill the policy's history.
  ``step`` is ``make_step``'s function, possibly compiled; by default an eager one.
  """
  step = step or make_step(heads, world_model, actor, cfg)
  b, gamma = start["state"].shape[0], cfg.gamma
  ctx, clip, frame = start["context"], start["clip"], start["frame"]

  if "history" in start:  # the sim's current observation history
    history, prev_action = start["history"], start["last_action"]
  else:  # rebuilt from the logged states before the start (D9)
    hs = start["history_state"]
    h = hs.shape[1]
    flat_state = hs.flatten(0, 1)
    kin = heads.kinematics(flat_state, start["history_pose"].flatten(0, 1))
    history = heads.observation(
      flat_state,
      kin,
      ctx[:, None].expand(-1, h, -1).flatten(0, 1),
      start["history_last_action"].flatten(0, 1),
    )
    if cfg.sensor_noise:
      history = heads.add_noise(history, generator)
    history = {n: o.unflatten(0, (b, h)) for n, o in history.items()}
    prev_action = start["history_last_action"][:, -1]

  s, p = start["state"], start["pose"]
  c = torch.ones(b, device=s.device)
  alive = torch.ones(b, device=s.device)
  objective = torch.zeros(b, device=s.device)
  keys = ("reward", "cont", "alive", "end", "head", "rule", "input", "action_rate",
          "joint_limit", "action_max")  # fmt: skip
  out = {k: [] for k in keys}
  for t in range(cfg.horizon):
    noise = heads.sample_noise(b, generator) if cfg.sensor_noise else {}
    x, action, s_next, p_next, terms, head, rule, history = step(
      s, p, prev_action, history, ctx, clip, frame, noise
    )
    reward = sum(terms.values())
    done_prob = head if cfg.termination == "learned" else rule
    clip_end = heads.clip_end(clip, frame).float()
    objective = objective + gamma**t * c * reward
    cont = (1.0 - done_prob) * (1.0 - clip_end)
    c = c * cont
    for key, value in (
      ("input", x), ("reward", reward), ("cont", cont), ("alive", alive),
      ("end", clip_end), ("head", head), ("rule", rule),
      ("action_rate", terms["action_rate"]), ("joint_limit", terms["joint_limit"]),
      ("action_max", action.abs().amax(-1)),
    ):  # fmt: skip
      out[key].append(value)
    alive = alive * (1.0 - clip_end)
    s, p, prev_action = s_next, p_next, action
    frame = heads.next_frame(clip, frame)
  root_quat = state_to_qpos_qvel(s, p)[0][:, 3:7]
  goal = heads.goal(root_quat, clip, frame)
  out["input"].append(heads.critic_input(s, p, goal, ctx, clip, frame))

  inputs = torch.stack(out["input"], dim=1)
  if cfg.use_critic:
    value_end = critic_value(critic, inputs[:, -1]).float()  # gradient flows into ŝ_H
    objective = objective + gamma**cfg.horizon * c * value_end
  with torch.no_grad():
    value = critic_value(critic, inputs.detach().float())

  def stack(name):
    return torch.stack(out[name], dim=1).detach().float()

  return Rollout(
    objective=objective,
    reward=stack("reward"),
    cont=stack("cont"),
    alive=stack("alive"),
    clip_end=stack("end"),
    head_prob=stack("head"),
    rule_prob=stack("rule"),
    action_rate=stack("action_rate"),
    joint_limit=stack("joint_limit"),
    action_max=stack("action_max"),
    critic_input=inputs.detach().float(),
    value=value,
  )


def survival_before(cont: torch.Tensor) -> torch.Tensor:
  """(B, H) ĉ_t: the product of the continuations before step t (1 at t = 0)."""
  return torch.cat([torch.ones_like(cont[:, :1]), cont.cumprod(1)[:, :-1]], dim=1)


def lambda_returns(
  reward: torch.Tensor,
  cont: torch.Tensor,
  value: torch.Tensor,
  gamma: float,
  lam: float,
) -> torch.Tensor:
  """Eq. 3: R_t = r_{t+1} + γ cont_t ((1 - λ) V_{t+1} + λ R_{t+1}), R_H = V_H."""
  ret = value[:, -1]
  out = []
  for t in reversed(range(reward.shape[1])):
    ret = reward[:, t] + gamma * cont[:, t] * ((1 - lam) * value[:, t + 1] + lam * ret)
    out.append(ret)
  return torch.stack(out[::-1], dim=1)


class SuperMimic:
  """World model, policy and critic with their optimizers (Algorithm A.1)."""

  def __init__(
    self,
    obs: TensorDict,
    heads: AnalyticHeads,
    num_actions: int,
    context_dim: int,
    cfg: SuperMimicCfg,
    device: str,
    compile: bool = False,
  ) -> None:
    self.cfg, self.heads, self.device = cfg, heads, device
    pc, cc, wc = cfg.policy, cfg.critic, cfg.world_model
    groups = {"actor": ["goal", "proprio"], "critic": ["critic"]}
    self.actor = EncoderDecoderModel(
      obs, groups, "actor", num_actions, hidden_dims=pc.hidden_dims,
      encoder_dims=pc.encoder_dims, latent_dim=pc.latent_dim, obs_normalization=True,
    ).to(device)  # fmt: skip
    self.critic = MLPModel(
      obs, groups, "critic", 1, hidden_dims=cc.hidden_dims, obs_normalization=True
    ).to(device)
    state_dim = obs["critic"].shape[-1] - obs["goal"].shape[-1] - context_dim - 5
    self.world_model = WorldModel(
      WorldModelCfg(
        state_dim=state_dim, action_dim=num_actions, context_dim=context_dim,
        hidden=wc.hidden, termination_hidden=wc.termination_hidden,
        termination_inputs=wc.termination_inputs,
      )
    ).to(device)  # fmt: skip
    self.wm_optimizer = torch.optim.Adam(
      self.world_model.parameters(), lr=wc.learning_rate
    )
    self.actor_optimizer = torch.optim.Adam(
      self.actor.parameters(), lr=pc.learning_rate, betas=pc.betas
    )
    self.critic_optimizer = torch.optim.Adam(
      self.critic.parameters(), lr=cc.learning_rate
    )
    self._optimizers = (
      (self.wm_optimizer, wc.learning_rate),
      (self.actor_optimizer, pc.learning_rate),
      (self.critic_optimizer, cc.learning_rate),
    )
    # The imagination step and the world model's window loss are long chains of small
    # ops: compiled, each runs as a few fused kernels instead of thousands of launches.
    self._step = make_step(heads, self.world_model, self.actor, pc)
    self._window_loss = window_loss
    if compile:
      self._step = torch.compile(self._step, dynamic=False)
      self._window_loss = torch.compile(window_loss, dynamic=False)

  def imagine(
    self, start: dict[str, torch.Tensor], generator: torch.Generator | None = None
  ) -> Rollout:
    return imagine(
      self.heads, self.world_model, self.actor, self.critic, start, self.cfg.policy,
      generator, self._step,
    )  # fmt: skip

  def set_matmul_precision(self) -> None:
    """TF32 matmuls (for "bf16", the ops autocast leaves in FP32)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

  def _autocast(self):
    enabled = self.cfg.precision == "bf16"
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)

  def set_iteration(self, iteration: int) -> None:
    """Cosine learning-rate decay to ``lr_final_ratio`` (Table A.5), then constant."""
    progress = min(iteration / self.cfg.lr_decay_iterations, 1.0)
    final = self.cfg.lr_final_ratio
    scale = final + (1.0 - final) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for opt, lr in self._optimizers:
      for group in opt.param_groups:
        group["lr"] = lr * scale

  # World model (Eq. 1) ----------------------------------------------------------

  def update_world_model(
    self, replay: ReplayBuffer, generator: torch.Generator | None = None
  ) -> dict[str, float]:
    wc, stats = self.cfg.world_model, {}
    for _ in range(wc.batches):
      w = replay.sample_windows(wc.batch_size, wc.window, generator)
      ref_root = self.heads.ref_root(w["frame"].flatten()).unflatten(
        0, w["frame"].shape
      )
      with self._autocast():
        loss, info = self._window_loss(
          self.world_model, w["states"], w["poses"], w["actions"], w["context"],
          ref_root, w["failure"],
        )  # fmt: skip
      self.wm_optimizer.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), wc.max_grad_norm)
      self.wm_optimizer.step()
      for k, v in {"loss": loss.detach(), **info}.items():
        stats[k] = stats.get(k, 0.0) + float(v) / wc.batches
    return stats

  # Policy (Eq. 2) and critic (Eq. 3) ---------------------------------------------

  def update_policy(
    self,
    replay: ReplayBuffer,
    generator: torch.Generator | None = None,
    update_actor: bool = True,
    clips: torch.Tensor | None = None,
  ) -> dict[str, float]:
    pc = self.cfg.policy
    start = replay.sample_starts(pc.num_rollouts, self.heads.history, generator, clips)
    frozen = [*self.world_model.parameters(), *self.critic.parameters()]
    if not update_actor:
      frozen += list(self.actor.parameters())
    for prm in frozen:
      prm.requires_grad_(False)
    try:
      with self._autocast():
        roll = self.imagine(start, generator)
      stats = {"objective": float(roll.objective.detach().mean())}
      if update_actor:
        self.actor_optimizer.zero_grad(set_to_none=True)
        (-roll.objective.mean()).backward()
        params = list(self.actor.parameters())
        norm = torch.nn.utils.clip_grad_norm_(params, pc.max_grad_norm)
        for prm in params:
          prm.grad.clamp_(-pc.max_grad_value, pc.max_grad_value)
        self.actor_optimizer.step()
        stats["actor_grad_norm"] = float(norm)
    finally:
      for prm in frozen:
        prm.requires_grad_(True)
    stats.update(self._fit_critic(roll))
    stats.update(_rollout_stats(roll, pc.gamma))
    return stats

  def _fit_critic(self, roll: Rollout) -> dict[str, float]:
    cc = self.cfg.critic
    target = lambda_returns(
      roll.reward, roll.cont, roll.value, self.cfg.policy.gamma, cc.lam
    )
    keep = roll.alive > 0  # steps after a clip end have no target (D8)
    x, y = roll.critic_input[:, :-1][keep], target[keep]
    w = survival_before(roll.cont)[keep] if cc.survival_weighted else torch.ones_like(y)
    n, losses = len(y), []
    for _ in range(cc.epochs):
      for batch in torch.randperm(n, device=y.device).chunk(cc.minibatches):
        err = (critic_value(self.critic, x[batch]) - y[batch]).square()
        loss = (w[batch] * err).sum() / w[batch].sum()
        self.critic_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cc.max_grad_norm)
        self.critic_optimizer.step()
        losses.append(loss.detach())
    return {
      "critic_loss": float(torch.stack(losses).mean()) if losses else float("nan"),
      "value": float(roll.value[:, 0].mean()),
      "lambda_return": float(target[:, 0].mean()),
    }

  # Checkpoints --------------------------------------------------------------------

  def state_dict(self) -> dict:
    return {
      "actor": self.actor.state_dict(),
      "critic": self.critic.state_dict(),
      "world_model": {
        "cfg": vars(self.world_model.cfg),
        "state_dict": self.world_model.state_dict(),
      },
      "wm_optimizer": self.wm_optimizer.state_dict(),
      "actor_optimizer": self.actor_optimizer.state_dict(),
      "critic_optimizer": self.critic_optimizer.state_dict(),
    }

  def load_state_dict(self, state: dict, optimizers: bool = True) -> None:
    self.actor.load_state_dict(state["actor"])
    self.critic.load_state_dict(state["critic"])
    self.world_model.load_state_dict(state["world_model"]["state_dict"])
    if optimizers:
      self.wm_optimizer.load_state_dict(state["wm_optimizer"])
      self.actor_optimizer.load_state_dict(state["actor_optimizer"])
      self.critic_optimizer.load_state_dict(state["critic_optimizer"])

  def load_ppo_actor(self, path: str) -> None:
    """Initialize the actor from a PPO-Sim checkpoint (same network, D17)."""
    ckpt = torch.load(path, map_location=self.device, weights_only=False)
    state = {
      k: v
      for k, v in ckpt["actor_state_dict"].items()
      if not k.startswith("distribution.")
    }
    self.actor.load_state_dict(state)


@torch.no_grad()
def horizon_return(roll: Rollout, prob: torch.Tensor, gamma: float) -> torch.Tensor:
  """(B,) imagined H-step return without the critic, under failure probability ``prob``."""
  cont = (1 - prob) * (1 - roll.clip_end)
  survival = torch.cat([torch.ones_like(cont[:, :1]), cont.cumprod(1)[:, :-1]], dim=1)
  discount = gamma ** torch.arange(cont.shape[1], device=cont.device)
  return (discount * survival * roll.reward).sum(1)


@torch.no_grad()
def _rollout_stats(roll: Rollout, gamma: float) -> dict[str, float]:
  """Imagined return, how the learned head and the failure rule compare (D31), and the
  extremes of the reward's unbounded penalties."""
  discount = gamma ** torch.arange(roll.reward.shape[1], device=roll.reward.device)
  survival = {}
  for name, prob in (("head", roll.head_prob), ("rule", roll.rule_prob)):
    survival[name] = (1 - prob * roll.alive).prod(1)  # failures only, not clip ends
  inside = roll.alive > 0
  extreme = inside & (roll.reward < EXTREME_REWARD)
  c = survival_before(roll.cont)
  return {
    "imagined_reward": float((roll.reward * roll.alive).sum() / roll.alive.sum()),
    "imagined_return": float((roll.reward * roll.alive * discount).sum(1).mean()),
    "survival_head": float(survival["head"].mean()),
    "survival_rule": float(survival["rule"].mean()),
    "survival_disagreement": float(
      ((survival["head"] < 0.5) != (survival["rule"] < 0.5)).float().mean()
    ),
    "reward_min": float(roll.reward[inside].min()),
    "action_rate_min": float(roll.action_rate[inside].min()),
    "joint_limit_min": float(roll.joint_limit[inside].min()),
    "action_max": float(roll.action_max[inside].max()),
    "extreme_share": float(extreme.any(1).float().mean()),
    "extreme_survival": float(c[extreme].mean()) if extreme.any() else float("nan"),
  }
