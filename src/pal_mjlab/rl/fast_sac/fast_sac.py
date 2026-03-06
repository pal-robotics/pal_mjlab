"""FastSAC algorithm compatible with the rsl-rl OnPolicyRunner.

Implements the PPO-like interface (act, process_env_step, compute_returns,
update) so the existing runner infrastructure (logging, checkpointing) works
out of the box.  The key difference is that FastSAC uses an off-policy replay
buffer and performs ``num_updates`` gradient steps per runner iteration.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import torch.optim as optim
from rsl_rl.env import VecEnv
from rsl_rl.utils import resolve_obs_groups
from tensordict import TensorDict
from torch import nn
from torch.amp import GradScaler, autocast

from pal_mjlab.rl.fast_sac.networks import Actor, Critic
from pal_mjlab.rl.fast_sac.replay_buffer import ReplayBuffer


class EmpiricalNormalization(nn.Module):
  """Welford-style online mean/variance normalization."""

  def __init__(self, shape: int, device: torch.device | str, eps: float = 1e-2):
    super().__init__()
    self.eps = eps
    self.register_buffer("_mean", torch.zeros(1, shape, device=device))
    self.register_buffer("_var", torch.ones(1, shape, device=device))
    self.register_buffer("_std", torch.ones(1, shape, device=device))
    self.register_buffer("count", torch.tensor(0, dtype=torch.long, device=device))

  @torch.no_grad()
  def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
    if self.training and update:
      self._update(x)
    return (x - self._mean) / (self._std + self.eps)

  def _update(self, x: torch.Tensor) -> None:
    batch_size = x.shape[0]
    batch_mean = x.mean(dim=0, keepdim=True)
    batch_var = x.var(dim=0, unbiased=False, keepdim=True)
    new_count = self.count + batch_size

    delta = batch_mean - self._mean
    self._mean.copy_(self._mean + delta * (batch_size / new_count))

    delta2 = batch_mean - self._mean
    m_a = self._var * self.count
    m_b = batch_var * batch_size
    M2 = m_a + m_b + delta2.pow(2) * (self.count * batch_size / new_count)
    self._var.copy_(M2 / new_count)
    self._std.copy_(self._var.sqrt())
    self.count.copy_(new_count)


class NoOpNormalization(nn.Module):
  """Identity normalizer with the same signature as EmpiricalNormalization."""

  def forward(self, x: torch.Tensor, update: bool = True) -> torch.Tensor:
    del update
    return x


class FastSAC:
  """FastSAC: efficient SAC with distributional critics.

  Exposes the same interface as ``rsl_rl.algorithms.PPO`` so that
  ``OnPolicyRunner`` can drive collection and logging.
  """

  def __init__(
    self,
    actor: Actor,
    actor_obs_groups_dict : dict[str, list[str]],
    obs_dict : TensorDict,
    critic: Critic,
    critic_target: Critic,
    replay_buffer: ReplayBuffer,
    obs_normalizer: nn.Module,
    critic_obs_normalizer: nn.Module,
    *,
    num_envs: int,
    n_act: int,
    # SAC hyper-parameters
    critic_lr: float = 3e-4,
    actor_lr: float = 3e-4,
    alpha_lr: float = 3e-4,
    gamma: float = 0.97,
    tau: float = 0.125,
    batch_size: int = 8192,
    learning_starts: int = 10,
    policy_frequency: int = 4,
    num_updates: int = 8,
    target_entropy_ratio: float = 0.0,
    alpha_init: float = 0.001,
    use_autotune: bool = True,
    max_grad_norm: float = 0.0,
    weight_decay: float = 0.001,
    use_amp: bool = True,
    amp_dtype: str = "bf16",
    compile: bool = True,
    num_steps: int = 1,
    device: str = "cpu",
  ) -> None:
    self.device = device
    self.actor = actor
    self.actor_obs_groups_dict = actor_obs_groups_dict
    self.obs_dict = obs_dict
    self.critic = critic
    self.critic_target = critic_target
    self.replay_buffer = replay_buffer
    self.obs_normalizer = obs_normalizer
    self.critic_obs_normalizer = critic_obs_normalizer

    self.num_envs = num_envs
    self.n_act = n_act
    self.gamma = gamma
    self.tau = tau
    self.batch_size = batch_size
    self.learning_starts = learning_starts
    self.policy_frequency = policy_frequency
    self.num_updates = num_updates
    self.max_grad_norm = max_grad_norm
    self.use_amp = use_amp
    self.amp_dtype = amp_dtype
    self._compile = compile
    self.num_steps = num_steps

    self.log_alpha = torch.tensor(
      [math.log(alpha_init)], requires_grad=True, device=device
    )
    self.target_entropy = -n_act * target_entropy_ratio
    self.use_autotune = use_autotune

    self.q_optimizer = optim.AdamW(
      self.critic.parameters(),
      lr=critic_lr,
      weight_decay=weight_decay,
      betas=(0.9, 0.95),
    )
    self.actor_optimizer = optim.AdamW(
      self.actor.parameters(),
      lr=actor_lr,
      weight_decay=weight_decay,
      betas=(0.9, 0.95),
    )
    self.alpha_optimizer = optim.AdamW(
      [self.log_alpha],
      lr=alpha_lr,
      betas=(0.9, 0.95),
    )

    self.scaler = GradScaler(enabled=use_amp)

    # Bookkeeping for the runner's collect loop
    self._last_actor_obs: torch.Tensor | None = None
    self._last_critic_obs: torch.Tensor | None = None
    self._last_actions: torch.Tensor | None = None
    self._global_step = 0  # counts runner iterations for warmup

    # Compiled / plain function references set up in train_mode()
    self._update_critic_fn = self._update_critic
    self._update_actor_fn = self._update_actor
    self._normalize_obs_fn = self.obs_normalizer.forward
    self._normalize_critic_obs_fn = self.critic_obs_normalizer.forward

    # Latest loss values for logging
    self._latest_losses: dict[str, float] = {}

    # Learning rate property (read by the runner logger)
    self.learning_rate = actor_lr

  @contextmanager
  def _maybe_amp(self):
    dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16
    with autocast(device_type="cuda", dtype=dtype, enabled=self.use_amp):
      yield

  def act(self, obs: TensorDict) -> torch.Tensor:
    """Select actions for all envs.  Called by the runner's collect loop."""
    flat_actor_obs = self._flatten_actor_obs(obs)
    flat_critic_obs = self._flatten_critic_obs(obs)
    self._last_actor_obs = flat_actor_obs
    self._last_critic_obs = flat_critic_obs

    with torch.no_grad(), self._maybe_amp():
      norm_obs = self._normalize_obs_fn(flat_actor_obs, update=False)
      actions = self.actor.explore(norm_obs)

    self._last_actions = actions
    return actions

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict,
  ) -> None:
    """Store the transition in the replay buffer."""
    flat_next_actor_obs = self._flatten_actor_obs(obs)
    flat_next_critic_obs = self._flatten_critic_obs(obs)
    truncations = extras.get("time_outs", torch.zeros_like(dones))

    # If true final observations are unavailable on timeouts, disable
    # timeout bootstrapping to avoid using post-reset observations.
    has_final_obs = False
    if torch.any(truncations.bool()):
      final_obs = extras.get("observations", {}).get("final", {})
      if isinstance(final_obs, dict):
        actor_keys = all(k in final_obs for k in self._actor_obs_group_names)
        critic_keys = all(k in final_obs for k in self._critic_obs_group_names)
        if actor_keys and critic_keys:
          final_actor_obs = torch.cat(
            [final_obs[k] for k in self._actor_obs_group_names],
            dim=-1,
          ).to(flat_next_actor_obs.device)
          final_critic_obs = torch.cat(
            [final_obs[k] for k in self._critic_obs_group_names],
            dim=-1,
          ).to(flat_next_critic_obs.device)
          timeout_mask = truncations.bool().unsqueeze(-1)
          flat_next_actor_obs = torch.where(
            timeout_mask, final_actor_obs, flat_next_actor_obs
          )
          flat_next_critic_obs = torch.where(
            timeout_mask, final_critic_obs, flat_next_critic_obs
          )
          has_final_obs = True
      if not has_final_obs:
        truncations = torch.zeros_like(truncations)

    assert self._last_actor_obs is not None
    assert self._last_critic_obs is not None
    assert self._last_actions is not None
    self.replay_buffer.extend(
      observations=self._last_actor_obs,
      critic_observations=self._last_critic_obs,
      actions=self._last_actions,
      rewards=rewards,
      dones=dones,
      truncations=truncations.long(),
      next_observations=flat_next_actor_obs,
      next_critic_observations=flat_next_critic_obs,
    )

    # Update normalizers during collection.
    self.obs_normalizer(flat_next_actor_obs, update=True)
    self.critic_obs_normalizer(flat_next_critic_obs, update=True)

    self._global_step += 1

  def compute_returns(self, obs: TensorDict) -> None:
    """No-op: SAC does not use GAE returns."""

  def update(self) -> dict[str, float]:
    """Perform ``num_updates`` SAC gradient steps using replay data."""
    if self._global_step <= self.learning_starts:
      return {}

    batch_per_env = max(self.batch_size // self.num_envs, 1)
    large_batch_size = batch_per_env * self.num_updates
    data = self.replay_buffer.sample(large_batch_size)

    # Normalize observations once for the whole large batch
    with torch.no_grad():
      data["observations"] = self._normalize_obs_fn(data["observations"], update=False)
      data["next_observations"] = self._normalize_obs_fn(
        data["next_observations"], update=False
      )
      data["critic_observations"] = self._normalize_critic_obs_fn(
        data["critic_observations"], update=False
      )
      data["next_critic_observations"] = self._normalize_critic_obs_fn(
        data["next_critic_observations"], update=False
      )

    samples_per_update = batch_per_env * self.num_envs

    # Accumulators for logging
    total_critic_loss = 0.0
    total_actor_loss = 0.0
    total_alpha_loss = 0.0
    total_q_max = 0.0
    total_q_min = 0.0
    total_entropy = 0.0
    total_alpha = 0.0
    actor_updates = 0

    for i in range(self.num_updates):
      start = i * samples_per_update
      end = (i + 1) * samples_per_update
      batch = {k: v[start:end] for k, v in data.items()}

      critic_loss, q_max, q_min, next_log_probs = self._update_critic_fn(batch)
      total_critic_loss += critic_loss.item()
      total_q_max += q_max.item()
      total_q_min += q_min.item()

      # Delayed policy updates
      if self.num_updates > 1:
        do_actor_update = i % self.policy_frequency == 1
      else:
        do_actor_update = self._global_step % self.policy_frequency == 0

      if do_actor_update:
        actor_loss, entropy = self._update_actor_fn(batch)
        total_actor_loss += actor_loss.item()
        total_entropy += entropy.item()
        actor_updates += 1

      # Alpha update
      if self.use_autotune:
        alpha_loss = self._update_alpha(next_log_probs)
        total_alpha_loss += alpha_loss

      total_alpha += self.log_alpha.exp().item()

      # Soft target update
      with torch.no_grad():
        src_ps = [p.data for p in self.critic.parameters()]
        tgt_ps = [p.data for p in self.critic_target.parameters()]
        torch._foreach_mul_(tgt_ps, 1.0 - self.tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=self.tau)

    n = self.num_updates
    loss_dict = {
      "critic_loss": total_critic_loss / n,
      "q_max": total_q_max / n,
      "q_min": total_q_min / n,
      "alpha": total_alpha / n,
    }
    if actor_updates > 0:
      loss_dict["actor_loss"] = total_actor_loss / actor_updates
      loss_dict["policy_entropy"] = total_entropy / actor_updates
    if self.use_autotune:
      loss_dict["alpha_loss"] = total_alpha_loss / n

    self._latest_losses = loss_dict
    return loss_dict

  def _update_critic(
    self, batch: dict[str, torch.Tensor]
  ) -> tuple[float, float, float, torch.Tensor]:
    with self._maybe_amp():
      critic_obs = batch["critic_observations"]
      actions = batch["actions"]
      next_actor_obs = batch["next_observations"]
      next_critic_obs = batch["next_critic_observations"]
      rewards = batch["rewards"]
      dones = batch["dones"].bool()
      truncations = batch["truncations"].bool()
      bootstrap = (truncations | ~dones).float()

      with torch.no_grad():
        next_actions, next_log_probs = self.actor.get_actions_and_log_probs(
          next_actor_obs
        )
        discount = self.gamma ** batch["effective_n_steps"]

        modified_rewards = (
          rewards - discount * bootstrap * self.log_alpha.exp() * next_log_probs
        )

        target_distributions = self.critic_target.projection(
          next_critic_obs,
          next_actions,
          modified_rewards,
          bootstrap,
          discount,
        )
        target_values = self.critic_target.get_value(target_distributions)

      q_outputs = self.critic(critic_obs, actions)
      critic_log_probs = F.log_softmax(q_outputs, dim=-1)
      critic_losses = -torch.sum(target_distributions * critic_log_probs, dim=-1)
      qf_loss = critic_losses.mean(dim=1).sum(dim=0)

    self.q_optimizer.zero_grad(set_to_none=True)
    self.scaler.scale(qf_loss).backward()
    self.scaler.unscale_(self.q_optimizer)
    if self.max_grad_norm > 0:
      torch.nn.utils.clip_grad_norm_(
        self.critic.parameters(), max_norm=self.max_grad_norm
      )
    self.scaler.step(self.q_optimizer)
    self.scaler.update()

    return (
      qf_loss.detach(),
      target_values.max().detach(),
      target_values.min().detach(),
      next_log_probs.detach(),
    )

  def _update_actor(self, batch: dict[str, torch.Tensor]) -> tuple[float, float]:
    with self._maybe_amp():
      actor_obs = batch["observations"]
      critic_obs = batch["critic_observations"]
      actions, log_probs = self.actor.get_actions_and_log_probs(actor_obs)

      q_outputs = self.critic(critic_obs, actions)
      q_probs = F.softmax(q_outputs, dim=-1)
      q_values = self.critic.get_value(q_probs).mean(dim=0)

      actor_loss = (self.log_alpha.exp().detach() * log_probs - q_values).mean()

    self.actor_optimizer.zero_grad(set_to_none=True)
    self.scaler.scale(actor_loss).backward()
    self.scaler.unscale_(self.actor_optimizer)
    if self.max_grad_norm > 0:
      torch.nn.utils.clip_grad_norm_(
        self.actor.parameters(), max_norm=self.max_grad_norm
      )
    self.scaler.step(self.actor_optimizer)
    self.scaler.update()

    return actor_loss.detach(), -log_probs.mean().detach()

  def _update_alpha(self, next_log_probs: torch.Tensor) -> float:
    self.alpha_optimizer.zero_grad(set_to_none=True)
    with self._maybe_amp():
      alpha_loss = (
        -self.log_alpha.exp() * (next_log_probs.detach() + self.target_entropy)
      ).mean()
    self.scaler.scale(alpha_loss).backward()
    self.scaler.unscale_(self.alpha_optimizer)
    self.scaler.step(self.alpha_optimizer)
    self.scaler.update()
    return alpha_loss.item()

  def train_mode(self) -> None:
    self.actor.train()
    self.critic.train()
    self.obs_normalizer.train()
    self.critic_obs_normalizer.train()

    if self._compile:
      self._update_critic_fn = torch.compile(self._update_critic)
      self._update_actor_fn = torch.compile(self._update_actor)
      self._normalize_obs_fn = torch.compile(self.obs_normalizer.forward)
      self._normalize_critic_obs_fn = torch.compile(self.critic_obs_normalizer.forward)

  def eval_mode(self) -> None:
    self.actor.eval()
    self.critic.eval()
    self.obs_normalizer.eval()
    self.critic_obs_normalizer.eval()

  def save(self) -> dict:
    return {
      "actor_state_dict": self.actor.state_dict(),
      "critic_state_dict": self.critic.state_dict(),
      "critic_target_state_dict": self.critic_target.state_dict(),
      "log_alpha": self.log_alpha.detach().cpu(),
      "obs_normalizer_state_dict": self.obs_normalizer.state_dict(),
      "critic_obs_normalizer_state_dict": (self.critic_obs_normalizer.state_dict()),
      "q_optimizer_state_dict": self.q_optimizer.state_dict(),
      "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
      "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
      "scaler_state_dict": self.scaler.state_dict(),
      "global_step": self._global_step,
    }

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    if load_cfg is None:
      load_cfg = {
        "actor": True,
        "critic": True,
        "optimizer": True,
        "iteration": True,
      }

    if load_cfg.get("actor"):
      self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
    if (load_cfg.get("actor") or load_cfg.get("critic")) and (
      "obs_normalizer_state_dict" in loaded_dict
    ):
      self.obs_normalizer.load_state_dict(
        loaded_dict["obs_normalizer_state_dict"], strict=strict
      )
    if load_cfg.get("critic"):
      self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
      if "critic_target_state_dict" in loaded_dict:
        self.critic_target.load_state_dict(
          loaded_dict["critic_target_state_dict"], strict=strict
        )
      if "critic_obs_normalizer_state_dict" in loaded_dict:
        self.critic_obs_normalizer.load_state_dict(
          loaded_dict["critic_obs_normalizer_state_dict"],
          strict=strict,
        )
    if load_cfg.get("optimizer"):
      if "q_optimizer_state_dict" in loaded_dict:
        self.q_optimizer.load_state_dict(loaded_dict["q_optimizer_state_dict"])
      if "actor_optimizer_state_dict" in loaded_dict:
        self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
      if "alpha_optimizer_state_dict" in loaded_dict:
        self.alpha_optimizer.load_state_dict(loaded_dict["alpha_optimizer_state_dict"])
      if "scaler_state_dict" in loaded_dict:
        self.scaler.load_state_dict(loaded_dict["scaler_state_dict"])
    if "log_alpha" in loaded_dict:
      self.log_alpha.data.copy_(loaded_dict["log_alpha"].to(self.device))
    if "global_step" in loaded_dict:
      self._global_step = loaded_dict["global_step"]

    return load_cfg.get("iteration", False)

  def get_policy(self) -> Actor:
    return self.actor

  def broadcast_parameters(self) -> None:
    """Broadcast parameters across GPUs.

    Distributed training is not supported for FastSAC, so calling
    this method indicates a misconfiguration.
    """
    raise NotImplementedError(
      "Distributed training is not supported for FastSAC; "
      "broadcast_parameters was called in a distributed setting."
    )

  @property
  def intrinsic_rewards(self) -> None:
    """No intrinsic rewards in FastSAC (no RND)."""
    return None

  @staticmethod
  def _flatten_groups(obs: TensorDict, group_names: list[str]) -> torch.Tensor:
    """Flatten configured observation groups into a single tensor."""
    parts = [obs[group] for group in group_names]
    return torch.cat(parts, dim=-1)

  def _flatten_actor_obs(self, obs: TensorDict) -> torch.Tensor:
    return self._flatten_groups(obs, self._actor_obs_group_names)

  def _flatten_critic_obs(self, obs: TensorDict) -> torch.Tensor:
    return self._flatten_groups(obs, self._critic_obs_group_names)

  @staticmethod
  def construct_algorithm(
    obs: TensorDict, env: VecEnv, cfg: dict, device: str
  ) -> "FastSAC":
    """Construct FastSAC from a config dict.

    Mirrors ``PPO.construct_algorithm`` so that ``OnPolicyRunner``
    can instantiate FastSAC via the standard ``resolve_callable`` path.
    """
    alg_cfg = dict(cfg["algorithm"])
    alg_cfg.pop("class_name", None)

    # Resolve observation groups
    default_sets = ["actor"]
    if "critic" in obs.keys():
      default_sets.append("critic")
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

    actor_group_names = list(cfg["obs_groups"]["actor"])
    critic_group_names = list(cfg["obs_groups"].get("critic", actor_group_names))
    n_actor_obs = sum(obs[group].shape[-1] for group in actor_group_names)
    n_critic_obs = sum(obs[group].shape[-1] for group in critic_group_names)
    n_act = env.num_actions
    num_envs = env.num_envs

    # Extract algorithm hyper-parameters
    actor_hidden_dim = alg_cfg.pop("actor_hidden_dim", 512)
    critic_hidden_dim = alg_cfg.pop("critic_hidden_dim", 768)
    num_atoms = alg_cfg.pop("num_atoms", 101)
    v_min = alg_cfg.pop("v_min", -20.0)
    v_max = alg_cfg.pop("v_max", 20.0)
    num_q_networks = alg_cfg.pop("num_q_networks", 2)
    use_layer_norm = alg_cfg.pop("use_layer_norm", True)
    use_tanh = alg_cfg.pop("use_tanh", True)
    log_std_min = alg_cfg.pop("log_std_min", -5.0)
    log_std_max = alg_cfg.pop("log_std_max", 0.0)
    obs_normalization = alg_cfg.pop("obs_normalization", True)
    buffer_size = alg_cfg.pop("buffer_size", 1024)
    num_steps = alg_cfg.pop("num_steps", 1)

    actor = Actor(
      n_obs=n_actor_obs,
      n_act=n_act,
      hidden_dim=actor_hidden_dim,
      log_std_min=log_std_min,
      log_std_max=log_std_max,
      use_tanh=use_tanh,
      use_layer_norm=use_layer_norm,
      device=device,
    )

    critic = Critic(
      n_obs=n_critic_obs,
      n_act=n_act,
      num_atoms=num_atoms,
      v_min=v_min,
      v_max=v_max,
      hidden_dim=critic_hidden_dim,
      use_layer_norm=use_layer_norm,
      num_q_networks=num_q_networks,
      device=device,
    )

    critic_target = Critic(
      n_obs=n_critic_obs,
      n_act=n_act,
      num_atoms=num_atoms,
      v_min=v_min,
      v_max=v_max,
      hidden_dim=critic_hidden_dim,
      use_layer_norm=use_layer_norm,
      num_q_networks=num_q_networks,
      device=device,
    )
    critic_target.load_state_dict(critic.state_dict())

    replay_buffer = ReplayBuffer(
      n_env=num_envs,
      buffer_size=buffer_size,
      n_obs=n_actor_obs,
      n_critic_obs=n_critic_obs,
      n_act=n_act,
      n_steps=num_steps,
      gamma=alg_cfg.get("gamma", 0.97),
      device=device,
    )

    if obs_normalization:
      obs_norm: nn.Module = EmpiricalNormalization(shape=n_actor_obs, device=device)
      critic_obs_norm: nn.Module = EmpiricalNormalization(
        shape=n_critic_obs, device=device
      )
    else:
      obs_norm = NoOpNormalization()
      critic_obs_norm = NoOpNormalization()

    print(f"FastSAC Actor: {actor}")
    print(f"FastSAC Critic: {critic}")

    obs_dict = obs
    actor_obs_groups_dict = dict(
      (actor_group_name, cfg["obs_groups"][actor_group_name])
      for actor_group_name in actor_group_names
    )  

    # Build the algorithm instance
    alg = FastSAC(
      actor=actor,
      actor_obs_groups_dict = actor_obs_groups_dict,
      obs_dict = obs_dict,
      critic=critic,
      critic_target=critic_target,
      replay_buffer=replay_buffer,
      obs_normalizer=obs_norm,
      critic_obs_normalizer=critic_obs_norm,
      num_envs=num_envs,
      n_act=n_act,
      critic_lr=alg_cfg.pop("critic_lr", 3e-4),
      actor_lr=alg_cfg.pop("actor_lr", 3e-4),
      alpha_lr=alg_cfg.pop("alpha_lr", 3e-4),
      gamma=alg_cfg.pop("gamma", 0.97),
      tau=alg_cfg.pop("tau", 0.125),
      batch_size=alg_cfg.pop("batch_size", 8192),
      learning_starts=alg_cfg.pop("learning_starts", 10),
      policy_frequency=alg_cfg.pop("policy_frequency", 4),
      num_updates=alg_cfg.pop("num_updates", 8),
      target_entropy_ratio=alg_cfg.pop("target_entropy_ratio", 0.0),
      alpha_init=alg_cfg.pop("alpha_init", 0.001),
      use_autotune=alg_cfg.pop("use_autotune", True),
      max_grad_norm=alg_cfg.pop("max_grad_norm", 0.0),
      weight_decay=alg_cfg.pop("weight_decay", 0.001),
      use_amp=alg_cfg.pop("amp", True),
      amp_dtype=alg_cfg.pop("amp_dtype", "bf16"),
      compile=alg_cfg.pop("compile", True),
      num_steps=num_steps,
      device=device,
    )

    # Store observation group names for flattening TensorDicts.
    alg._actor_obs_group_names = actor_group_names
    alg._critic_obs_group_names = critic_group_names

    return alg