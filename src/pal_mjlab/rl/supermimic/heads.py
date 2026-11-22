"""Analytic heads of SuperMimic's world model (paper Section 3.2).

The paper's world model is a learned dynamics model plus reward, observation and
termination heads, "each either learned or computed analytically". Here the reward r̂, the
noise-free observation ō and the goal g are computed from a predicted state with forward
kinematics and the training env's own definitions and parameters (read from its config);
the termination d̂ is the learned head of ``WorldModel``. Every term is the env's own,
written once on a ``Batch`` (``imagination.py``): the heads name none of them.

States are heading states ŝ (``state.py``) with world poses p = (x, y, yaw) in the env's
local frame; ξ is the sim context; c is the command state (``CommandModel``). A term
without a single definition raises, so imagination never silently optimizes a different
objective than the simulator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.noise import UniformNoiseCfg

from .command_model import CommandModel
from .imagination import Batch, Memory, batch_source, imagined_fn, memory_init, term_memory
from .names import DEFAULT_CRITIC_GROUP, ImaginationCfg, imagination_cfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class AnalyticHeads:
  def __init__(
    self,
    env: ManagerBasedRlEnv,
    command_model: CommandModel,
    compile_fk: bool = True,
    names: ImaginationCfg | None = None,
    critic_group: str = DEFAULT_CRITIC_GROUP,
  ) -> None:
    """``command_model``: the task's command twin;
    ``names``: the env's term and group names (the env's ``imagination`` if None);
    ``critic_group``: the critic's observation group. Terms are found by their single
    definition (``imagined_fn``), so their names are free."""
    names = names or imagination_cfg(env)
    self.command_model = command_model
    self.source = batch_source(env)
    """Builds the imagined batches (``imagination.py``), with the env's FK and layout."""
    if compile_fk:
      self.source.fk_fn = torch.compile(self.source.fk.forward)
    cfg, self.device = env.cfg, env.device
    self.noise: dict[str, tuple[float, float]] = {}
    """Proprio name: additive uniform sensor noise (low, high)."""

    # Observations: every term of the proprio, goal and critic groups, by its single
    # definition, with the env's params. Proprioception (Table A.1) also gets the env's
    # sensor noise and a history of h frames per term; the goal and the critic input are
    # their groups' terms concatenated, as the env's observation manager does.
    om = env.observation_manager
    proprio = names.proprio_group
    group = cfg.observations[proprio]
    self.obs_terms: list[str] = list(group.terms)
    self.history = group.history_length or 0
    self.obs = self._group_terms(env, proprio, noisy=True)
    """Proprio name: (single definition, resolved params)."""
    self.goal_terms = self._group_terms(env, names.goal_group)
    self.critic_terms = self._group_terms(env, critic_group)
    term_dims = dict(
      zip(om.active_terms[proprio], om.group_obs_term_dim[proprio], strict=True)
    )
    self.obs_dims = {
      n: term_dims[n][-1] // max(self.history, 1) for n in self.obs_terms
    }

    # Reward (Table A.2): every weighted env term, by its single definition, with the
    # env's weights, params and dt scaling.
    self.rewards: dict[str, tuple] = {}
    """name: (single definition, resolved params, weight, memory init or None)."""
    missing = []
    for name, term in cfg.rewards.items():
      if term is None or term.weight == 0.0:
        continue
      fn = imagined_fn(term.func)
      if fn is None:
        missing.append(f"{name} ({term.func.__module__}.{term.func.__qualname__})")
        continue
      cfg_r = env.reward_manager.get_term_cfg(name)  # entities resolved
      self.rewards[name] = (fn, cfg_r.params, term.weight, memory_init(cfg_r.func))
    if missing:
      raise NotImplementedError(
        "Reward terms without a single definition, so imagination cannot compute them: "
        f"{', '.join(missing)}. Write them on a Batch with @imaginable "
        "(rl/supermimic/imagination.py)."
      )
    self.reward_scale = env.step_dt if cfg.scale_rewards_by_dt else 1.0

    # Failures (``ImaginationCfg.failure_terms``) with a single definition end imagined
    # episodes as the simulator does; the world model's learned head covers the others.
    tm = env.termination_manager
    self.failures: dict[str, tuple] = {}
    """name: (single definition, resolved params) of the analytic failure terms."""
    for name in names.failure_terms:
      cfg_t = tm.get_term_cfg(name)
      fn = imagined_fn(cfg_t.func)
      if fn is not None:
        self.failures[name] = (fn, cfg_t.params)
    self.all_failures_analytic = len(self.failures) == len(names.failure_terms)


  def _group_terms(self, env: ManagerBasedRlEnv, group_name: str, noisy: bool = False) -> dict:
    """``{term: (single definition, resolved params)}`` of an observation group, raising
    once for every term imagination cannot compute. ``noisy``: the group's additive
    uniform noise is imagined too (``self.noise``); otherwise the group must be noise-free
    and without history, as imagination computes it once per step."""
    group = env.cfg.observations[group_name]
    om = env.observation_manager
    terms, missing = {}, []
    for name, term in group.terms.items():
      fn = imagined_fn(term.func)
      if fn is None:
        missing.append(f"{name} ({term.func.__module__}.{term.func.__qualname__})")
        continue
      if memory_init(term.func) is not None:
        raise NotImplementedError(f"{group_name}/{name}: memory is for reward terms only")
      terms[name] = (fn, om.get_term_cfg(group_name, name).params)  # entities resolved
      corrupted = group.enable_corruption and term.noise is not None
      if noisy and corrupted:
        if not (isinstance(term.noise, UniformNoiseCfg) and term.noise.operation == "add"):
          raise NotImplementedError(f"{group_name}/{name}: only additive uniform noise")
        self.noise[name] = (float(term.noise.n_min), float(term.noise.n_max))
      elif not noisy and (corrupted or term.history_length or group.history_length):
        raise NotImplementedError(
          f"{group_name}/{name}: noise and history are only imagined for proprioception"
        )
    if missing:
      raise NotImplementedError(
        f"Observation terms of {group_name!r} without a single definition, so "
        f"imagination cannot compute them: {', '.join(missing)}. Write them on a Batch "
        "with @imaginable (rl/supermimic/imagination.py)."
      )
    return terms

  # Observation ō and sensor noise η ------------------------------------------------

  def goal(self, b: Batch) -> torch.Tensor:
    """The policy's goal: the goal group's terms on a batch, concatenated."""
    return torch.cat([fn(b, **p) for fn, p in self.goal_terms.values()], dim=-1)

  def critic_input(self, b: Batch) -> torch.Tensor:
    """The critic's input: the critic group's terms on a batch, concatenated."""
    return torch.cat([fn(b, **p) for fn, p in self.critic_terms.values()], dim=-1)

  def observation(self, b: Batch) -> dict[str, torch.Tensor]:
    """Noise-free proprioceptive frame ō per term, (B, dim) each, on a batch: the state,
    its last action and the command state after the step (as the env observes them)."""
    return {name: fn(b, **params) for name, (fn, params) in self.obs.items()}

  def add_noise(
    self, frame: dict[str, torch.Tensor], generator: torch.Generator | None = None
  ) -> dict[str, torch.Tensor]:
    noise = self.sample_noise(len(next(iter(frame.values()))), generator)
    return {n: x + noise[n] if n in noise else x for n, x in frame.items()}

  def sample_noise(
    self, batch: int, generator: torch.Generator | None = None
  ) -> dict[str, torch.Tensor]:
    """Additive sensor noise η per noisy term, (batch, dim) each."""
    out = {}
    for name, (lo, hi) in self.noise.items():
      shape = (batch, self.obs_dims[name])
      u = torch.rand(shape, device=self.device, generator=generator)
      out[name] = lo + (hi - lo) * u
    return out

  def flatten_history(self, history: dict[str, torch.Tensor]) -> torch.Tensor:
    """(B, h, dim) per term -> the env's flat layout: per-term history, oldest first."""
    return torch.cat([history[n].flatten(1) for n in self.obs_terms], dim=-1)

  def failure_probability(
    self, b: Batch, learned: torch.Tensor
  ) -> torch.Tensor:
    """The probability that the step to ``b`` ended the episode in a failure: the
    analytic failure terms on ``b`` combined with the learned head's ``learned`` (for the
    failures without a single definition; alone if there is none)."""
    if not self.failures:
      return learned
    alive = torch.ones_like(learned)
    for fn, params in self.failures.values():
      alive = alive * (1.0 - fn(b, **params))
    if not self.all_failures_analytic:
      alive = alive * (1.0 - learned)
    return 1.0 - alive

  # Reward r̂ ---------------------------------------------------------------------

  def reward_terms(
    self, b: Batch, memory: Memory | None = None
  ) -> tuple[dict[str, torch.Tensor], Memory]:
    """Each env reward term on an imagined batch: weight * value (* dt), as the env's
    reward manager scales it, and the stateful terms' next memory. ``b`` holds the
    step's resulting state, its action and the command state the step targeted;
    ``memory`` the stateful terms' memory before the step (zero if missing)."""
    memory = dict(memory or {})
    terms = {}
    for name, (fn, params, weight, init) in self.rewards.items():
      if init is None:
        value = fn(b, **params)
      else:
        mem = term_memory(memory, name) or init(b, **params)
        value, mem = fn(b, mem, **params)
        memory.update({f"{name}/{k}": v for k, v in mem.items()})
      terms[name] = value * (weight * self.reward_scale)
    return terms, memory
