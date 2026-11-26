"""SuperMimic's contract with a task, checked on a simulator rollout.

Single-definition terms (``imagination.py``) make the env and imagination run the same
functions; what can still differ is their inputs. ``check_contract`` drives the env with
random actions (episodes end and reset) and, at every step, compares what imagination
would compute with what the env computed:

  reward/<term>     each weighted reward term (as the reward manager weighed it) vs its
                    single definition on the step's resulting state rebuilt as a
                    prediction (from the heading state and pose), with the action and the
                    command state the step targeted
  obs/<group>/<term>  each observation term (noise-free) vs its single definition on the
                    same rebuilt state with the advanced command state
  command/advance   ``advance(c_t)`` vs the command state after the step (envs not reset)
  command/done      1 where ``done(c_t)`` and the env's episode-end terms disagree (if
                    given), or where ``done(c_t)`` holds and the episode did not end
  termination/<term>  1 where a failure term with a single definition, on the rebuilt
                    state, disagrees with the env's termination (above 0.5 vs fired)

A task's test calls it and bounds the errors; a term imagination cannot compute (no single
definition) is listed under ``missing`` rather than silently skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from .command_model import command_model_of
from .imagination import batch_source, imagined_fn, memory_init, reward_memory, term_memory
from .names import imagination_cfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass
class ContractReport:
  errors: dict[str, float] = field(default_factory=dict)
  """Max absolute error per check over the rollout."""
  missing: list[str] = field(default_factory=list)
  """Terms without a single definition (imagination cannot compute them)."""
  steps: int = 0
  resets: int = 0
  resampled: int = 0
  """Env steps left out because the env resampled the command (``resampled_commands``)."""
  """Episodes that ended during the rollout (the checks covered resets)."""

  def worst(self, prefix: str = "") -> tuple[str, float]:
    """The largest error among the checks starting with ``prefix``."""
    items = [(k, v) for k, v in self.errors.items() if k.startswith(prefix)]
    return max(items, key=lambda kv: kv[1]) if items else ("", 0.0)

  def _add(self, name: str, err: torch.Tensor) -> None:
    value = float(err.abs().max()) if err.numel() else 0.0
    self.errors[name] = max(self.errors.get(name, 0.0), value)


@torch.no_grad()
def check_contract(
  env: ManagerBasedRlEnv,
  steps: int = 60,
  seed: int = 0,
  action_std: float = 2.0,
  end_terms: tuple[str, ...] = (),
  resampled_commands: bool = False,
) -> ContractReport:
  """Roll the env out with Gaussian actions (std ``action_std``) for ``steps`` steps,
  resetting finished episodes, and report the contract's errors. ``end_terms``: the
  termination terms that end an episode exactly when ``CommandModel.done`` says so (e.g. a
  motion's end); without them only "done implies the episode ended" is checked.
  ``resampled_commands``: the env may resample the command (e.g. a new velocity every few
  seconds), which imagination does not foresee; envs whose command changed otherwise than
  ``advance`` says are left out of the observation and command checks for that step
  (counted in ``report.resampled``)."""
  names = imagination_cfg(env)
  source = batch_source(env)
  model = command_model_of(env, names.command)
  report = ContractReport()

  rewards = {}
  for name in env.reward_manager.active_terms:
    cfg = env.reward_manager.get_term_cfg(name)
    if cfg.weight == 0.0:
      continue
    fn = imagined_fn(cfg.func)
    if fn is None:
      report.missing.append(f"reward/{name}")
    else:
      rewards[name] = (fn, cfg)
  failures = {}
  for name in names.failure_terms:
    fn = imagined_fn(env.termination_manager.get_term_cfg(name).func)
    if fn is not None:
      failures[name] = (fn, env.termination_manager.get_term_cfg(name))
  observations = {}
  om = env.observation_manager
  for group in om.active_terms:
    for name in om.active_terms[group]:
      cfg = om.get_term_cfg(group, name)
      fn = imagined_fn(cfg.func)
      if fn is None:
        report.missing.append(f"obs/{group}/{name}")
      else:
        observations[f"{group}/{name}"] = (fn, cfg)

  gen = torch.Generator(device=env.device).manual_seed(seed)
  rm, tm = env.reward_manager, env.termination_manager
  env.reset()
  for _ in range(steps):
    c = model.state()  # the command state the step targets
    before = source.sim()
    memory = reward_memory(env)  # what the stateful reward terms start from
    action = action_std * torch.randn(
      env.num_envs, env.action_manager.total_action_dim, device=env.device, generator=gen
    )
    _, _, terminated, truncated, _ = env.step(action)
    after = source.sim()  # the resulting state, advanced command state
    ended = terminated | truncated

    # Rewards: the resulting state as a prediction, the step's action, its target.
    scored = source.imagined(
      after.state, after.pose, c, after.action, before.action, after.context
    )
    for name, (fn, cfg) in rewards.items():
      i = rm.active_terms.index(name)
      if memory_init(cfg.func) is None:
        value = fn(scored, **cfg.params)
      else:
        value, _ = fn(scored, term_memory(memory, name), **cfg.params)
      report._add(f"reward/{name}", value * cfg.weight - rm._step_reward[:, i])
    # Failures with a single definition: the imagined probability vs the env's verdict.
    for name, (fn, cfg) in failures.items():
      imagined = fn(scored, **cfg.params) > 0.5
      report._add(f"termination/{name}", (imagined != tm.get_term(name)).double())
    # Observations: the same state with the advanced command state.
    c_next = model.advance(c)
    kept = torch.ones_like(ended)
    if resampled_commands:
      for k, v in c_next.items():
        kept &= (v == after.c[k]).reshape(len(v), -1).all(-1)
      report.resampled += int((~kept).sum())
    observed = scored._replace(c=c_next, memo={})
    for key, (fn, cfg) in observations.items():
      env_value = cfg.func(env, **cfg.params)
      report._add(f"obs/{key}", (fn(observed, **cfg.params) - env_value)[kept])
    # The command model: how c moves, and when it ends the episode.
    live = ~ended & kept
    for k, v in c_next.items():
      report._add(f"command/advance/{k}", (v[live] - after.c[k][live]).double())
    done = model.done(c)
    if end_terms:
      env_end = torch.zeros_like(done)
      for term in end_terms:
        env_end |= tm.get_term(term)
      report._add("command/done", (done != env_end).double())
    else:
      report._add("command/done", (done & ~ended).double())

    report.steps += 1
    if ended.any():
      report.resets += int(ended.sum())
      env.reset(env_ids=ended.nonzero().squeeze(-1))
  return report
