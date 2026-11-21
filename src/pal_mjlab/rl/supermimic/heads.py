"""Analytic heads of SuperMimic's world model (paper Section 3.2).

The paper's world model is a learned dynamics model plus reward, observation and
termination heads, "each either learned or computed analytically". Here the reward r̂, the
noise-free observation ō and the goal g are computed from a predicted state with forward
kinematics and the training env's own definitions and parameters (read from its config);
the termination d̂ is the learned head of ``WorldModel``. The command's own quantities
(goal, episode end, termination reference, critic features) are the ``CommandModel``'s.

States are heading states ŝ (``state.py``) with world poses p = (x, y, yaw) in the env's
local frame; ξ is the sim context; c is the command state (``CommandModel``). Only the
env terms SuperMimic uses are supported; anything else raises, so
imagination never silently optimizes a different objective than the simulator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.noise import UniformNoiseCfg

from pal_mjlab.utils.math.kinematics import Kinematics

from .command_model import CommandModel
from .imagination import Batch, batch_source, imagined_fn
from .names import DEFAULT_CRITIC_GROUP, ImaginationCfg, imagination_cfg
from .state import state_to_qpos_qvel

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
    ``critic_group``: the critic's observation group. Observation terms are recognized by
    their function, so their names are free."""
    names = names or imagination_cfg(env)
    # TODO(M2): the heads still use the motion-tracking task's terms; the task boundary
    # removes this. Imported here, not at module level: importing pal_mjlab.tasks loads
    # every task, the SuperMimic one included, and that one imports this package.
    from pal_mjlab.tasks.supermimic.mdp import observations as sm_obs

    self.command_model = command_model
    self.source = batch_source(env)
    """Builds the imagined batches (``imagination.py``), with the env's FK and layout."""
    if compile_fk:
      self.source.fk_fn = torch.compile(self.source.fk.forward)
    cfg, self.device = env.cfg, env.device

    # Proprioception (Table A.1): every term of the proprio group, by its single
    # definition, with the env's params and noise; history of h frames per term.
    om = env.observation_manager
    proprio = names.proprio_group
    group = cfg.observations[proprio]
    self.obs_terms: list[str] = list(group.terms)
    self.history = group.history_length or 0
    self.obs: dict[str, tuple] = {}
    """name: (single definition, resolved params)."""
    self.noise: dict[str, tuple[float, float]] = {}
    missing = []
    for name, term in group.terms.items():
      fn = imagined_fn(term.func)
      if fn is None:
        missing.append(f"{name} ({term.func.__module__}.{term.func.__qualname__})")
        continue
      self.obs[name] = (fn, om.get_term_cfg(proprio, name).params)  # entities resolved
      if group.enable_corruption and term.noise is not None:
        if not (isinstance(term.noise, UniformNoiseCfg) and term.noise.operation == "add"):
          raise NotImplementedError(f"proprio term {name!r}: only additive uniform noise")
        self.noise[name] = (float(term.noise.n_min), float(term.noise.n_max))
    if missing:
      raise NotImplementedError(
        f"Observation terms of {proprio!r} without a single definition, so imagination "
        f"cannot compute them: {', '.join(missing)}. Write them on a Batch with "
        "@imaginable (rl/supermimic/imagination.py)."
      )
    term_dims = dict(
      zip(om.active_terms[proprio], om.group_obs_term_dim[proprio], strict=True)
    )
    self.obs_dims = {
      n: term_dims[n][-1] // max(self.history, 1) for n in self.obs_terms
    }
    critic = cfg.observations[critic_group].terms

    # Reward (Table A.2): every weighted env term, by its single definition, with the
    # env's weights, params and dt scaling.
    self.rewards: dict[str, tuple] = {}
    """name: (single definition, resolved params, weight)."""
    missing = []
    for name, term in cfg.rewards.items():
      if term is None or term.weight == 0.0:
        continue
      fn = imagined_fn(term.func)
      if fn is None:
        missing.append(f"{name} ({term.func.__module__}.{term.func.__qualname__})")
        continue
      params = env.reward_manager.get_term_cfg(name).params  # entities resolved
      self.rewards[name] = (fn, params, term.weight)
    if missing:
      raise NotImplementedError(
        "Reward terms without a single definition, so imagination cannot compute them: "
        f"{', '.join(missing)}. Write them on a Batch with @imaginable "
        "(rl/supermimic/imagination.py)."
      )
    self.reward_scale = env.step_dt if cfg.scale_rewards_by_dt else 1.0

    # The critic's input layout (``critic_input``), recognized by the terms' functions; the
    # reference-reading terms must read this command.
    self.critic_terms = list(critic)
    layout = [sm_obs.robot_heading_state, sm_obs.motion_goal, sm_obs.sim_context,
              sm_obs.motion_root_pose_error, sm_obs.motion_time_left]  # fmt: skip
    if [t.func for t in critic.values()] != layout:
      raise ValueError(
        f"The critic group {critic_group!r} must hold, in order: heading state, motion "
        f"goal, sim context, root pose error and time left; got {self.critic_terms}."
      )
    for name, term in critic.items():
      if term.params.get("command_name", names.command) != names.command:
        raise ValueError(f"critic term {name!r} reads another command than {names.command!r}")

  # Kinematics -------------------------------------------------------------------

  def kinematics(self, state: torch.Tensor, pose: torch.Tensor) -> Kinematics:
    """The batch's FK of a (predicted) state."""
    return self.source.fk_fn(*state_to_qpos_qvel(state, pose))

  # Observation ō and sensor noise η ------------------------------------------------

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

  # Reward r̂ ---------------------------------------------------------------------

  def reward_terms(self, b: Batch) -> dict[str, torch.Tensor]:
    """Each env reward term on an imagined batch: weight * value (* dt), as the env's
    reward manager scales it. ``b`` holds the step's resulting state, its action and the
    command state the step targeted."""
    return {
      name: fn(b, **params) * (weight * self.reward_scale)
      for name, (fn, params, weight) in self.rewards.items()
    }
