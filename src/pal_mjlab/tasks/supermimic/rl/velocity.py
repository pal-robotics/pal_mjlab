"""Post-training of SuperMimic's velocity interface (Appendix B and G, D56).

A velocity encoder maps the base twist command and the proprio history into the shared latent
of a frozen motion-tracking policy, whose decoder acts:

  collect   every env tracks a velocity command (``mdp/velocity.py``); the replay rows also
            hold the command source (held command, reference flag)
  fit       the world model, loaded from the motion run, keeps fitting online (Eq. 1)
  improve   the encoder by BPTT through the frozen world model and decoder on the
            velocity-tracking reward (Eq. 2, Table A.6, and D58's rest pose), plus 0.1 x the latent
            alignment to the motion encoder on reference commands (Appendix G); then the velocity
            critic (Eq. 3)

The velocity reward never updates the decoder (paper Section 6), so this is the paper's joint
training without the motion half.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, cast

import torch

from ..mdp import velocity as vel
from ..state import state_to_qpos_qvel
from .heads import AnalyticHeads
from .models import VelocityActor
from .runner import SimCollector, SuperMimicRunner, SuperMimicRunnerCfg
from .supermimic import (
  PolicyUpdateCfg,
  Rollout,
  SuperMimic,
  SuperMimicCfg,
  critic_value,
  start_history,
)
from .world_model import WorldModel

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from rsl_rl.models import MLPModel

_VELOCITY = {
  vel.track_lin_vel_exp: vel.lin_vel_exp,
  vel.track_yaw_rate_exp: vel.yaw_rate_exp,
  vel.torso_lean_l2: vel.torso_lean,
  vel.torso_rate_l2: vel.torso_rate,
}


class VelocityHeads(AnalyticHeads):
  """The analytic heads with the velocity interface's goal, reward and critic input.

  The goal is the command: a replay row's held command, or its reference command at the
  current frame. The learned failure head takes the upright root (default height,
  gravity-z -1) as its reference, so it sees the fall rule's two quantities (D39, D56).
  """

  CRITIC_TERMS = ["heading_state", "velocity_command", "sim_context"]

  def __init__(self, env: ManagerBasedRlEnv, compile_fk: bool = True) -> None:
    self.velocity_terms, self.rest_weight = [], 0.0
    super().__init__(env, compile_fk)
    cmd = cast(vel.VelocityCommand, env.command_manager.get_term("velocity"))
    self.reference_command = cmd.reference
    self.command_scale = cmd.command_scale
    height = env.scene[cmd.motion.cfg.entity_name].data.default_root_state[0, 2]
    self.upright = torch.tensor([float(height), -1.0], device=self.device)

  def _add_reward_term(self, env: ManagerBasedRlEnv, name: str, term) -> None:
    if term.func is vel.joint_deviation_l2:
      asset_cfg = env.reward_manager.get_term_cfg(name).params["asset_cfg"]
      self.rest_weight = term.weight
      self.rest_joints = torch.as_tensor(asset_cfg.joint_ids, device=self.device)
      default = env.scene[asset_cfg.name].data.default_joint_pos
      self.rest_pose = default[0, self.rest_joints]
      return None
    if term.func not in _VELOCITY:
      return super()._add_reward_term(env, name, term)
    params = {k: v for k, v in term.params.items() if k != "command_name"}
    if "body_name" in params:
      params = {"torso": self.body_names.index(params["body_name"])}
    self.velocity_terms.append((term.weight, partial(_VELOCITY[term.func], **params)))

  def command(self, row: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """(B, 3) the command of replay rows (``VelocityCommand.replay_row``) at ``frame``."""
    return torch.where(row[:, 3:] > 0.5, self.reference_command[frame], row[:, :3])

  def ref_root(self, frame: torch.Tensor) -> torch.Tensor:
    return self.upright.expand(len(frame), 2)

  def critic_input(self, state, pose, goal, context, clip, frame) -> torch.Tensor:
    """(ŝ, c, ξ): ``goal`` is the command."""
    return torch.cat([state, goal, context], dim=-1)

  def reward_terms(
    self, state, kin, ref, action, prev_action
  ) -> dict[str, torch.Tensor]:
    """Table A.6 for the command ``ref``, plus the rest pose (D58): ``tracking`` holds the
    task, torso and rest terms."""
    terms = super().reward_terms(state, kin, ref, action, prev_action)
    task = sum(w * f(kin.bodies, ref) for w, f in self.velocity_terms)
    if self.rest_weight:
      nj = (state.shape[-1] - 13) // 2
      q = state[:, 13 : 13 + nj][:, self.rest_joints]
      task = task + self.rest_weight * vel.joint_deviation(q, self.rest_pose)
    terms["tracking"] = task * self.reward_scale
    return terms


def make_velocity_step(heads: VelocityHeads):
  """``make_step`` for the velocity interface: the actor acts on the command, and the step
  also returns the latent alignment error to the motion encoder's latent of the same reference
  frame."""

  def step(
    world_model, actor, s, p, prev_action, history, ctx, clip, frame, row, noise
  ):
    command = heads.command(row, frame)
    x = heads.critic_input(s, p, command, ctx, clip, frame)
    proprio = heads.flatten_history(history)
    proprio_n = actor.normalize(proprio)
    z = actor.latent(command, proprio_n)
    # bf16 autocast: keep the state and reward math in fp32
    action = actor.decode(z, proprio_n).float()
    with torch.no_grad():  # stop-gradient on the motion latent
      root_quat = state_to_qpos_qvel(s, p)[0][:, 3:7]
      z_motion = actor.motion_latent(heads.goal(root_quat, clip, frame), proprio)
    align = (z.float() - z_motion.float()).square().mean(-1)
    s_next, p_next = world_model.step(s, p, action, ctx)
    kin = heads.kinematics(s_next, p_next)
    terms = heads.reward_terms(s_next, kin, command, action, prev_action)
    logit = world_model.termination_logit(s_next, heads.ref_root(frame)).float()
    obs = heads.observation(s_next, kin, ctx, action)
    history = heads.push_history(history, obs, noise)
    return x, action, s_next, p_next, terms, torch.sigmoid(logit), history, align

  return step


def imagine_velocity(
  heads: VelocityHeads,
  world_model: WorldModel,
  actor: VelocityActor,
  critic: MLPModel,
  start: dict[str, torch.Tensor],
  cfg: PolicyUpdateCfg,
  generator: torch.Generator | None = None,
  step=None,
) -> tuple[Rollout, torch.Tensor]:
  """``imagine`` for the velocity interface: the episode ends only at a predicted fall.

  Returns the rollout and the latent alignment loss: the squared latent error averaged over
  the steps of the rollouts with a reference command.
  """
  step = step or make_velocity_step(heads)
  b, gamma = start["state"].shape[0], cfg.gamma
  ctx, clip, frame = start["context"], start["clip"], start["frame"]
  row = start["command"]  # the held command and the reference flag
  history, prev_action = start_history(heads, start, generator)
  s, p = start["state"], start["pose"]
  c = torch.ones(b, device=s.device)
  objective = torch.zeros(b, device=s.device)
  out = {k: [] for k in ("input", "reward", "head", "align")}
  for t in range(cfg.horizon):
    noise = heads.sample_noise(b, generator)
    x, action, s, p, terms, head, history, align = step(
      world_model, actor, s, p, prev_action, history, ctx, clip, frame, row, noise
    )
    reward = sum(terms.values())
    objective = objective + gamma**t * c * reward
    c = c * (1.0 - head)
    for key, value in (
      ("input", x),
      ("reward", reward),
      ("head", head),
      ("align", align),
    ):
      out[key].append(value)
    prev_action = action
    frame = heads.next_frame(clip, frame)
  out["input"].append(
    heads.critic_input(s, p, heads.command(row, frame), ctx, clip, frame)
  )

  inputs = torch.stack(out["input"], dim=1)
  if cfg.use_critic:
    value_end = critic_value(critic, inputs[:, -1]).float()  # gradient flows into ŝ_H
    objective = objective + gamma**cfg.horizon * c * value_end
  with torch.no_grad():
    value = critic_value(critic, inputs.detach().float())
  reference = (row[:, 3] > 0.5).float()
  align = torch.stack(out["align"], dim=1).mean(1)
  align = (align * reference).sum() / reference.sum().clamp_min(1.0)

  def stack(name):
    return torch.stack(out[name], dim=1).detach().float()

  head = stack("head")
  roll = Rollout(
    objective=objective,
    reward=stack("reward"),
    cont=1.0 - head,
    alive=torch.ones_like(head),
    clip_end=torch.zeros_like(head),
    head_prob=head,
    critic_input=inputs.detach().float(),
    value=value,
  )
  return roll, align


@dataclass
class VelocityCfg(SuperMimicCfg):
  encoder_dims: tuple[int, ...] = (256, 256)
  """The velocity encoder (Table A.4)."""
  align_weight: float = 0.1
  """Latent alignment to the motion encoder on reference commands (Appendix G)."""


class VelocitySuperMimic(SuperMimic):
  """SuperMimic's updates with the velocity encoder as the only trained part of the actor."""

  cfg_cls = VelocityCfg
  cfg: VelocityCfg
  heads: VelocityHeads
  actor: VelocityActor

  def __init__(
    self,
    obs,
    heads: VelocityHeads,
    num_actions: int,
    state_dim: int,
    context_dim: int,
    cfg: VelocityCfg,
    device: str,
    compile: bool = False,
  ) -> None:
    super().__init__(
      obs, heads, num_actions, state_dim, context_dim, cfg, device, compile
    )
    pc = cfg.policy
    self.actor = VelocityActor(self.actor, heads.command_scale, cfg.encoder_dims)
    self.actor.to(device)
    self.actor_optimizer = torch.optim.Adam(
      self.actor.encoder.parameters(), lr=pc.learning_rate, betas=pc.betas
    )
    self._optimizers = (
      (self.wm_optimizer, cfg.world_model.learning_rate),
      (self.actor_optimizer, pc.learning_rate),
      (self.critic_optimizer, cfg.critic.learning_rate),
    )
    self._step = make_velocity_step(heads)
    if compile:
      self._step = torch.compile(self._step, dynamic=False)

  def load_motion(self, path: str) -> None:
    """The motion policy (frozen) and the world model with its optimizer, from a SuperMimic
    checkpoint; the velocity critic starts from scratch."""
    path = os.path.expanduser(path)
    alg = torch.load(path, map_location=self.device, weights_only=False)["alg"]
    self.actor.motion.load_state_dict(alg["actor"])
    self.world_model.load_state_dict(alg["world_model"]["state_dict"])
    self.wm_optimizer.load_state_dict(alg["wm_optimizer"])

  def _imagine(self, start, generator) -> tuple[Rollout, torch.Tensor]:
    return imagine_velocity(
      self.heads, self.world_model, self.actor, self.critic, start, self.cfg.policy,
      generator, self._step,
    )  # fmt: skip

  def imagine(self, start, generator=None) -> Rollout:
    return self._imagine(start, generator)[0]

  def _imagined_loss(self, start, generator):
    roll, align = self._imagine(start, generator)
    loss = -roll.objective.mean() + self.cfg.align_weight * align
    return roll, loss, {"align": float(align.detach())}


class VelocityCollector(SimCollector):
  def __init__(self, env: ManagerBasedRlEnv) -> None:
    super().__init__(env)
    self.velocity = cast(vel.VelocityCommand, env.command_manager.get_term("velocity"))

  def command(self) -> torch.Tensor:
    return self.velocity.replay_row()


@dataclass
class VelocityRunnerCfg(SuperMimicRunnerCfg):
  class_name: str = "VelocityRunner"
  algorithm: VelocityCfg = field(default_factory=VelocityCfg)
  motion_checkpoint: str = ""
  """The SuperMimic checkpoint (model_N.pt) whose motion policy and world model the velocity
  encoder is post-trained on; required for a new run."""


class VelocityRunner(SuperMimicRunner):
  collector_cls = VelocityCollector
  heads_cls = VelocityHeads
  algorithm_cls = VelocitySuperMimic
  alg: VelocitySuperMimic

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    if self.current_learning_iteration == 0:  # a new run, not a resumed one
      if not self.cfg["motion_checkpoint"]:
        raise ValueError("Set --agent.motion-checkpoint: the motion run's model_N.pt.")
      self.alg.load_motion(self.cfg["motion_checkpoint"])
    super().learn(num_learning_iterations, init_at_random_ep_len)
