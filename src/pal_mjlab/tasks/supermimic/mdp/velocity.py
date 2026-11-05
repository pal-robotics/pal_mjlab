"""SuperMimic's velocity interface (Appendix G, Table A.6): base twist command, reward and fall.

The command c = (vx, vy, ωz) is the root's planar velocity in its heading frame and its world
yaw rate. Each env draws a command source at every reset and every 5 s: with probability 0.5
the reference command, read from its clip at every step (the centered 25-frame mean of the
reference root's twist), otherwise a command held fixed, uniform over the ranges and zero
(standing) with probability 0.1. Episodes still start at frames of the motion library.

The reward and fall terms are pure functions of the tracked bodies' ``BodyState`` and the
command, shared by the env terms below and imagination (``rl/velocity.py``), like
``tracking.py`` for motion tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers import CommandTerm, CommandTermCfg

from ..kinematics import quat_rotate, quat_rotate_inv, yaw_quat
from ..library import MotionLibrary
from .commands import MotionLibraryCommand
from .tracking import BodyState, projected_gravity_z

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg

__all__ = [
  "VelocityCommand",
  "VelocityCommandCfg",
  "velocity_command",
  "track_lin_vel_exp",
  "track_yaw_rate_exp",
  "torso_lean_l2",
  "torso_rate_l2",
  "joint_deviation_l2",
  "fall",
]


# Pure terms: body 0 is the root ----------------------------------------------------


def reference_twist(library: MotionLibrary, window: int = 25) -> torch.Tensor:
  """(frames, 3) the reference command at every library frame: the centered ``window``-frame
  mean, within the clip, of the reference root's planar velocity in its heading frame and its
  world yaw rate."""
  quat, vel = library.qpos[:, 3:7], library.qvel
  twist = torch.cat(
    [
      quat_rotate_inv(yaw_quat(quat), vel[:, :3])[:, :2],
      quat_rotate(quat, vel[:, 3:6])[
        :, 2:
      ],  # the free joint's angular velocity: body frame
    ],
    dim=-1,
  ).double()
  csum = torch.cat([twist.new_zeros(1, 3), twist.cumsum(0)])
  order = library.clip_start.argsort()
  clip = torch.repeat_interleave(order, library.clip_length[order])  # per frame
  frame = torch.arange(library.num_frames, device=quat.device)
  lo = torch.maximum(frame - window // 2, library.clip_start[clip])
  hi = torch.minimum(frame + window // 2, library.clip_end[clip] - 1)
  return ((csum[hi + 1] - csum[lo]) / (hi - lo + 1)[:, None]).float()


def _in_heading(bodies: BodyState, v: torch.Tensor) -> torch.Tensor:
  """World vectors ``v`` in the root's heading frame."""
  return quat_rotate_inv(yaw_quat(bodies.quat[:, 0]), v)


def twist(bodies: BodyState) -> torch.Tensor:
  """(B, 3) the root's planar velocity in its heading frame and its world yaw rate."""
  lin = _in_heading(bodies, bodies.lin_vel[:, 0])[:, :2]
  return torch.cat([lin, bodies.ang_vel[:, 0, 2:]], dim=-1)


def lin_vel_exp(bodies: BodyState, command: torch.Tensor, std: float) -> torch.Tensor:
  err = (twist(bodies)[:, :2] - command[:, :2]).square().sum(-1)
  return torch.exp(-err / std**2)


def yaw_rate_exp(bodies: BodyState, command: torch.Tensor, std: float) -> torch.Tensor:
  return torch.exp(-(bodies.ang_vel[:, 0, 2] - command[:, 2]).square() / std**2)


def torso_lean(bodies: BodyState, command: torch.Tensor, torso: int) -> torch.Tensor:
  """sin² of the torso's sideways lean: the lateral part of its z-axis in the heading frame
  (forward lean is free)."""
  up = bodies.pos.new_tensor([0.0, 0.0, 1.0]).expand_as(bodies.pos[:, torso])
  return _in_heading(bodies, quat_rotate(bodies.quat[:, torso], up))[:, 1].square()


def torso_rate(bodies: BodyState, command: torch.Tensor, torso: int) -> torch.Tensor:
  """The torso's squared roll and pitch rates in the heading frame."""
  return _in_heading(bodies, bodies.ang_vel[:, torso])[:, :2].square().sum(-1)


def joint_deviation(q: torch.Tensor, default: torch.Tensor) -> torch.Tensor:
  """Sum over joints of the squared distance from the default pose."""
  return (q - default).square().sum(-1)


def root_fall(bodies: BodyState, min_height: float, max_tilt: float) -> torch.Tensor:
  """The root below ``min_height`` or 1 + u_z above ``max_tilt`` (u: gravity in the root
  frame; 0 upright, 2 upside down)."""
  tilt = 1.0 + projected_gravity_z(bodies.quat[:, 0])
  return (bodies.pos[:, 0, 2] < min_height) | (tilt > max_tilt)


# Command ------------------------------------------------------------------------------


class VelocityCommand(CommandTerm):
  cfg: VelocityCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: VelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    r = cfg.ranges
    self._ranges = torch.tensor(
      [r.lin_vel_x, r.lin_vel_y, r.ang_vel_z], device=self.device
    )
    self.held = torch.zeros(self.num_envs, 3, device=self.device)
    self.from_reference = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  @cached_property
  def motion(self) -> MotionLibraryCommand:
    """The motion command, for the reference's clip and frame (looked up on first use: the
    command manager exists once all its terms are built)."""
    term = self._env.command_manager.get_term(self.cfg.motion_command)
    return cast(MotionLibraryCommand, term)

  @cached_property
  def reference(self) -> torch.Tensor:
    """(library frames, 3) the reference command per frame."""
    return reference_twist(self.motion.library, self.cfg.window)

  @property
  def command_scale(self) -> torch.Tensor:
    """(3,) the largest held command per component: the policy's normalization."""
    return self._ranges.abs().amax(-1)

  @property
  def command(self) -> torch.Tensor:
    ref = self.reference[self.motion.frame]
    return torch.where(self.from_reference[:, None], ref, self.held)

  @property
  def robot_bodies(self) -> BodyState:
    return self.motion.robot_bodies

  def body_index(self, name: str) -> int:
    return self.motion.cfg.body_names.index(name)

  def replay_row(self) -> torch.Tensor:
    """(E, 4) what fixes the command along a rollout: the held command and the reference
    flag (``rl/velocity.py``)."""
    return torch.cat([self.held, self.from_reference[:, None].float()], dim=-1)

  def _update_metrics(self) -> None:
    pass  # the reward manager logs each tracking term

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n, lo, hi = len(env_ids), self._ranges[:, 0], self._ranges[:, 1]
    held = lo + (hi - lo) * torch.rand(n, 3, device=self.device)
    standing = torch.rand(n, 1, device=self.device) < self.cfg.standing_prob
    self.held[env_ids] = torch.where(standing, 0.0, held)
    reference = torch.rand(n, device=self.device) < self.cfg.reference_prob
    self.from_reference[env_ids] = reference

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    pass  # read from the motion command's frame


@dataclass(kw_only=True)
class VelocityCommandCfg(CommandTermCfg):
  @dataclass
  class Ranges:
    lin_vel_x: tuple[float, float] = (-1.5, 2.0)
    lin_vel_y: tuple[float, float] = (-1.32, 1.32)
    ang_vel_z: tuple[float, float] = (-3.35, 3.35)

  ranges: Ranges = field(default_factory=Ranges)
  """Held commands: uniform over these ranges [m/s, m/s, rad/s] (the paper's)."""
  reference_prob: float = 0.5
  """Probability of the reference command source."""
  standing_prob: float = 0.1
  """Probability that a held command is zero (standing)."""
  window: int = 25
  """Frames of the centered mean that makes the reference command."""
  motion_command: str = "motion"
  resampling_time_range: tuple[float, float] = (5.0, 5.0)

  def build(self, env: ManagerBasedRlEnv) -> VelocityCommand:
    return VelocityCommand(self, env)


# Env terms ----------------------------------------------------------------------------


def _velocity(env: ManagerBasedRlEnv, command_name: str) -> VelocityCommand:
  return cast(VelocityCommand, env.command_manager.get_term(command_name))


def velocity_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """The base twist command (raw; the policy normalizes it)."""
  return _velocity(env, command_name).command


def track_lin_vel_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  c = _velocity(env, command_name)
  return lin_vel_exp(c.robot_bodies, c.command, std)


def track_yaw_rate_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  c = _velocity(env, command_name)
  return yaw_rate_exp(c.robot_bodies, c.command, std)


def torso_lean_l2(
  env: ManagerBasedRlEnv, command_name: str, body_name: str
) -> torch.Tensor:
  c = _velocity(env, command_name)
  return torso_lean(c.robot_bodies, c.command, c.body_index(body_name))


def torso_rate_l2(
  env: ManagerBasedRlEnv, command_name: str, body_name: str
) -> torch.Tensor:
  c = _velocity(env, command_name)
  return torso_rate(c.robot_bodies, c.command, c.body_index(body_name))


def joint_deviation_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  data = env.scene[asset_cfg.name].data
  ids = asset_cfg.joint_ids
  return joint_deviation(data.joint_pos[:, ids], data.default_joint_pos[:, ids])


def fall(
  env: ManagerBasedRlEnv, command_name: str, min_height: float, max_tilt: float
) -> torch.Tensor:
  """The velocity interface's termination: a fall of the root (Table A.6)."""
  return root_fall(_velocity(env, command_name).robot_bodies, min_height, max_tilt)
