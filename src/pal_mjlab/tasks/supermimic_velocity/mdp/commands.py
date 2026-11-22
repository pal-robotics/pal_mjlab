"""The velocity command, with its twin for SuperMimic's imagination.

mjlab's uniform velocity command (resampled every few seconds, some envs standing still),
with ``command_model()``. Heading control is off: the yaw rate is commanded directly, so
the command is a function of the command alone and imagination can carry it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

from pal_mjlab.rl.supermimic.command_model import CommandState

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["SuperMimicVelocityCommand", "SuperMimicVelocityCommandCfg", "VelocityCommandModel"]


class VelocityCommandModel:
  """``c = {"command": (B, 3)}``: the body-frame (vx, vy, yaw rate) command.

  Constant over imagination's horizon (the env's resampling every few seconds is not
  foreseen); it never ends the episode (the env's time limit does); the termination
  head's reference is the nominal root height, upright.
  """

  def __init__(self, cmd: SuperMimicVelocityCommand) -> None:
    self.cmd = cmd
    self.root_height = float(cmd.robot.data.default_root_state[0, 2])

  def state(self) -> CommandState:
    return {"command": self.cmd.command.clone()}

  def advance(self, c: CommandState) -> CommandState:
    return c

  def done(self, c: CommandState) -> torch.Tensor:
    return torch.zeros(len(c["command"]), dtype=torch.bool, device=c["command"].device)

  def termination_ref(self, c: CommandState) -> torch.Tensor:
    ref = torch.empty(len(c["command"]), 2, device=c["command"].device)
    ref[:, 0], ref[:, 1] = self.root_height, -1.0
    return ref


class SuperMimicVelocityCommand(UniformVelocityCommand):
  def __init__(self, cfg: SuperMimicVelocityCommandCfg, env: ManagerBasedRlEnv):
    if cfg.heading_command:
      raise ValueError("SuperMimic's velocity command needs heading_command=False.")
    super().__init__(cfg, env)
    self._command_model: VelocityCommandModel | None = None

  def step_metrics(self) -> dict[str, torch.Tensor]:
    """Per-env tracking this step, for SuperMimic's step logs (mjlab's ``metrics`` sum
    over an episode instead). ``standing_error_xy`` is what standing still would score."""
    cmd, vel = self.vel_command_b, self.robot.data.root_link_lin_vel_b
    return {
      "error_vel_xy": torch.norm(cmd[:, :2] - vel[:, :2], dim=-1),
      "error_vel_yaw": torch.abs(cmd[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]),
      "speed_xy": torch.norm(vel[:, :2], dim=-1),
      "standing_error_xy": torch.norm(cmd[:, :2], dim=-1),
    }

  def command_model(self) -> VelocityCommandModel:
    """This command's tensor twin, for SuperMimic's imagination (one per command)."""
    if self._command_model is None:
      self._command_model = VelocityCommandModel(self)
    return self._command_model


@dataclass(kw_only=True)
class SuperMimicVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> SuperMimicVelocityCommand:
    return SuperMimicVelocityCommand(self, env)
