"""Custom actuator models for PAL robots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity


@dataclass(kw_only=True)
class BacklashPdActuatorCfg(IdealPdActuatorCfg):
  """Explicit PD actuator with a gear backlash (dead zone) on the output side.

  The motor side tracks the PD target, but the link is only engaged once it
  has travelled half the backlash away from it. Inside the gap the
  proportional term transmits no torque, so the link can be moved freely by
  ``backlash`` radians in total (as seen on the real robot when the legs are
  moved by hand in position control).
  """

  backlash: float = 0.0
  """Total backlash (play) in rad. The link is free within +-backlash/2."""

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> BacklashPdActuator:
    return BacklashPdActuator(self, entity, target_ids, target_names)


class BacklashPdActuator(IdealPdActuator[BacklashPdActuatorCfg]):
  """Ideal PD actuator with a dead zone of width ``backlash`` on the position error.

  torque = kp * deadzone(q_target - q, backlash / 2) + kd * (qd_target - qd) + tau_ff

  The damping term is kept active inside the gap for numerical robustness; it
  does not resist quasi-static displacement, which is what the backlash is
  about. Keeping the stateless ``control_law`` (and not overriding
  ``compute``) keeps the actuator fusable.
  """

  param_names = ("stiffness", "damping", "force_limit", "backlash")

  def __init__(
    self,
    cfg: BacklashPdActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self.backlash: torch.Tensor | None = None
    self.default_backlash: torch.Tensor | None = None

  @staticmethod
  def control_law(params: dict[str, torch.Tensor], cmd: ActuatorCmd) -> torch.Tensor:
    half_gap = 0.5 * params["backlash"]
    pos_error = cmd.position_target - cmd.pos
    engaged_error = pos_error - torch.clamp(pos_error, -half_gap, half_gap)
    torque = params["stiffness"] * engaged_error
    torque += params["damping"] * (cmd.velocity_target - cmd.vel)
    torque += cmd.effort_target
    force_limit = params["force_limit"]
    return torch.clamp(torque, -force_limit, force_limit)

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)
    num_envs = data.nworld
    num_joints = len(self._target_names)
    self.backlash = torch.full(
      (num_envs, num_joints), self.cfg.backlash, dtype=torch.float, device=device
    )
    self.default_backlash = self.backlash.clone()

  def set_backlash(self, env_ids: torch.Tensor | slice, backlash: torch.Tensor) -> None:
    """Set the backlash for the given environments (in place, fusion-safe).

    Args:
      env_ids: Environment indices to update.
      backlash: Shape (len(env_ids), num_targets) or (len(env_ids),).
    """
    assert self.backlash is not None
    if backlash.ndim == 1:
      backlash = backlash.unsqueeze(-1)
    self.backlash[env_ids] = backlash
