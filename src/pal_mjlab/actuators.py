"""Custom actuator models for PAL robots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.dc_actuator import dc_motor_clip
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg, pd_torque

if TYPE_CHECKING:
  from mjlab.entity import Entity


def speed_limit_clip(
  torque: torch.Tensor,
  force_limit: torch.Tensor,
  velocity_limit: torch.Tensor,
  full_torque_ratio: torch.Tensor,
  vel: torch.Tensor,
) -> torch.Tensor:
  """Clip a torque to a transmission speed limit.

  Full ``force_limit`` up to ``full_torque_ratio * velocity_limit``, then falling
  linearly to zero at ``velocity_limit`` (and reversing above it) in the driving
  direction. Braking (torque against the motion) is always available up to
  ``force_limit``. An infinite ``velocity_limit`` reduces to a plain clamp.
  """
  saturation_effort = force_limit / (1.0 - full_torque_ratio)
  clipped = dc_motor_clip(torque, saturation_effort, velocity_limit, force_limit, vel)
  # An infinite force_limit makes the curve NaN (inf / inf): no limit at all then.
  return torch.where(torch.isfinite(force_limit), clipped, torque)


def _per_env_tensor(value: float, num_envs: int, num_joints: int, device: str):
  return torch.full((num_envs, num_joints), value, dtype=torch.float, device=device)


@dataclass(kw_only=True)
class _SpeedLimitCfgMixin:
  velocity_limit: float = float("inf")
  """Joint speed (rad/s or m/s) at which the actuator can no longer drive the joint."""

  full_torque_ratio: float = 0.8
  """Fraction of ``velocity_limit`` up to which the full effort limit is available."""

  def _validate_speed_limit(self) -> None:
    if not self.velocity_limit > 0.0:
      raise ValueError(f"velocity_limit must be > 0, got {self.velocity_limit}.")
    if not 0.0 <= self.full_torque_ratio < 1.0:
      raise ValueError(
        f"full_torque_ratio must be in [0, 1), got {self.full_torque_ratio}."
      )


@dataclass(kw_only=True)
class SpeedLimitedPdActuatorCfg(_SpeedLimitCfgMixin, IdealPdActuatorCfg):
  """Explicit PD actuator whose torque vanishes at the transmission speed limit.

  MuJoCo position actuators only have a constant force range, so joints can be
  driven at any speed. This one applies :func:`speed_limit_clip` to the PD torque.
  """

  def __post_init__(self) -> None:
    super().__post_init__()
    self._validate_speed_limit()

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> SpeedLimitedPdActuator:
    return SpeedLimitedPdActuator(self, entity, target_ids, target_names)


class SpeedLimitedPdActuator(IdealPdActuator[SpeedLimitedPdActuatorCfg]):
  """Ideal PD actuator with a speed-dependent torque limit (fusable)."""

  param_names = (*IdealPdActuator.param_names, "velocity_limit", "full_torque_ratio")

  def __init__(
    self,
    cfg: SpeedLimitedPdActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self.velocity_limit: torch.Tensor | None = None
    self.full_torque_ratio: torch.Tensor | None = None

  @staticmethod
  def control_law(params: dict[str, torch.Tensor], cmd: ActuatorCmd) -> torch.Tensor:
    torque = pd_torque(params["stiffness"], params["damping"], cmd)
    return speed_limit_clip(
      torque,
      params["force_limit"],
      params["velocity_limit"],
      params["full_torque_ratio"],
      cmd.vel,
    )

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)
    n, j = data.nworld, len(self._target_names)
    self.velocity_limit = _per_env_tensor(self.cfg.velocity_limit, n, j, device)
    self.full_torque_ratio = _per_env_tensor(self.cfg.full_torque_ratio, n, j, device)


@dataclass(kw_only=True)
class BacklashPdActuatorCfg(_SpeedLimitCfgMixin, IdealPdActuatorCfg):
  """Explicit PD actuator with a gear backlash (dead zone) on the output side.

  The motor side tracks the PD target, but the link is only engaged once it
  has travelled half the backlash away from it. Inside the gap the
  proportional term transmits no torque, so the link can be moved freely by
  ``backlash`` radians in total (as seen on the real robot when the legs are
  moved by hand in position control).
  """

  backlash: float = 0.0
  """Total backlash (play) in rad. The link is free within +-backlash/2."""

  def __post_init__(self) -> None:
    super().__post_init__()
    self._validate_speed_limit()

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

  param_names = (
    "stiffness",
    "damping",
    "force_limit",
    "backlash",
    "velocity_limit",
    "full_torque_ratio",
  )

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
    self.velocity_limit: torch.Tensor | None = None
    self.full_torque_ratio: torch.Tensor | None = None

  @staticmethod
  def control_law(params: dict[str, torch.Tensor], cmd: ActuatorCmd) -> torch.Tensor:
    half_gap = 0.5 * params["backlash"]
    pos_error = cmd.position_target - cmd.pos
    engaged_error = pos_error - torch.clamp(pos_error, -half_gap, half_gap)
    torque = params["stiffness"] * engaged_error
    torque += params["damping"] * (cmd.velocity_target - cmd.vel)
    torque += cmd.effort_target
    # Plain clamp with the default (infinite) velocity limit.
    return speed_limit_clip(
      torque,
      params["force_limit"],
      params["velocity_limit"],
      params["full_torque_ratio"],
      cmd.vel,
    )

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
    self.velocity_limit = _per_env_tensor(
      self.cfg.velocity_limit, num_envs, num_joints, device
    )
    self.full_torque_ratio = _per_env_tensor(
      self.cfg.full_torque_ratio, num_envs, num_joints, device
    )

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
