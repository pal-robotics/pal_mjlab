"""Scripted arm action term.

An ``ActionTerm`` with ``action_dim == 0`` that does not consume any policy
outputs but writes joint position targets for a configured set of joints
(typically the upper body) every decimation substep. The targets follow a
per-environment motion profile randomized at reset and periodically re-rolled
during the episode. Profiles supported: sinusoid, random walk, hold.

The terms are written via ``Entity.set_joint_position_target``, so MuJoCo's
built-in position actuators apply real PD torques to track them, producing
realistic angular-momentum coupling on the rest of the body.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_MODE_SINUSOID = 0
_MODE_RANDOM_WALK = 1
_MODE_HOLD = 2
_MODE_NAME_TO_CODE = {
  "sinusoid": _MODE_SINUSOID,
  "random_walk": _MODE_RANDOM_WALK,
  "hold": _MODE_HOLD,
}


@dataclass(kw_only=True)
class ScriptedArmActionCfg(ActionTermCfg):
  """Configuration for the scripted arm action term."""

  entity_name: str = "robot"
  joint_names: tuple[str, ...] = (r"arm_.*",)
  """Regex patterns selecting the joints to drive."""

  amplitude_range: tuple[float, float] = (0.0, 0.5)
  """Per-joint sinusoid amplitude range (rad)."""

  frequency_range_hz: tuple[float, float] = (0.3, 1.5)
  """Per-joint sinusoid frequency range (Hz)."""

  phase_range: tuple[float, float] = (0.0, 2.0 * math.pi)
  """Per-joint sinusoid phase range (rad)."""

  bias_range: tuple[float, float] = (-0.3, 0.3)
  """Per-joint additive bias range around the default pose (rad)."""

  resample_interval_s_range: tuple[float, float] = (2.0, 6.0)
  """Range for the time (s) between mid-episode motion re-rolls."""

  mode_weights: dict[str, float] = field(
    default_factory=lambda: {"sinusoid": 0.5, "random_walk": 0.3, "hold": 0.2}
  )
  """Categorical weights over motion modes. Allowed keys: sinusoid, random_walk, hold."""

  random_walk_step_std: float = 0.02
  """Std of the Gaussian step (rad per substep) used in random_walk mode."""

  def build(self, env: ManagerBasedRlEnv) -> ScriptedArmAction:
    return ScriptedArmAction(self, env)


class ScriptedArmAction(ActionTerm):
  """Drive a subset of joints with a per-env randomized motion profile."""

  cfg: ScriptedArmActionCfg

  def __init__(self, cfg: ScriptedArmActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    joint_ids, joint_names = self._entity.find_joints(list(cfg.joint_names))
    if not joint_ids:
      raise ValueError(
        f"ScriptedArmAction: no joints matched patterns {cfg.joint_names}"
      )
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._joint_names = joint_names
    self._num_arm_joints = len(joint_ids)

    for name in cfg.mode_weights:
      if name not in _MODE_NAME_TO_CODE:
        raise ValueError(
          f"ScriptedArmAction: unknown mode '{name}'. "
          f"Allowed: {sorted(_MODE_NAME_TO_CODE)}"
        )
    modes = list(cfg.mode_weights.keys())
    weights = torch.tensor(
      [cfg.mode_weights[m] for m in modes], device=self.device, dtype=torch.float
    )
    if weights.sum() <= 0:
      raise ValueError("ScriptedArmAction: mode_weights must sum to > 0.")
    self._mode_probs = weights / weights.sum()
    self._mode_codes = torch.tensor(
      [_MODE_NAME_TO_CODE[m] for m in modes],
      device=self.device,
      dtype=torch.long,
    )

    self._dt = self._env.physics_dt

    N, J = self.num_envs, self._num_arm_joints
    device = self.device
    self._amplitude = torch.zeros(N, J, device=device)
    self._frequency_hz = torch.zeros(N, J, device=device)
    self._phase = torch.zeros(N, J, device=device)
    self._bias = torch.zeros(N, J, device=device)
    self._current_target = torch.zeros(N, J, device=device)
    self._time = torch.zeros(N, device=device)
    self._next_resample_t = torch.zeros(N, device=device)
    self._mode = torch.zeros(N, dtype=torch.long, device=device)

    self._raw_action_buf = torch.zeros(N, 0, device=device)

    self._resample(torch.arange(N, device=device))

  @property
  def action_dim(self) -> int:
    return 0

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_action_buf

  def process_actions(self, actions: torch.Tensor) -> None:
    del actions

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None or isinstance(env_ids, slice):
      ids = torch.arange(self.num_envs, device=self.device)
    else:
      ids = env_ids
    if ids.numel() == 0:
      return
    self._time[ids] = 0.0
    self._resample(ids)

  def _resample(self, env_ids: torch.Tensor) -> None:
    n = env_ids.numel()
    if n == 0:
      return
    J = self._num_arm_joints
    device = self.device

    mode_idx = torch.multinomial(self._mode_probs, n, replacement=True)
    self._mode[env_ids] = self._mode_codes[mode_idx]

    self._amplitude[env_ids] = _uniform(
      self.cfg.amplitude_range, (n, J), device=device
    )
    self._frequency_hz[env_ids] = _uniform(
      self.cfg.frequency_range_hz, (n, J), device=device
    )
    self._phase[env_ids] = _uniform(self.cfg.phase_range, (n, J), device=device)
    self._bias[env_ids] = _uniform(self.cfg.bias_range, (n, J), device=device)

    default = self._entity.data.default_joint_pos[env_ids][:, self._joint_ids]
    self._current_target[env_ids] = default + self._bias[env_ids]

    interval = _uniform(
      self.cfg.resample_interval_s_range, (n,), device=device
    )
    self._next_resample_t[env_ids] = self._time[env_ids] + interval

  def apply_actions(self) -> None:
    self._time += self._dt

    needs_resample = self._time >= self._next_resample_t
    if needs_resample.any():
      self._resample(needs_resample.nonzero(as_tuple=True)[0])

    default = self._entity.data.default_joint_pos[:, self._joint_ids]
    t = self._time.unsqueeze(-1)
    sin_target = default + self._bias + self._amplitude * torch.sin(
      2.0 * math.pi * self._frequency_hz * t + self._phase
    )

    rw_step = (
      torch.randn_like(self._current_target) * self.cfg.random_walk_step_std
    )
    rw_mask = (self._mode == _MODE_RANDOM_WALK).unsqueeze(-1)
    self._current_target = torch.where(
      rw_mask, self._current_target + rw_step, self._current_target
    )

    sin_mask = (self._mode == _MODE_SINUSOID).unsqueeze(-1)
    target = torch.where(sin_mask, sin_target, self._current_target)

    lo = self._entity.data.soft_joint_pos_limits[:, self._joint_ids, 0]
    hi = self._entity.data.soft_joint_pos_limits[:, self._joint_ids, 1]
    target = torch.clamp(target, lo, hi)
    self._current_target = torch.clamp(self._current_target, lo, hi)

    self._entity.set_joint_position_target(target, joint_ids=self._joint_ids)


def _uniform(
  rng: tuple[float, float], shape: tuple[int, ...], device: str
) -> torch.Tensor:
  lo, hi = rng
  return torch.empty(shape, device=device).uniform_(lo, hi)
