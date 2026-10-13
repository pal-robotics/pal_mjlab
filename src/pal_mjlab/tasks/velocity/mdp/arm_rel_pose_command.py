from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from .safe_position_sampling import push_xy_outside_radius, validate_base_position

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


def _quat_from_rpy(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor):
  """Intrinsic ZYX (yaw, pitch, roll) -> quaternion (w, x, y, z)."""
  cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
  cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
  cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
  return torch.stack(
    (
      cr * cp * cy + sr * sp * sy,
      sr * cp * cy - cr * sp * sy,
      cr * sp * cy + sr * cp * sy,
      cr * cp * sy - sr * sp * cy,
    ),
    dim=-1,
  )


class UniformHandPoseCommand(CommandTerm):
  """Hand position + orientation command, expressed in the base frame.

  The hand's pointing axis (local +x) always has a non-negative component along
  the base's forward axis (+x), so it never points behind the robot.
  """

  cfg: UniformHandPoseCommandCfg

  def __init__(self, cfg: UniformHandPoseCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.hand_pose_command = torch.zeros(self.num_envs, 7, device=self.device)
    self.hand_pose_command[:, 3] = 1.0  # identity quaternion

    self.is_base_pose_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    self._base_pose = torch.tensor(
      [*cfg.base_position, *cfg.base_orientation], device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return self.hand_pose_command

  def _update_metrics(self) -> None:
    pass

  def _sample(self, n: int, rng: tuple[float, float]) -> torch.Tensor:
    return torch.empty(n, device=self.device).uniform_(*rng)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    r = self.cfg.ranges

    xy = torch.stack((self._sample(n, r.x), self._sample(n, r.y)), dim=-1)
    xy = push_xy_outside_radius(xy, self.cfg.min_xy_radius)
    self.hand_pose_command[env_ids, :2] = xy
    self.hand_pose_command[env_ids, 2] = self._sample(n, r.z)

    roll = self._sample(n, r.roll)
    pitch = self._sample(n, r.pitch)
    yaw = self._sample(n, r.yaw)
    self.hand_pose_command[env_ids, 3:7] = _quat_from_rpy(roll, pitch, yaw)

    self.is_base_pose_env[env_ids] = self._sample(n, (0.0, 1.0)) <= self.cfg.rel_base_pose

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    ids = self.is_base_pose_env.nonzero(as_tuple=False).flatten()
    self.hand_pose_command[ids] = self._base_pose


@dataclass(kw_only=True)
class UniformHandPoseCommandCfg(CommandTermCfg):

  @dataclass
  class Ranges:
    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]
    # Radians, expressed in the base frame (x forward). Defaults keep the hand
    # facing forward or sideways, with everything in between.
    yaw: tuple[float, float] = (-math.pi / 2, math.pi / 2)
    pitch: tuple[float, float] = (-math.pi / 4, math.pi / 4)
    roll: tuple[float, float] = (-math.pi / 4, math.pi / 4)

  ranges: Ranges

  base_position: tuple[float, float, float]
  base_orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz

  rel_base_pose: float = 0.1
  min_xy_radius: float = 0.2

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> UniformHandPoseCommand:
    return UniformHandPoseCommand(self, env)

  def __post_init__(self):
    validate_base_position(self.base_position, self.min_xy_radius)
    half_pi = math.pi / 2
    for name in ("yaw", "pitch"):
        lo, hi = getattr(self.ranges, name)
        if lo < -half_pi or hi > half_pi:
            raise ValueError(f"ranges.{name}={lo, hi} must lie within [-pi/2, pi/2].")