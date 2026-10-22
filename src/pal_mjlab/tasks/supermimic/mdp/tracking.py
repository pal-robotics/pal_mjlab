"""Pure, differentiable tracking terms on body kinematics.

Shared by the simulator env (reward and termination terms in this package) and, later, the
world-model imagination, so both optimize exactly the same functions. Inputs are
``BodyState``s computed by ``TorchKinematics`` from (qpos, qvel); body 0 is the anchor.
Definitions follow BeyondMimic as used in SuperMimic (Table A.2, Appendix A).
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from ..kinematics import (
  quat_conj,
  quat_error_magnitude,
  quat_mul,
  quat_rotate,
  quat_rotate_inv,
  yaw_quat,
)


class BodyState(NamedTuple):
  """World-frame poses and velocities of the tracked bodies, (B, nb, ·), quaternions wxyz."""

  pos: torch.Tensor
  quat: torch.Tensor
  lin_vel: torch.Tensor
  ang_vel: torch.Tensor


def relative_reference(
  ref: BodyState, robot: BodyState
) -> tuple[torch.Tensor, torch.Tensor]:
  """Reference body poses re-anchored on the robot (BeyondMimic's ``body_*_relative_w``).

  The reference is translated so its anchor sits at the robot anchor's xy (keeping the
  reference height) and rotated by the heading difference, so body tracking ignores
  global drift (which the root terms penalize separately).
  """
  ref_anchor_pos, ref_anchor_quat = ref.pos[:, :1], ref.quat[:, :1]
  delta_pos = torch.cat([robot.pos[:, :1, :2], ref_anchor_pos[..., 2:]], dim=-1)
  delta_quat = yaw_quat(quat_mul(robot.quat[:, :1], quat_conj(ref_anchor_quat)))
  delta_quat = delta_quat.expand_as(ref.quat)
  pos = delta_pos + quat_rotate(delta_quat, ref.pos - ref_anchor_pos)
  quat = quat_mul(delta_quat, ref.quat)
  return pos, quat


# Squared errors, each (B,).


def anchor_pos_error(robot: BodyState, ref: BodyState) -> torch.Tensor:
  return (ref.pos[:, 0] - robot.pos[:, 0]).square().sum(-1)


def anchor_ori_error(robot: BodyState, ref: BodyState) -> torch.Tensor:
  return quat_error_magnitude(ref.quat[:, 0], robot.quat[:, 0]).square()


def body_pos_error(robot: BodyState, ref: BodyState) -> torch.Tensor:
  rel_pos, _ = relative_reference(ref, robot)
  return (rel_pos - robot.pos).square().sum(-1).mean(-1)


def body_ori_error(robot: BodyState, ref: BodyState) -> torch.Tensor:
  _, rel_quat = relative_reference(ref, robot)
  return quat_error_magnitude(rel_quat, robot.quat).square().mean(-1)


def body_lin_vel_error(robot: BodyState, ref: BodyState) -> torch.Tensor:
  return (ref.lin_vel - robot.lin_vel).square().sum(-1).mean(-1)


def body_ang_vel_error(robot: BodyState, ref: BodyState) -> torch.Tensor:
  return (ref.ang_vel - robot.ang_vel).square().sum(-1).mean(-1)


def projected_gravity_z(quat: torch.Tensor) -> torch.Tensor:
  """z component of the unit gravity direction expressed in the frame of ``quat``."""
  g = torch.zeros(quat.shape[:-1] + (3,), dtype=quat.dtype, device=quat.device)
  g[..., 2] = -1.0
  return quat_rotate_inv(quat, g)[..., 2]


def anchor_failure(
  robot: BodyState, ref: BodyState, z_threshold: float, tilt_threshold: float
) -> torch.Tensor:
  """SuperMimic's termination: anchor height error or tilt error beyond its threshold."""
  z_err = (ref.pos[:, 0, 2] - robot.pos[:, 0, 2]).abs()
  tilt = (
    projected_gravity_z(ref.quat[:, 0]) - projected_gravity_z(robot.quat[:, 0])
  ).abs()
  return (z_err > z_threshold) | (tilt > tilt_threshold)


def joint_limit_violation(
  q: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> torch.Tensor:
  """Sum over joints of the distance outside [lower, upper]."""
  return ((lower - q).clamp_min(0.0) + (q - upper).clamp_min(0.0)).sum(-1)
