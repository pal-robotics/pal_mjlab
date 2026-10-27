"""Heading-invariant robot state (D4) and root-pose helpers.

The state is what the world model will predict and what the critic sees: root height,
root tilt (6D rotation of the root in its heading frame), root linear and angular velocity
in the root frame, and all joint positions and velocities (including passive joints, so
the state is Markov). World x, y and yaw are left out: the dynamics do not depend on them.

The root's world pose (x, y, yaw) is carried separately and integrated from per-step
displacements expressed in the heading frame, which is how the world model predicts it.
"""

from __future__ import annotations

import torch

from .kinematics import (
  matrix_from_rot6d,
  quat_conj,
  quat_from_matrix,
  quat_mul,
  quat_rotate,
  quat_rotate_inv,
  rot6d,
  yaw_angle,
  yaw_quat,
)


def heading_state(qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
  """(B, 13 + 2 nj) state from MuJoCo-layout qpos/qvel of a floating-base robot."""
  quat = qpos[:, 3:7]
  tilt = quat_mul(quat_conj(yaw_quat(quat)), quat)
  lin_vel_b = quat_rotate_inv(quat, qvel[:, :3])
  return torch.cat(
    [qpos[:, 2:3], rot6d(tilt), lin_vel_b, qvel[:, 3:6], qpos[:, 7:], qvel[:, 6:]],
    dim=-1,
  )


def heading_pose(qpos: torch.Tensor) -> torch.Tensor:
  """(B, 3) world x, y and yaw of the root."""
  return torch.cat([qpos[:, :2], yaw_angle(qpos[:, 3:7])[:, None]], dim=-1)


def state_to_qpos_qvel(
  state: torch.Tensor, pose: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Inverse of (heading_state, heading_pose): MuJoCo-layout qpos/qvel, differentiable."""
  nj = (state.shape[-1] - 13) // 2
  tilt = quat_from_matrix(matrix_from_rot6d(state[:, 1:7]))
  half = 0.5 * pose[:, 2:3]
  zero = torch.zeros_like(half)
  heading = torch.cat([torch.cos(half), zero, zero, torch.sin(half)], dim=-1)
  quat = quat_mul(heading, tilt)
  qpos = torch.cat([pose[:, :2], state[:, :1], quat, state[:, 13 : 13 + nj]], dim=-1)
  qvel = torch.cat(
    [quat_rotate(quat, state[:, 7:10]), state[:, 10:13], state[:, 13 + nj :]], dim=-1
  )
  return qpos, qvel


def wrap_angle(a: torch.Tensor) -> torch.Tensor:
  return torch.atan2(torch.sin(a), torch.cos(a))


def pose_delta(pose: torch.Tensor, next_pose: torch.Tensor) -> torch.Tensor:
  """(B, 3) planar displacement in the heading frame of ``pose``, and the yaw change."""
  c, s = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
  d = next_pose[:, :2] - pose[:, :2]
  dx = c * d[:, 0] + s * d[:, 1]
  dy = -s * d[:, 0] + c * d[:, 1]
  return torch.stack([dx, dy, wrap_angle(next_pose[:, 2] - pose[:, 2])], dim=-1)


def integrate_pose(pose: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
  """Inverse of ``pose_delta``."""
  c, s = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
  x = pose[:, 0] + c * delta[:, 0] - s * delta[:, 1]
  y = pose[:, 1] + s * delta[:, 0] + c * delta[:, 1]
  return torch.stack([x, y, wrap_angle(pose[:, 2] + delta[:, 2])], dim=-1)


def root_pose_error(
  robot_pos: torch.Tensor,
  robot_quat: torch.Tensor,
  ref_pos: torch.Tensor,
  ref_quat: torch.Tensor,
) -> torch.Tensor:
  """(B, 4): reference root position in the robot's heading frame, and the yaw error.

  Privileged critic input (D7): the heading-invariant state cannot see global drift, but
  the reward has a global root-position term.
  """
  heading = yaw_quat(robot_quat)
  dpos = quat_rotate_inv(heading, ref_pos - robot_pos)
  dq = quat_mul(quat_conj(heading), yaw_quat(ref_quat))
  dyaw = 2.0 * torch.atan2(dq[:, 3], dq[:, 0])
  dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))
  return torch.cat([dpos, dyaw[:, None]], dim=-1)


def reference_goal(
  qpos: torch.Tensor,
  qvel: torch.Tensor,
  robot_quat: torch.Tensor,
  joint_ids: torch.Tensor,
) -> torch.Tensor:
  """The policy's goal (D6): (..., K * F) from K reference frames.

  ``qpos`` (..., K, nq) and ``qvel`` (..., K, nv) are the reference in MuJoCo layout, from the
  target frame (the one the next action should reach) on, the last frame repeated past the clip
  end; ``robot_quat`` (..., 4) is the robot's root orientation (wxyz). Per frame: the reference
  positions and velocities of the ``joint_ids`` joints, and the reference root orientation in the
  robot root frame, rot6d(conj(q_robot) * q_ref). The training env's goal, and the goal builder for
  deployment (rl/export.py, D48).
  """
  ref_quat = qpos[..., 3:7]
  robot_quat = robot_quat[..., None, :].expand_as(ref_quat)
  feats = (
    qpos[..., 7:][..., joint_ids],
    qvel[..., 6:][..., joint_ids],
    rot6d(quat_mul(quat_conj(robot_quat), ref_quat)),
  )
  return torch.cat(feats, dim=-1).flatten(-2)
