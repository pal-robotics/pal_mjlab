"""Heading-invariant robot state (D4) and root-pose helpers.

The state is what the world model will predict and what the critic sees: root height,
root tilt (6D rotation of the root in its heading frame), root linear and angular velocity
in the root frame, and all joint positions and velocities (including passive joints, so
the state is Markov). World x, y and yaw are left out: the dynamics do not depend on them.
"""

from __future__ import annotations

import torch

from .kinematics import quat_conj, quat_mul, quat_rotate_inv, rot6d, yaw_quat


def heading_state(qpos: torch.Tensor, qvel: torch.Tensor) -> torch.Tensor:
  """(B, 13 + 2 nj) state from MuJoCo-layout qpos/qvel of a floating-base robot."""
  quat = qpos[:, 3:7]
  tilt = quat_mul(quat_conj(yaw_quat(quat)), quat)
  lin_vel_b = quat_rotate_inv(quat, qvel[:, :3])
  return torch.cat(
    [qpos[:, 2:3], rot6d(tilt), lin_vel_b, qvel[:, 3:6], qpos[:, 7:], qvel[:, 6:]],
    dim=-1,
  )


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
