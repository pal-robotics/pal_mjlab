from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class PalMotionCommand(MotionCommand):
  """Custom motion command for PAL robots with per-joint randomization."""

  cfg: PalMotionCommandCfg

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori = quat_mul(orientations_delta, root_ori)

    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel += rand_samples[:, :3]
    root_ang_vel += rand_samples[:, 3:]

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]

    if self.cfg.joint_position_ranges:
      # Build per-joint randomization tensors.
      num_joints = joint_pos.shape[1]
      lower = torch.full(
        (num_joints,), self.cfg.joint_position_range[0], device=self.device
      )
      upper = torch.full(
        (num_joints,), self.cfg.joint_position_range[1], device=self.device
      )

      joint_names = self.robot.joint_names
      for pattern, r in self.cfg.joint_position_ranges.items():
        for i, name in enumerate(joint_names):
          if re.fullmatch(pattern, name):
            lower[i] = r[0]
            upper[i] = r[1]

      joint_pos += sample_uniform(
        lower=lower.unsqueeze(0),
        upper=upper.unsqueeze(0),
        size=joint_pos.shape,
        device=self.device,
      )
    else:
      # Fallback to base behavior.
      joint_pos += sample_uniform(
        lower=self.cfg.joint_position_range[0],
        upper=self.cfg.joint_position_range[1],
        size=joint_pos.shape,
        device=self.device,
      )

    # Write reference state to simulation
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    self.robot.clear_state(env_ids=env_ids)


@dataclass(kw_only=True)
class PalMotionCommandCfg(MotionCommandCfg):
  """Configuration for the custom PAL motion command."""

  joint_position_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
  """Dictionary mapping joint name regex patterns to randomization ranges."""

  def build(self, env: ManagerBasedRlEnv) -> PalMotionCommand:
    return PalMotionCommand(self, env)
