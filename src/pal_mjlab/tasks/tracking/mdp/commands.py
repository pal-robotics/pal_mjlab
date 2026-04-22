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

    # Pre-seed history buffers with actual reference motion data [t-H, ..., t]
    self._prime_history_buffers(env_ids)

  def _prime_history_buffers(self, env_ids: torch.Tensor):
    """Pre-seed actor_history circular buffers with reference motion data.

    After RSI resets the robot to timestep t, this fills each term's
    CircularBuffer with frames [t - history_length, ..., t] from the
    reference motion file.

    If the 'actor_history' group doesn't exist (standard task), this is a
    no-op — switching between task IDs requires no code changes.
    """
    obs_mgr = self._env.observation_manager
    history_group = "actor_history"

    if history_group not in obs_mgr._group_obs_term_history_buffer:
      return  # Standard task — nothing to do

    history_buffers = obs_mgr._group_obs_term_history_buffer[history_group]
    if not history_buffers:
      return

    # Determine history length from any buffer in the group
    any_buf = next(iter(history_buffers.values()))
    history_length = any_buf.max_length

    # Save and temporarily shift time_steps to pull ref frames without
    # touching simulation state.
    original_time_steps = self.time_steps.clone()

    # Iterate oldest-to-newest: [t-history_length, ..., t-1, t]
    for lag in range(history_length, -1, -1):
      lagged_t = torch.clamp(self.time_steps[env_ids] - lag, min=0)
      self.time_steps[env_ids] = lagged_t

      for term_name, buf in history_buffers.items():
        group_term_names = obs_mgr._group_obs_term_names[history_group]
        group_term_cfgs = obs_mgr._group_obs_term_cfgs[history_group]
        if term_name not in group_term_names:
          continue
        idx = group_term_names.index(term_name)
        term_cfg = group_term_cfgs[idx]

        # Compute raw obs value; skip noise so we inject clean reference data
        obs_val = term_cfg.func(self._env, **term_cfg.params)
        if term_cfg.scale is not None:
          obs_val = obs_val * term_cfg.scale

        # On the oldest lag, zero-out the env_ids slots so the next append
        # triggers the CircularBuffer's backfill logic properly
        if lag == history_length:
          buf.reset(batch_ids=env_ids.tolist())

        # Build a full-batch tensor; only env_ids rows matter
        if not buf.is_initialized:
          buf.append(obs_val)
        else:
          full_batch = buf.buffer[:, -1].clone()  # (batch, ...)
          full_batch[env_ids] = obs_val[env_ids]
          buf.append(full_batch)

    # Restore time_steps
    self.time_steps.copy_(original_time_steps)


@dataclass(kw_only=True)
class PalMotionCommandCfg(MotionCommandCfg):
  """Configuration for the custom PAL motion command."""

  joint_position_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
  """Dictionary mapping joint name regex patterns to randomization ranges."""

  def build(self, env: ManagerBasedRlEnv) -> PalMotionCommand:
    return PalMotionCommand(self, env)
