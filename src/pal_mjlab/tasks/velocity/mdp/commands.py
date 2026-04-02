from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class UniformVelocityCommandWithProgressTracking(UniformVelocityCommand):
  """Velocity command sampler with episode-level progress tracking."""

  def __init__(
    self,
    cfg: UniformVelocityCommandWithProgressTrackingCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg, env)

    self.prev_root_pos_w_xy = self.robot.data.root_link_pos_w[:, :2].clone()
    self._needs_progress_sync = torch.ones(
      self.num_envs, dtype=torch.bool, device=self.device
    )

    self.episode_desired_progress = torch.zeros(self.num_envs, device=self.device)
    self.episode_achieved_progress = torch.zeros(self.num_envs, device=self.device)

    self.metrics["terrain_desired_progress"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["terrain_achieved_progress"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["terrain_progress_ratio"] = torch.zeros(
      self.num_envs, device=self.device
    )

  @property
  def progress_cfg(self) -> UniformVelocityCommandWithProgressTrackingCfg:
    """Typed view of cfg for subclass-specific fields."""
    return cast(UniformVelocityCommandWithProgressTrackingCfg, self.cfg)

  def compute(self, dt: float) -> None:
    """Update progress using the previous command, then refresh commands."""
    sync_mask = self._needs_progress_sync.clone()
    sync_env_ids = sync_mask.nonzero(as_tuple=False).flatten()
    if sync_env_ids.numel() > 0:
      self._sync_progress_tracking(sync_env_ids)

    track_env_ids = (~sync_mask).nonzero(as_tuple=False).flatten()
    self._update_episode_progress(dt, track_env_ids)
    super().compute(dt)

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    assert isinstance(env_ids, torch.Tensor)
    self.reset_progress_tracking(env_ids, sync_prev_root=False)
    return extras

  def flush_episode_progress(self, env_ids: torch.Tensor, dt: float) -> None:
    """Flush progress before curriculum evaluates completed episodes."""
    sync_mask = self._needs_progress_sync[env_ids].clone()
    sync_env_ids = env_ids[sync_mask]
    if sync_env_ids.numel() > 0:
      self._sync_progress_tracking(sync_env_ids)

    track_env_ids = env_ids[~sync_mask]
    self._update_episode_progress(dt, track_env_ids)

  def _sync_progress_tracking(self, env_ids: torch.Tensor) -> None:
    if env_ids.numel() == 0:
      return

    self.prev_root_pos_w_xy[env_ids] = self.robot.data.root_link_pos_w[env_ids, :2]
    self._needs_progress_sync[env_ids] = False

  def _update_episode_progress(
    self,
    dt: float,
    env_ids: torch.Tensor | None = None,
  ) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

    if env_ids.numel() == 0:
      self._refresh_progress_metrics()
      return

    curr_root_pos_w_xy = self.robot.data.root_link_pos_w[env_ids, :2]
    delta_pos_w_xy = curr_root_pos_w_xy - self.prev_root_pos_w_xy[env_ids]
    self.prev_root_pos_w_xy[env_ids] = curr_root_pos_w_xy

    cmd_b_xy = self.vel_command_b[env_ids, :2]
    cmd_speed = torch.linalg.vector_norm(cmd_b_xy, dim=1)

    cfg = self.progress_cfg

    active = cmd_speed > cfg.progress_min_speed

    if not cfg.include_standing_in_progress:
      active &= ~self.is_standing_env[env_ids]

    if not cfg.include_heading_in_progress:
      active &= ~self.is_heading_env[env_ids]

    active_env_ids = env_ids[active]
    if active_env_ids.numel() == 0:
      self._refresh_progress_metrics()
      return

    desired_step_progress = cmd_speed[active] * dt
    self.episode_desired_progress[active_env_ids] += desired_step_progress

    cmd_dir_b_xy = cmd_b_xy[active] / cmd_speed[active].unsqueeze(1).clamp_min(
      cfg.progress_eps
    )

    zeros = torch.zeros(
      (active_env_ids.numel(), 1),
      device=self.device,
      dtype=cmd_dir_b_xy.dtype,
    )
    cmd_dir_w_xy = quat_apply(
      self.robot.data.root_link_quat_w[active_env_ids],
      torch.cat([cmd_dir_b_xy, zeros], dim=1),
    )[:, :2]
    cmd_dir_w_xy = cmd_dir_w_xy / torch.linalg.vector_norm(
      cmd_dir_w_xy, dim=1, keepdim=True
    ).clamp_min(cfg.progress_eps)

    achieved_step_progress = torch.sum(
      delta_pos_w_xy[active] * cmd_dir_w_xy,
      dim=1,
    )

    if not cfg.allow_backward_progress:
      achieved_step_progress = achieved_step_progress.clamp_min(0.0)

    if cfg.cap_step_progress_to_desired:
      achieved_step_progress = torch.minimum(
        achieved_step_progress,
        desired_step_progress,
      )

    self.episode_achieved_progress[active_env_ids] += achieved_step_progress
    self._refresh_progress_metrics()

  def _refresh_progress_metrics(self) -> None:
    cfg = self.progress_cfg
    self.metrics["terrain_desired_progress"].copy_(self.episode_desired_progress)
    self.metrics["terrain_achieved_progress"].copy_(self.episode_achieved_progress)
    self.metrics["terrain_progress_ratio"].copy_(
      self.episode_achieved_progress
      / self.episode_desired_progress.clamp_min(cfg.progress_eps)
    )

  def get_episode_progress_ratio(self, env_ids: torch.Tensor) -> torch.Tensor:
    cfg = self.progress_cfg
    return self.episode_achieved_progress[env_ids] / self.episode_desired_progress[
      env_ids
    ].clamp_min(cfg.progress_eps)

  def reset_progress_tracking(
    self,
    env_ids: torch.Tensor | None = None,
    *,
    sync_prev_root: bool = True,
  ) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

    self.episode_desired_progress[env_ids] = 0.0
    self.episode_achieved_progress[env_ids] = 0.0
    if sync_prev_root:
      self._sync_progress_tracking(env_ids)
    else:
      self._needs_progress_sync[env_ids] = True
    self._refresh_progress_metrics()


@dataclass(kw_only=True)
class UniformVelocityCommandWithProgressTrackingCfg(UniformVelocityCommandCfg):
  progress_min_speed: float = 0.1
  include_standing_in_progress: bool = False
  include_heading_in_progress: bool = False
  allow_backward_progress: bool = False
  cap_step_progress_to_desired: bool = True
  progress_eps: float = 1e-6

  def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommandWithProgressTracking:
    return UniformVelocityCommandWithProgressTracking(self, env)
