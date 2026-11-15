"""Reward terms of the stand-up task.

Progress terms cover the stages of getting up; care terms penalize what a
violent motion shows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.string import resolve_matching_names_values

from .contacts import (
  GroundContactCfg,
  StandCfg,
  get_ground_contact,
  height_progress,
  is_standing,
  upright_progress,
)
from .events import get_start_action

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_ACTUATOR_ATTR = "_stand_up_actuator_params"


##
# Progress.
##


def torso_upright(env: ManagerBasedRlEnv, stand_cfg: StandCfg) -> torch.Tensor:
  """0 upside down, 0.25 horizontal, 1 upright."""
  return upright_progress(env, stand_cfg)


def base_height_progress(env: ManagerBasedRlEnv, stand_cfg: StandCfg) -> torch.Tensor:
  """Base height between lying (0) and standing (1), scaled by how upright
  the torso is."""
  return height_progress(env, stand_cfg) * upright_progress(env, stand_cfg)


def feet_load(env: ManagerBasedRlEnv, stand_cfg: StandCfg) -> torch.Tensor:
  """Share of the weight carried by the feet, scaled by how upright the torso
  is."""
  contact = get_ground_contact(env, stand_cfg.contact_cfg)
  return contact.feet_load_fraction() * upright_progress(env, stand_cfg)


def standing(env: ManagerBasedRlEnv, stand_cfg: StandCfg) -> torch.Tensor:
  return is_standing(env, stand_cfg).float()


class standing_posture:
  """Joints at their default position, scaled by how close the robot is to
  standing. ``std`` maps joint name regexes to kernel widths."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    joint_ids, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    _, _, std = resolve_matching_names_values(cfg.params["std"], joint_names)
    self._joint_ids = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)
    self._std = torch.tensor(std, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    stand_cfg: StandCfg,
    std: dict[str, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del std  # Resolved in __init__.
    asset: Entity = env.scene[asset_cfg.name]
    error = (
      asset.data.joint_pos[:, self._joint_ids]
      - asset.data.default_joint_pos[:, self._joint_ids]
    )
    closeness = torch.exp(-torch.mean(torch.square(error / self._std), dim=-1))
    return base_height_progress(env, stand_cfg) * closeness


def standing_still(
  env: ManagerBasedRlEnv,
  stand_cfg: StandCfg,
  lin_speed_std: float,
  ang_speed_std: float,
) -> torch.Tensor:
  """Base at rest, scaled by how close the robot is to standing."""
  asset: Entity = env.scene[stand_cfg.entity_name]
  lin = torch.sum(torch.square(asset.data.root_link_lin_vel_w), dim=-1)
  ang = torch.sum(torch.square(asset.data.root_link_ang_vel_w), dim=-1)
  rest = torch.exp(-lin / lin_speed_std**2 - ang / ang_speed_std**2)
  return base_height_progress(env, stand_cfg) * rest


##
# Care.
##


def _action_history(
  env: ManagerBasedRlEnv,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """The last three actions, newest first. Before the episode it holds the
  action that held the start pose, not the zeros of the reset."""
  manager = env.action_manager
  start = get_start_action(env)
  # Rewards run after the episode length was incremented.
  length = env.episode_length_buf.unsqueeze(-1)
  previous = torch.where(length <= 1, start, manager.prev_action)
  before_previous = torch.where(length <= 2, start, manager.prev_prev_action)
  return manager.action, previous, before_previous


def episode_action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  action, previous, _ = _action_history(env)
  return torch.sum(torch.square(action - previous), dim=1)


def episode_action_acc_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  action, previous, before_previous = _action_history(env)
  return torch.sum(torch.square(action - 2 * previous + before_previous), dim=1)


class joint_speed_excess:
  """Joint speed beyond a limit, relative to the limit, summed over the
  joints. ``speed_limits`` maps joint name regexes to limits."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    joint_ids, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    _, _, limits = resolve_matching_names_values(
      cfg.params["speed_limits"], joint_names
    )
    self._joint_ids = torch.as_tensor(joint_ids, device=env.device, dtype=torch.long)
    self._limits = torch.tensor(limits, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    speed_limits: dict[str, float],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del speed_limits  # Resolved in __init__.
    asset: Entity = env.scene[asset_cfg.name]
    speed = asset.data.joint_vel[:, self._joint_ids].abs()
    return torch.sum((speed / self._limits - 1.0).clamp(min=0.0), dim=-1)


def _actuator_params(env: ManagerBasedRlEnv, asset: Entity) -> tuple:
  """Stiffness, damping, force limit and joint addresses of the actuators."""
  params = getattr(env, _ACTUATOR_ATTR, None)
  if params is None:
    model = env.sim.mj_model
    ctrl_ids = asset.indexing.ctrl_ids.cpu().numpy()
    joint_ids = model.actuator_trnid[ctrl_ids, 0]

    def tensor(values) -> torch.Tensor:
      return torch.as_tensor(values, dtype=torch.float32, device=env.device)

    params = (
      tensor(model.actuator_gainprm[ctrl_ids, 0]),
      tensor(-model.actuator_biasprm[ctrl_ids, 2]),
      tensor(model.actuator_forcerange[ctrl_ids]).abs().max(dim=-1).values,
      torch.as_tensor(model.jnt_qposadr[joint_ids], device=env.device).long(),
      torch.as_tensor(model.jnt_dofadr[joint_ids], device=env.device).long(),
    )
    setattr(env, _ACTUATOR_ATTR, params)
  return params


def actuator_effort_l2(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Mean squared actuator force, each as a fraction of its limit."""
  asset: Entity = env.scene[asset_cfg.name]
  limit = _actuator_params(env, asset)[2]
  return torch.mean(torch.square(asset.data.actuator_force / limit), dim=-1)


def actuator_overload(
  env: ManagerBasedRlEnv,
  threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Force the position controllers ask for beyond ``threshold`` of the
  actuators' limits, in limits, summed over the actuators.

  Measured on what is asked for, ``kp * (target - q) - kd * qvel``, not on
  the clipped force: a target far out of reach keeps a gradient to come back.
  """
  asset: Entity = env.scene[asset_cfg.name]
  stiffness, damping, limit, qpos_ids, qvel_ids = _actuator_params(env, asset)
  data = env.sim.data
  target = data.ctrl[:, asset.indexing.ctrl_ids]
  demand = (
    stiffness * (target - data.qpos[:, qpos_ids]) - damping * (data.qvel[:, qvel_ids])
  )
  return torch.sum(((demand / limit).abs() - threshold).clamp(min=0.0), dim=-1)


def ground_force_excess(
  env: ManagerBasedRlEnv,
  contact_cfg: GroundContactCfg,
  body_threshold: float,
  feet_threshold: float,
) -> torch.Tensor:
  """Ground force peaks, in robot weights, beyond ``body_threshold`` on any
  body other than a foot and ``feet_threshold`` on the feet together."""
  contact = get_ground_contact(env, contact_cfg)
  load = contact.peak_force / contact.weight
  body = (load[:, ~contact.is_foot] - body_threshold).clamp(min=0.0).sum(dim=-1)
  feet = (load[:, contact.is_foot].sum(dim=-1) - feet_threshold).clamp(min=0.0)
  # Nothing was commanded yet in the step an episode starts with.
  return torch.where(env.episode_length_buf <= 1, torch.zeros_like(body), body + feet)


def base_speed_l2(
  env: ManagerBasedRlEnv,
  ang_scale: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared linear speed of the base plus ``ang_scale`` times its angular
  speed, squared."""
  asset: Entity = env.scene[asset_cfg.name]
  lin = torch.sum(torch.square(asset.data.root_link_lin_vel_w), dim=-1)
  ang = torch.sum(torch.square(asset.data.root_link_ang_vel_w), dim=-1)
  return lin + ang_scale**2 * ang


def base_acceleration_excess(
  env: ManagerBasedRlEnv, sensor_name: str, threshold: float
) -> torch.Tensor:
  """Accelerometer reading beyond ``threshold`` [m/s^2] away from gravity, in g."""
  acc = env.scene[sensor_name].data
  deviation = (torch.linalg.vector_norm(acc, dim=-1) - 9.81).abs()
  cost = (deviation - threshold).clamp(min=0.0) / 9.81
  return torch.where(env.episode_length_buf <= 1, torch.zeros_like(cost), cost)
