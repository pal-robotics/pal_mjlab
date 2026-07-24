from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply, quat_inv, quat_mul

from pal_mjlab.tasks.manipulation.mdp.commands import LiftingCommand
from pal_mjlab.tasks.manipulation.mdp.contact_sensor import site_contact_both_fingers

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def object_position_in_robot_root_frame(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command: LiftingCommand = env.command_manager.get_term(command_name)
  return quat_apply(
    quat_inv(robot.data.root_link_quat_w),
    command.object_pos_w - robot.data.root_link_pos_w,
  )


def object_orientation_in_robot_root_frame(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command: LiftingCommand = env.command_manager.get_term(command_name)
  return quat_mul(quat_inv(robot.data.root_link_quat_w), command.object_quat_w)


def target_position_in_robot_base_frame(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command: LiftingCommand = env.command_manager.get_term(command_name)
  return quat_apply(
    quat_inv(robot.data.root_link_quat_w),
    command.target_pos - robot.data.root_link_pos_w,
  )


def ee_position_in_robot_base_frame(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  return quat_apply(
    quat_inv(robot.data.root_link_quat_w),
    ee_pos_w - robot.data.root_link_pos_w,
  )


def object_yaw_in_robot_root_frame(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Returns the relative yaw angle of the object represented as [cos(yaw), sin(yaw)]."""
  if asset_cfg is None:
    asset_cfg = SceneEntityCfg("robot")
  obj_quat = object_orientation_in_robot_root_frame(env, command_name, asset_cfg)
  _, _, yaw = euler_xyz_from_quat(obj_quat)
  return torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)


def object_both__contact_fingers(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  site_names: list[str],
  threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  min_dist: float = 0.0,
  false_negative_rate: float = 0.0,
) -> torch.Tensor:
  """Returns a binary flag [B, 1] indicating if both fingertips are in contact with the object."""
  contact_both = site_contact_both_fingers(
    env=env,
    sensor_name=sensor_name,
    site_names=site_names,
    threshold=threshold,
    asset_cfg=asset_cfg,
    min_dist=min_dist,
  )

  # Check physical collision sensors
  try:
    sensor = env.scene[sensor_name]
    if getattr(sensor, "data", None) is not None:
      data = sensor.data
      if data.found is not None:
        actual_contact = (data.found > 0).all(dim=-1).float()
        contact_both = contact_both * actual_contact
  except KeyError:
    pass

  if false_negative_rate > 0.0:
    mask = (torch.rand_like(contact_both) >= false_negative_rate).float()
    contact_both = contact_both * mask
  return contact_both.unsqueeze(-1)


object_both_contact_fingers = object_both__contact_fingers


def reached_flag(
  env: ManagerBasedRlEnv,
  command_name: str = "lift_height",
) -> torch.Tensor:
  """Returns a binary flag [B, 1] that is 1.0 once the object has reached the target goal.

  This is a privileged-but-real-robot-observable signal: the robot knows the target
  position and has contact/force sensing, so this flag can be reconstructed on hardware.
  Exposing it explicitly helps both the actor (to switch behavioral modes) and the
  critic (to accurately estimate value across the phase boundary).
  """
  command: LiftingCommand = env.command_manager.get_term(command_name)
  reached = getattr(
    command, "reached", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  )
  return reached.float().unsqueeze(-1)
