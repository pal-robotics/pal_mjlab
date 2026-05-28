from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from pal_mjlab.tasks.manipulation.mdp.commands import LiftingCommand
from pal_mjlab.tasks.manipulation.mdp.contact_sensor import site_contact_both_fingers


def object_ee_distance(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg | None = None,
  min_reaching_reward: float = 0.0,
) -> torch.Tensor:
  if asset_cfg is None:
    asset_cfg = SceneEntityCfg("robot")
  robot: Entity = env.scene[asset_cfg.name]
  command: LiftingCommand = env.command_manager.get_term(command_name)
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  distance = torch.norm(ee_pos_w - command.object_pos_w, dim=-1)

  distance_reward = 1.0 - torch.tanh(distance / std)

  # Never be a penalization (min clip) and maintain 1.0 max
  return torch.clamp(distance_reward, min=min_reaching_reward, max=1.0)





def object_is_lifted(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  site_names: list[str],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  min_weight: float = 0.5,
  max_weight: float = 5.0,
  lift_threshold: float = 0.05,
) -> torch.Tensor:
  command: LiftingCommand = env.command_manager.get_term(command_name)

  fingers_close = site_contact_both_fingers(
    env, sensor_name, site_names, asset_cfg=asset_cfg
  ).bool()

  elevation = command.object_bottom_z - command.table_surface_z
  elevation = torch.clamp(elevation, min=0.0, max=lift_threshold)

  ratio = max_weight / min_weight
  scale = min_weight * (ratio ** (elevation / lift_threshold))

  is_lifted = (~command.object_on_table & fingers_close).float()
  return is_lifted * scale


def object_goal_distance(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  sensor_name: str,
  site_names: list[str],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  command: LiftingCommand = env.command_manager.get_term(command_name)
  contact_both = site_contact_both_fingers(
    env, sensor_name, site_names, asset_cfg=asset_cfg
  ).bool()

  distance = torch.norm(command.target_pos - command.object_pos_w, dim=-1)
  return (~command.object_on_table & contact_both) * (1.0 - torch.tanh(distance / std))


def contact_penalty(env: ManagerBasedRlEnv, sensor_names: list[str]) -> torch.Tensor:
  contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for name in sensor_names:
    sensor: ContactSensor = env.scene[name]
    contact |= sensor.data.found.any(dim=-1)
  return contact.float()


def arm_contact_while_lifting_term(
  env: ManagerBasedRlEnv,
  sensor_names: list[str],
  command_name: str,
  sensor_name: str,
  site_names: list[str],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for name in sensor_names:
    sensor: ContactSensor = env.scene[name]
    contact |= sensor.data.found.any(dim=-1)
  lifted = object_is_lifted(
    env=env,
    command_name=command_name,
    sensor_name=sensor_name,
    site_names=site_names,
    asset_cfg=asset_cfg,
  ).bool()
  return (contact & lifted).float()


def object_contact_both_fingers(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  return sensor.data.found.all(dim=-1).float()


def contact_sensor_found(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  return sensor.data.found.float()


def action_rate_l2(
  env: ManagerBasedRlEnv, action_indices: list[int] | None = None
) -> torch.Tensor:
  if action_indices is None:
    action_diff = env.action_manager.action - env.action_manager.prev_action
  else:
    action_diff = (
      env.action_manager.action[:, action_indices]
      - env.action_manager.prev_action[:, action_indices]
    )
  return torch.sum(torch.square(action_diff), dim=1)


def ee_vel_penalty(
  env: ManagerBasedRlEnv,
  threshold: float = 0.06,
  scale: float = 50.0,
  max_penalty: float = 10.0,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  ee_lin_vel_w = robot.data.site_lin_vel_w[:, asset_cfg.site_ids].squeeze(1)
  ee_vel_norm = torch.linalg.norm(ee_lin_vel_w, dim=-1)

  excess_vel = torch.clamp(ee_vel_norm - threshold, min=0.0)
  penalty = torch.exp(scale * excess_vel) - 1.0
  return torch.clamp(penalty, max=max_penalty)
