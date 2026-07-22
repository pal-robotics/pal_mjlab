from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from pal_mjlab.tasks.manipulation.mdp.commands import LiftingCommand
from pal_mjlab.tasks.manipulation.mdp.contact_sensor import site_contact_both_fingers

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def top_surface_penetration_term(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float = 0.008,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminates the episode if either fingertip penetrates the top surface of the cube beyond the threshold."""
  robot: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  box = command.object
  geom_id = box.indexing.geom_ids[0]
  box_sizes = env.sim.model.geom_size[:, geom_id]

  # Locate fingertip sites
  fingertip_site_names = [s for s in robot.site_names if "fingertip" in s]
  assert len(fingertip_site_names) == 2, "Expected exactly 2 fingertip sites"

  left_idx = robot.site_names.index(fingertip_site_names[0])
  right_idx = robot.site_names.index(fingertip_site_names[1])

  p_left = robot.data.site_pos_w[:, left_idx]
  p_right = robot.data.site_pos_w[:, right_idx]

  box_pos = (
    box.data.root_pos_w
    if hasattr(box.data, "root_pos_w")
    else box.data.geom_pos_w[:, 0]
  )
  box_quat = (
    box.data.root_quat_w
    if hasattr(box.data, "root_quat_w")
    else box.data.geom_quat_w[:, 0]
  )

  # Transform to local frame
  p_left_local = quat_apply(quat_inv(box_quat), p_left - box_pos)
  p_right_local = quat_apply(quat_inv(box_quat), p_right - box_pos)

  half_x = box_sizes[:, 0]
  half_y = box_sizes[:, 1]
  half_z = box_sizes[:, 2]

  # Helper to compute penetration depth for a given fingertip
  def get_top_penetration_depth(p_local: torch.Tensor) -> torch.Tensor:
    x = p_local[:, 0]
    y = p_local[:, 1]
    z = p_local[:, 2]

    is_inside = (
      (torch.abs(x) <= half_x) & (torch.abs(y) <= half_y) & (torch.abs(z) <= half_z)
    )

    # Distances to each face
    dist_x = half_x - torch.abs(x)
    dist_y = half_y - torch.abs(y)
    dist_z = half_z - torch.abs(z)

    dists = torch.stack([dist_x, dist_y, dist_z], dim=-1)
    min_dist, min_axis = torch.min(dists, dim=-1)

    # Condition: inside, closest to top face (min_axis == 2) and in upper half (z > 0)
    is_top_penetration = is_inside & (min_axis == 2) & (z > 0)

    # Return penetration depth if top penetration, else 0.0
    return torch.where(is_top_penetration, min_dist, torch.zeros_like(min_dist))

  left_depth = get_top_penetration_depth(p_left_local)
  right_depth = get_top_penetration_depth(p_right_local)

  # Terminate if left or right penetration depth exceeds threshold
  terminated = (left_depth > threshold) | (right_depth > threshold)
  return terminated


def object_released_on_floor_term(
  env: ManagerBasedRlEnv,
  command_name: str,
  floor_z: float = 0.1,
) -> torch.Tensor:
  """Terminates the episode when the object has reached the target and fallen to the floor."""
  command: LiftingCommand = env.command_manager.get_term(command_name)
  on_floor = command.object_pos_w[:, 2] < floor_z

  contact_both = site_contact_both_fingers(
    env,
    sensor_name="box_fingertip_contact",
    site_names=["gripper_right_fingertip_.*_site"],
  ).bool()

  return command.reached & on_floor & ~contact_both


def cube_contact_with_table_after_reached_term(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Terminates the episode when the cube contacts the table after the target has been reached."""
  command: LiftingCommand = env.command_manager.get_term(command_name)
  return command.reached & command.object_on_table


def cube_fell_off_table_term(
  env: ManagerBasedRlEnv,
  command_name: str,
  floor_z: float = 0.1,
) -> torch.Tensor:
  """Terminates the episode early when the cube falls to the floor before the goal is reached.

  This is a failure condition: the robot dropped the object before lifting it to the
  target height. It is distinct from `object_released_on_floor_term` (the success
  termination) which only fires *after* `reached` is True.
  """
  command: LiftingCommand = env.command_manager.get_term(command_name)
  on_floor = command.object_pos_w[:, 2] < floor_z
  return ~command.reached & on_floor
