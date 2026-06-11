from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply, quat_inv, quat_mul

from pal_mjlab.tasks.manipulation.mdp.commands import LiftingCommand


def object_position_in_robot_root_frame(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
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
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command: LiftingCommand = env.command_manager.get_term(command_name)
  return quat_mul(quat_inv(robot.data.root_link_quat_w), command.object_quat_w)


def object_pose_6d_in_robot_root_frame(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  obj_pos = object_position_in_robot_root_frame(env, command_name, asset_cfg)
  obj_quat = object_orientation_in_robot_root_frame(env, command_name, asset_cfg)
  roll, pitch, yaw = euler_xyz_from_quat(obj_quat)
  obj_euler = torch.stack([roll, pitch, yaw], dim=-1)
  return torch.cat([obj_pos, obj_euler], dim=-1)


def target_position_in_robot_base_frame(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  command: LiftingCommand = env.command_manager.get_term(command_name)
  return quat_apply(
    quat_inv(robot.data.root_link_quat_w),
    command.target_pos - robot.data.root_link_pos_w,
  )


def ee_position_in_robot_base_frame(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  return quat_apply(
    quat_inv(robot.data.root_link_quat_w),
    ee_pos_w - robot.data.root_link_pos_w,
  )


def camera_rgbd(
  env: ManagerBasedRlEnv, sensor_name: str, cutoff_distance: float = 1.5
) -> torch.Tensor:
  rgb = manipulation_mdp.camera_rgb(env, sensor_name)
  depth = manipulation_mdp.camera_depth(
    env, sensor_name, cutoff_distance=cutoff_distance
  )
  return torch.cat([rgb, depth], dim=1)


def head_camera_keypoints(
  env: ManagerBasedRlEnv,
  camera_name: str = "head_realsense_camera",
  noise_std: float = 0.0,
  box_entity_name: str = "box",
  robot_entity_name: str = "robot",
) -> torch.Tensor:
  # 1. Box corners
  box = env.scene[box_entity_name]
  geom_id = box.indexing.geom_ids[0]
  box_sizes = env.sim.model.geom_size[:, geom_id]

  num_envs = env.num_envs
  device = env.device

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

  # Find which local axis of the box points "up" (aligned with world Z axis)
  unit_x = torch.tensor([1.0, 0.0, 0.0], device=device).expand(num_envs, -1)
  unit_y = torch.tensor([0.0, 1.0, 0.0], device=device).expand(num_envs, -1)
  unit_z = torch.tensor([0.0, 0.0, 1.0], device=device).expand(num_envs, -1)

  axis_x_w = quat_apply(box_quat, unit_x)
  axis_y_w = quat_apply(box_quat, unit_y)
  axis_z_w = quat_apply(box_quat, unit_z)

  align_x = axis_x_w[:, 2]
  align_y = axis_y_w[:, 2]
  align_z = axis_z_w[:, 2]

  alignments = torch.stack([align_x.abs(), align_y.abs(), align_z.abs()], dim=1)
  up_axis = torch.argmax(alignments, dim=1)

  up_align = torch.where(
    up_axis == 0, align_x, torch.where(up_axis == 1, align_y, align_z)
  )
  up_sign = torch.sign(up_align)

  h1_axis = torch.where(up_axis == 0, 1, 0)
  h2_axis = torch.where(up_axis == 2, 1, 2)

  size_up = torch.gather(box_sizes, 1, up_axis.unsqueeze(1)).squeeze(1)
  size_h1 = torch.gather(box_sizes, 1, h1_axis.unsqueeze(1)).squeeze(1)
  size_h2 = torch.gather(box_sizes, 1, h2_axis.unsqueeze(1)).squeeze(1)

  local_corners = torch.zeros(num_envs, 4, 3, device=device)
  env_indices = torch.arange(num_envs, device=device)

  up_val = up_sign * 1.5 * size_up
  local_corners[env_indices, 0, up_axis] = up_val
  local_corners[env_indices, 1, up_axis] = up_val
  local_corners[env_indices, 2, up_axis] = up_val
  local_corners[env_indices, 3, up_axis] = up_val

  local_corners[env_indices, 0, h1_axis] = size_h1
  local_corners[env_indices, 0, h2_axis] = size_h2

  local_corners[env_indices, 1, h1_axis] = size_h1
  local_corners[env_indices, 1, h2_axis] = -size_h2

  local_corners[env_indices, 2, h1_axis] = -size_h1
  local_corners[env_indices, 2, h2_axis] = size_h2

  local_corners[env_indices, 3, h1_axis] = -size_h1
  local_corners[env_indices, 3, h2_axis] = -size_h2

  box_pos_exp = box_pos.unsqueeze(1).expand(-1, 4, -1)
  box_quat_exp = box_quat.unsqueeze(1).expand(-1, 4, -1)
  corners_3d_w = box_pos_exp + quat_apply(box_quat_exp, local_corners)

  # 2. Fingertip sites
  robot = env.scene[robot_entity_name]
  fingertip_site_names = [s for s in robot.site_names if "fingertip" in s]
  fingertip_pos_w = robot.data.site_pos_w[
    :, [robot.site_names.index(name) for name in fingertip_site_names]
  ]

  keypoints_3d_w = torch.cat([corners_3d_w, fingertip_pos_w], dim=1)

  # 3. Camera Kinematics
  cam_idx = env.sim.mj_model.camera(f"robot/{camera_name}").id
  cam_pos = env.sim.data.cam_xpos[:, cam_idx]
  cam_xmat = env.sim.data.cam_xmat[:, cam_idx]
  cam_fovy = env.sim.mj_model.cam_fovy[cam_idx]

  # 4. Transform to Camera coordinates
  diff = keypoints_3d_w - cam_pos.unsqueeze(1)
  points_c = torch.bmm(diff, cam_xmat)

  # 5. Projection to NDC [-1, 1]
  import math

  fovy_rad = cam_fovy * (math.pi / 180.0)
  focal_length = 1.0 / math.tan(fovy_rad / 2.0)

  x_c = points_c[..., 0]
  y_c = points_c[..., 1]
  z_c = points_c[..., 2]
  z_depth = -z_c

  u_norm = (x_c / (z_depth + 1e-6)) * focal_length
  v_norm = (-y_c / (z_depth + 1e-6)) * focal_length
  keypoints_2d = torch.stack([v_norm, u_norm], dim=-1)

  # 6. Noise curriculum addition
  if noise_std > 0.0:
    keypoints_2d = keypoints_2d + torch.randn_like(keypoints_2d) * noise_std

  return keypoints_2d.reshape(num_envs, -1)


def object_width(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Returns the horizontal full width of the object."""
  command = env.command_manager.get_term(command_name)
  box = command.object
  geom_id = box.indexing.geom_ids[0]
  box_sizes = env.sim.model.geom_size[:, geom_id]
  return box_sizes[:, 0:1] * 2.0


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
