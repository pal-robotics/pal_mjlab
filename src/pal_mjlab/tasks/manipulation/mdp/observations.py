from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply, quat_inv, quat_mul

from pal_mjlab.tasks.manipulation.mdp.commands import LiftingCommand
from pal_mjlab.tasks.manipulation.mdp.contact_sensor import site_contact_both_fingers


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

  # Check for fingertip penetration during play (1 env)
  if env.num_envs == 1:
    try:
      robot = env.scene["robot"]
      fingertip_site_names = [s for s in robot.site_names if "fingertip" in s]
      if len(fingertip_site_names) == 2:
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

        # Helper to check and print penetration for a fingertip
        def check_finger_penetration(name: str, p_local: torch.Tensor):
          x = p_local[:, 0]
          y = p_local[:, 1]
          z = p_local[:, 2]

          is_inside = (torch.abs(x) <= half_x) & (torch.abs(y) <= half_y) & (torch.abs(z) <= half_z)

          if is_inside.any():
            dist_x = half_x - torch.abs(x)
            dist_y = half_y - torch.abs(y)
            dist_z = half_z - torch.abs(z)

            # Stack distances to find which face is closest (where it penetrated)
            dists = torch.stack([dist_x, dist_y, dist_z], dim=-1)
            min_axis = torch.argmin(dists, dim=-1).item()
            min_dist = dists[0, min_axis].item()

            if min_axis == 2 and z.item() > 0:
              print(f"[PENETRATION DETECTED] {name} fingertip penetrated TOP surface! Depth: {min_dist:.4f} m")

        check_finger_penetration("Left", p_left_local)
        check_finger_penetration("Right", p_right_local)
    except Exception as e:
      pass

  return box_sizes[:, 1:2] * 2.0


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


# ---------------------------------------------------------------------------
# Occlusion-dropout wrappers
# ---------------------------------------------------------------------------

def _get_shared_occlusion_mask(
  env: ManagerBasedRlEnv,
  p_drop: float | str,
  command_name: str = "lift_height",
) -> torch.Tensor:
  """Return a per-environment boolean occlusion mask, shared across obs terms.

  The mask is resampled exactly once per policy step (keyed on
  ``env.common_step_counter``), so every obs function that calls this within
  the same step will see the **same** mask — guaranteeing all-or-nothing
  dropout across position, yaw, and width simultaneously.
  """
  step = env.common_step_counter
  cached = getattr(env, "_occlusion_cache", None)
  if cached is None or cached[0] != step:
    if isinstance(p_drop, str) and p_drop == "dynamic":
      # Compute dynamic p_drop based on distance to the object
      robot = env.scene["robot"]
      ee_idx = robot.site_names.index("gripper_right_grasping_site")
      ee_pos_w = robot.data.site_pos_w[:, ee_idx]
      
      command = env.command_manager.get_term(command_name)
      object_pos_w = command.object_pos_w
      
      dist = torch.norm(ee_pos_w - object_pos_w, dim=-1)
      
      # Linear mapping: far (>= 0.6m) -> 0.0, close (<= 0.05m) -> 0.75
      p_drop_tensor = 0.4 * (1.0 - (dist - 0.05) / (0.6 - 0.05))
      p_drop_tensor = torch.clamp(p_drop_tensor, min=0.05, max=0.4)
    else:
      p_drop_tensor = p_drop

    if isinstance(p_drop_tensor, float) and p_drop_tensor <= 0.0:
      mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    else:
      mask = torch.rand(env.num_envs, device=env.device) < p_drop_tensor
    env._occlusion_cache = (step, mask)
  return env._occlusion_cache[1]  # shape: (num_envs,)


def _apply_shared_occlusion_dropout(
  obs: torch.Tensor,
  env: ManagerBasedRlEnv,
  p_drop: float | str,
  command_name: str = "lift_height",
) -> torch.Tensor:
  """Zero the observation for environments selected by the shared mask.

  All obs functions that call this within the same policy step share one
  Bernoulli draw, so position / yaw / width are always occluded together.
  """
  if isinstance(p_drop, float) and p_drop <= 0.0:
    return obs
  mask = _get_shared_occlusion_mask(env, p_drop, command_name=command_name).unsqueeze(-1)  # (num_envs, 1)
  return torch.where(mask, torch.zeros_like(obs), obs)


def object_position_in_robot_root_frame_dropout(
  env: ManagerBasedRlEnv,
  command_name: str,
  p_drop: float | str = 0.3,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Object position in robot root frame with shared stochastic occlusion dropout."""
  obs = object_position_in_robot_root_frame(env, command_name, asset_cfg)
  return _apply_shared_occlusion_dropout(obs, env, p_drop, command_name=command_name)


def object_yaw_in_robot_root_frame_dropout(
  env: ManagerBasedRlEnv,
  command_name: str,
  p_drop: float | str = 0.3,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Object yaw (cos/sin) in robot root frame with shared stochastic occlusion dropout."""
  obs = object_yaw_in_robot_root_frame(env, command_name, asset_cfg)
  return _apply_shared_occlusion_dropout(obs, env, p_drop, command_name=command_name)


def object_width_dropout(
  env: ManagerBasedRlEnv,
  command_name: str,
  p_drop: float | str = 0.3,
) -> torch.Tensor:
  """Object width with shared stochastic occlusion dropout."""
  obs = object_width(env, command_name)
  return _apply_shared_occlusion_dropout(obs, env, p_drop, command_name=command_name)


def object_both__contact_fingers(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  site_names: list[str],
  threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  min_dist: float = 0.0,
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
  return contact_both.unsqueeze(-1)


object_both_contact_fingers = object_both__contact_fingers
