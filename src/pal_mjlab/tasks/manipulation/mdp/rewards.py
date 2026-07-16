from __future__ import annotations

import functools

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from pal_mjlab.tasks.manipulation.mdp.commands import LiftingCommand
from pal_mjlab.tasks.manipulation.mdp.contact_sensor import site_contact_both_fingers


def freeze_on_reached(fn):
  """Decorator to freeze a reward term when the command.reached status is True.

  The reward will be locked to the value it had at the moment reached became True,
  preventing changes or decay during the post-reached phases (e.g. releasing/falling).
  """
  @functools.wraps(fn)
  def wrapper(env: ManagerBasedRlEnv, *args, **kwargs):
    command_name = kwargs.get("command_name", None)
    if command_name is None and len(args) > 0 and isinstance(args[0], str):
      command_name = args[0]
    if command_name is None:
      command_name = "lift_height"

    command = env.command_manager.get_term(command_name)
    current_reward = fn(env, *args, **kwargs)

    if not hasattr(command, "frozen_rewards"):
      command.frozen_rewards = {}

    func_name = fn.__name__
    if func_name not in command.frozen_rewards:
      command.frozen_rewards[func_name] = torch.zeros(env.num_envs, device=env.device)

    frozen_val = torch.where(
      command.reached,
      command.frozen_rewards[func_name],
      current_reward
    )
    command.frozen_rewards[func_name] = frozen_val
    return frozen_val

  return wrapper


def contact_penalty(env: ManagerBasedRlEnv, sensor_names: list[str]) -> torch.Tensor:
  contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for name in sensor_names:
    sensor: ContactSensor = env.scene[name]
    contact |= sensor.data.found.any(dim=-1)
  return contact.float()


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


def fingertip_cube_alignment_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg | None = None,
  std: float = 0.15,
  power: int = 1,
  as_penalty: bool = False,
  sensor_name: str | None = None,
  site_names: list[str] | None = None,
) -> torch.Tensor:
  """Rewards/penalizes the relative yaw misalignment between the gripper squeeze axis and the cube in the robot base frame.

  The penalty/reward remains active throughout the episode (even after contact).
  """
  import math

  from mjlab.utils.lab_api.math import euler_xyz_from_quat

  if asset_cfg is None:
    asset_cfg = SceneEntityCfg("robot")
  robot: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)

  # 1. Get cube orientation in robot root frame, and extract its yaw
  from pal_mjlab.tasks.manipulation.mdp.observations import (
    object_orientation_in_robot_root_frame,
  )
  box_quat_root = object_orientation_in_robot_root_frame(env, command_name, asset_cfg)
  _, _, box_yaw_root = euler_xyz_from_quat(box_quat_root)

  # 2. Locate fingertip sites and calculate the squeeze axis in the robot root frame
  fingertip_site_names = [s for s in robot.site_names if "fingertip" in s]
  assert len(fingertip_site_names) == 2, "Expected exactly 2 fingertip sites"

  left_idx = robot.site_names.index(fingertip_site_names[0])
  right_idx = robot.site_names.index(fingertip_site_names[1])

  p_left = robot.data.site_pos_w[:, left_idx]
  p_right = robot.data.site_pos_w[:, right_idx]

  v_squeeze_w = p_left - p_right
  
  # Rotate squeeze vector to robot root frame and compute its yaw
  root_rot_w = robot.data.root_link_quat_w
  v_squeeze_root = quat_apply(quat_inv(root_rot_w), v_squeeze_w)
  ee_yaw_root = torch.atan2(v_squeeze_root[:, 1], v_squeeze_root[:, 0])

  # 3. Compute relative yaw between the squeeze axis and the box
  yaw_diff = ee_yaw_root - box_yaw_root

  # Wrap difference to [-pi/4, pi/4] due to 90-degree rotational symmetry of the cube
  wrapped_yaw_diff = (yaw_diff + math.pi / 4.0) % (math.pi / 2.0) - math.pi / 4.0
  angle_rad = torch.abs(wrapped_yaw_diff)

  # 4. Scale by distance
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  distance = torch.norm(ee_pos_w - command.object_pos_w, dim=-1)
  distance_scale = torch.exp(-distance / std)

  # Check if contact is established
  if sensor_name is not None and site_names is not None:
    from pal_mjlab.tasks.manipulation.mdp.observations import (
      object_both__contact_fingers,
    )
    contact = object_both__contact_fingers(
      env, sensor_name, site_names, asset_cfg=asset_cfg
    ).squeeze(-1)
  else:
    contact = torch.zeros(env.num_envs, device=env.device)

  if as_penalty:
    # Penalize the yaw misalignment angle (in radians) directly
    reward = angle_rad * distance_scale
  else:
    # Reward yaw alignment (mapping angle to [0, pi/4])
    reward = (math.pi / 4.0 - angle_rad) * distance_scale

  return reward * (1.0 - contact)


@freeze_on_reached
def gripper_open_during_approach_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  std: float = 0.08,
  max_open: float = 0.07,
) -> torch.Tensor:
  """Rewards keeping the gripper open when far from the object.

  The reward is proportional to the gripper joint opening and scaled by the distance to the object.
  As the robot gets closer to the object, this reward decays to 0, allowing the gripper to close.
  """
  robot = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)

  # Get distance between end-effector and object
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  distance = torch.norm(ee_pos_w - command.object_pos_w, dim=-1)

  # Get gripper joint position
  joint_ids, _ = robot.find_joints("gripper_right_finger_joint")
  gripper_pos = robot.data.joint_pos[:, joint_ids[0]]

  # Scale that goes to 1.0 when far and 0.0 when close
  approach_scale = 1.0 - torch.exp(-distance / std)

  # Normalized gripper opening
  normalized_open = torch.clamp(gripper_pos / max_open, 0.0, 1.0)

  return approach_scale * normalized_open


def top_surface_penetration_term(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float = 0.008,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
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

    is_inside = (torch.abs(x) <= half_x) & (torch.abs(y) <= half_y) & (torch.abs(z) <= half_z)

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



def arm_right_1_joint_limit_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  threshold: float = -0.4,
  scale: float = 4.0,
  max_penalty: float = 10.0,
) -> torch.Tensor:
  """Penalizes the right arm joint 1 when its position relative to the default configuration drops below a threshold.

  The penalty grows exponentially with respect to the violation.
  """
  robot: Entity = env.scene[asset_cfg.name]
  joint_ids, _ = robot.find_joints("arm_right_1_joint")
  joint_pos = robot.data.joint_pos[:, joint_ids[0]]
  default_joint_pos = robot.data.default_joint_pos[:, joint_ids[0]]

  joint_pos_rel = joint_pos - default_joint_pos
  violation = torch.clamp(threshold - joint_pos_rel, min=0.0)
  penalty = torch.exp(scale * violation) - 1.0
  return torch.clamp(penalty, max=max_penalty)



def object_released_on_floor_term(
  env: ManagerBasedRlEnv,
  command_name: str,
  floor_z: float = 0.1,
) -> torch.Tensor:
  """Terminates the episode when the object has reached the target and fallen to the floor."""
  command: LiftingCommand = env.command_manager.get_term(command_name)
  on_floor = command.object_pos_w[:, 2] < floor_z

  from pal_mjlab.tasks.manipulation.mdp.contact_sensor import site_contact_both_fingers
  contact_both = site_contact_both_fingers(
    env,
    sensor_name="box_fingertip_contact",
    site_names=["gripper_right_fingertip_.*_site"],
  ).bool()

  return command.reached & on_floor & ~contact_both


@freeze_on_reached
def object_goal_distance_adaptive(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  sensor_name: str,
  site_names: list[str],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  coordinate_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
  command: LiftingCommand = env.command_manager.get_term(command_name)
  contact_both = site_contact_both_fingers(
    env, sensor_name, site_names, asset_cfg=asset_cfg
  ).bool()

  diff = command.target_pos - command.object_pos_w
  weights = torch.tensor(coordinate_weights, device=env.device)
  weighted_diff = diff * weights
  distance = torch.norm(weighted_diff, dim=-1)

  return (~command.object_on_table & contact_both) * (1.0 - torch.tanh(distance / std))


@freeze_on_reached
def object_is_lifted_adaptive(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  site_names: list[str],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  min_weight: float = 0.1,
  max_weight: float = 1.5,
  lift_threshold: float = 0.03,
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


@freeze_on_reached
def object_ee_distance_adaptive(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg | None = None,
  min_reaching_reward: float = 0.0,
  deactivate_on_contact: bool = False,
  sensor_name: str | None = None,
  site_names: list[str] | None = None,
) -> torch.Tensor:
  if asset_cfg is None:
    asset_cfg = SceneEntityCfg("robot")
  robot: Entity = env.scene[asset_cfg.name]
  command: LiftingCommand = env.command_manager.get_term(command_name)
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  distance = torch.norm(ee_pos_w - command.object_pos_w, dim=-1)

  distance_reward = 1.0 - torch.tanh(distance / std)
  reward = torch.clamp(distance_reward, min=min_reaching_reward, max=1.0)

  if deactivate_on_contact and sensor_name is not None and site_names is not None:
    from pal_mjlab.tasks.manipulation.mdp.observations import (
      object_both__contact_fingers,
    )
    contact = object_both__contact_fingers(
      env, sensor_name, site_names, asset_cfg=asset_cfg
    ).squeeze(-1)
    reward = reward * (1.0 - contact)

  return reward


@freeze_on_reached
def fingertip_cube_alignment_reward_adaptive(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg | None = None,
  std: float = 0.15,
  power: int = 1,
  as_penalty: bool = False,
  sensor_name: str | None = None,
  site_names: list[str] | None = None,
) -> torch.Tensor:
  reward = fingertip_cube_alignment_reward(
    env=env,
    command_name=command_name,
    asset_cfg=asset_cfg,
    std=std,
    power=power,
    as_penalty=as_penalty,
    sensor_name=sensor_name,
    site_names=site_names,
  )
  return reward


def release_cube_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  max_open: float = 0.08,
) -> torch.Tensor:
  command: LiftingCommand = env.command_manager.get_term(command_name)
  reached = getattr(command, "reached", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)).float()

  # Get gripper joint velocity
  robot = env.scene[asset_cfg.name]
  joint_ids, _ = robot.find_joints("gripper_right_finger_joint")
  gripper_vel = robot.data.joint_vel[:, joint_ids[0]]

  # Reward only active opening (positive velocity) when reached is True
  active_opening = torch.clamp(gripper_vel, min=0.0)
  return reached * active_opening


def object_falling_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  command: LiftingCommand = env.command_manager.get_term(command_name)
  reached = getattr(command, "reached", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)).float()

  from pal_mjlab.tasks.manipulation.mdp.contact_sensor import site_contact_both_fingers
  contact_both = site_contact_both_fingers(
    env,
    sensor_name="box_fingertip_contact",
    site_names=["gripper_right_fingertip_.*_site"],
  ).bool()

  # Binary reward: 1.0 if the object's velocity along the Z axis is negative (falling down)
  z_velocity = command.object.data.root_link_lin_vel_w[:, 2]
  z_decreasing = (z_velocity < -1e-5).float()
  return reached * (~contact_both).float() * z_decreasing


def object_contact_both_fingers_adaptive(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  site_names: list[str],
  command_name: str = "lift_height",
) -> torch.Tensor:
  contact = site_contact_both_fingers(
    env, sensor_name, site_names
  ).float()
  return contact


@freeze_on_reached
def object_table_sliding_penalty_adaptive(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Penalizes the linear velocity of the object (in the XY plane) when it is on the table,
  only before the target has been reached.
  """
  command: LiftingCommand = env.command_manager.get_term(command_name)
  
  obj = command.object
  lin_vel_xy = obj.data.root_link_lin_vel_w[:, :2]
  speed_xy = torch.norm(lin_vel_xy, dim=-1)
  
  return command.object_on_table.float() * speed_xy
