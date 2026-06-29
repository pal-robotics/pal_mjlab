from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_inv

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


def object_goal_distance(
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


def fingertip_cube_alignment_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg | None = None,
  std: float = 0.15,
  power: int = 1,
) -> torch.Tensor:
  """Rewards the alignment of the fingertips' squeeze direction with the cube's principal axes, computed in the robot root frame.

  This only constrains the line connecting the two fingertips to be perpendicular to the cube's faces.
  """
  if asset_cfg is None:
    asset_cfg = SceneEntityCfg("robot")
  robot: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_term(command_name)

  # 1. Locate fingertip sites and calculate the squeeze axis in the robot root frame
  fingertip_site_names = [s for s in robot.site_names if "fingertip" in s]
  assert len(fingertip_site_names) == 2, "Expected exactly 2 fingertip sites"

  left_idx = robot.site_names.index(fingertip_site_names[0])
  right_idx = robot.site_names.index(fingertip_site_names[1])

  p_left = robot.data.site_pos_w[:, left_idx]
  p_right = robot.data.site_pos_w[:, right_idx]

  v_squeeze_w = p_left - p_right
  # Rotate squeeze vector to robot root frame
  v_squeeze_root = quat_apply(quat_inv(robot.data.root_link_quat_w), v_squeeze_w)
  v_squeeze_root_norm = v_squeeze_root / torch.norm(
    v_squeeze_root, dim=-1, keepdim=True
  ).clamp(min=1e-6)

  # 2. Get relative box orientation and axes in the robot root frame
  from pal_mjlab.tasks.manipulation.mdp.observations import (
    object_orientation_in_robot_root_frame,
  )

  box_quat_root = object_orientation_in_robot_root_frame(env, command_name, asset_cfg)

  B = env.num_envs
  device = env.device
  unit_x = torch.tensor([1.0, 0.0, 0.0], device=device).expand(B, -1)
  unit_y = torch.tensor([0.0, 1.0, 0.0], device=device).expand(B, -1)

  box_axes_root = [quat_apply(box_quat_root, unit) for unit in (unit_x, unit_y)]

  # 3. Calculate max absolute similarity between the squeeze axis and box axes in the root frame
  similarities = torch.stack(
    [
      torch.abs(torch.sum(v_squeeze_root_norm * box_axis_root, dim=-1))
      for box_axis_root in box_axes_root
    ],
    dim=-1,
  )
  alignment = torch.max(similarities, dim=-1).values
  if power != 1:
    alignment = torch.pow(alignment, power)

  # 4. Scale by distance
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  distance = torch.norm(ee_pos_w - command.object_pos_w, dim=-1)
  distance_scale = torch.exp(-distance / std)

  return alignment * distance_scale


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


def object_table_sliding_penalty(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Penalizes the linear velocity of the object (in the XY plane) when it is on the table."""
  command: LiftingCommand = env.command_manager.get_term(command_name)
  obj: Entity = command.object
  # Linear velocity in the horizontal (XY) plane in world frame
  lin_vel_xy = obj.data.root_link_lin_vel_w[:, :2]
  speed_xy = torch.norm(lin_vel_xy, dim=-1)
  # Only penalize when the object is on the table
  return command.object_on_table.float() * speed_xy


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


_CRITICAL_CONFIGS_CACHE = {}


def occlusion_similarity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  sigma: float = 0.4,
  num_configs: int = 35,
  config_file_path: str | None = None,
) -> torch.Tensor:
  """Penalizes the robot joint configuration if it is similar to any of the top critical self-occlusion configurations.

  Uses a weighted sum of Gaussian RBFs: Sum_i w_i * exp( -||theta - c_i||^2 / (2 * sigma^2) )
  """
  global _CRITICAL_CONFIGS_CACHE

  import os

  if config_file_path is None or not os.path.exists(config_file_path):
    # Fallback to the package-relative path
    package_dir = os.path.dirname(os.path.abspath(__file__))
    fallback_path = os.path.join(package_dir, "critical_configurations.json")
    if os.path.exists(fallback_path):
      config_file_path = fallback_path

  cache_key = (config_file_path, num_configs, env.device)
  if cache_key not in _CRITICAL_CONFIGS_CACHE:
    import json

    if config_file_path is None or not os.path.exists(config_file_path):
      raise FileNotFoundError(f"Critical configurations file not found at {config_file_path}")

    with open(config_file_path, "r") as f:
      data = json.load(f)

    selected_configs = data["critical_configurations"][:num_configs]
    configs_list = []
    weights_list = []

    total_percentage = sum(c["percentage"] for c in selected_configs)

    for c in selected_configs:
      configs_list.append(c["joints"])
      weights_list.append(c["percentage"] / total_percentage)

    # Convert to PyTorch tensors and move to device
    configs_tensor = torch.tensor(configs_list, dtype=torch.float32, device=env.device)
    weights_tensor = torch.tensor(weights_list, dtype=torch.float32, device=env.device)

    _CRITICAL_CONFIGS_CACHE[cache_key] = (configs_tensor, weights_tensor)

  configs_tensor, weights_tensor = _CRITICAL_CONFIGS_CACHE[cache_key]

  robot: Entity = env.scene[asset_cfg.name]
  joint_names = [f"arm_right_{i}_joint" for i in range(1, 8)]
  joint_ids = []
  for name in joint_names:
    ids, _ = robot.find_joints(name)
    joint_ids.append(ids[0])

  # Get robot's current right arm joint positions (shape: num_envs, 7)
  joint_pos = robot.data.joint_pos[:, joint_ids]

  # Broadcast subtraction to get differences (shape: num_envs, num_configs, 7)
  diff = joint_pos.unsqueeze(1) - configs_tensor.unsqueeze(0)

  # Squared L2 distance (shape: num_envs, num_configs)
  dist_sq = torch.sum(torch.square(diff), dim=-1)

  # RBF similarity (shape: num_envs, num_configs)
  rbf = torch.exp(-dist_sq / (2 * sigma**2))

  # Weighted sum (shape: num_envs,)
  penalty = torch.sum(weights_tensor.unsqueeze(0) * rbf, dim=-1)

  return penalty


# Per-env success-hold counters, keyed by env id so multiple envs don't share state.
_SUCCESS_HOLD_COUNTERS: dict[int, torch.Tensor] = {}


def object_held_at_goal_term(
  env: ManagerBasedRlEnv,
  command_name: str,
  hold_time_s: float = 1.0,
) -> torch.Tensor:
  """Terminates the episode once the object has been continuously held at the goal
  for at least *hold_time_s* seconds.

  A counter is incremented each step the success condition is met (object within
  ``LiftingCommand.cfg.success_threshold`` of the goal and not on the table), and
  reset to zero when the condition breaks.  The episode terminates when the counter
  reaches ``ceil(hold_time_s / env.step_dt)``.

  Args:
    env: The RL environment.
    command_name: Name of the ``LiftingCommand`` term.
    hold_time_s: Seconds the object must be continuously held at the goal before
      the episode is terminated.  Default: 1.0 s.

  Returns:
    Bool tensor of shape ``(num_envs,)`` — True for environments that have held
    the object at the goal long enough.
  """
  global _SUCCESS_HOLD_COUNTERS

  env_id = id(env)
  if env_id not in _SUCCESS_HOLD_COUNTERS:
    _SUCCESS_HOLD_COUNTERS[env_id] = torch.zeros(
      env.num_envs, dtype=torch.float32, device=env.device
    )

  counter = _SUCCESS_HOLD_COUNTERS[env_id]

  command: LiftingCommand = env.command_manager.get_term(command_name)
  # compute_success() returns a bool tensor (num_envs,)
  at_goal = command.compute_success()

  # Increment counter where success, reset where not
  counter = torch.where(at_goal, counter + 1.0, torch.zeros_like(counter))
  _SUCCESS_HOLD_COUNTERS[env_id] = counter

  # Number of env steps required
  hold_steps = hold_time_s / env.step_dt
  return counter >= hold_steps
