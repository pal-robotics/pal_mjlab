from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity

from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from pal_mjlab.tasks.lift.mdp.commands import LiftingCommand
from mjlab.utils.lab_api.math import (
    quat_conjugate,
    quat_mul,
    quat_error_magnitude
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def position_command_error(
    env: ManagerBasedRlEnv,
    command_name: str,
    site_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)

    des_pos_b = command.command[:, :3]

    root_pos_w = asset.data.site_pos_w[:, 0]  # Root site position
    root_quat_w = asset.data.site_quat_w[:, 0]  # Root site quaternion

    # Transform position: p_w = p_root + R_root * p_b
    pos_rotated = quat_mul(
        quat_mul(
            root_quat_w,
            torch.cat(
                [torch.zeros(env.num_envs, 1, device=env.device), des_pos_b], dim=1
            ),
        ),
        quat_conjugate(root_quat_w),
    )[:, 1:]  # Extract xyz from quaternion product
    des_pos_w = root_pos_w + pos_rotated

    current_site_pos_w = asset.data.site_pos_w[:, asset.site_names.index(site_name)]

    pos_error = current_site_pos_w - des_pos_w

    return torch.norm(pos_error, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRlEnv,
    command_name: str,
    site_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)

    des_pos_b = command.command[:, :3]

    root_pos_w = asset.data.site_pos_w[:, 0]  # Root site position
    root_quat_w = asset.data.site_quat_w[:, 0]  # Root site quaternion

    pos_rotated = quat_mul(
        quat_mul(
            root_quat_w,
            torch.cat(
                [torch.zeros(env.num_envs, 1, device=env.device), des_pos_b], dim=1
            ),
        ),
        quat_conjugate(root_quat_w),
    )[:, 1:]  # Extract xyz from quaternion product
    des_pos_w = root_pos_w + pos_rotated

    current_site_pos_w = asset.data.site_pos_w[:, asset.site_names.index(site_name)]

    pos_error = current_site_pos_w - des_pos_w
    distance = torch.norm(pos_error, dim=1)

    return 1 - torch.tanh(distance / std)

def orientation_command_error(
    env: "ManagerBasedRlEnv",
    command_name: str,
    site_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    # Get robot entity and command
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)

    # Desired orientation in base frame (qw, qx, qy, qz)
    des_quat_b = command.command[:, 3:] 

    # Root (base) orientation in world frame
    root_quat_w = asset.data.site_quat_w[:, 0] 

    # Transform desired orientation from base -> world:
    # q_des_w = q_root_w ⊗ q_des_b
    des_quat_w = quat_mul(root_quat_w, des_quat_b)  

    # Current site orientation in world frame
    site_idx = asset.site_names.index(site_name)
    current_quat_w = asset.data.site_quat_w[:, site_idx]  

    # Quaternion error magnitude (angle between quaternions)
    ori_error = quat_error_magnitude(des_quat_w, current_quat_w)  

    return ori_error

def stand_still_joint_deviation_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    abs_error = torch.abs(error)

    # amount beyond the 0.1 margin
    excess = torch.relu(abs_error - 0.005) 

    # per-env penalty: sum of excess across all monitored joints
    penalty = torch.sum(excess, dim=1)
    return penalty

def ee_object_distance(
    env: ManagerBasedRlEnv,
    std: float,
    object_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward the agent for reaching the object using a tanh kernel.

    Returns values in (0, 1], with:
      - ~1 when the EE is on the cube
      - smoothly decaying towards 0 as distance increases
    """
    robot: Entity = env.scene[asset_cfg.name]
    obj: Entity = env.scene[object_name]

    # End-effector world position (assumes a single EE site id).
    ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)  # [N, 3]

    # Object world position (root link).
    obj_pos_w = obj.data.root_link_pos_w  # [N, 3]

    # Geometric distance EE–object.
    dist = torch.norm(ee_pos_w - obj_pos_w, dim=-1)  # [N]

    # IsaacLab-style tanh kernel: 1 when close, ~0 when far.
    return 1.0 - torch.tanh(dist / std)


def object_is_lifted_binary(
    env: ManagerBasedRlEnv,
    minimal_height: float,
    object_name: str,
) -> torch.Tensor:
    """Binary reward: object is above minimal height."""
    obj: Entity = env.scene[object_name]
    obj_height = obj.data.root_link_pos_w[:, 2]          # [N]
    lifted = (obj_height > minimal_height).float()       # 0 or 1
    return lifted

def object_goal_gaussian_distance(
    env: ManagerBasedRlEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    object_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """
    Stage-3 reward: object close to goal, only when lifted.

    Very similar idea to Isaac Lab's `object_goal_distance`, but with a Gaussian
    kernel instead of tanh.
    """
    robot: Entity = env.scene[asset_cfg.name]
    obj: Entity = env.scene[object_name]
    command = env.command_manager.get_term(command_name)

    obj_pos_w = obj.data.root_link_pos_w                 # [N, 3]
    target_pos_w = command.target_pos                    # [N, 3] (world frame)

    # Distance object–target (squared)
    bring_error = torch.sum((target_pos_w - obj_pos_w) ** 2, dim=-1)  # [N]
    bringing = torch.exp(-bring_error / (std ** 2))                   # (0, 1]

    # Lifted gate
    obj_height = obj_pos_w[:, 2]
    lifted = (obj_height > minimal_height).float()                    # [N]

    return lifted * bringing

def fingertips_grasp_binary(
    env: ManagerBasedRlEnv,
    left_sensor_name: str,
    right_sensor_name: str,
) -> torch.Tensor:
    """
    Binary reward: 1 if BOTH fingertip–cube contact sensors report contact,
    otherwise 0.

    Assumes each ContactSensor produces a `.data.found` field of shape:
      [num_envs] or [num_envs, num_slots]
    """
    # Look up the two contact sensors from the scene
    left_sensor = env.scene.sensors[left_sensor_name]
    right_sensor = env.scene.sensors[right_sensor_name]

    # Extract "found" flags
    left_found = left_sensor.data.found         
    right_found = right_sensor.data.found      

    if left_found.ndim > 1:
        left_found = left_found.any(dim=-1)   
    if right_found.ndim > 1:
        right_found = right_found.any(dim=-1) 

    # Both fingertips must be in contact with the cube
    both = torch.logical_and(left_found, right_found)  # [N] boolean

    return both.float()  

def joint_velocity_hinge_penalty(
  env: ManagerBasedRlEnv,
  max_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Quadratic hinge penalty on joint velocities exceeding a symmetric limit.
  Penalizes only the amount by which |v| exceeds max_vel. Returns a negative
  penalty, shaped as the negative squared L2 norm of the excess velocities.
  """
  robot: Entity = env.scene[asset_cfg.name]
  joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  excess = (joint_vel.abs() - max_vel).clamp_min(0.0)
  return (excess**2).sum(dim=-1)




class action_rate_l2_louis:
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]

        _, joint_names = asset.find_joints(
            cfg.params["asset_cfg"].joint_names,
        )
        self._joint_ids = [
            asset.actuator_names.index(jname)
            for jname in joint_names
            if jname in asset.actuator_names
        ]

    def __call__(
        self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        return torch.sum(
            torch.square(
                env.action_manager.action[:, self._joint_ids]
                - env.action_manager.prev_action[:, self._joint_ids]
            ),
            dim=1,
        )
