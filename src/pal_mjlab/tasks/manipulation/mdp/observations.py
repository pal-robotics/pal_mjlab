from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul
from mjlab.tasks.manipulation import mdp as manipulation_mdp
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

def camera_rgbd(env: ManagerBasedRlEnv, sensor_name: str, cutoff_distance: float = 1.0) -> torch.Tensor:
    rgb = manipulation_mdp.camera_rgb(env, sensor_name)
    depth = manipulation_mdp.camera_depth(env, sensor_name, cutoff_distance=cutoff_distance)
    return torch.cat([rgb, depth], dim=1)
