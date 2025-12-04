"""Useful methods for MDP terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.nan_guard import NanGuard

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

def object_out_of_bounds_box_local(
    env: ManagerBasedRlEnv,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    object_name: str = "cube",
    robot_name: str = "robot",
) -> torch.Tensor:
    """True if object leaves the [x_min, x_max] x [y_min, y_max] box
    around the robot base, in the robot's local XY frame.
    """
    robot: Entity = env.scene[robot_name]
    obj: Entity = env.scene[object_name]

    # Robot base and object positions in world frame: [N, 3]
    robot_pos_w = robot.data.root_link_pos_w
    obj_pos_w = obj.data.root_link_pos_w

    # Position of cube relative to robot base in XY: [N, 2]
    rel_xy = obj_pos_w[:, :2] - robot_pos_w[:, :2]

    dx = rel_xy[:, 0]
    dy = rel_xy[:, 1]

    out_x = (dx < x_min) | (dx > x_max)
    out_y = (dy < y_min) | (dy > y_max)
    out = out_x | out_y  # [N] bool

    print(
        f"[DEBUG] cube pos (x,y): ({dx[0].item():.3f}, {dy[0].item():.3f}), "
        f"out_x={out_x[0].item()}, out_y={out_y[0].item()}, out={out[0].item()}"
    )

    return out_x | out_y  # [N] bool


