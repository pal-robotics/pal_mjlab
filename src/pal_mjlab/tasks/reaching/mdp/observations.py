from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.third_party.isaaclab.isaaclab.utils.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_conjugate,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_mul,
    quat_unique,
    sample_uniform,
    quat_apply,
)
if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def commands_gen(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)

    des_pos_b = command.command[:, :3]

    return des_pos_b
