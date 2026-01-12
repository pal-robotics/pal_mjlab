from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def commands_gen(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)

    des_pos_b = command.command

    return des_pos_b
