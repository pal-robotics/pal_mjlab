"""Useful methods for MDP observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def box_out_bounds(
    env: ManagerBasedRlEnv,
) -> torch.Tensor:
    box: Entity = env.scene["box"]
    table: Entity = env.scene["table"]

    # table half-extents (x, y) — from size="0.25 1.0 0.5"
    table_half_x = 0.25
    table_half_y = 1.0

    # box position relative to table origin, in world-frame axes
    box_pos_rel = box.data.root_link_pos_w - table.data.root_link_pos_w

    out_x = box_pos_rel[:, 0].abs() > table_half_x
    out_y = box_pos_rel[:, 1].abs() > table_half_y

    return out_x | out_y