"""Extension of mjlab's joint position action"""
from __future__ import annotations

from mjlab.envs.mdp.actions.actions import JointPositionActionCfg, JointPositionActionCfg

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  apply_delta_pose,
  quat_apply,
  combine_frame_transforms
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv



# Two possibilities :
# 1 : Arbitrary positions
# 2 : Rework command as to also include command for pelvis (maybe overdoing it)

