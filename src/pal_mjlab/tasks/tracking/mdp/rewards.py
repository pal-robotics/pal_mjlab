from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.sensor import ContactSensor
from mjlab.tasks.tracking.mdp import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_global_anchor_velocity_z_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  '''Reward for having the robot's anchor match the motion's anchor in the Z axis
  Very useful for motions where Z coordinate is important, such as jumping
  '''
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.square(
    command.anchor_lin_vel_w[:, 2] - command.robot_anchor_lin_vel_w[:, 2]
  )
  return torch.exp(-error / std**2)


def all_feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.02,
) -> torch.Tensor:
  '''Rewards both feet being in the air long enough
  Mostly used for jumping, to encourage both feet leaving the ground longer
  '''
  sensor: ContactSensor = env.scene[sensor_name]
  air_time = sensor.data.current_air_time  # [B, F]

  # foot is considered airborne if it has been off contact long enough
  in_air = air_time > threshold  # [B, F]

  # both feet airborne → actual jump
  all_feet_airborne = in_air.all(dim=1).float()  # [B]

  # reward = average airtime only when fully airborne
  mean_air_time = air_time.mean(dim=1)

  return all_feet_airborne * mean_air_time
