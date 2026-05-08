from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.sensor import ContactSensor
from mjlab.tasks.tracking.mdp import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_foot_contact(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  foot_body_names: tuple[str, ...],
  height_threshold: float = 0.05,
) -> torch.Tensor:
  """Reward foot contact state matching the motion reference height.

  When the reference foot z-height is below *height_threshold*, the foot
  should be in contact with the ground and contact is rewarded.  Above the
  threshold the foot should be airborne and no contact is rewarded.

  Returns a per-environment mean over feet in [0, 1].
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  sensor: ContactSensor = env.scene[sensor_name]

  cmd_foot_idxs = [command.cfg.body_names.index(name) for name in foot_body_names]
  sensor_primary_names = list(dict.fromkeys(slot.primary_name for slot in sensor._slots))
  sensor_foot_idxs = [sensor_primary_names.index(name) for name in foot_body_names]

  # Reference foot heights in world frame: [B, num_feet]
  foot_heights = command.body_pos_w[:, cmd_foot_idxs, 2]
  should_contact = foot_heights < height_threshold

  assert sensor.data.found is not None
  in_contact = sensor.data.found[:, sensor_foot_idxs] > 0

  return (should_contact == in_contact).float().mean(dim=-1)

def motion_global_anchor_velocity_z_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.square(
    command.anchor_lin_vel_w[:, 2] - command.robot_anchor_lin_vel_w[:, 2]
  )
  return torch.exp(-error / std**2)

def feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float = 0.02,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    air_time = sensor.data.current_air_time  # [B, F]

    # foot is considered airborne if it has been off contact long enough
    in_air = air_time > threshold  # [B, F]

    # both feet airborne → actual jump
    all_feet_airborne = in_air.all(dim=1).float()  # [B]

    # reward = average airtime only when fully airborne
    mean_air_time = air_time.mean(dim=1)

    return all_feet_airborne * mean_air_time
