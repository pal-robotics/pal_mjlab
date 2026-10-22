from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor

from ..kinematics import quat_rotate_inv
from .commands import MotionLibraryCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

__all__ = [
  "motion_goal",
  "robot_heading_state",
  "motion_root_pose_error",
  "motion_time_left",
  "sim_context",
  "imu_projected_gravity",
]


def _command(env: ManagerBasedRlEnv, command_name: str) -> MotionLibraryCommand:
  return cast(MotionLibraryCommand, env.command_manager.get_term(command_name))


def motion_goal(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """K future reference frames (D6)."""
  return _command(env, command_name).goal


def robot_heading_state(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Heading-invariant robot state (D4): the world model's state, seen by the critic."""
  return _command(env, command_name).heading_state


def motion_root_pose_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference root position (robot heading frame) and yaw error (privileged, D7)."""
  return _command(env, command_name).root_pose_error


def motion_time_left(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Normalized steps left in the clip: the return ends at the clip end (privileged, D7)."""
  return _command(env, command_name).clip_time_left


def sim_context(
  env: ManagerBasedRlEnv,
  friction_asset_cfg: SceneEntityCfg,
  com_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Domain-randomization parameters ξ: foot friction, joint calibration offsets, CoM offset.

  Friction is shared by all foot geoms, so the first one is read.
  """
  asset: Entity = env.scene[friction_asset_cfg.name]
  geom = asset.indexing.geom_ids[friction_asset_cfg.geom_ids][0]
  friction = env.sim.model.geom_friction[..., geom, 0]
  friction = friction.expand(env.num_envs)[:, None]
  body = asset.indexing.body_ids[com_asset_cfg.body_ids]
  ipos = env.sim.model.body_ipos
  ipos = ipos.expand(env.num_envs, *ipos.shape[-2:])
  com = ipos[:, body] - env.sim.get_default_field("body_ipos")[body]
  return torch.cat([friction, asset.data.encoder_bias, com.flatten(1)], dim=-1)


def imu_projected_gravity(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Gravity direction in the IMU frame, from a ``framequat`` sensor on the IMU site."""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, BuiltinSensor)
  quat = sensor.data
  gravity = torch.zeros(quat.shape[0], 3, device=quat.device)
  gravity[:, 2] = -1.0
  return quat_rotate_inv(quat, gravity)
