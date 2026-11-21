from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pal_mjlab.rl.supermimic.imagination import ContextPart

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
]


def _command(env: ManagerBasedRlEnv, command_name: str) -> MotionLibraryCommand:
  return cast(MotionLibraryCommand, env.command_manager.get_term(command_name))


def motion_goal(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """The next K reference frames: the policy's goal."""
  return _command(env, command_name).goal


def robot_heading_state(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Heading-invariant robot state: the world model's state, seen by the critic."""
  return _command(env, command_name).heading_state


def motion_root_pose_error(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Reference root position in the robot heading frame, and the yaw error (privileged)."""
  return _command(env, command_name).root_pose_error


def motion_time_left(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Normalized steps left in the clip: the return ends at the clip end (privileged)."""
  return _command(env, command_name).clip_time_left


def sim_context(
  env: ManagerBasedRlEnv,
  friction_asset_cfg: SceneEntityCfg,
  bias_asset_cfg: SceneEntityCfg,
  com_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Domain-randomization parameters ξ: foot friction, joint calibration offsets of the
  measured joints, torso CoM offset.

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
  bias = asset.data.encoder_bias[:, bias_asset_cfg.joint_ids]
  return torch.cat([friction, bias, com.flatten(1)], dim=-1)


def _sim_context_layout(
  env: ManagerBasedRlEnv,
  friction_asset_cfg: SceneEntityCfg,
  bias_asset_cfg: SceneEntityCfg,
  com_asset_cfg: SceneEntityCfg,
) -> dict[str, ContextPart]:
  """``sim_context``'s named parts: friction (1), the encoder offsets of
  ``bias_asset_cfg``'s joints (one per joint), the CoM offset (3 per body)."""
  asset: Entity = env.scene[bias_asset_cfg.name]
  ids = bias_asset_cfg.joint_ids
  joints = list(range(len(asset.joint_names)))[ids] if isinstance(ids, slice) else list(ids)
  bodies = len(asset.indexing.body_ids[com_asset_cfg.body_ids])
  nb = len(joints)
  return {
    "friction": ContextPart(slice(0, 1)),
    "encoder_bias": ContextPart(slice(1, 1 + nb), [int(j) for j in joints]),
    "com_offset": ContextPart(slice(1 + nb, 1 + nb + 3 * bodies)),
  }


sim_context.context_layout = _sim_context_layout  # type: ignore[attr-defined]
