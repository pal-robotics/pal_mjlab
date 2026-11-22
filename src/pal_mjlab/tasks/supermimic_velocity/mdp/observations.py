"""The velocity task's observations: the command (the policy's goal) and ξ.

The proprioception and the heading state are the generic single-definition terms
(``pal_mjlab/rl/supermimic/terms.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pal_mjlab.rl.supermimic.imagination import Batch, ContextPart, imaginable
from pal_mjlab.tasks.supermimic.mdp.observations import sim_context

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

__all__ = ["velocity_command", "velocity_context"]

_DOF_FIELDS = ("dof_frictionloss", "dof_damping", "dof_armature")


@imaginable
def velocity_command(b: Batch) -> torch.Tensor:
  """The commanded body-frame (vx, vy, yaw rate)."""
  return b.c["command"]


def velocity_context(
  env: ManagerBasedRlEnv,
  friction_asset_cfg: SceneEntityCfg,
  bias_asset_cfg: SceneEntityCfg,
  com_asset_cfg: SceneEntityCfg,
  dof_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """ξ: the motion task's (foot friction, encoder offsets, torso CoM offset), then the
  randomized joint friction loss, damping and armature of ``dof_asset_cfg``'s joints."""
  parts = [sim_context(env, friction_asset_cfg, bias_asset_cfg, com_asset_cfg)]
  dofs = _dof_ids(env, dof_asset_cfg)
  for name in _DOF_FIELDS:
    value = getattr(env.sim.model, name)
    parts.append(value.expand(env.num_envs, value.shape[-1])[:, dofs])
  return torch.cat(parts, dim=-1)


def _dof_ids(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.indexing.joint_v_adr[asset_cfg.joint_ids]


def _joint_ids(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> list[int]:
  asset: Entity = env.scene[asset_cfg.name]
  ids = asset_cfg.joint_ids
  return list(range(len(asset.joint_names)))[ids] if isinstance(ids, slice) else list(ids)


def _velocity_context_layout(
  env: ManagerBasedRlEnv,
  friction_asset_cfg: SceneEntityCfg,
  bias_asset_cfg: SceneEntityCfg,
  com_asset_cfg: SceneEntityCfg,
  dof_asset_cfg: SceneEntityCfg,
) -> dict[str, ContextPart]:
  layout = sim_context.context_layout(  # type: ignore[attr-defined]
    env, friction_asset_cfg, bias_asset_cfg, com_asset_cfg
  )
  start = max(p.columns.stop for p in layout.values())
  joints = [int(j) for j in _joint_ids(env, dof_asset_cfg)]
  for name in _DOF_FIELDS:
    key = name.removeprefix("dof_")
    layout[key] = ContextPart(slice(start, start + len(joints)), joints)
    start += len(joints)
  return layout


velocity_context.context_layout = _velocity_context_layout  # type: ignore[attr-defined]
