"""Single-definition versions of the mjlab terms SuperMimic tasks use (``imagination.py``).

Each is the mjlab term of the same name, written on a ``Batch`` so that imagination
computes it exactly as the env does. SuperMimic tasks use these in their env configs;
other tasks keep mjlab's.
"""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from .imagination import Batch, imaginable

__all__ = [
  "action_rate_l2",
  "context",
  "heading_state",
  "joint_pos_rel",
  "joint_vel_rel",
  "last_action",
  "site_ang_vel",
  "site_projected_gravity",
]

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


@imaginable
def heading_state(b: Batch) -> torch.Tensor:
  """The heading-invariant robot state: the world model's state (a critic input)."""
  return b.state


@imaginable
def context(b: Batch) -> torch.Tensor:
  """The sim context ξ (``ImaginationCfg.context``; a critic input)."""
  return b.context


@imaginable
def joint_pos_rel(
  b: Batch, biased: bool = False, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """mjlab's ``joint_pos_rel``. ``biased``: as the encoders read the joints, with the
  calibration offsets of ξ's ``encoder_bias`` part."""
  idx = b.spec.joint_index(asset_cfg.joint_ids)
  q = b.qpos[:, 7:][:, idx]
  if biased:
    q = q + b.context[:, b.spec.context_joint_index("encoder_bias", asset_cfg.joint_ids)]
  return q - b.spec.q0[idx]


@imaginable
def joint_vel_rel(
  b: Batch, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """mjlab's ``joint_vel_rel``."""
  idx = b.spec.joint_index(asset_cfg.joint_ids)
  return b.qvel[:, 6:][:, idx] - b.spec.qd0[idx]


@imaginable
def last_action(b: Batch, action_name: str | None = None) -> torch.Tensor:
  """mjlab's ``last_action``, of the whole action (the policy's output)."""
  if action_name is not None:
    raise ValueError("last_action: only the whole action (action_name=None) is imagined.")
  return b.action


@imaginable
def site_ang_vel(b: Batch, site: str) -> torch.Tensor:
  """Angular velocity of ``site`` in its own frame: what a gyro there measures (mjlab's
  ``builtin_sensor`` of a gyro), from the batch's FK."""
  i = b.spec.site(site)
  return quat_apply_inverse(b.kin.site_quat[:, i], b.kin.site_ang_vel[:, i])


@imaginable
def site_projected_gravity(b: Batch, site: str) -> torch.Tensor:
  """Gravity direction in the frame of ``site`` (an IMU's orientation estimate)."""
  quat = b.kin.site_quat[:, b.spec.site(site)]
  gravity = torch.zeros_like(quat[:, :3])
  gravity[:, 2] = -1.0
  return quat_apply_inverse(quat, gravity)


@imaginable
def action_rate_l2(b: Batch) -> torch.Tensor:
  """mjlab's ``action_rate_l2``: squared change of the raw policy action."""
  return (b.action - b.prev_action).square().sum(-1)
