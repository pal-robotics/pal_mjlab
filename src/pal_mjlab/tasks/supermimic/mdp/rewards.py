"""Tracking reward of SuperMimic (Table A.2): BeyondMimic's terms without self-collision.

Each term is ``exp(-error / std**2)`` of a squared error from ``tracking.py``, written once
on a ``Batch`` (``@imaginable``): the env computes it on the simulator's state, SuperMimic's
imagination on predicted states. The robot's bodies are the batch's FK, the reference the
command model's at the command state the step targeted.
"""

from __future__ import annotations

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pal_mjlab.rl.supermimic.imagination import Batch, imaginable

from . import tracking
from .tracking import BodyState

__all__ = [
  "motion_root_pos_exp",
  "motion_root_ori_exp",
  "motion_body_pos_exp",
  "motion_body_ori_exp",
  "motion_body_lin_vel_exp",
  "motion_body_ang_vel_exp",
  "joint_pos_limits_hard",
]

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def tracked_bodies(b: Batch) -> tuple[BodyState, BodyState]:
  """The robot's tracked bodies (from the batch's FK) and the reference's, same order;
  computed once per batch for the tracking terms (and the failure termination)."""
  if "motion_bodies" not in b.memo:
    b.memo["motion_bodies"] = _compute_robot_and_reference(b)
  return b.memo["motion_bodies"]


def _compute_robot_and_reference(b: Batch) -> tuple[BodyState, BodyState]:
  model = b.command_model  # the motion command's model (``command_model.py``)
  idx = model.body_index(b.spec)
  k = b.kin
  robot = BodyState(k.body_pos, k.body_quat, k.body_lin_vel, k.body_ang_vel)
  if idx is not None:
    robot = BodyState(*(t[:, idx] for t in robot))
  return robot, model.reference(b.c)


def _exp_term(b: Batch, error_fn, std: float) -> torch.Tensor:
  robot, ref = tracked_bodies(b)
  return torch.exp(-error_fn(robot, ref) / std**2)


@imaginable
def motion_root_pos_exp(b: Batch, std: float) -> torch.Tensor:
  return _exp_term(b, tracking.anchor_pos_error, std)


@imaginable
def motion_root_ori_exp(b: Batch, std: float) -> torch.Tensor:
  return _exp_term(b, tracking.anchor_ori_error, std)


@imaginable
def motion_body_pos_exp(b: Batch, std: float) -> torch.Tensor:
  return _exp_term(b, tracking.body_pos_error, std)


@imaginable
def motion_body_ori_exp(b: Batch, std: float) -> torch.Tensor:
  return _exp_term(b, tracking.body_ori_error, std)


@imaginable
def motion_body_lin_vel_exp(b: Batch, std: float) -> torch.Tensor:
  return _exp_term(b, tracking.body_lin_vel_error, std)


@imaginable
def motion_body_ang_vel_exp(b: Batch, std: float) -> torch.Tensor:
  return _exp_term(b, tracking.body_ang_vel_error, std)


TRACKING_ERRORS = {
  motion_root_pos_exp: tracking.anchor_pos_error,
  motion_root_ori_exp: tracking.anchor_ori_error,
  motion_body_pos_exp: tracking.body_pos_error,
  motion_body_ori_exp: tracking.body_ori_error,
  motion_body_lin_vel_exp: tracking.body_lin_vel_error,
  motion_body_ang_vel_exp: tracking.body_ang_vel_error,
}
"""The squared error of each tracking term (``scripts/evaluate.py`` scores with them)."""


@imaginable
def joint_pos_limits_hard(
  b: Batch, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Distance outside the hard joint limits: references often exceed the soft ones."""
  cols = b.spec.joint_cols(asset_cfg.joint_ids)
  limits = b.spec.joint_limits[cols]
  q = b.qpos[:, 7:][:, cols]
  return tracking.joint_limit_violation(q, limits[..., 0], limits[..., 1])
