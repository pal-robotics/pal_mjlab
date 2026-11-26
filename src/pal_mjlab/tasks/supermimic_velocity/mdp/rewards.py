"""The velocity task's rewards, as single-definition terms (``@imaginable``).

Each is the PPO task's term (mjlab's velocity rewards and pal's joint-limit terms), with
the same formula, written on a ``Batch`` so that SuperMimic's imagination computes it
exactly as the env does. Where the PPO term reads a sensor a predicted state does not
have, it reads the batch instead:

- the robot's root velocities: the heading state; body orientations and velocities: FK;
- the command: the command state ``c`` (the step's target, as the env's reward manager
  sees it);
- foot heights: the foot sites' FK height above the terrain (``terrain_height``: 0 on
  the flat plane, a heightfield lookup on rough terrain later), instead of the height
  scan;
- foot contacts: a smooth kinematic proxy, ``sigmoid((contact_height - foot height) /
  contact_width)``, instead of the contact sensor (on rough terrain, a learned contact
  head later); its parameters are calibrated against the contact sensor
  (``FOOT_CONTACT``);
- landing forces: the foot's vertical speed at touchdown (``soft_landing``).

The terms that keep state across steps (a foot's time in the air, its peak height over a
swing) are stateful single definitions (``@imaginable(memory=...)``).
"""

from __future__ import annotations

import re

import numpy as np
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import resolve_matching_names_values

from pal_mjlab.rl.supermimic.imagination import (
  Batch,
  capsule_bottoms,
  imaginable,
  terrain_height,
)

__all__ = [
  "FOOT_CONTACT",
  "FOOT_SOLES",
  "FOOT_SOLE_BODIES",
  "track_linear_velocity",
  "track_angular_velocity",
  "upright",
  "variable_posture",
  "posture_l2",
  "body_angular_velocity_penalty",
  "joint_pos_limits",
  "joint_limits_convex_hull",
  "joint_vel_limits",
  "feet_clearance",
  "feet_swing_height",
  "feet_air_time",
  "feet_slip",
  "soft_landing",
]

FOOT_CONTACT = {"contact_height": 0.0002, "contact_width": 0.0002}
"""The contact proxy's parameters (m): a foot whose lowest sole point is this high above
the terrain is half in contact; the transition's width. The sensor reports contact exactly
when that point is at or below the terrain (median -0.8 mm): a 0 threshold agrees with it
100% on a walking policy."""

FOOT_SOLES = {"left_foot": ("leg_left_foot_link",), "right_foot": ("leg_right_foot_link",)}
"""Each foot site's sole: the bodies whose collision capsules touch the ground."""
FOOT_SOLE_BODIES = tuple(body for bodies in FOOT_SOLES.values() for body in bodies)

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_ROOT_LIN_VEL, _ROOT_ANG_VEL = slice(7, 10), slice(10, 13)  # in the heading state


def _active(b: Batch, threshold: float) -> torch.Tensor:
  """1 where the command's speed (linear norm + |yaw rate|) exceeds ``threshold``."""
  cmd = b.c["command"]
  return ((torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])) > threshold).float()


def _body(b: Batch, asset_cfg: SceneEntityCfg) -> int:
  names = asset_cfg.body_names
  name = names[0] if isinstance(names, (list, tuple)) else names
  return b.spec.body(name)


def sole_heights(b: Batch, sites: tuple[str, ...]) -> torch.Tensor:
  """(B, F) height above the terrain of each foot's lowest sole point (its collision
  capsules, ``FOOT_SOLES``); once per batch."""
  key = ("soles", sites)
  if key not in b.memo:
    b.memo[key] = torch.stack(
      [capsule_bottoms(b, FOOT_SOLES[s]).min(dim=-1).values for s in sites], dim=-1
    )
  return b.memo[key]


def _feet(b: Batch, asset_cfg: SceneEntityCfg, contact_height: float, contact_width: float):
  """(height above the terrain, linear velocity, contact probability) of the feet,
  (B, F) / (B, F, 3) / (B, F); once per batch. Height and velocity are the foot sites'
  (as mjlab's foot height scan); contact comes from the sole's lowest point, not the site:
  a foot touching down on an edge of its sole has its site centimetres up (a policy
  dragged a tilted foot to pass as airborne)."""
  names = tuple(asset_cfg.site_names)
  key = ("feet", names, contact_height, contact_width)
  if key not in b.memo:
    ids = [b.spec.site(n) for n in names]
    pos, vel = b.kin.site_pos[:, ids], b.kin.site_lin_vel[:, ids]
    height = pos[..., 2] - terrain_height(b, pos[..., :2])
    contact = torch.sigmoid((contact_height - sole_heights(b, names)) / contact_width)
    b.memo[key] = (height, vel, contact)
  return b.memo[key]


# Velocity tracking and posture ------------------------------------------------------


@imaginable
def track_linear_velocity(b: Batch, std: float) -> torch.Tensor:
  actual = b.state[:, _ROOT_LIN_VEL]
  cmd = b.c["command"]
  error = torch.sum(torch.square(cmd[:, :2] - actual[:, :2]), dim=1)
  error = error + torch.square(actual[:, 2])
  return torch.exp(-error / std**2)


@imaginable
def track_angular_velocity(b: Batch, std: float) -> torch.Tensor:
  actual = b.state[:, _ROOT_ANG_VEL]
  cmd = b.c["command"]
  error = torch.square(cmd[:, 2] - actual[:, 2])
  error = error + torch.sum(torch.square(actual[:, :2]), dim=1)
  return torch.exp(-error / std**2)


@imaginable
def upright(b: Batch, std: float, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  quat = b.kin.body_quat[:, _body(b, asset_cfg)]
  gravity = torch.zeros_like(quat[:, :3])
  gravity[:, 2] = -1.0
  projected = quat_apply_inverse(quat, gravity)
  return torch.exp(-torch.sum(torch.square(projected[:, :2]), dim=1) / std**2)


def _posture_error(
  b: Batch,
  std_standing,
  std_walking,
  std_running,
  asset_cfg: SceneEntityCfg,
  walking_threshold: float,
  running_threshold: float,
) -> torch.Tensor:
  """mean((q - q0)² / std²) over the joints, std by the commanded speed's regime."""
  spec = b.spec
  idx = spec.joint_index(asset_cfg.joint_ids)

  def stds():
    names = [spec.joint_names[c] for c in spec.joint_cols(asset_cfg.joint_ids)]
    return tuple(
      torch.tensor(
        resolve_matching_names_values(data=d, list_of_strings=names)[2],
        device=b.state.device,
      )
      for d in (std_standing, std_walking, std_running)
    )

  standing, walking, running = spec.cached(("posture", id(asset_cfg)), stds)
  cmd = b.c["command"]
  speed = torch.norm(cmd[:, :2], dim=1) + torch.abs(cmd[:, 2])
  stand = (speed < walking_threshold).float()[:, None]
  walk = ((speed >= walking_threshold) & (speed < running_threshold)).float()[:, None]
  run = (speed >= running_threshold).float()[:, None]
  std = standing * stand + walking * walk + running * run
  error = torch.square(b.qpos[:, 7:][:, idx] - spec.q0[idx])
  return torch.mean(error / std**2, dim=1)


@imaginable
def variable_posture(
  b: Batch,
  std_standing,
  std_walking,
  std_running,
  asset_cfg: SceneEntityCfg,
  walking_threshold: float = 0.5,
  running_threshold: float = 1.5,
) -> torch.Tensor:
  """mjlab's ``variable_posture``: exp(-mean((q - q0)² / std²))."""
  return torch.exp(
    -_posture_error(
      b, std_standing, std_walking, std_running, asset_cfg, walking_threshold,
      running_threshold,
    )  # fmt: skip
  )


@imaginable
def posture_l2(
  b: Batch,
  std_standing,
  std_walking,
  std_running,
  asset_cfg: SceneEntityCfg,
  walking_threshold: float = 0.5,
  running_threshold: float = 1.5,
) -> torch.Tensor:
  """``variable_posture``'s exponent, mean((q - q0)² / std²), as a penalty: its gradient
  grows with the deviation, where the kernel's has faded."""
  return _posture_error(
    b, std_standing, std_walking, std_running, asset_cfg, walking_threshold,
    running_threshold,
  )  # fmt: skip


@imaginable
def body_angular_velocity_penalty(
  b: Batch, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  ang_vel = b.kin.body_ang_vel[:, _body(b, asset_cfg)]
  return torch.sum(torch.square(ang_vel[:, :2]), dim=1)


# Joint limits -------------------------------------------------------------------------


@imaginable
def joint_pos_limits(b: Batch, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG) -> torch.Tensor:
  """mjlab's ``joint_pos_limits``: distance outside the soft limits."""
  idx = b.spec.joint_index(asset_cfg.joint_ids)
  q, limits = b.qpos[:, 7:][:, idx], b.spec.soft_joint_limits[idx]
  out = -(q - limits[:, 0]).clip(max=0.0) + (q - limits[:, 1]).clip(min=0.0)
  return torch.sum(out, dim=1)


def _state_cols(b: Batch, patterns) -> list[int]:
  """State columns of the joints matching ``patterns``, in the robot's order."""
  return [
    c for c, name in enumerate(b.spec.joint_names)
    if any(re.fullmatch(p, name) for p in patterns)
  ]  # fmt: skip


@imaginable
def joint_limits_convex_hull(
  b: Batch,
  asset_cfg: SceneEntityCfg,
  metrics_suffix: str,
  margin: float,
  joint_names_group: list[list[str]],
  hull_points: torch.Tensor,
) -> torch.Tensor:
  """pal's ``joint_limits_convex_hull``: squared distance of each joint group outside its
  convex hull (shrunk by ``margin``)."""
  del asset_cfg, metrics_suffix

  def hull():
    from scipy.spatial import ConvexHull

    eq = torch.from_numpy(ConvexHull(hull_points.cpu().numpy()).equations).to(b.state)
    normals, offsets = eq[:, :-1], eq[:, -1]
    offsets = offsets + margin * torch.norm(normals, dim=1)
    groups = [_state_cols(b, g) for g in joint_names_group]
    return normals, offsets, groups

  normals, offsets, groups = b.spec.cached(("hull", id(hull_points), margin), hull)
  q = b.qpos[:, 7:]
  penalty = torch.zeros_like(q[:, 0])
  for cols in groups:
    dist = torch.clamp(q[:, cols] @ normals.T + offsets, min=0.0).max(dim=1)[0]
    penalty = penalty + torch.square(dist)
  return penalty


@imaginable
def joint_vel_limits(
  b: Batch, velocity_limits, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """pal's ``joint_vel_limits``: distance outside 90% of the velocity limits."""
  spec = b.spec

  def limits():
    cols = spec.joint_cols(asset_cfg.joint_ids)
    names = [spec.joint_names[c] for c in cols]
    index, _, values = resolve_matching_names_values(data=velocity_limits, list_of_strings=names)
    soft = 0.9 * torch.tensor(np.asarray(values, dtype=np.float32), device=b.state.device)
    return [cols[i] for i in index], soft

  cols, soft = spec.cached(("vel_limits", id(asset_cfg)), limits)
  qd = b.qvel[:, 6:][:, cols]
  out = -(qd - soft[:, 0]).clip(max=0.0) + (qd - soft[:, 1]).clip(min=0.0)
  return torch.sum(out, dim=1)


# Feet ---------------------------------------------------------------------------------


@imaginable
def feet_clearance(
  b: Batch,
  target_height: float,
  asset_cfg: SceneEntityCfg,
  command_threshold: float = 0.01,
  contact_height: float = FOOT_CONTACT["contact_height"],
  contact_width: float = FOOT_CONTACT["contact_width"],
) -> torch.Tensor:
  """mjlab's ``feet_clearance``: |foot height - target| weighted by the foot's xy speed."""
  height, vel, _ = _feet(b, asset_cfg, contact_height, contact_width)
  cost = torch.sum(torch.abs(height - target_height) * torch.norm(vel[..., :2], dim=-1), dim=1)
  return cost * _active(b, command_threshold)


@imaginable
def feet_slip(
  b: Batch,
  asset_cfg: SceneEntityCfg,
  command_threshold: float = 0.01,
  contact_height: float = FOOT_CONTACT["contact_height"],
  contact_width: float = FOOT_CONTACT["contact_width"],
) -> torch.Tensor:
  """mjlab's ``feet_slip``: squared xy speed of the feet in contact."""
  _, vel, contact = _feet(b, asset_cfg, contact_height, contact_width)
  cost = torch.sum(torch.square(torch.norm(vel[..., :2], dim=-1)) * contact, dim=1)
  return cost * _active(b, command_threshold)


def _air_memory(b: Batch, asset_cfg: SceneEntityCfg, **_) -> dict[str, torch.Tensor]:
  n = len(asset_cfg.site_names)
  return {"air_time": b.state.new_zeros(len(b.state), n)}


@imaginable(memory=_air_memory)
def feet_air_time(
  b: Batch,
  memory: dict,
  asset_cfg: SceneEntityCfg,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_threshold: float = 0.5,
  contact_height: float = FOOT_CONTACT["contact_height"],
  contact_width: float = FOOT_CONTACT["contact_width"],
  edge_width: float = 0.01,
) -> tuple[torch.Tensor, dict]:
  """mjlab's ``feet_air_time`` (the feet whose current time in the air is in range), with
  soft edges: mjlab's hard window has no gradient, so imagination could not see it. Each
  foot scores sigmoid((air - min) / w) * sigmoid((max - air) / w), w = ``edge_width`` (s);
  the air time follows the smooth contact proxy, so the foot's height carries the gradient."""
  _, _, contact = _feet(b, asset_cfg, contact_height, contact_width)
  air = (memory["air_time"] + b.spec.step_dt) * (1.0 - contact)
  in_range = torch.sigmoid((air - threshold_min) / edge_width) * torch.sigmoid(
    (threshold_max - air) / edge_width
  )
  reward = torch.sum(in_range, dim=1) * _active(b, command_threshold)
  return reward, {"air_time": air}


def _swing_memory(b: Batch, asset_cfg: SceneEntityCfg, **_) -> dict[str, torch.Tensor]:
  zeros = b.state.new_zeros(len(b.state), len(asset_cfg.site_names))
  return {"peak_height": zeros, "in_air": zeros.clone()}


@imaginable(memory=_swing_memory)
def feet_swing_height(
  b: Batch,
  memory: dict,
  target_height: float,
  asset_cfg: SceneEntityCfg,
  command_threshold: float = 0.05,
  contact_height: float = FOOT_CONTACT["contact_height"],
  contact_width: float = FOOT_CONTACT["contact_width"],
) -> tuple[torch.Tensor, dict]:
  """mjlab's ``feet_swing_height``: (peak height over the swing / target - 1)² at
  touchdown."""
  height, _, contact = _feet(b, asset_cfg, contact_height, contact_width)
  in_air = 1.0 - contact
  peak = memory["peak_height"] + in_air * torch.relu(height - memory["peak_height"])
  touchdown = contact * memory["in_air"]  # in contact now, in the air before
  error = torch.square(peak / target_height - 1.0)
  cost = torch.sum(error * touchdown, dim=1) * _active(b, command_threshold)
  return cost, {"peak_height": peak * (1.0 - touchdown), "in_air": in_air}


def _landing_memory(b: Batch, asset_cfg: SceneEntityCfg, **_) -> dict[str, torch.Tensor]:
  return {"in_air": b.state.new_zeros(len(b.state), len(asset_cfg.site_names))}


@imaginable(memory=_landing_memory)
def soft_landing(
  b: Batch,
  memory: dict,
  asset_cfg: SceneEntityCfg,
  command_threshold: float = 0.05,
  contact_height: float = FOOT_CONTACT["contact_height"],
  contact_width: float = FOOT_CONTACT["contact_width"],
) -> tuple[torch.Tensor, dict]:
  """mjlab's ``soft_landing`` with the foot's vertical speed at touchdown in place of the
  landing force (which a predicted state does not have)."""
  _, vel, contact = _feet(b, asset_cfg, contact_height, contact_width)
  touchdown = contact * memory["in_air"]
  cost = torch.sum(torch.abs(vel[..., 2]) * touchdown, dim=1) * _active(b, command_threshold)
  return cost, {"in_air": 1.0 - contact}
