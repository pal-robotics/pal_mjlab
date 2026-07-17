"""Useful methods for MDP events."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_SE3_KEYS = ("x", "y", "z", "roll", "pitch", "yaw")


def _sample_se3_range(
  range_dict: dict[str, tuple[float, float]] | None,
  shape: tuple[int, ...],
  device: str,
) -> torch.Tensor:
  """Sample uniform ``[x, y, z, roll, pitch, yaw]`` offsets.

  ``range_dict`` maps any subset of those keys to ``(min, max)`` ranges; missing
  keys default to ``(0.0, 0.0)`` (no offset). ``None`` is treated as empty. The
  returned tensor has the requested ``shape`` whose last dimension must be 6.
  """
  range_dict = range_dict or {}
  range_list = [range_dict.get(key, (0.0, 0.0)) for key in _SE3_KEYS]
  ranges = torch.tensor(range_list, device=device)
  return sample_uniform(ranges[:, 0], ranges[:, 1], shape, device=device)


def resolve_env_ids(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> torch.Tensor:
  """Return ``env_ids`` unchanged, or all environment indices if ``None``.

  Event functions receive ``env_ids=None`` to mean "all environments" (a full
  reset, or a global-time interval term). This normalizes that sentinel to a
  concrete index tensor so the function body can assume a real ``torch.Tensor``.
  """
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  return env_ids

def reset_box(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
) -> None:

  env_ids = resolve_env_ids(env, env_ids)

  asset: Entity = env.scene["box"]

  # Pose.
  pose_samples = _sample_se3_range(pose_range, (len(env_ids), 6), env.device)

  # Floating-base entities.
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  root_states = default_root_state[env_ids].clone()

  positions = (
    root_states[:, 0:3] + pose_samples[:, 0:3] + env.scene.env_origins[env_ids]
  )
  orientations_delta = quat_from_euler_xyz(
    pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
  )
  orientations = quat_mul(root_states[:, 3:7], orientations_delta)

  asset.write_root_link_pose_to_sim(
    torch.cat([positions, orientations], dim=-1), env_ids=env_ids
  )