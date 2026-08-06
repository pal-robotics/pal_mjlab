from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import DualBandVelocityCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


def commanded_displacement(
  commands: torch.Tensor, durations: torch.Tensor
) -> torch.Tensor:
  """Add up the straight-line hops that a sequence of twist commands asks for.

  Each segment holds a single twist, so the robot traces a circular arc. What moves it
  from where the segment started to where it ended is the arc's chord: a straight hop
  pointing at the middle of the turn, shorter than the distance walked along the arc by
  ``sin(turn / 2) / (turn / 2)`` -- 1 for a command that goes straight, 0 for one that
  closes a full circle.

  Args:
    commands: Per-segment (vx, vy, wz) commands, shape (num_envs, num_segments, 3).
    durations: Per-segment duration in seconds, shape (num_envs, num_segments).

  Returns:
    Net displacement, shape (num_envs, 2). The heading the episode started from only
    rotates the whole path, so it leaves the norm unchanged and is taken as zero.
  """
  forward, lateral, yaw_rate = commands.unbind(-1)

  # The actual translation speed
  speed = torch.hypot(forward, lateral)

  # A sideways command makes the robot travel at an angle to the way it faces.
  slip = torch.atan2(lateral, forward)

  # A segment starts out facing whatever the segments before it turned through.
  turn = yaw_rate * durations
  heading = torch.cumsum(turn, dim=-1) - turn

  # The chord points that far past the heading the segment started at, and it is
  # shorter than the arc by sin(half_turn) / half_turn.
  half_turn = 0.5 * turn
  # Dividing that out directly is 0/0 for a command that does not turn -- the common
  # case here, and true of every unused segment, so it would poison the result with
  # NaNs on every call. torch.sinc carries the limit of 1 at 0; its convention is
  # sin(pi * x) / (pi * x), hence passing the angle divided by pi.
  chord_shortening = torch.sinc(half_turn / math.pi)

  # Each segment is one straight hop: this long, in this direction.
  hop_length = speed * durations * chord_shortening
  hop_heading = heading + slip + half_turn

  # Walk the hops one after the other. Dimension -2 is the segment axis.
  direction = torch.stack((torch.cos(hop_heading), torch.sin(hop_heading)), dim=-1)
  return (hop_length.unsqueeze(-1) * direction).sum(dim=-2)


def terrain_levels_vel_with_history(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  promotion_frac: float = 0.8,
  demotion_frac: float = 0.5,
  min_expected_distance: float = 0.25,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> dict[str, torch.Tensor]:
  """Move environments up or down a terrain level based on how far they walked.

  The yardstick is the whole episode's command record rather than the command that
  happens to be active at reset: every command sampled during the episode is integrated
  into the net displacement it asked for over a full episode, which is what the walked
  distance can fairly be compared against.

  Args:
    env: The environment.
    env_ids: Environments being reset.
    command_name: Name of the velocity command term holding the record.
    promotion_frac: Fraction of the commanded displacement that must be achieved to
      move up a level, capped at half a sub-terrain.
    demotion_frac: Fraction of the promotion distance below which an environment moves
      down a level.
    min_expected_distance: Environments commanded to travel less than this over an
      episode keep their level, since they say nothing about the terrain.

  Returns:
    Terrain level statistics for logging.
  """
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command_term = env.command_manager.get_term(command_name)
  assert isinstance(command_term, DualBandVelocityCommand)
  history = command_term.command_history

  # Distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
    dim=1,
  )

  # Distance the commands asked for: every command the episode sampled, each held for
  # as long as it was active and the last one held to the end of the episode. An
  # environment that terminated early is measured fairly against a full episode of commands
  # (this envs shouldn't promote)
  segment_durations = history.durations(end_time=env.max_episode_length_s)
  displacement = commanded_displacement(
    history.commands[env_ids], segment_durations[env_ids]
  )
  expected_distance = torch.norm(displacement, dim=1)

  # Crossing half a sub-terrain proves the level regardless of how fast the command
  # was, so promotion never asks for more than that.
  promotion_distance = (expected_distance * promotion_frac).clamp(
    max=terrain_generator.size[0] / 2
  )
  demotion_distance = promotion_distance * demotion_frac

  # Environments barely commanded to move say nothing about the terrain, so they keep
  # their level. The record is empty on the initial reset, which freezes every
  # environment there and preserves ``max_init_terrain_level``.
  is_commanded_to_move = expected_distance >= min_expected_distance
  move_up = is_commanded_to_move & (distance > promotion_distance)
  move_down = is_commanded_to_move & (distance < demotion_distance)

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  # Compute per-terrain-type mean levels.
  levels = terrain.terrain_levels.float()
  result: dict[str, torch.Tensor] = {
    "mean": torch.mean(levels),
    "max": torch.max(levels),
    "walked_distance": torch.mean(distance),
    "expected_distance": torch.mean(expected_distance),
  }

  # In curriculum mode num_cols == num_terrains (one column per type),
  # so the column index directly maps to the sub-terrain name.
  sub_terrain_names = list(terrain_generator.sub_terrains.keys())
  terrain_origins = terrain.terrain_origins
  assert terrain_origins is not None
  num_cols = terrain_origins.shape[1]
  if num_cols == len(sub_terrain_names):
    types = terrain.terrain_types
    for i, name in enumerate(sub_terrain_names):
      mask = types == i
      if mask.any():
        result[name] = torch.mean(levels[mask])

  return result
