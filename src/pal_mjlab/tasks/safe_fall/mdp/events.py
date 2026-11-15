"""Reset and disturbance events for the safe-fall task."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp
from mjlab.entity import Entity
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_angle_axis,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
)

from .faults import ActuatorFaultCfg, ActuatorFaults, get_actuator_faults

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

FAMILY_STANDING = 0
FAMILY_MOVING = 1

# What the walking policy was going through when a state was recorded.
STATE_WALKING = 0
STATE_TRIPPED = 1
STATE_JOINT_DOWN = 2

# Gap [m] left between the ground and the lowest point of a tipped robot.
_TIP_GROUND_CLEARANCE = 0.005


class reset_fall_start:
  """Reset to the instant the fallback policy takes over, then kick the robot.

  Each environment starts from one of two families:

  - **standing**: the default pose with a small offset on ``asset_cfg``'s
    joints. With a zero kick this is the plain emergency squat.
  - **moving**: a full state (root height, orientation, velocities and joint
    state) sampled from a buffer recorded while the walking policy was
    walking, so the robot is mid-gait and already has momentum. Used for a
    ``moving_fraction`` of the environments when ``state_buffer_path`` names an
    existing file; otherwise every environment starts standing.

  The buffer can also hold states recorded a detection time after something
  went wrong with the walking policy, which is when a supervisor hands over:

  - **joint down**: an actuator went limp. A moving start whose episode has
    an actuator down from the first step (see ``fault_cfg``) is taken from
    the states recorded with that same actuator down in a
    ``fault_state_fraction`` of the cases. The actuator faults of the episode
    must therefore be drawn before this event runs.
  - **tripped**: a swinging foot was caught. A ``trip_state_fraction`` of the
    other moving starts is taken from these.

  Such a state already holds a fall in the making, so it is never tipped, and
  an ``as_recorded_fraction`` of them is kicked by at most
  ``as_recorded_max_kick_speed`` instead of the full range.

  A ``tip_fraction`` of the other starts is then tipped over: the
  whole robot is rotated about a horizontal axis of uniform direction by an
  angle in ``tip_angle_range``, set down so that the edge of its feet touches
  the ground, and given a tipping rate in ``tip_rate_range`` about that
  axis. Beyond roughly 25 degrees the center of mass is far outside the feet
  and the fall cannot be avoided, whatever the kick: this is the state a fall
  detector hands over. ``ground_geom_cfg`` selects the foot geoms (capsules or
  spheres) that are set down on the ground; it is required for tipping.

  Every start then receives a horizontal velocity kick of speed in
  ``[min_kick_speed, max_kick_speed]`` plus a roll/pitch rate in
  ``[-max_kick_ang_speed, max_kick_ang_speed]``. Small kicks are recoverable
  on the feet; large ones make a fall inevitable. A robot that falls while
  walking or running falls forwards: in a ``forward_fraction`` of the episodes
  the kick, and the tipping if there is one, point within
  ``forward_half_angle`` of where the robot faces. In the others the
  directions are uniform, which stands for external pushes. Most handovers
  are mild: in a ``gentle_fraction`` of the episodes the kick speed is drawn
  below ``gentle_max_kick_speed`` only.

  ``asset_cfg.joint_names`` must not select joints of the closed leg chain
  whose offset would violate the knee linkage (leg length, femur, knee).

  The sampled kick speed, start family and tip angle (0 if not tipped) of
  each environment are kept in ``kick_speed``, ``family`` and ``tip_angle`` so
  evaluation can bin results by them, as are the kind of recorded state of a
  moving start (``state_kind``) and whether it was left as recorded
  (``as_recorded``).
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self.kick_speed = torch.zeros(env.num_envs, device=env.device)
    self.family = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self.tip_angle = torch.zeros(env.num_envs, device=env.device)
    self.state_kind = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    self.as_recorded = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    # Foot geoms that are set down on the ground after tipping.
    self._ground_geom_ids: torch.Tensor | None = None
    ground_geom_cfg: SceneEntityCfg | None = cfg.params.get("ground_geom_cfg")
    if ground_geom_cfg is not None:
      entity: Entity = env.scene[ground_geom_cfg.name]
      local_ids = entity.find_geoms(ground_geom_cfg.geom_names)[0]
      global_ids = entity.indexing.geom_ids[local_ids].cpu().numpy()
      model = env.sim.mj_model
      types = model.geom_type[global_ids]
      capsule = types == mujoco.mjtGeom.mjGEOM_CAPSULE
      if not np.all(capsule | (types == mujoco.mjtGeom.mjGEOM_SPHERE)):
        raise ValueError("ground_geom_cfg must select capsule or sphere geoms.")
      sizes = model.geom_size[global_ids]
      self._ground_geom_ids = torch.as_tensor(global_ids, device=env.device).long()
      self._ground_geom_radius = torch.as_tensor(
        sizes[:, 0], dtype=torch.float32, device=env.device
      )
      self._ground_geom_half_length = torch.as_tensor(
        np.where(capsule, sizes[:, 1], 0.0), dtype=torch.float32, device=env.device
      )

    self._buffer: dict[str, torch.Tensor] | None = None
    path = cfg.params.get("state_buffer_path")
    if path is not None and not Path(path).is_file():
      warnings.warn(
        f"Walker state buffer '{Path(path).resolve()}' not found: every episode "
        "will start standing still. Record it with `python -m "
        "pal_mjlab.scripts.record_walker_states --checkpoint <walker.pt>` to "
        "also train on starts taken from walking.",
        stacklevel=2,
      )
      path = None
    if path is not None:
      asset: Entity = env.scene[cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name]
      with np.load(Path(path)) as data:
        joint_names = tuple(str(n) for n in data["joint_names"])
        if joint_names != tuple(asset.joint_names):
          raise ValueError(
            f"State buffer '{path}' was recorded with a different joint order."
          )
        self._buffer = {
          key: torch.as_tensor(data[key], dtype=torch.float32, device=env.device)
          for key in ("root_height", "root_quat", "root_vel", "joint_pos", "joint_vel")
        }
        num_states = len(data["root_height"])
        # Buffers from before the trip and joint-down states hold walking only.
        kind = data["kind"] if "kind" in data else np.zeros(num_states, dtype=np.int64)
        fault_joint = (
          data["fault_joint"]
          if "fault_joint" in data
          else np.full(num_states, -1, dtype=np.int64)
        )
        fault_joint_names = (
          tuple(str(n) for n in data["fault_joint_names"])
          if "fault_joint_names" in data
          else ()
        )
      self._index_buffer(
        torch.as_tensor(kind, dtype=torch.long, device=env.device),
        torch.as_tensor(fault_joint, dtype=torch.long, device=env.device),
        fault_joint_names,
      )

  def _index_buffer(
    self,
    kind: torch.Tensor,
    fault_joint: torch.Tensor,
    fault_joint_names: tuple[str, ...],
  ) -> None:
    """Sort the buffer's states by kind and, for joint-down states, by joint."""
    self._walking_ids = (kind == STATE_WALKING).nonzero().flatten()
    self._tripped_ids = (kind == STATE_TRIPPED).nonzero().flatten()
    if len(self._walking_ids) == 0:
      raise ValueError("The walker state buffer holds no plain walking states.")
    down = (kind == STATE_JOINT_DOWN).nonzero().flatten()
    joint = fault_joint[down]
    self._joint_down_ids = down[torch.argsort(joint, stable=True)]
    # States with actuator column j down: _joint_down_ids[start[j] : start[j] + count[j]].
    self._joint_down_count = torch.bincount(joint, minlength=len(fault_joint_names))
    self._joint_down_start = (
      torch.cumsum(self._joint_down_count, dim=0) - self._joint_down_count
    )
    self._fault_joint_names = fault_joint_names
    self._fault_joint_names_checked = False

  def _joint_down_states(
    self, faults: ActuatorFaults, env_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """For each environment, a state recorded with its first failed actuator
    down: (whether there is one [n], its index in the buffer [n])."""
    if not self._fault_joint_names_checked:
      if tuple(faults.joint_names) != self._fault_joint_names:
        raise ValueError(
          "The walker state buffer was recorded with a different actuator order."
        )
      self._fault_joint_names_checked = True
    first = faults.first_failed[env_ids]
    column = first.clamp(min=0)
    count = self._joint_down_count[column]
    available = (first >= 0) & (count > 0)
    offset = (torch.rand(len(env_ids), device=first.device) * count).long()
    position = self._joint_down_start[column] + torch.minimum(offset, count - 1)
    position = position.clamp(min=0, max=len(self._joint_down_ids) - 1)
    return available, self._joint_down_ids[position]

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    del env_ids  # Stateless between episodes.

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    max_kick_speed: float,
    min_kick_speed: float = 0.0,
    max_kick_ang_speed: float = 0.0,
    position_range: tuple[float, float] = (0.0, 0.0),
    height_range: tuple[float, float] = (0.0, 0.0),
    joint_pos_noise: float = 0.0,
    moving_fraction: float = 0.0,
    state_buffer_path: str | None = None,
    tip_fraction: float = 0.0,
    tip_angle_range: tuple[float, float] = (0.0, 0.0),
    tip_rate_range: tuple[float, float] = (0.0, 0.0),
    forward_fraction: float = 0.0,
    forward_half_angle: float = math.radians(30.0),
    gentle_fraction: float = 0.0,
    gentle_max_kick_speed: float = 2.5,
    trip_state_fraction: float = 0.0,
    fault_state_fraction: float = 0.0,
    fault_cfg: ActuatorFaultCfg | None = None,
    as_recorded_fraction: float = 0.0,
    as_recorded_max_kick_speed: float = 0.5,
    ground_geom_cfg: SceneEntityCfg | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> None:
    del state_buffer_path, ground_geom_cfg  # Used in __init__.
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    device = env.device
    asset: Entity = env.scene[asset_cfg.name]

    # -- Standing family: default state plus joint noise.
    default_root = asset.data.default_root_state[env_ids]
    root_height = default_root[:, 2].clone()
    root_quat = default_root[:, 3:7].clone()
    root_vel = default_root[:, 7:13].clone()
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    if joint_pos_noise > 0.0:
      joint_ids = asset_cfg.joint_ids
      noise_shape = joint_pos[:, joint_ids].shape
      noisy = joint_pos[:, joint_ids] + sample_uniform(
        -joint_pos_noise, joint_pos_noise, noise_shape, device
      )
      limits = asset.data.soft_joint_pos_limits[env_ids][:, joint_ids]
      joint_pos[:, joint_ids] = noisy.clamp(limits[..., 0], limits[..., 1])

    # -- Moving family: overwrite with recorded walking states.
    family = torch.full((n,), FAMILY_STANDING, device=device, dtype=torch.long)
    state_kind = torch.full((n,), STATE_WALKING, device=device, dtype=torch.long)
    if self._buffer is not None and moving_fraction > 0.0:
      moving = torch.rand(n, device=device) < moving_fraction

      def pick(ids: torch.Tensor) -> torch.Tensor:
        return ids[torch.randint(0, len(ids), (n,), device=device)]

      idx = pick(self._walking_ids)
      if fault_state_fraction > 0.0 and len(self._joint_down_ids) > 0:
        if fault_cfg is None:
          raise ValueError("fault_state_fraction > 0 needs fault_cfg.")
        available, down_idx = self._joint_down_states(
          get_actuator_faults(env, fault_cfg), env_ids
        )
        use = available & (torch.rand(n, device=device) < fault_state_fraction)
        idx = torch.where(use, down_idx, idx)
        state_kind[use] = STATE_JOINT_DOWN
      if trip_state_fraction > 0.0 and len(self._tripped_ids) > 0:
        use = (state_kind == STATE_WALKING) & (
          torch.rand(n, device=device) < trip_state_fraction
        )
        idx = torch.where(use, pick(self._tripped_ids), idx)
        state_kind[use] = STATE_TRIPPED
      state_kind[~moving] = STATE_WALKING
      root_height = torch.where(moving, self._buffer["root_height"][idx], root_height)
      root_quat = torch.where(
        moving.unsqueeze(-1), self._buffer["root_quat"][idx], root_quat
      )
      root_vel = torch.where(
        moving.unsqueeze(-1), self._buffer["root_vel"][idx], root_vel
      )
      joint_pos = torch.where(
        moving.unsqueeze(-1), self._buffer["joint_pos"][idx], joint_pos
      )
      joint_vel = torch.where(
        moving.unsqueeze(-1), self._buffer["joint_vel"][idx], joint_vel
      )
      family = torch.where(moving, FAMILY_MOVING, family)

    # -- Random heading: rotate the whole state about the vertical axis.
    yaw = sample_uniform(-math.pi, math.pi, (n,), device)
    zeros = torch.zeros_like(yaw)
    yaw_quat = quat_from_euler_xyz(zeros, zeros, yaw)
    root_quat = quat_mul(yaw_quat, root_quat)
    lin_vel = quat_apply(yaw_quat, root_vel[:, 0:3])
    ang_vel = quat_apply(yaw_quat, root_vel[:, 3:6])

    # -- Falls in the direction of travel: one direction per episode, near
    # where the robot faces, for both the tipping and the kick.
    forward = torch.rand(n, device=device) < forward_fraction
    x_axis = torch.zeros(n, 3, device=device)
    x_axis[:, 0] = 1.0
    facing = quat_apply(root_quat, x_axis)
    forward_direction = torch.atan2(facing[:, 1], facing[:, 0]) + sample_uniform(
      -forward_half_angle, forward_half_angle, (n,), device
    )

    # A state recorded after a trip or with a joint down is a fall already.
    disturbed = state_kind != STATE_WALKING
    as_recorded = disturbed & (torch.rand(n, device=device) < as_recorded_fraction)

    # -- Tipping: rotate the whole state about a horizontal axis through the
    # ground point below the base, and add a rotation rate about that axis.
    tip_angle = torch.zeros(n, device=device)
    tip_offset = torch.zeros(n, 2, device=device)
    if tip_fraction > 0.0:
      if self._ground_geom_ids is None:
        raise ValueError("tip_fraction > 0 needs ground_geom_cfg.")
      tipped = (torch.rand(n, device=device) < tip_fraction) & ~disturbed
      tip_angle = torch.where(
        tipped, sample_uniform(*tip_angle_range, (n,), device), tip_angle
      )
      tip_rate = torch.where(
        tipped, sample_uniform(*tip_rate_range, (n,), device), torch.zeros_like(yaw)
      )
      heading = sample_uniform(-math.pi, math.pi, (n,), device)
      heading = torch.where(forward, forward_direction, heading)
      # The robot falls towards ``toward``; the rotation axis is z x toward.
      toward = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
      axis = torch.stack([-toward[:, 1], toward[:, 0], zeros], dim=-1)
      tip_quat = quat_from_angle_axis(tip_angle, axis)
      root_quat = quat_mul(tip_quat, root_quat)
      lin_vel = quat_apply(tip_quat, lin_vel)
      ang_vel = quat_apply(tip_quat, ang_vel) + tip_rate.unsqueeze(-1) * axis
      sin, cos = torch.sin(tip_angle), torch.cos(tip_angle)
      tip_offset = (root_height * sin).unsqueeze(-1) * toward
      lin_vel[:, 0:2] += (tip_rate * root_height * cos).unsqueeze(-1) * toward
      lin_vel[:, 2] -= tip_rate * root_height * sin
      root_height = root_height * cos

    # -- Kick: horizontal velocity, plus a tipping rate.
    speed = sample_uniform(min_kick_speed, max_kick_speed, (n,), device)
    gentle_max = min(gentle_max_kick_speed, max_kick_speed)
    speed = torch.where(
      torch.rand(n, device=device) < gentle_fraction,
      sample_uniform(min(min_kick_speed, gentle_max), gentle_max, (n,), device),
      speed,
    )
    speed = torch.where(
      as_recorded,
      sample_uniform(0.0, as_recorded_max_kick_speed, (n,), device),
      speed,
    )
    direction = sample_uniform(-math.pi, math.pi, (n,), device)
    direction = torch.where(forward, forward_direction, direction)
    lin_vel[:, 0] += speed * torch.cos(direction)
    lin_vel[:, 1] += speed * torch.sin(direction)
    if max_kick_ang_speed > 0.0:
      ang_vel[:, 0:2] += sample_uniform(
        -max_kick_ang_speed, max_kick_ang_speed, (n, 2), device
      )

    position = env.scene.env_origins[env_ids].clone()
    position[:, 0:2] += sample_uniform(*position_range, (n, 2), device) + tip_offset
    position[:, 2] += root_height + sample_uniform(*height_range, (n,), device)

    asset.write_root_link_pose_to_sim(
      torch.cat([position, root_quat], dim=-1), env_ids=env_ids
    )
    asset.write_root_link_velocity_to_sim(
      torch.cat([lin_vel, ang_vel], dim=-1), env_ids=env_ids
    )
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # -- Set tipped robots down: the rotation above lifts one edge of the feet
    # and pushes the other into the ground.
    if bool((tip_angle > 0.0).any()):
      lowest = self._lowest_ground_geom_height(env, env_ids)
      shift = _TIP_GROUND_CLEARANCE - (lowest - env.scene.env_origins[env_ids, 2])
      position[:, 2] += torch.where(tip_angle > 0.0, shift, torch.zeros_like(shift))
      asset.write_root_link_pose_to_sim(
        torch.cat([position, root_quat], dim=-1), env_ids=env_ids
      )

    self.kick_speed[env_ids] = speed
    self.family[env_ids] = family
    self.tip_angle[env_ids] = tip_angle
    self.state_kind[env_ids] = state_kind
    self.as_recorded[env_ids] = as_recorded

  def _lowest_ground_geom_height(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor
  ) -> torch.Tensor:
    """World height [len(env_ids)] of the lowest point of the ground geoms."""
    assert self._ground_geom_ids is not None
    with wp.ScopedDevice(env.sim.wp_device):
      mjwarp.kinematics(env.sim.wp_model, env.sim.wp_data)
    data = env.sim.data
    center_z = data.geom_xpos[env_ids][:, self._ground_geom_ids, 2]
    # A capsule's axis is its local z; its world z component is xmat[2, 2].
    axis_z = data.geom_xmat[env_ids][:, self._ground_geom_ids, 2, 2]
    lowest = (
      center_z - self._ground_geom_half_length * axis_z.abs() - self._ground_geom_radius
    )
    return lowest.min(dim=-1).values


def shove(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  speed_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  """Add a horizontal velocity of random direction to the whole robot.

  A second disturbance in the middle of the episode, for ``mode="interval"``:
  it also arrives after the robot has settled, so a posture that only looks
  stable is tested and one that leaves the chest open is punished through
  the impact it leads to. Direction is uniform, speed [m/s] uniform in
  ``speed_range``.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  n = len(env_ids)
  speed = sample_uniform(speed_range[0], speed_range[1], (n,), env.device)
  heading = sample_uniform(-math.pi, math.pi, (n,), env.device)
  vel_w = asset.data.root_link_vel_w[env_ids].clone()
  vel_w[:, 0] += speed * torch.cos(heading)
  vel_w[:, 1] += speed * torch.sin(heading)
  asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)
