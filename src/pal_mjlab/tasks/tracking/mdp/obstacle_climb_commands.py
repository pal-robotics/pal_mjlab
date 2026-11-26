"""Motion command for a get-on-box motion tracked on boxes of a sampled height.

The reference motion (e.g. ``get_on_box``) is recorded stepping onto a box of height
``reference_obstacle_height`` and is used as is: every environment tracks the same
frames, only the box under it changes. Each environment is placed on the terrain row
whose box has the sampled height (one row per height, see
``BoxSingleObstacleTerrainCfg.box_height_range``), and the policy has to absorb the
difference between the box and the reference.

The reference state written at a reset is moved vertically so that no foot starts
inside the box, which the reference does not respect (a higher box than the reference,
and its soles sink into the box while pushing up): see ``climb_spawn_offset``.

The reference does not respect the box with the rest of the body either (the knees
pass through its front face while stepping up), so at construction every frame is
checked against every box in the CPU model of the scene, and a reset that would
start a body inside the box starts at the nearest frame that does not (see
``nearest_true``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def feet_over_obstacle(
  foot_xy: torch.Tensor,
  footprint: tuple[float, float, float, float],
  margin: float,
) -> torch.Tensor:
  """``[T, num_feet]``: the foot (link origin, ``[T, num_feet, 2]``) is over the box.

  ``footprint`` is the box ``(x_min, x_max, y_min, y_max)`` relative to the env origin,
  grown by ``margin`` (about half a foot) so a foot partly over the edge counts.
  """
  x0, x1, y0, y1 = footprint
  x, y = foot_xy[..., 0], foot_xy[..., 1]
  return (x > x0 - margin) & (x < x1 + margin) & (y > y0 - margin) & (y < y1 + margin)


def climb_spawn_offset(
  feet_over: torch.Tensor,
  sole_z: torch.Tensor,
  time_steps: torch.Tensor,
  obstacle_height: torch.Tensor,
  clearance: float = 0.0,
) -> torch.Tensor:
  """Vertical offset (m) of the reference state written at a reset, per env.

  ``feet_over`` (``feet_over_obstacle``) and ``sole_z`` (sole height of each foot in
  the reference) are ``[T, num_feet]``. The lowest sole over the box is put
  ``clearance`` above ``obstacle_height``: with every foot over the box the robot is
  moved by that much either way; with only some, it is only lifted (the foot on the
  ground stays on or above the ground); with none, it is left as is.
  """
  over = feet_over[time_steps]
  lowest = torch.where(over, sole_z[time_steps], torch.inf).min(dim=-1).values
  dz = torch.where(over.any(dim=-1), obstacle_height + clearance - lowest, 0.0)
  return torch.where(over.all(dim=-1), dz, dz.clamp(min=0.0))


def nearest_true(mask: torch.Tensor) -> torch.Tensor:
  """``[R, T]`` bool -> ``[R, T]`` long: per row, the nearest frame where ``mask``.

  Ties go to the later frame. Raises if a row has no such frame.
  """
  t = torch.arange(mask.shape[1], device=mask.device)
  out = torch.empty(mask.shape, dtype=torch.long, device=mask.device)
  for r in range(mask.shape[0]):
    ok = torch.nonzero(mask[r]).flatten()
    if ok.numel() == 0:
      raise ValueError(f"Row {r} has no valid frame.")
    dist = (ok[None, :] - t[:, None]).abs() * 2 - (ok[None, :] > t[:, None]).long()
    out[r] = ok[dist.argmin(dim=1)]
  return out


class ObstacleClimbMotionCommand(MotionCommand):
  cfg: ObstacleClimbMotionCommandCfg

  def __init__(self, cfg: ObstacleClimbMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    terrain = env.scene.terrain
    if terrain is None or terrain.terrain_origins is None:
      raise ValueError("ObstacleClimbMotionCommand needs a generator terrain.")
    num_rows = terrain.terrain_origins.shape[0]
    if num_rows != len(cfg.obstacle_heights):
      raise ValueError(
        f"The terrain has {num_rows} rows but {len(cfg.obstacle_heights)} obstacle "
        "heights were given; row i must hold the box of height obstacle_heights[i]."
      )

    foot_ids = self.robot.find_bodies(cfg.foot_body_names, preserve_order=True)[0]
    foot_pos = self.motion._body_pos_w[:, foot_ids]
    self.feet_over = feet_over_obstacle(
      foot_pos[..., :2], cfg.obstacle_footprint, cfg.footprint_margin
    )
    # Sole height: the foot link height above its height at frame 0 (on the ground).
    self.sole_z = foot_pos[..., 2] - foot_pos[0, :, 2]
    if self.feet_over[0].any() or not self.feet_over[-1].all():
      raise ValueError(
        "The motion must start with no foot and end with every foot over the box "
        f"footprint {cfg.obstacle_footprint}: check obstacle_footprint."
      )
    self.obstacle_heights = torch.tensor(cfg.obstacle_heights, device=self.device)
    self.env_obstacle_heights = self.obstacle_heights[terrain.terrain_levels].clone()
    self.metrics["obstacle_height"] = self.env_obstacle_heights.clone()

    clean = self._collision_free_frames(env, terrain.terrain_origins[:, 0])
    self.spawn_frames = nearest_true(clean)
    for h, row in zip(cfg.obstacle_heights, clean, strict=True):
      print(
        f"[ObstacleClimbMotionCommand] box {h:.3f} m: "
        f"{int((~row).sum())}/{row.numel()} reference frames start inside the box, "
        "resets there use the nearest frame that does not"
      )
    if not clean[:, 0].all():
      raise ValueError(
        "The first frame of the motion starts inside the box: move the box away "
        "(obstacle_footprint and the terrain)."
      )

  def _collision_free_frames(
    self, env: ManagerBasedRlEnv, origins: torch.Tensor
  ) -> torch.Tensor:
    """``[rows, T]``: the reset state of the frame on that row's box (with
    ``climb_spawn_offset``) has no robot geom deeper than the tolerance in the
    terrain, in the CPU model of the scene."""
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    prefix = f"{self.cfg.entity_name}/"
    joint_adr = np.array(
      [model.jnt_qposadr[model.joint(prefix + n).id] for n in self.robot.joint_names]
    )
    free = next(
      j
      for j in range(model.njnt)
      if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
      and model.joint(j).name.startswith(prefix)
    )
    free_adr = model.jnt_qposadr[free]
    robot_geom = np.array(
      [
        model.body(model.geom_bodyid[g]).name.startswith(prefix)
        for g in range(model.ngeom)
      ]
    )

    n_frames = self.motion.time_step_total
    root_pos = self.motion._body_pos_w[:, 0].cpu().numpy()
    root_quat = self.motion._body_quat_w[:, 0].cpu().numpy()
    joint_pos = self.motion.joint_pos.cpu().numpy()
    frames = torch.arange(n_frames, device=self.device)
    clean = torch.ones(len(origins), n_frames, dtype=torch.bool)
    tol = self.cfg.spawn_penetration_tolerance
    for r, origin in enumerate(origins.cpu().numpy()):
      dz = (
        climb_spawn_offset(
          self.feet_over,
          self.sole_z,
          frames,
          torch.full((n_frames,), self.cfg.obstacle_heights[r], device=self.device),
          self.cfg.spawn_clearance,
        )
        .cpu()
        .numpy()
      )
      for t in range(n_frames):
        data.qpos[:] = model.qpos0
        data.qpos[free_adr : free_adr + 3] = origin + root_pos[t] + [0.0, 0.0, dz[t]]
        data.qpos[free_adr + 3 : free_adr + 7] = root_quat[t]
        data.qpos[joint_adr] = joint_pos[t]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        for i in range(data.ncon):
          c = data.contact[i]
          if robot_geom[c.geom1] != robot_geom[c.geom2] and c.dist < -tol:
            clean[r, t] = False
            break
    return clean.to(self.device)

  def _to_spawnable(self, env_ids: torch.Tensor) -> None:
    terrain = self._env.scene.terrain
    assert terrain is not None
    rows = terrain.terrain_levels[env_ids]
    self.time_steps[env_ids] = self.spawn_frames[rows, self.time_steps[env_ids]]

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    super()._adaptive_sampling(env_ids)
    self._to_spawnable(env_ids)

  def _uniform_sampling(self, env_ids: torch.Tensor):
    super()._uniform_sampling(env_ids)
    self._to_spawnable(env_ids)

  def _sample_obstacle_heights(self, env_ids: torch.Tensor) -> None:
    """Pick a height per env and move it to the terrain row with that box."""
    terrain = self._env.scene.terrain
    assert terrain is not None and terrain.terrain_origins is not None
    rows = torch.randint(
      0, len(self.cfg.obstacle_heights), (len(env_ids),), device=self.device
    )
    terrain.terrain_levels[env_ids] = rows
    terrain.env_origins[env_ids] = terrain.terrain_origins[
      rows, terrain.terrain_types[env_ids]
    ]
    self.env_obstacle_heights[env_ids] = self.obstacle_heights[rows]
    self.metrics["obstacle_height"][env_ids] = self.obstacle_heights[rows]

  def _resample_command(self, env_ids: torch.Tensor):
    self._sample_obstacle_heights(env_ids)
    return super()._resample_command(env_ids)

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    dz = climb_spawn_offset(
      self.feet_over,
      self.sole_z,
      self.time_steps[env_ids],
      self.env_obstacle_heights[env_ids],
      self.cfg.spawn_clearance,
    )
    root_pos = root_pos.clone()
    root_pos[:, 2] += dz
    super()._write_reference_state_to_sim(
      env_ids, root_pos, root_ori, root_lin_vel, root_ang_vel, joint_pos, joint_vel
    )


@dataclass(kw_only=True)
class ObstacleClimbMotionCommandCfg(MotionCommandCfg):
  obstacle_heights: tuple[float, ...]
  """Box height of each terrain row, in meters (row i <-> obstacle_heights[i])."""
  reference_obstacle_height: float
  """Height of the box the reference motion was recorded on, in meters."""
  obstacle_footprint: tuple[float, float, float, float]
  """Box ``(x_min, x_max, y_min, y_max)`` relative to the env origin, in meters."""
  foot_body_names: tuple[str, ...] = ("leg_left_foot_link", "leg_right_foot_link")
  footprint_margin: float = 0.12
  """A foot counts as over the box within this distance of its footprint (m)."""
  spawn_clearance: float = 0.01
  """Height of the lowest sole over the box above it at a reset (m)."""
  spawn_penetration_tolerance: float = 0.01
  """A reset frame is rejected when a robot geom is deeper than this in the terrain."""

  def build(self, env: ManagerBasedRlEnv) -> ObstacleClimbMotionCommand:
    return ObstacleClimbMotionCommand(self, env)
