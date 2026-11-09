from __future__ import annotations

from typing import TYPE_CHECKING, cast

# from mjlab.tests.test_runner import env
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse


from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def motion_joint_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float | None = None,
  *,
  kappa: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  jnt_ids = asset_cfg.joint_ids
  ref_joint = command.joint_pos[:, jnt_ids]
  robot_joint = command.robot_joint_pos[:, jnt_ids]
  sq_err = torch.square(ref_joint - robot_joint)

  return torch.exp(-kappa * torch.sum(sq_err, dim=-1) / std**2)


def motion_joint_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float | None = 1.0,
  *,
  kappa: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  jnt_ids = asset_cfg.joint_ids
  sq_err = torch.square(
    command.joint_vel[:, jnt_ids] - command.robot_joint_vel[:, jnt_ids]
  )

  return torch.exp(-kappa * torch.sum(sq_err, dim=-1) / std**2)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  *,
  axes: str = "xy",
) -> torch.Tensor:
  angmom = env.scene[sensor_name].data
  if axes == "xy":
    return torch.sum(torch.square(angmom[..., :2]), dim=-1)
  if axes == "xyz":
    return torch.sum(torch.square(angmom), dim=-1)
  raise ValueError(f"Unsupported axes {axes!r}; use 'xy' or 'xyz'.")


def motion_anchor_linear_velocity_body_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float, *, kappa: float = 1.0
) -> torch.Tensor:
  """Root linear velocity tracking in the robot anchor (base) frame."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  anchor_quat = command.robot_anchor_quat_w
  ref_lin_b = quat_apply_inverse(anchor_quat, command.anchor_lin_vel_w)
  robot_lin_b = quat_apply_inverse(anchor_quat, command.robot_anchor_lin_vel_w)
  error = torch.sum(torch.square(ref_lin_b - robot_lin_b), dim=-1)

  return torch.exp(-kappa * error / std**2)


def motion_anchor_angular_velocity_body_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float, *, kappa: float = 1.0
) -> torch.Tensor:
  """Root angular velocity tracking in the robot anchor (base) frame."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  anchor_quat = command.robot_anchor_quat_w
  ref_ang_b = quat_apply_inverse(anchor_quat, command.anchor_ang_vel_w)
  robot_ang_b = quat_apply_inverse(anchor_quat, command.robot_anchor_ang_vel_w)
  error = torch.sum(torch.square(ref_ang_b - robot_ang_b), dim=-1)

  return torch.exp(-kappa * error / std**2)

class zmp_support_polygon_penalty:
  """Keep the Zero Moment Point inside the support polygon for static motions.

  At construction, every clip in the motion library is inspected: if any of the
  ``foot_body_names`` rises more than ``lift_threshold`` above its lowest height in
  that clip, the clip is considered *dynamic* (it contains steps) and this term is
  disabled for it. Envs holding a fixed frame (``moving_envs == False``) are always
  treated as static.

  For static envs, the ZMP of the simulated robot is computed on the ground plane
  from the whole-body centre of mass ``c``, its acceleration ``c_dd`` (finite
  difference of the COM velocity) and the rate of change of the centroidal angular
  momentum ``L_d``::

    zmp_x = c_x - (z_c * m * c_dd_x + L_d_y) / (m * (g + c_dd_z))
    zmp_y = c_y - (z_c * m * c_dd_y - L_d_x) / (m * (g + c_dd_z))

  with ``z_c`` the COM height above the ground. The support polygon is the convex
  hull of the end points of the foot capsule geoms that are touching the ground.
  The returned penalty (use a negative weight) is ``relu(margin - d)`` where ``d`` is
  the signed distance from the ZMP to the polygon boundary (positive inside).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    asset_cfg: SceneEntityCfg = params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    self.robot = env.scene[asset_cfg.name]
    self.device = env.device
    self.num_envs = env.num_envs
    self.dt = env.step_dt

    command = cast(MotionCommand, env.command_manager.get_term(params["command_name"]))

    # --- 1. Static / dynamic classification of every clip in the library. ---
    foot_body_names = params["foot_body_names"]
    foot_idx = [command.cfg.body_names.index(n) for n in foot_body_names]
    foot_z = command.motion.body_pos_w[:, foot_idx, 2]  # (T_total, F)
    lift_threshold = params.get("lift_threshold", 0.03)
    clip_lifts_foot = torch.zeros(
      command.motion.num_trajectories, dtype=torch.bool, device=self.device
    )
    for i, (s, e) in enumerate(
      zip(
        command.motion.segment_start_idx.tolist(),
        command.motion.segment_end_idx.tolist(),
        strict=True,
      )
    ):
      z = foot_z[s:e]
      lift = (z - z.min(dim=0, keepdim=True).values).amax()
      clip_lifts_foot[i] = lift > lift_threshold
    self.clip_is_static = ~clip_lifts_foot
    static_names = [
      n for n, st in zip(command.motion.segment_names, self.clip_is_static.tolist()) if st
    ]
    print(
      f"[zmp_support_polygon_penalty] {len(static_names)}/"
      f"{command.motion.num_trajectories} static clips: {static_names}"
    )

    # --- 2. Foot capsule geometry (end points in the geom frame). ---
    geom_ids, _ = self.robot.find_geoms(params["foot_geom_names"], preserve_order=True)
    self.geom_ids = torch.tensor(geom_ids, dtype=torch.long, device=self.device)
    global_gids = self.robot.indexing.geom_ids[self.geom_ids].cpu().numpy()
    mj_model = env.sim.mj_model
    # Capsule: size[0] = radius, size[1] = half length along the local z axis.
    self.geom_radius = torch.tensor(
      mj_model.geom_size[global_gids, 0], dtype=torch.float32, device=self.device
    )
    self.geom_half_len = torch.tensor(
      mj_model.geom_size[global_gids, 1], dtype=torch.float32, device=self.device
    )

    root_body_id = self.robot.indexing.root_body_id
    self.root_body_id = root_body_id
    self.mass = float(mj_model.body_subtreemass[int(root_body_id)])
    self.gravity = float(abs(mj_model.opt.gravity[2]))

    # --- 3. Finite-difference buffers. ---
    self.prev_com_vel = torch.zeros(self.num_envs, 3, device=self.device)
    self.prev_angmom = torch.zeros(self.num_envs, 3, device=self.device)
    self.has_prev = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # Exposed for debugging / visualisation.
    self.zmp_w = torch.zeros(self.num_envs, 2, device=self.device)
    self.zmp_distance = torch.zeros(self.num_envs, device=self.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.has_prev[env_ids] = False

  # ------------------------------------------------------------------ helpers

  def _support_points(
    self, contact_height: float, ground_z: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Capsule end points in world xy, and whether each touches the ground."""
    pose = self.robot.data.geom_pose_w[:, self.geom_ids]  # (N, G, 7)
    pos, quat = pose[..., :3], pose[..., 3:]
    z_axis = torch.zeros_like(pos)
    z_axis[..., 2] = 1.0
    axis_w = quat_apply(quat, z_axis) * self.geom_half_len[None, :, None]
    pts = torch.cat([pos + axis_w, pos - axis_w], dim=1)  # (N, 2G, 3)
    radius = torch.cat([self.geom_radius, self.geom_radius])[None, :]
    in_contact = (pts[..., 2] - ground_z[:, None] - radius) < contact_height
    return pts[..., :2], in_contact

  @staticmethod
  def _signed_distance_to_hull(
    point: torch.Tensor, pts: torch.Tensor, valid: torch.Tensor, eps: float = 1e-6
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Signed distance (positive inside) from ``point`` to the convex hull of ``pts``.

    Batched brute-force hull: the directed edge i->j belongs to the CCW hull iff
    every valid point lies on its left. Inside a convex polygon, the distance to
    the boundary equals the min signed distance to the hull edge lines; outside it
    is a (tight enough) lower bound on the true distance.

    Args:
      point: (N, 2). pts: (N, P, 2). valid: (N, P) bool.

    Returns:
      dist: (N,), has_hull: (N,) bool (False if fewer than 2 distinct points).
    """
    edge = pts[:, None, :, :] - pts[:, :, None, :]  # (N, i, j, 2) = p_j - p_i
    edge_len = torch.linalg.norm(edge, dim=-1)  # (N, i, j)
    # cross(p_j - p_i, p_k - p_i) for all (i, j, k): (N, i, j, k). Note that
    # p_k - p_i is the same tensor as ``edge`` indexed by (i, k).
    cross = (
      edge[:, :, :, None, 0] * edge[:, :, None, :, 1]
      - edge[:, :, :, None, 1] * edge[:, :, None, :, 0]
    )
    left_or_on = (cross >= -eps * edge_len[..., None]) | ~valid[:, None, None, :]
    is_edge = (
      left_or_on.all(dim=-1)
      & valid[:, :, None]
      & valid[:, None, :]
      & (edge_len > eps)
    )
    to_point = point[:, None, :] - pts  # (N, i, 2)
    edge_cross = (
      edge[..., 0] * to_point[:, :, None, 1] - edge[..., 1] * to_point[:, :, None, 0]
    )
    line_dist = edge_cross / edge_len.clamp_min(eps)
    line_dist = torch.where(is_edge, line_dist, torch.full_like(line_dist, float("inf")))
    dist = line_dist.flatten(1).amin(dim=-1)
    has_hull = is_edge.flatten(1).any(dim=-1)
    return torch.where(has_hull, dist, torch.zeros_like(dist)), has_hull

  # ------------------------------------------------------------------ call

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    foot_body_names: tuple[str, ...],
    foot_geom_names: tuple[str, ...],
    margin: float = 0.02,
    lift_threshold: float = 0.03,
    contact_height: float = 0.01,
    use_angular_momentum: bool = True,
    max_com_acc: float = 20.0,
    max_penalty: float = 0.3,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del foot_body_names, lift_threshold, asset_cfg  # Used in __init__.
    command = cast(MotionCommand, env.command_manager.get_term(command_name))
    data = self.robot.data.data

    # Whole-body centroidal quantities (world frame).
    com = data.subtree_com[:, self.root_body_id]
    com_vel = data.subtree_linvel[:, self.root_body_id]
    angmom = data.subtree_angmom[:, self.root_body_id]

    # Finite differences; zero on the first step after a reset.
    has_prev = self.has_prev[:, None]
    com_acc = torch.where(has_prev, (com_vel - self.prev_com_vel) / self.dt, 0.0)
    angmom_dot = torch.where(has_prev, (angmom - self.prev_angmom) / self.dt, 0.0)
    # Guard against spikes when the command teleports the robot mid-episode.
    acc_norm = torch.linalg.norm(com_acc, dim=-1, keepdim=True)
    com_acc = com_acc * torch.clamp(max_com_acc / acc_norm.clamp_min(1e-6), max=1.0)
    self.prev_com_vel.copy_(com_vel)
    self.prev_angmom.copy_(angmom)
    self.has_prev[:] = True

    # Dynamic ZMP on the (flat) ground plane.
    ground_z = env.scene.env_origins[:, 2]
    z_c = com[:, 2] - ground_z
    denom = self.mass * torch.clamp(self.gravity + com_acc[:, 2], min=0.1 * self.gravity)
    tau_x = z_c * self.mass * com_acc[:, 0]
    tau_y = z_c * self.mass * com_acc[:, 1]
    if use_angular_momentum:
      tau_x = tau_x + angmom_dot[:, 1]
      tau_y = tau_y - angmom_dot[:, 0]
    zmp = torch.stack([com[:, 0] - tau_x / denom, com[:, 1] - tau_y / denom], dim=-1)

    # Support polygon from the foot capsules that touch the ground.
    pts, in_contact = self._support_points(contact_height, ground_z)
    dist, has_hull = self._signed_distance_to_hull(zmp, pts, in_contact)

    self.zmp_w = zmp
    self.zmp_distance = dist

    static = self.clip_is_static[command.rand_motion] | ~command.moving_envs
    penalty = torch.clamp(margin - dist, min=0.0, max=max_penalty)
    return penalty * (static & has_hull).float()
