from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


@requires_model_fields(
  "body_pos",
  "geom_size",
  "geom_rbound",
  "geom_aabb",
  recompute=RecomputeLevel.set_const_0,
)
def randomize_table_height(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  table_asset_cfg: SceneEntityCfg,
  height_range: tuple[float, float] = (-0.025, 0.025),
) -> None:
  """Randomize the table's working surface height while keeping the table grounded.

  Samples a height offset per environment and atomically applies it to both the
  table body's local Z position and the table geom's Z half-size, so that:
    - Table bottom remains flush with the floor (z = 0 in local mocap frame).
    - Table top varies by ``height_range`` (default ±2.5 cm).

  Both the body Z position and the geom half-height are shifted by ``delta / 2``,
  because the top surface = body_z + geom_half_z = (base + d) + (base + d) = 2*(base+d),
  so a ±2.5 cm top-surface change requires d = ±0.0125 m.

  Args:
      env: The RL environment.
      env_ids: Environment indices to randomize. If None, all envs.
      table_asset_cfg: SceneEntityCfg for the table entity (must resolve to the
          table body and its box geom).
      height_range: Full range of table-top height change in metres. The default
          ``(-0.025, 0.025)`` gives ±2.5 cm variation on the working surface.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  asset = env.scene[table_asset_cfg.name]
  n_envs = len(env_ids)

  # Sample a top-surface delta per env in [height_range[0], height_range[1]].
  lo, hi = height_range
  top_delta = torch.rand(n_envs, device=env.device) * (hi - lo) + lo  # (n_envs,)
  # The half-height and body-Z both shift by half of top_delta.
  half_delta = top_delta / 2.0  # (n_envs,)

  # --- Geom half-height (axis 2) ---
  geom_ids = asset.indexing.geom_ids[table_asset_cfg.geom_ids]  # (n_geoms,)
  env_g, geom_g = torch.meshgrid(env_ids, geom_ids, indexing="ij")
  default_geom_size = env.sim.get_default_field("geom_size")[geom_ids]  # (n_geoms, 3)
  # New half-height = default + half_delta, broadcast over geoms.
  new_half_z = (
    default_geom_size[None, :, 2] + half_delta[:, None]  # (n_envs, n_geoms)
  )
  # Write full geom_size: keep X/Y from default, update Z.
  new_geom_size = default_geom_size.unsqueeze(0).expand(n_envs, -1, -1).clone()
  new_geom_size[:, :, 2] = new_half_z
  env.sim.model.geom_size[env_g, geom_g] = new_geom_size

  # Recompute broadphase bounds (rbound and aabb) for the new geom size.
  from mjlab.envs.mdp.dr.geom import _recompute_geom_bounds

  _recompute_geom_bounds(env, env_ids, table_asset_cfg)

  # --- Body local Z (axis 2) ---
  body_ids = asset.indexing.body_ids[table_asset_cfg.body_ids]  # (n_bodies,)
  env_b, body_b = torch.meshgrid(env_ids, body_ids, indexing="ij")
  default_body_pos = env.sim.get_default_field("body_pos")[body_ids]  # (n_bodies, 3)
  new_body_pos = default_body_pos.unsqueeze(0).expand(n_envs, -1, -1).clone()
  new_body_pos[:, :, 2] = default_body_pos[None, :, 2] + half_delta[:, None]
  env.sim.model.body_pos[env_b, body_b] = new_body_pos


def reset_joints_mixed(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  goal_joint_pos: dict[str, float] | None = None,
  goal_prob: float = 0.5,
) -> None:
  """Resets robot joints to either the default init state or a goal configuration, with a random offset."""
  from mjlab.tasks.velocity.mdp import resolve_env_ids
  from mjlab.utils.lab_api.math import sample_uniform

  env_ids = resolve_env_ids(env, env_ids)

  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  # Start with default joint positions for these envs and joints
  joint_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()

  # Identify which joint names are being randomized
  joint_names = [asset.joint_names[idx] for idx in asset_cfg.joint_ids]

  # Check if we should initialize some envs in the goal neighborhood
  if goal_joint_pos and len(env_ids) > 0:
    # Sample which of the resetted envs should use the goal configuration
    use_goal = torch.rand(len(env_ids), device=env.device) < goal_prob
    if use_goal.any():
      # Map joint_names to their index in the asset_cfg.joint_ids
      for idx_in_sub, name in enumerate(joint_names):
        if name in goal_joint_pos:
          goal_val = goal_joint_pos[name]
          joint_pos[use_goal, idx_in_sub] += goal_val

  # Add random uniform offset
  joint_pos += sample_uniform(*position_range, joint_pos.shape, env.device)

  # Clamp to soft limits
  joint_pos_limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
  joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])

  # Randomize velocities
  joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
  joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)

  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)

  asset.write_joint_state_to_sim(
    joint_pos.view(len(env_ids), -1),
    joint_vel.view(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=joint_ids,
  )

