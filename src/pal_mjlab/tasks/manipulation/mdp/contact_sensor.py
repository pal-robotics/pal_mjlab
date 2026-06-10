import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor


def site_contact_found(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  site_names: list[str],
  threshold: float = 0.015,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Returns a mask [B, P] of contacts that are close to the specified sites.

  Args:
      env: The environment.
      sensor_name: Name of the ContactSensor.
      site_names: List of site names (one per primary in the sensor).
      threshold: Distance threshold to consider a contact as being "at the site".
      asset_cfg: The robot entity config.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.found is None or data.pos is None:
    return torch.zeros(env.num_envs, len(site_names), device=env.device)

  robot: Entity = env.scene[asset_cfg.name]

  # Get site indices and positions
  # We assume site_names are already prefixed or we handle them here
  # But in env_cfgs.py they are usually unprefixed in SceneEntityCfg
  site_ids, _ = robot.find_sites(site_names, preserve_order=True)
  site_pos_w = robot.data.site_pos_w[:, site_ids]  # [B, P, 3]

  # data.pos is [B, N, 3]. With num_slots=1, N=P.
  contact_pos_w = data.pos  # [B, P, 3]

  # Calculate distance
  dist = torch.norm(contact_pos_w - site_pos_w, dim=-1)  # [B, P]

  # Thresholding
  # data.found > 0 checks if any contact was detected by the sensor at all
  is_near = dist < threshold
  contact_at_site = (data.found > 0) & is_near

  return contact_at_site.float()


def site_contact_both_fingers(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  site_names: list[str],
  threshold: float = 0.03,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  min_dist: float = 0.035,
) -> torch.Tensor:
  """Returns 1.0 if all specified sites are within threshold distance of the object, 0.0 otherwise."""
  robot: Entity = env.scene[asset_cfg.name]
  box: Entity = env.scene["box"]

  site_ids, _ = robot.find_sites(site_names, preserve_order=True)
  site_pos_w = robot.data.site_pos_w[:, site_ids]  # [B, P, 3]

  obj_pos_w = box.data.geom_pos_w[:, 0].unsqueeze(1)  # [B, 1, 3]

  dist_to_obj = torch.norm(site_pos_w - obj_pos_w, dim=-1)  # [B, P]
  both_contact = (dist_to_obj < threshold).all(dim=-1)

  if site_pos_w.shape[1] >= 2:
    dist_between = torch.norm(site_pos_w[:, 0] - site_pos_w[:, 1], dim=-1)
    apart = dist_between >= min_dist
    both_contact = both_contact & apart

  # if both_contact.any():
  #     print("\033[92mFLAG: site_contact_both_fingers is TRUE\033[0m")

  return both_contact.float()


def site_contact_single_finger(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  site_names: list[str],
  threshold: float = 0.03,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Returns 1.0 if exactly one specified site is within threshold distance of the object, 0.0 otherwise."""
  robot: Entity = env.scene[asset_cfg.name]
  box: Entity = env.scene["box"]

  site_ids, _ = robot.find_sites(site_names, preserve_order=True)
  site_pos_w = robot.data.site_pos_w[:, site_ids]  # [B, P, 3]

  obj_pos_w = box.data.geom_pos_w[:, 0].unsqueeze(1)  # [B, 1, 3]

  dist_to_obj = torch.norm(site_pos_w - obj_pos_w, dim=-1)  # [B, P]
  single_contact = (dist_to_obj < threshold).sum(dim=-1) == 1

  return single_contact.float()
