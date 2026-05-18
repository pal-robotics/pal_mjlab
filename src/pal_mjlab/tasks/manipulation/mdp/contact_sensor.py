import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactSensor
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

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
  site_pos_w = robot.data.site_pos_w[:, site_ids] # [B, P, 3]
  
  # data.pos is [B, N, 3]. With num_slots=1, N=P.
  contact_pos_w = data.pos # [B, P, 3]
  
  # Calculate distance
  dist = torch.norm(contact_pos_w - site_pos_w, dim=-1) # [B, P]
  
  # Thresholding
  # data.found > 0 checks if any contact was detected by the sensor at all
  is_near = dist < threshold
  contact_at_site = (data.found > 0) & is_near
  
  return contact_at_site.float()

def site_contact_both_fingers(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  site_names: list[str],
  threshold: float = 0.02,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Returns 1.0 if all specified sites are in contact, 0.0 otherwise."""
  contact_mask = site_contact_found(env, sensor_name, site_names, threshold, asset_cfg)
  return contact_mask.all(dim=-1).float()
