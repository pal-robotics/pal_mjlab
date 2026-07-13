import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg


def site_contact_both_fingers(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  site_names: list[str],
  threshold: float = 0.05,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  min_dist: float = 0.0,  # 0.035
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

  return both_contact.float()

