from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.utils.lab_api.math import subtract_frame_transforms
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_BOX_CFG = SceneEntityCfg("box")


def hands_to_box(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg,
  box_cfg: SceneEntityCfg = _DEFAULT_BOX_CFG,
) -> torch.Tensor:
  """`asset_cfg.body_ids` should select the hand/wrist bodies to draw
  toward the box (replaces genesis's per-link_name loop with a single
  batched body-index lookup).
  """
  asset: Entity = env.scene[asset_cfg.name]
  box: Entity = env.scene[box_cfg.name]

  box_pos = box.data.root_link_pos_w  # (N, 3)

  hand_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids]  # (N, n_hands, 3)
  err = torch.norm(hand_pos - box_pos.unsqueeze(1), dim=-1)  # (N, n_hands)
  min_err = torch.min(err, dim=1).values  # (N,) — closest hand only
  reward = torch.exp(-torch.square(min_err) / std**2)

  return reward


def hand_contact_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)

  reward = torch.tanh(force_magnitude / 15.0 - 2.0) + 1.0

  return torch.sum(reward, dim=-1)


def track_box_position(
  env: ManagerBasedRlEnv,
  std: float,
  target_pos,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  box_cfg: SceneEntityCfg = _DEFAULT_BOX_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  box: Entity = env.scene[box_cfg.name]

  robot_pos_w = asset.data.root_link_pos_w  # (N, 3)
  robot_quat_w = asset.data.root_link_quat_w  # (N, 4)
  box_pos_w = box.data.root_link_pos_w  # (N, 3)

  # box position expressed in the robot's local frame
  box_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, box_pos_w)

  target_pos_b = torch.tensor(target_pos, device=env.device)

  err = torch.sum(torch.square(box_pos_b - target_pos_b), dim=-1)
  return torch.exp(-err / std**2)


class VariablePostureGripperManipulation:
  """Like `VariablePosture`, but the two regimes (walking / lifting) are
  gated by distance to the box rather than commanded speed.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    asset: Entity = env.scene[asset_cfg.name]
    _, joint_names = asset.find_joints(asset_cfg.joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params.get("std_standing"), list_of_strings=joint_names
    )

    self.std_standing = torch.tensor(std_standing, device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del std_standing

    asset: Entity = env.scene[asset_cfg.name]

    std = self.std_standing

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def table_contact_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data

  cost = torch.sum(sensor_data.found, dim=-1)

  return cost
