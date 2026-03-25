"""Useful methods for MDP Metrics."""

from mjlab.sensor.contact_sensor import ContactSensor
import torch
from mjlab.entity.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

def joint_velocity_magnitude(env, asset_cfg):
  """L1 norm of joint velocities."""
  return env.scene[asset_cfg.name].data.joint_vel.abs().sum(dim=-1)  # (num_envs,)


def joint_accelerations_magnitude(env, asset_cfg):
  """L1 norm of joint velocities."""
  return env.scene[asset_cfg.name].data.joint_acc.abs().sum(dim=-1)  # (num_envs,)


def joint_torques_magnitude(env, asset_cfg):
  """L1 norm of joint torques."""
  return env.scene[asset_cfg.name].data.actuator_force.abs().sum(dim=-1)  # (num_envs,)


def action_rate_l2(env) -> torch.Tensor:
  """Penalize the rate of change of the actions using L2 squared kernel."""
  return torch.sum(
    torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1
  )


def action_acc_l2(env) -> torch.Tensor:
  """Penalize the acceleration of the actions using L2 squared kernel."""
  action_acc = (
    env.action_manager.action
    - 2 * env.action_manager.prev_action
    + env.action_manager.prev_prev_action
  )
  return torch.sum(torch.square(action_acc), dim=1)


class max_feet_delta_velocity_along_gravity:
  """Calculate the maximum change in velocity of the feet along the gravity direction."""

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    # self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.prev_site_velocities = torch.zeros(
      (env.num_envs, len(self.site_names), 3),
      device=env.device,
      dtype=torch.float32,
    )

  def __call__(self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    site_velocities = asset.data.site_lin_vel_w[:, asset_cfg.site_ids]
    change_in_site_velocities = site_velocities - self.prev_site_velocities
    self.prev_site_velocities = site_velocities.clone()

    term = change_in_site_velocities * asset.data.gravity_vec_w.unsqueeze(1)
    squared_term_along_gravity = torch.square(term)
    max_term = torch.max(squared_term_along_gravity, dim=1).values
    max_term_z = max_term[:, 2]  # (num_envs,)

    return max_term_z
  
