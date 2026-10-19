"""Useful methods for MDP Metrics."""

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


class joint_torques_magnitude:
  """Abs torque of the actuator driving the selected joint(s). Metric term."""

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
    # Actuators are named after their joints in your constants file.
    ids, names = asset.find_actuators(joint_names)
    assert len(ids) == len(joint_names), (
      f"Joints {joint_names} did not all map to actuators (got {names})"
    )
    self._actuator_ids = ids

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.actuator_force[:, self._actuator_ids].abs().sum(dim=-1)  # (num_envs,)


class joint_raw_action:
  """Abs raw action of ONE joint, per env. Metric term.

  The action index is looked up from the joint name, so it follows whatever
  order the action term uses.
  """

  def __init__(self, cfg: MetricsTermCfg, env: ManagerBasedRlEnv):
    term = env.action_manager.get_term("joint_pos")  # key used in cfg.actions
    names = list(getattr(term, "target_names", None) or getattr(term, "_target_names"))
    self._idx = names.index(cfg.params["joint_name"])

  def __call__(self, env: ManagerBasedRlEnv, joint_name: str) -> torch.Tensor:
    return env.action_manager.action[:, self._idx].abs()  # (num_envs,)


def raw_action_metrics(joint_names) -> dict:
  """One raw-action metric per joint, keyed raw_action/<joint_name>."""
  return {
    f"raw_action/{n}": MetricsTermCfg(func=joint_raw_action, params={"joint_name": n})
    for n in joint_names
  }


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