"""Useful methods for MDP Metrics."""

import torch


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
