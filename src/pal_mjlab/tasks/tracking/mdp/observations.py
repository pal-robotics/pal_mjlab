from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_phase(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Return the normalized phase of the motion clip as [sin(2πφ), cos(2πφ)]."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  # Normalized phase φ in [0, 1): current_step / total_steps
  # command.time_steps has shape (num_envs,)
  phase = command.time_steps.float() / command.motion.time_step_total
  phase_rad = 2.0 * math.pi * phase

  # Return encoded periodic phase as (num_envs, 2)
  return torch.stack([torch.sin(phase_rad), torch.cos(phase_rad)], dim=-1)


def external_parameters(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Collect privileged environment parameters (e_t) for A-RMA.

  Total dimension: 167.

  Breakdown:
  - Base COM Offset (3D): 3
  - Foot Friction (Left, Right): 2
  - Control Delay (22 actuators): 22
  - P Gain Scale (22 actuators): 22
  - Encoder Bias (22 actuators): 22
  - Joint Friction Offset (26 DOFs): 26
  - Link Mass Scale (11 Bodies): 11
  - Link COM Offset (11 Bodies * 3D): 33
  - Joint Damping Scale (26 DOFs): 26
  """
  robot = env.scene["robot"]
  sim = env.sim

  # 1. Base COM Offset (3D) - Using pelvis_2_link
  pelvis_id = robot.find_bodies("pelvis_2_link")[0]
  default_ipos = sim.get_default_field("body_ipos")[pelvis_id]
  current_ipos = sim.model.body_ipos[:, pelvis_id]
  base_com = (current_ipos - default_ipos).reshape(env.num_envs, -1) # (N, 3)

  # 2. Foot Friction (2D) - Left and Right
  left_foot_id = robot.find_geoms("left_foot0_collision")[0]
  right_foot_id = robot.find_geoms("right_foot0_collision")[0]
  foot_friction = torch.cat([
      sim.model.geom_friction[:, left_foot_id, 0:1],
      sim.model.geom_friction[:, right_foot_id, 0:1]
  ], dim=-1).reshape(env.num_envs, -1) # (N, 2)

  # 3. Control Delay (22D)
  control_delay = sim.model.actuator_dynprm[:, :, 0].reshape(env.num_envs, -1) # (N, 22)

  # 4. P Gain Scale (22D)
  curr_gain = sim.model.actuator_gainprm[:, :, 0]
  def_gain = sim.get_default_field("actuator_gainprm")[:, 0]
  p_gain_scale = (curr_gain / (def_gain + 1e-6)).reshape(env.num_envs, -1) # (N, 22)

  # 5. Encoder Bias (22D) - Placeholder (zeros for now)
  encoder_bias = torch.zeros((env.num_envs, 22), device=env.device)

  # 6. Joint Friction Offset (26D)
  curr_f_loss = sim.model.dof_frictionloss
  def_f_loss = sim.get_default_field("dof_frictionloss")
  joint_friction_offset = (curr_f_loss - def_f_loss).reshape(env.num_envs, -1) # (N, 26)

  # 7. Link Mass Scale (11 Targeted Bodies)
  arma_bodies = (
      "base_link", "pelvis_1_link", "pelvis_2_link",
      "leg_left_1_link", "leg_right_1_link",
      "leg_left_3_link", "leg_right_3_link",
      "leg_left_femur_link", "leg_right_femur_link",
      "leg_left_knee_link", "leg_right_knee_link"
  )
  arma_body_ids = robot.find_bodies(arma_bodies)[0]
  curr_mass = sim.model.body_mass[:, arma_body_ids]
  def_mass = sim.get_default_field("body_mass")[arma_body_ids]
  link_mass_scale = (curr_mass / (def_mass + 1e-6)).reshape(env.num_envs, -1) # (N, 11)

  # 8. Link COM Offset (33D)
  curr_ipos_all = sim.model.body_ipos[:, arma_body_ids]
  def_ipos_all = sim.get_default_field("body_ipos")[arma_body_ids]
  link_com_offset = (curr_ipos_all - def_ipos_all).reshape(env.num_envs, -1) # (N, 33)

  # 9. Joint Damping Scale (26D)
  curr_damp = sim.model.dof_damping
  def_damp = sim.get_default_field("dof_damping")
  joint_damping_scale = (curr_damp / (def_damp + 1e-6)).reshape(env.num_envs, -1) # (N, 26)

  return torch.cat([
      base_com,                # 3
      foot_friction,           # 2
      control_delay,           # 22
      p_gain_scale,            # 22
      encoder_bias,            # 22
      joint_friction_offset,   # 26
      link_mass_scale,         # 11
      link_com_offset,         # 33
      joint_damping_scale      # 26
  ], dim=-1)                   # Total: 167







