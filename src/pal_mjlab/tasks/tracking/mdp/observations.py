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

  Automatically detects active 'startup' randomization events and collects
  the corresponding parameters from the simulation.
  """
  robot = env.scene["robot"]
  sim = env.sim
  obs = []

  # Iterate through sorted events to ensure deterministic observation order
  for event_name in sorted(env.cfg.events.keys()):
    cfg = env.cfg.events[event_name]
    if cfg.mode != "startup":
      continue

    func_name = cfg.func.__name__
    params = cfg.params
    asset_cfg = params.get("asset_cfg", None)

    if func_name == "geom_friction":
      geom_ids = robot.find_geoms(asset_cfg.geom_names)[0]
      if params.get("shared_random", False):
        # If shared, all geoms have the same friction; take only the first
        friction = sim.model.geom_friction[:, geom_ids[0:1], 0:1]
      else:
        friction = sim.model.geom_friction[:, geom_ids, 0:1]
      obs.append(friction.reshape(env.num_envs, -1))

    elif func_name in ["body_mass", "link_mass"]:
      body_ids = robot.find_bodies(asset_cfg.body_names)[0]
      curr_mass = sim.model.body_mass[:, body_ids]
      def_mass = sim.get_default_field("body_mass")[body_ids]
      link_mass_scale = (curr_mass / (def_mass + 1e-6)).reshape(env.num_envs, -1)
      obs.append(link_mass_scale)

    elif func_name in ["body_ipos", "body_com_offset"]:
      body_ids = robot.find_bodies(asset_cfg.body_names)[0]
      curr_ipos = sim.model.body_ipos[:, body_ids]
      def_ipos = sim.get_default_field("body_ipos")[body_ids]
      link_com_offset = (curr_ipos - def_ipos).reshape(env.num_envs, -1)
      obs.append(link_com_offset)

    elif func_name in ["joint_damping", "dof_damping"]:
      # damping is NV (num_dofs)
      curr_damp = sim.model.dof_damping
      def_damp = sim.get_default_field("dof_damping")
      joint_damping_scale = (curr_damp / (def_damp + 1e-6)).reshape(env.num_envs, -1)
      obs.append(joint_damping_scale)

    elif func_name in ["joint_friction", "dof_frictionloss"]:
      # frictionloss is NV (num_dofs)
      curr_fric = sim.model.dof_frictionloss
      obs.append(curr_fric.reshape(env.num_envs, -1))


    elif func_name == "encoder_bias":
      encoder_bias = robot.data.encoder_bias.reshape(env.num_envs, -1)
      obs.append(encoder_bias)

    elif func_name == "control_delay":
      # actuator_dynprm[..., 0] is the filter time constant used for delay
      control_delay = sim.model.actuator_dynprm[:, :, 0].reshape(env.num_envs, -1)
      obs.append(control_delay)

    elif func_name == "p_gain":
      curr_gain = sim.model.actuator_gainprm[:, :, 0]
      def_gain = sim.get_default_field("actuator_gainprm")[:, 0]
      p_gain_scale = (curr_gain / (def_gain + 1e-6)).reshape(env.num_envs, -1)
      obs.append(p_gain_scale)

  if not obs:
    return torch.zeros(env.num_envs, 0, device=env.device)

  return torch.cat(obs, dim=-1)

