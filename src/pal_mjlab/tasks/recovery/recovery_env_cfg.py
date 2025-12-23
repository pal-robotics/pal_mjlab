"""Velocity task configuration.

This module provides a factory function to create a base velocity task config.
Robot-specific configurations call the factory and customize as needed.
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
  ActionTermCfg,
  CommandTermCfg,
  CurriculumTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from pal_mjlab.tasks.recovery import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_recovery_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base fall_recovery tracking task configuration."""

  ##
  # Observations
  ##

  policy_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    # imu proj grav instead
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    **policy_terms,
    "base_lin_acc": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_acc"},
    ),
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "torso_height": ObservationTermCfg(
      func=mdp.torso_height_obs,
    ),
  }

  observations = {
    "policy": ObservationGroupCfg(
      terms=policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  ##
  # Actions
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": mdp.JointPositionActionCfg(
      asset_name="robot",
      actuator_names=(".*",),
      scale=0.5,  # Override per-robot.
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {}

  ##
  # Events
  ##

  events = {
    # "reset_base": EventTermCfg(
    #   func=mdp.reset_root_state_uniform,
    #   mode="reset",
    #   params={
    #     "pose_range": {
    #       "x": (-0.5, 0.5), 
    #       "y": (-0.5, 0.5), 
    #       # Dropping it from high.
    #       "z": (0.5, 0.5),
    #       # Randomizing its orientation.
    #       "roll": (-math.pi, math.pi),
    #       "pitch": (-math.pi, math.pi),
    #       "yaw": (-math.pi, math.pi),
    #     },
    #     "velocity_range": {
    #       "lin_vel_x": (-0.0, 0.0),
    #       "lin_vel_y": (-0.0, 0.0),
    #       "lin_vel_z": (-0.0, 0.0),
    #       "ang_vel_x": (-0.0, 0.0),
    #       "ang_vel_y": (-0.0, 0.0),
    #       "ang_vel_z": (-0.0, 0.0),
    #     },
    #   },
    # ),
    # "reset_robot_joints": EventTermCfg(
    #   func=mdp.reset_joints_by_offset,
    #   mode="reset",
    #   params={
    #     "position_range": (-math.pi / 2, math.pi / 2),
    #     "velocity_range": (0.0, 0.0),
    #     "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    #   },
    # ),
    "reset_robot_qpos": EventTermCfg(
      func=mdp.reset_joints_random_or_default,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "field": "geom_friction",
        "ranges": (0.3, 1.2),
      },
    ),
  }

  ##
  # Rewards
  ##

  rewards = {
    # -- getup rewards --
    "orientation": RewardTermCfg(
      func=mdp.orientation,
      weight=1.0,
      params={"std": math.sqrt(0.5)},
    ),
    "torso_height": RewardTermCfg(
      func=mdp.torso_height,
      weight=4.0,
      params={
        "std": math.sqrt(2 / 3),
        "z_des": 1.0,
      },
    ),
    "posture": RewardTermCfg(
      func=mdp.getup_posture,
      weight=0.5,
    ),
    # -- regularization --
    "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-0.1),
    # "dof_vel_limits": RewardTermCfg(
    #   func=mdp.joint_vel_limits, 
    #   weight=-0.01,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    #     "velocity_limits": {},
    #     "soft_ratio": 0.9,
    #   },
    # ),
    "action_rate_l2": RewardTermCfg(
        func=mdp.action_rate_l2, weight=-0.01,
    ),
    "power": RewardTermCfg(
        func=mdp.power_limit, weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        },
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-1.0,
      params={"sensor_name": ""}, # Set per-robot.
    )
  }

  ##
  # Terminations
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    # TODO Louis: Could it be good to add it as a termination? In the
    # sense that when I tested it, the torque limitation issue was still
    # present, so I suppose it had an impact on this kind of task.
  }

  ##
  # Curriculum
  ##

  curriculum = {
    "action_rate": CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "action_rate_l2",
        "weight_stages": [
            {"step": 0, "weight": -0.01},
            {"step": 2500 * 24, "weight": -0.1},
        ],
      },
    ),
    "power": CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
          "reward_name": "power",
          "weight_stages": [
              {"step": 0, "weight": 0.0},
              {"step": 5000 * 24, "weight": -0.01},
              {"step": 7500 * 24, "weight": -0.1},
          ],
      },
    ),
  }

  ##
  # Assemble and return
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=5,
      ),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      asset_name="robot",
      body_name="",  # Set per-robot.
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
  )