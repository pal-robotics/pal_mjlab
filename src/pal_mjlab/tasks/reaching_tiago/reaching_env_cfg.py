"""Reaching task configuration."""

import math

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
from pal_mjlab.tasks.reaching import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

def make_reaching_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base reaching task configuration."""

    ## --------------------------------------------------------
    # Observations
    ## --------------------------------------------------------

    policy_terms = {
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "commands": ObservationTermCfg(
            func=mdp.commands_gen,
            params={"command_name": "pose_command_left"},
        ),
    }

    critic_terms = {
        **policy_terms,
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

    ## --------------------------------------------------------
    # Actions
    ## --------------------------------------------------------

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            asset_name="robot",
            actuator_names=(".*",),
            scale=0.5,  # Override per-robot.
            use_default_offset=True,
        )
    }

    ## --------------------------------------------------------
    # Commands
    ## --------------------------------------------------------

    commands: dict[str, CommandTermCfg] = {
        "pose_command_right": mdp.UniformPoseCommandCfg(
            asset_name="robot",
            debug_vis=True,
            resampling_time_range=(3.0, 8.0),
            site_name="ee_left",
            ranges=mdp.PoseRanges(
                pos_x=(0.0, 0.0),  # Set per-robot.
                pos_y=(0.0, 0.0),  # Set per-robot.
                pos_z=(0.0, 0.0),  # Set per-robot.
            ),
        )
    }

    ## --------------------------------------------------------
    # Events
    ## --------------------------------------------------------

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
    }

    ## --------------------------------------------------------
    # Rewards
    ## --------------------------------------------------------

    rewards = {
        "upright": RewardTermCfg(
            func=mdp.flat_orientation,
            weight=1.0,
            params={
                "std": math.sqrt(0.2),
                "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per-robot.
            },
        ),
        "pos_left": RewardTermCfg(
            func=mdp.position_command_error,
            weight=-0.5,
            params={
                "site_name": "ee_left",
                "command_name": "pose_command_left",
            },
        ),
        "pos_left_fine_grained": RewardTermCfg(
            func=mdp.position_command_error_tanh,
            weight=0.5,
            params={
                "site_name": "ee_left",
                "command_name": "pose_command_left",
                "std": 0.1,
            },
        ),
        "pose": RewardTermCfg(
            func=mdp.posture,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "std": {},  # Set per-robot.
            },
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "action_rate_body_l2": RewardTermCfg(
            func=mdp.action_rate_l2_louis,
            weight=-0.01,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=(".*",)
                ),  # Set per-robot.
            },
        ),
        "action_rate_left_arm_l2": RewardTermCfg(
            func=mdp.action_rate_l2_louis,
            weight=-0.0001,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=(".*",)
                ),  # Set per-robot.
            },
        ),
    }

    ## --------------------------------------------------------
    # Terminations
    ## --------------------------------------------------------

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    }

    ## --------------------------------------------------------
    # Curriculum
    ## --------------------------------------------------------
    curriculum = {
    }

    ## --------------------------------------------------------
    # Assemble final configuration
    ## --------------------------------------------------------

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

