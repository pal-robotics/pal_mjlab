"""Reaching task configuration."""

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
            noise=Unoise(n_min=-0.07, n_max=0.07),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "pose_command_left": ObservationTermCfg(
            func=mdp.commands_gen,
            params={"command_name": "pose_command_left"},
        ),
        "pose_command_right": ObservationTermCfg(
            func=mdp.commands_gen,
            params={"command_name": "pose_command_right"},
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
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,  # Override per-robot.
            use_default_offset=True,
        )
    }

    ## --------------------------------------------------------
    # Commands
    ## --------------------------------------------------------

    commands: dict[str, CommandTermCfg] = {
        "pose_command_left": mdp.UniformPoseCommandCfg(
            entity_name="robot",
            debug_vis=True,
            resampling_time_range=(5.0, 10.0),
            site_name="ee_left",
            ranges=mdp.PoseRanges(
                pos_x=(0.0, 0.0),  # Set per-robot.
                pos_y=(0.0, 0.0),  # Set per-robot.
                pos_z=(0.0, 0.0),  # Set per-robot.
            ),
        ),
        "pose_command_right": mdp.UniformPoseCommandCfg(
            entity_name="robot",
            debug_vis=True,
            resampling_time_range=(5.0, 10.0),
            site_name="ee_right",
            ranges=mdp.PoseRanges(
                pos_x=(0.0, 0.0),  # Set per-robot.
                pos_y=(0.0, 0.0),  # Set per-robot.
                pos_z=(0.0, 0.0),  # Set per-robot.
            ),
        ),
    }

    ## --------------------------------------------------------
    # Events
    ## --------------------------------------------------------

    events = {
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.1, 0.1),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "reset_frictionloss": EventTermCfg(
            mode="reset",
            func=mdp.randomize_field,
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "field": "dof_frictionloss",
                "ranges": (0.5, 2.0),
                "operation": "abs",
            },
        ),
        "reset_ineratia": EventTermCfg(
            mode="reset",
            func=mdp.randomize_field,
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "field": "body_inertia",
                "ranges": (0.95, 1.05),
                "operation": "scale",
            },
        ),
        "reset_masses": EventTermCfg(
            mode="reset",
            func=mdp.randomize_field,
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(".*",)),
                "field": "body_mass",
                "ranges": (0.95, 1.05),
                "operation": "scale",
            },
        ),
    }

    ## --------------------------------------------------------
    # Rewards
    ## --------------------------------------------------------

    rewards = {
        "pos_left": RewardTermCfg(
            func=mdp.position_command_error,
            weight=-2.0,
            params={
                "site_name": "ee_left",
                "command_name": "pose_command_left",
            },
        ),
        "pos_left_fine_grained": RewardTermCfg(
            func=mdp.position_command_error_tanh,
            weight=2.0,
            params={
                "site_name": "ee_left",
                "command_name": "pose_command_left",
                "std": 0.05,
            },
        ),
        "ee_left_orientation": RewardTermCfg(
            func=mdp.orientation_command_error,
            weight=-0.2,
            params={
                "site_name": "ee_left",
                "command_name": "pose_command_left",
            },
        ),
        "pos_right": RewardTermCfg(
            func=mdp.position_command_error,
            weight=-2.0,
            params={
                "site_name": "ee_right",
                "command_name": "pose_command_right",
            },
        ),
        "pos_right_fine_grained": RewardTermCfg(
            func=mdp.position_command_error_tanh,
            weight=3.0,
            params={
                "site_name": "ee_right",
                "command_name": "pose_command_right",
                "std": 0.05,
            },
        ),
        "ee_right_orientation": RewardTermCfg(
            func=mdp.orientation_command_error,
            weight=-0.2,
            params={
                "site_name": "ee_right",
                "command_name": "pose_command_right",
            },
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "action_rate_l2": RewardTermCfg(
            func=mdp.action_rate_l2_louis,
            weight=-0.003,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=(".*",)
                ),  # Set per-robot.
            },
        ),
        "joint_vel_hinge": RewardTermCfg(
            func=mdp.joint_velocity_hinge_penalty,
            weight=-0.05,
            params={
                "max_vel": 0.5,
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
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
        "action_rate_curr": CurriculumTermCfg(
            func=mdp.reward_weight,
            params={
                "reward_name": "action_rate_l2",
                "weight_stages": [
                    {"step": 0, "weight": -0.003},
                    {"step": 5_000 * 24, "weight": -0.01},
                ],
            },
        ),
        "orientation_curr_right": CurriculumTermCfg(
            func=mdp.reward_weight,
            params={
                "reward_name": "ee_right_orientation",
                "weight_stages": [
                    {"step": 0, "weight": -0.3},
                    {"step": 7_500 * 24, "weight": -0.6},
                ],
            },
        ),
        "orientation_curr_left": CurriculumTermCfg(
            func=mdp.reward_weight,
            params={
                "reward_name": "ee_left_orientation",
                "weight_stages": [
                    {"step": 0, "weight": -0.3},
                    {"step": 7_500 * 24, "weight": -0.6},
                ],
            },
        ),
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
            entity_name="robot",
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