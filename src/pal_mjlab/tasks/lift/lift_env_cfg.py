"""Lift task configuration."""

import math
import mujoco

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
from pal_mjlab.tasks.lift import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
from mjlab.envs.mdp.terminations import nan_detection
from mjlab.managers.manager_term_config import TerminationTermCfg



def make_lift_env_cfg() -> ManagerBasedRlEnvCfg:
    """Create base lift task configuration."""

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
        "ee_to_cube": ObservationTermCfg(
            func=mdp.ee_to_object_distance,
            params={
                "object_name": "cube",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    site_names=(), 
                ),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "cube_to_goal": ObservationTermCfg(
            func=mdp.object_position_error,
            params={
                "object_name": "cube",
                "command_name": "lift_height",
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
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
            actuator_names=("arm_.*_joint","torso_lift_joint"), 
            scale=0.5,  # Override per-robot.
            use_default_offset=True,
        ),
        "gripper_pos": JointPositionActionCfg(
            asset_name="robot",
            actuator_names=("yam_.*_joint"), 
            scale=0.5,  # Override per-robot.
            use_default_offset=True,
        ),
        # "gripper_pos": mdp.MirroredJointPositionActionCfg(
        #     asset_name="robot",
        #     actuator_names=[
        #         "gripper_left_outer_finger_left_joint",
        #         "gripper_right_outer_finger_left_joint",
        #     ],
        #     mirror_actuator_names=[
        #         "gripper_left_outer_finger_right_joint",
        #         "gripper_right_outer_finger_right_joint",
        #     ],
        #     mirror_pairs={
        #         "gripper_left_outer_finger_right_joint": "gripper_left_outer_finger_left_joint",
        #         "gripper_right_outer_finger_right_joint": "gripper_right_outer_finger_left_joint",
        #     },
        #     # If one axis is flipped, set mirror_sign=[-1.0, 1.0] etc.
        #     mirror_sign=None,
        #     scale=3.0,
        #     use_default_offset=True,
        # ),
    }

    ## --------------------------------------------------------
    # Commands
    ## --------------------------------------------------------

    commands: dict[str, CommandTermCfg] = {
        "lift_height": mdp.LiftingCommandCfg(
            asset_name="cube",
            resampling_time_range=(8.0, 12.0),
            debug_vis=True,
            difficulty="dynamic",
            object_pose_range=mdp.LiftingCommandCfg.ObjectPoseRangeCfg(
                x = (0.65, 0.8),
                y = (-0.3, 0.2),
                z = (0.035, 0.035),
                yaw=(3.14, 3.14),
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
                    "x": (-0.0, 0.0),
                    "y": (-0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "velocity_range": {},
            },
        ),
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
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "field": "dof_frictionloss",
                "ranges": (0.5, 2.0),
                "operation": "abs",
            },
        )
    }

    ## --------------------------------------------------------
    # Rewards
    ## --------------------------------------------------------

    rewards = {
        # 1) Reach: EE–cube distance (returns distance in meters → use NEGATIVE weight)
        "ee_object_distance": RewardTermCfg(
            func=mdp.ee_object_distance,  
            weight=0.1,                           
            params={
                "std": 0.3,
                "object_name": "cube",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    site_names=(),                  #
                ),
            },
        ),

        # 2) Lift: binary bonus when cube is above minimal height
        "object_is_lifted": RewardTermCfg(
            func=mdp.object_is_lifted_binary,
            weight=10.0,                            
            params={
                "minimal_height": 0.08,
                "object_name": "cube",
            },
        ),

        # 3) Bring: object-to-goal Gaussian, only when lifted
        "object_goal_distance": RewardTermCfg(
            func=mdp.object_goal_gaussian_distance,
            weight=30.0,                           
            params={
                "std": 0.3,                         
                "minimal_height": 0.08,
                "command_name": "lift_height",      
                "object_name": "cube",
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    site_names=(),                
                ),
            },
        ),
        # "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, 
        #     weight=-1.0,
        #     params={
        #         "asset_cfg": SceneEntityCfg(
        #             "robot", 
        #             joint_names=("arm_.*_joint","torso_lift_joint",)),  # Set per-robot.
        #     },
        #     ),
        # "action_rate_l2": RewardTermCfg(
        #     func=mdp.action_rate_l2_louis,
        #     weight=-0.0001,
        #     params={
        #         "asset_cfg": SceneEntityCfg(
        #             "robot", joint_names=(".*",)),  # Set per-robot.
        #     },
        # ),
        # "joint_acc_l2": RewardTermCfg(
        # func=mdp.joint_acc_l2,
        # weight=-0.1e-4,
        # params={
        #     "asset_cfg": SceneEntityCfg(
        #         "robot",
        #         joint_names=(".*",),),
        #     },
        # ),
        "stand_still_joint_deviation_l1": RewardTermCfg(
            func=mdp.stand_still_joint_deviation_l1,
            weight=-5.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(".*",),),
            },
        ),
        # "joint_vel_hinge": RewardTermCfg(
        #     func=mdp.joint_velocity_hinge_penalty,
        #     weight=-0.0001,
        #     params={
        #         "max_vel": 0.5,
        #         "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        #     },
        # ),
    }

    ## --------------------------------------------------------
    # Terminations
    ## --------------------------------------------------------

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "ee_ground_collision": TerminationTermCfg(  # <<< NEW
            func=mdp.illegal_contact,
            params={"sensor_name": "ee_ground_collision"},
        ),
        "nan_term": TerminationTermCfg(
            func=nan_detection,
            time_out=False,
        ),
        "cube_out_of_bounds": TerminationTermCfg(
            func=mdp.object_out_of_bounds_box_local,
            params={
                "x_min": 0.4,
                "x_max": 0.8,
                "y_min": -0.3,
                "y_max": 0.3,
                "object_name": "cube",
                "robot_name": "robot",
            },
        ),

    }


    ## --------------------------------------------------------
    # Curriculum
    ## --------------------------------------------------------
    curriculum = {
        # "action_rate_curr": CurriculumTermCfg(
        #     func=mdp.reward_weight,
        #     params={
        #         "reward_name": "action_rate_l2",
        #         "weight_stages": [
        #             {"step": 0, "weight": -0.0001},
        #             {"step": 5_000 * 24, "weight": -0.005},
        #             {"step": 15_000 * 24, "weight": -0.01},
        #         ],
        #     },
        # ),
        # "dof_pos_limits_curr": CurriculumTermCfg(
        #     func=mdp.reward_weight,
        #     params={
        #         "reward_name": "dof_pos_limits",
        #         "weight_stages": [
        #             {"step": 0, "weight": -0.0},
        #             {"step": 1_000 * 24, "weight": -1.0},
        #             ],
        #             },
        #             ),
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
            body_name="base_footprint",  # Set per-robot.
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

