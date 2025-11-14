"""Reaching task configuration."""

import math
from copy import deepcopy

from mjlab.entity.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
    ActionTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    CommandTermCfg,
    CurriculumTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from pal_mjlab.tasks.reaching import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

SCENE_CFG = SceneCfg(
    terrain=TerrainImporterCfg(
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=1,
    ),
    num_envs=1,
    extent=2.0,
)

VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    asset_name="robot",
    body_name="",  # Override in robot cfg.
    distance=3.0,
    elevation=-5.0,
    azimuth=90.0,
)

SIM_CFG = SimulationCfg(
    nconmax=35,
    njmax=400,
    mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
    ),
)


def create_reaching_env_cfg(
    robot_cfg: EntityCfg,
    action_scale: float | dict[str, float],
    viewer_body_name: str,
    posture_jn: tuple[str, ...],
    action_rate_body_jn: tuple[str, ...],
    action_rate_leftarm_jn: tuple[str, ...],
    posture_std: dict[str, float],
    foot_friction_geom_names: tuple[str, ...] | str,
    pos_x: tuple[float, float] = (-0.5, 0.5),
    pos_y: tuple[float, float] = (0.1, 0.5),
    pos_z: tuple[float, float] = (0.0, 1.0),
) -> ManagerBasedRlEnvCfg:
    """Create a basic balancing task configuration.

    Args:
      robot_cfg: Robot configuration (with sensors).
      action_scale: Action scaling factor(s).
      viewer_body_name: Body for camera tracking.
      foot_friction_geom_names: Geometry names for friction randomization.

    Returns:
      Complete ManagerBasedRlEnvCfg for velocity task.
    """
    scene = deepcopy(SCENE_CFG)
    scene.entities = {"robot": robot_cfg}

    viewer = deepcopy(VIEWER_CONFIG)
    viewer.body_name = viewer_body_name

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            asset_name="robot",
            actuator_names=(".*",),
            scale=action_scale,
            use_default_offset=True,
        )
    }
    commands: dict[str, CommandTermCfg] = {
        "pose_command_left": mdp.UniformPoseCommandCfg(
            asset_name="robot",
            debug_vis=True,
            resampling_time_range=(3.0, 8.0),
            site_name="ee_left",
            ranges=mdp.PoseRanges(
                pos_x=pos_x,
                pos_y=pos_y,
                pos_z=pos_z,
            ),
        )
    }

    policy_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            noise=Unoise(n_min=-0.5, n_max=0.5),
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        ),
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
        "commands": ObservationTermCfg(
            func=mdp.commands_gen,
            params={"command_name": "pose_command_left"},
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
        # "push_robot": EventTermCfg(
        #  func=mdp.push_by_setting_velocity,
        #  mode="interval",
        #  interval_range_s=(1.0, 3.0),
        #  params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        # ),
        # "foot_friction": EventTermCfg(
        #  mode="startup",
        #  func=mdp.randomize_field,
        #  domain_randomization=True,
        #  params={
        #    "asset_cfg": SceneEntityCfg("robot", geom_names=foot_friction_geom_names),
        #    "operation": "abs",
        #    "field": "geom_friction",
        #    "ranges": (0.3, 1.2),
        #  },
        # ),
    }

    rewards = {
        "upright": RewardTermCfg(
            func=mdp.flat_orientation,
            weight=1.0,
            params={
                "std": math.sqrt(0.2),
                "asset_cfg": SceneEntityCfg("robot", body_names=(viewer_body_name,)),
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
        # TODO: make it not kang specific
        "pose": RewardTermCfg(
            func=mdp.posture,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=posture_jn),
                "std": posture_std,
            },
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-1.0),
        "action_rate_body_l2": RewardTermCfg(
            func=mdp.action_rate_l2_louis,
            weight=-0.01,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=action_rate_body_jn),
            },
        ),
        "action_rate_left_arm_l2": RewardTermCfg(
            func=mdp.action_rate_l2_louis,
            weight=-0.0001,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=action_rate_leftarm_jn
                ),
            },
        ),
    }

    curriculum = {
        # "pos_left_curr": CurriculumTermCfg(
        #    func=mdp.reward_weight,
        #    params={
        #        "reward_name": "pos_left",
        #        "weight_stages": [
        #            {"step": 0, "weight": -0.5},
        #            {"step": 2500 * 24, "weight": -2.0},
        #        ],
        #    },
        # ),
        # "pos_left_fine_grained_curr": CurriculumTermCfg(
        #    func=mdp.reward_weight,
        #    params={
        #        "reward_name": "pos_left_fine_grained",
        #        "weight_stages": [
        #            {"step": 0, "weight": 0.5},
        #            {"step": 2500 * 24, "weight": 2.0},
        #        ],
        #    },
        # ),
        "action_rate_left_arm_l2_curr": CurriculumTermCfg(
            func=mdp.reward_weight,
            params={
                "reward_name": "action_rate_left_arm_l2",
                "weight_stages": [
                    {"step": 0, "weight": -0.0001},
                    {"step": 5_000 * 24, "weight": -0.005},
                ],
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
    }

    return ManagerBasedRlEnvCfg(
        scene=scene,
        observations=observations,
        commands=commands,
        actions=actions,
        rewards=rewards,
        curriculum=curriculum,
        terminations=terminations,
        events=events,
        sim=SIM_CFG,
        viewer=viewer,
        decimation=4,
        episode_length_s=20.0,
    )
