"""PAL Robotics Kangaroo Flat terrain tracking configuration."""

import copy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pal_mjlab.tasks.tracking.mdp.commands import PalMotionCommandCfg

from pal_mjlab.robots import (
    ANKLE_XY_CONVEX_HULL_POINTS,
    HIP_XY_CONVEX_HULL_POINTS,
    KANGAROO_ACTION_SCALE,
    KANGAROO_ACTUATOR_NAMES,
    get_kangaroo_robot_cfg,
)
from pal_mjlab.tasks.tracking import mdp as tracking_mdp
from pal_mjlab.tasks.velocity import mdp


def pal_kangaroo_flat_tracking_env_cfg(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics Kangaroo flat terrain tracking configuration."""
    cfg = make_tracking_env_cfg()

    # =========================================================================
    # 0. DATA DEFINITIONS
    # =========================================================================
    body_names = (
        "base_link",
        "pelvis_2_link",
        "leg_left_3_link",
        "leg_left_4_link",
        "leg_left_5_link",
        "leg_right_3_link",
        "leg_right_4_link",
        "leg_right_5_link",
        "arm_left_2_link",
        "arm_left_3_link",
        "arm_left_tip_link",
        "arm_right_2_link",
        "arm_right_3_link",
        "arm_right_tip_link",
    )

    geom_names = tuple(
        f"{side}_foot{i}_collision"
        for side in ("left", "right")
        for i in [0, 2, 4, 6, 8, 10]
    )

    body_geoms = (
        "leg_left_femur_collision",
        "leg_right_femur_collision",
        "leg_left_knee_collision",
        "leg_left_knee_bar_collision",
        "leg_right_knee_collision",
        "leg_right_knee_bar_collision",
        "arm_left_4_collision",
        "arm_right_4_collision",
        "pelvis_2_collision",
    )

    # =========================================================================
    # 1. SCENE & SIMULATION SETUP
    # =========================================================================
    cfg.scene.entities = {"robot": get_kangaroo_robot_cfg()}
    cfg.sim.mujoco.timestep = 0.002
    cfg.decimation = 10

    # Contact sensors
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            # Link pattern matches leg_left_5_link and leg_right_5_link
            mode="subtree",
            pattern=r"^(leg_left_5_link|leg_right_5_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    cfg.scene.sensors = (self_collision_cfg, feet_ground_cfg)

    # =========================================================================
    # 2. ACTIONS
    # =========================================================================
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = KANGAROO_ACTION_SCALE
    joint_pos_action.actuator_names = KANGAROO_ACTUATOR_NAMES

    # = =========================================================================
    # 3. COMMANDS
    # =========================================================================
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = "base_link"
    motion_cmd.body_names = body_names

    # =========================================================================
    # 4. REWARDS
    # =========================================================================
    # Split tracking rewards into Legs and Upper Body/Arms
    leg_bodies = (
        "leg_left_3_link", "leg_left_4_link", "leg_left_5_link",
        "leg_right_3_link", "leg_right_4_link", "leg_right_5_link"
    )
    other_bodies = (
        "base_link", "pelvis_2_link", 
        "arm_left_2_link", "arm_left_3_link", "arm_left_tip_link",
        "arm_right_2_link", "arm_right_3_link", "arm_right_tip_link"
    )

    # 1. Position tracking (High precision for legs, more slack for arms)
    cfg.rewards["motion_body_pos"].params["std"] = 0.4 
    cfg.rewards["motion_body_pos"].params["body_names"] = leg_bodies
    
    cfg.rewards["motion_body_pos_other"] = RewardTermCfg(
        func=tracking_mdp.motion_relative_body_position_error_exp,
        weight=0.5, # Lower weight for arms
        params={"command_name": "motion", "std": 0.6, "body_names": other_bodies},
    )

    # 2. Orientation tracking
    cfg.rewards["motion_body_ori"].params["std"] = 0.4
    cfg.rewards["motion_body_ori"].params["body_names"] = leg_bodies
    
    cfg.rewards["motion_body_ori_other"] = RewardTermCfg(
        func=tracking_mdp.motion_relative_body_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.6, "body_names": other_bodies},
    )

    # 3. Soft Landing (Penalize high-impact forces in the feet)
    cfg.rewards["soft_landing"] = RewardTermCfg(
        func=mdp.soft_landing,
        weight=-5e-5, 
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "motion",
            "command_threshold": 0.05,
        },
    )

    # 4. Foot Slip (Penalize foot sliding while in contact)
    # cfg.rewards["foot_slip"] = RewardTermCfg(
    #     func=tracking_mdp.feet_slip,
    #     weight=-0.1,
    #     params={
    #         "sensor_name": "feet_ground_contact",
    #         "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    #     },
    # )

    # Tighten tracking precision for velocities
    cfg.rewards["motion_body_lin_vel"].params["std"] = 0.5
    cfg.rewards["motion_body_ang_vel"].params["std"] = 0.5

    # Convex Hull limits for Hip
    cfg.rewards["convex_hull_joint_limits_hip"] = RewardTermCfg(
        func=mdp.joint_limits_convex_hull,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
            "metrics_suffix": "hipXY",
            "joint_names_group": [
                [r"leg_left_2_joint", r"leg_left_3_joint"],
                [r"leg_right_2_joint", r"leg_right_3_joint"],
            ],
            "margin": 0.02,
            "hull_points": HIP_XY_CONVEX_HULL_POINTS,
        },
    )

    # Convex Hull limits for Ankle
    cfg.rewards["convex_hull_joint_limits_ankle"] = RewardTermCfg(
        func=mdp.joint_limits_convex_hull,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
            "margin": 0.02,
            "metrics_suffix": "ankleXY",
            "joint_names_group": [
                [r"leg_left_4_joint", r"leg_left_5_joint"],
                [r"leg_right_4_joint", r"leg_right_5_joint"],
            ],
            "hull_points": ANKLE_XY_CONVEX_HULL_POINTS,
        },
    )

    # =========================================================================
    # 5. EVENTS (Domain Randomization)
    # =========================================================================
    cfg.events["left_foot_friction"] = EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=tuple(f"left_foot{i}_collision" for i in [0, 2, 4, 6, 8, 10])),
            "operation": "abs",
            "ranges": (0.3, 1.8),
            "shared_random": True,
        },
    )
    cfg.events["right_foot_friction"] = EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=tuple(f"right_foot{i}_collision" for i in [0, 2, 4, 6, 8, 10])),
            "operation": "abs",
            "ranges": (0.3, 1.8),
            "shared_random": True,
        },
    )
    cfg.events["encoder_bias"] = EventTermCfg(
        mode="startup",
        func=dr.encoder_bias,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "bias_range": (-0.015, 0.015),
        },
    )
    cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis_2_link",)


    # cfg.events["control_delay"] = EventTermCfg(
    #     mode="startup",
    #     func=tracking_mdp.control_delay,
    #     params={
    #         "delay_range": (0.0, 0.04),  # 0–40 ms
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    # cfg.events["p_gain"] = EventTermCfg(
    #     mode="startup",
    #     func=tracking_mdp.p_gain,
    #     params={
    #         "kp_range": (0.925, 1.05),
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )

    # cfg.events["joint_friction"] = EventTermCfg(
    #     mode="startup",
    #     func=dr.dof_frictionloss,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    #         "operation": "add",
    #         "ranges": (-0.008, 0.008),
    #         "shared_random": False,
    #     },
    # )

    # Added randomizations for A-RMA
    arma_bodies = (
        "base_link", "pelvis_1_link", "pelvis_2_link",
        "leg_left_1_link", "leg_right_1_link",
        "leg_left_3_link", "leg_right_3_link",
        "leg_left_femur_link", "leg_right_femur_link",
        "leg_left_knee_link", "leg_right_knee_link"
    )
    cfg.events["link_mass"] = EventTermCfg(
        mode="startup",
        func=dr.body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=arma_bodies),
            "operation": "scale",
            "ranges": (0.8, 1.2),
            "shared_random": False,
        },
    )
    cfg.events["link_com"] = EventTermCfg(
        mode="startup",
        func=dr.body_ipos,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=arma_bodies),
            "operation": "add",
            "ranges": {i: (-0.03, 0.03) for i in range(3)},
            "shared_random": False,
        },
    )

    cfg.events["joint_damping"] = EventTermCfg(

        mode="startup",
        func=dr.joint_damping,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "scale",
            "ranges": (0.3, 3.0),
            "shared_random": False,
        },
    )



    # =========================================================================
    # 6. TERMINATIONS
    # =========================================================================
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "leg_left_5_link",
        "leg_right_5_link",
        "arm_left_5_link",
        "arm_right_5_link",
    )

    # =========================================================================
    # 7. VIEWER
    # =========================================================================
    cfg.viewer.body_name = "base_link"

    # =========================================================================
    # 8. OBSERVATIONS
    # =========================================================================
    # A-RMA Phase 0: History Buffer Enabled for Actor (75 frames = 1.5s context)
    cfg.observations["actor"].history_length = 75
    cfg.observations["actor"].flatten_history_dim = False

    # Add IMU observations and motion phase to both Actor and Critic
    for group_name in ["actor", "critic"]:
        cfg.observations[group_name].nan_policy = "sanitize"
        cfg.observations[group_name].terms["motion_phase"] = ObservationTermCfg(
            func=tracking_mdp.motion_phase,
            params={"command_name": "motion"},
        )
        # Add IMU observations (Noiseless for Critic)
        cfg.observations[group_name].terms["base_lin_acc"] = ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_acc"},
            noise=Unoise(n_min=-0.05, n_max=0.05) if group_name == "actor" else None,
        )
        cfg.observations[group_name].terms["imu_projected_gravity"] = ObservationTermCfg(
            func=mdp.imu_projected_gravity,
            params={"sensor_name": "robot/imu_quat"},
            noise=Unoise(n_min=-0.05, n_max=0.05) if group_name == "actor" else None,
        )
        if group_name == "critic":
            cfg.observations["critic"].terms["foot_contact_forces"] = ObservationTermCfg(
                func=mdp.foot_contact_forces,
                params={"sensor_name": "feet_ground_contact"},
            )

    # Privileged Observation Group (e_t)
    cfg.observations["privileged"] = ObservationGroupCfg(
        nan_policy="sanitize",
        terms={
            "external_parameters": ObservationTermCfg(
                func=tracking_mdp.external_parameters,
            ),
        }
    )

    # NOTE: Critic does NOT include e_t directly in its obs group.
    # ArmaCriticModel reads obs["privileged"] separately and concatenates
    # [o_t || e_t] internally. Adding it here would double-count e_t.

    # Safety: Explicitly ensure ALL critic terms are noiseless
    for term in cfg.observations["critic"].terms.values():
        term.noise = None

    # Reduce IMU noise for existing terms in actor
    cfg.observations["actor"].terms["base_ang_vel"].noise.n_min = -0.04
    cfg.observations["actor"].terms["base_ang_vel"].noise.n_max = 0.04

    # Remove world-frame velocity and reference anchor from actor (spatial tracking strictly in critic)
    cfg.observations["actor"].terms.pop("base_lin_vel", None)
    cfg.observations["actor"].terms.pop("motion_anchor_pos_b", None)
    cfg.observations["actor"].terms.pop("motion_anchor_ori_b", None)

    # =========================================================================
    # 9. PLAY MODE OVERRIDES
    # =========================================================================
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.events.pop("control_delay", None)
        cfg.events.pop("p_gain", None)
        cfg.events.pop("link_mass", None)
        cfg.events.pop("link_com", None)
        cfg.events.pop("joint_damping", None)
        cfg.events.pop("left_foot_friction", None)
        cfg.events.pop("right_foot_friction", None)
        cfg.events.pop("encoder_bias", None)
        cfg.events.pop("base_com", None)
        cfg.events.pop("joint_friction", None)

        # Disable RSI randomization
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.sampling_mode = "start"

    return cfg