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
    use_history: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics Talos flat terrain tracking configuration."""
    cfg = make_tracking_env_cfg()

    # =========================================================================
    # 1. SCENE & SIMULATION SETUP
    # =========================================================================
    cfg.scene.entities = {"robot": get_kangaroo_robot_cfg()}
    cfg.sim.mujoco.timestep = 0.002
    # cfg.sim.njmax = 450
    # cfg.sim.nconmax = 100
    # cfg.sim.contact_sensor_maxmatch = 128
    cfg.decimation = 10

    # Self-collision configuration
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (self_collision_cfg,)

    # =========================================================================
    # 2. ACTIONS
    # =========================================================================
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = KANGAROO_ACTION_SCALE
    joint_pos_action.actuator_names = KANGAROO_ACTUATOR_NAMES

    assert cfg.commands is not None
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = "base_link"
    motion_cmd.body_names = (
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


    # motion_cmd.joint_position_range = (-0.2, 0.2)
    # The hull points should correspond to the respective joints defined in the joint_names_group order
    # leg_*_2_joint corresponds to Hip Pitch and leg_*_3_joint corresponds to Hip roll
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

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    cfg.events["body_friction"] = EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=body_geoms),  # Set per-robot.
        "operation": "abs",
        "ranges": (0.3, 2.0),
        "shared_random": False,  # All body geoms share the same friction.
        },
    )

    cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis_2_link",)

    cfg.events["control_delay"] = EventTermCfg(
        mode="startup",
        func=tracking_mdp.control_delay,
        params={
        "delay_range": (0.0, 0.04),  # 0–40 ms
        "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.events["p_gain"] = EventTermCfg(
        mode="startup",
        func=tracking_mdp.p_gain,
        params={
        "kp_range": (0.925, 1.05),
        "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.events["joint_friction"] = EventTermCfg(
        mode="startup",
        func=dr.dof_frictionloss,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "operation": "add",
        "ranges": (-0.008, 0.008),
        "shared_random": False,
        },
    )


    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "leg_left_5_link",
        "leg_right_5_link",
        "arm_left_5_link",
        "arm_right_5_link",
    )

    cfg.viewer.body_name = "base_link"

    # Uncomment to enable 5-step observation history for latency compensation:
    cfg.observations["actor"].history_length = 5
    cfg.observations["critic"].history_length = 5

    # Add periodic motion phase and restructure actor/critic observations
    for group_name in ["actor", "critic"]:
        cfg.observations[group_name].terms["motion_phase"] = ObservationTermCfg(
        func=tracking_mdp.motion_phase,
        params={"command_name": "motion"},


        # =========================================================================
        # 3. COMMANDS
        # =========================================================================

        # --- Original Command Configuration (Commented Out) ---
        motion_cmd = cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)
        motion_cmd.anchor_body_name = "base_link"
        motion_cmd.body_names = body_names

    # --- New Per-Joint Randomized Command ---
    # old_motion_cmd = cfg.commands["motion"]
    # cfg.commands["motion"] = PalMotionCommandCfg(
    #     entity_name=old_motion_cmd.entity_name,
    #     motion_file=old_motion_cmd.motion_file,
    #     anchor_body_name="base_link",
    #     body_names=body_names,
    #     resampling_time_range=old_motion_cmd.resampling_time_range,
    #     pose_range=old_motion_cmd.pose_range,
    #     velocity_range=old_motion_cmd.velocity_range,
    #     sampling_mode=old_motion_cmd.sampling_mode,
    #     debug_vis=old_motion_cmd.debug_vis,
    #     joint_position_range=old_motion_cmd.joint_position_range,  # Default fallback
    #     joint_position_ranges={
    #         r"arm_.*_joint": (-0.5, 0.5),
    #         r"leg_.*_length_joint|leg_.*_femur_joint|leg_.*_knee_joint": (-0.1, 0.1),
    #         r"leg_.*_joint": (-0.3, 0.3),
    #         r"pelvis_.*_joint": (-0.3, 0.3),
    #     },
    # )
    # motion_cmd = cfg.commands["motion"]
    # motion_cmd.joint_position_range = (-0.2, 0.2)
    # motion_cmd.joint_position_range = (-0.2, 0.2)

    # =========================================================================
    # 4. REWARDS
    # =========================================================================
    # The hull points correspond to the respective joints defined in the joint_names_group order.
    # leg_*_2_joint corresponds to Hip Pitch and leg_*_3_joint corresponds to Hip Roll.
    # Convex Hull limits
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

    # Tighten tracking precision for velocities
    cfg.rewards["motion_body_lin_vel"].params["std"] = 0.5
    cfg.rewards["motion_body_ang_vel"].params["std"] = 1.0

    # =========================================================================
    # 5. EVENTS (Domain Randomization)
    # =========================================================================
    geom_names = tuple(
        f"{side}_foot{i}_collision"
        for side in ("left", "right")
        for i in [0, 2, 4, 6, 8, 10]
    )
    
    body_geoms = (
        "leg_left_femur_collision", "leg_right_femur_collision",
        "leg_left_knee_collision", "leg_left_knee_bar_collision",
        "leg_right_knee_collision", "leg_right_knee_bar_collision",
        "arm_left_4_collision", "arm_right_4_collision",
        "pelvis_2_collision",
    )

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    
    cfg.events["body_friction"] = EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=body_geoms),
            "operation": "abs",
            "ranges": (0.3, 2.0),
            "shared_random": False,
        },
    )

    cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis_2_link",)

    cfg.events["control_delay"] = EventTermCfg(
        mode="startup",
        func=tracking_mdp.control_delay,
        params={
            "delay_range": (0.0, 0.04),  # 0–40 ms
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    cfg.events["p_gain"] = EventTermCfg(
        mode="startup",
        func=tracking_mdp.p_gain,
        params={
            "kp_range": (0.925, 1.05),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    
    cfg.events["joint_friction"] = EventTermCfg(
        mode="startup",
        func=dr.dof_frictionloss,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "add",
            "ranges": (-0.008, 0.008),
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
    # Latency compensation history
    # cfg.observations["actor"].history_length = 5
    # cfg.observations["critic"].history_length = 5

    # Add IMU observations
    for group_name in ["actor", "critic"]:
        cfg.observations[group_name].terms["motion_phase"] = ObservationTermCfg(
            func=tracking_mdp.motion_phase,
            params={"command_name": "motion"},
        )
        cfg.observations[group_name].terms["base_lin_acc"] = ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_acc"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        cfg.observations[group_name].terms["imu_projected_gravity"] = ObservationTermCfg(
            func=mdp.imu_projected_gravity,
            params={"sensor_name": "robot/imu_quat"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

    # Reduce IMU noise for existing terms
    cfg.observations["actor"].terms["base_ang_vel"].noise.n_min = -0.04
    cfg.observations["actor"].terms["base_ang_vel"].noise.n_max = 0.04

    # Modify observations based on state estimation and actor requirements
    # Note: base_lin_acc and imu_projected_gravity are used instead of world-frame velocity in actor.
    cfg.observations["actor"].terms.pop("base_lin_vel", None)

    # Move reference anchor tracking (spatial error) to the critic only
    cfg.observations["actor"].terms.pop("motion_anchor_pos_b", None)
    cfg.observations["actor"].terms.pop("motion_anchor_ori_b", None)

    # --- Old Observation Logic (Commented Out) ---
    # if not has_state_estimation:
    #   new_actor_terms = {
    #     k: v
    #     for k, v in cfg.observations["actor"].terms.items()
    #     if k not in ["motion_anchor_pos_b", "motion_anchor_ori_b", "base_lin_vel"]
    #   }
    #   cfg.observations["actor"] = ObservationGroupCfg(
    #     terms=new_actor_terms,
    #     concatenate_terms=True,
    #     enable_corruption=True,
    #   )

    # -------------------------------------------------------------------------
    # History Encoder Groups
    # -------------------------------------------------------------------------
    if use_history:
        # Create explicit history buffers for our custom TCN Encoder.
        # We maintain the normal actor/critic groups to represent the current 1-step obs.
        cfg.observations["actor_history"] = copy.deepcopy(cfg.observations["actor"])
        cfg.observations["actor_history"].history_length = 30
        cfg.observations["actor_history"].flatten_history_dim = False
        
        cfg.observations["critic_history"] = copy.deepcopy(cfg.observations["critic"])
        cfg.observations["critic_history"].history_length = 30
        cfg.observations["critic_history"].flatten_history_dim = False

    # =========================================================================
    # 9. PLAY MODE OVERRIDES
    # =========================================================================
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        # Disable noise and specific events
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.events.pop("control_delay", None)
        cfg.events.pop("p_gain", None)

        # Disable RSI randomization
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}
        motion_cmd.sampling_mode = "start"

    return cfg