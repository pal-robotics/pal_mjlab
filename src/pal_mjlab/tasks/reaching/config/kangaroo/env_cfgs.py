"""PAL Robotics KANGAROO locomotion + dual-arm reaching environment configurations.

Uses reaching task as base and adds velocity tracking for locomotion.
"""

import math

from pal_mjlab.robots import (
    get_kangaroo_robot_cfg,
    get_kangaroo_hands_robot_cfg,
    get_kangaroo_grippers_robot_cfg,
    KANGAROO_ACTION_SCALE,
    KANGAROO_HANDS_ACTION_SCALE,
    KANGAROO_GRIPPERS_ACTION_SCALE,
    KANGAROO_ACTUATOR_NAMES,
    KANGAROO_HANDS_ACTUATOR_NAMES,
    KANGAROO_GRIPPERS_ACTUATOR_NAMES,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
    RewardTermCfg,
    TerminationTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    CurriculumTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

# Base reaching task
from pal_mjlab.tasks.reaching.reaching_env_cfg import make_reaching_env_cfg
from pal_mjlab.tasks.reaching import mdp as reach_mdp

# Locomotion mdp functions
from mjlab.tasks.velocity import mdp as loco_mdp


def pal_kangaroo_flat_loco_reaching_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics KANGAROO flat terrain locomotion + dual-arm reaching."""
    
    # Start from reaching base
    cfg = make_reaching_env_cfg()
    cfg.scene.entities = {"robot": get_kangaroo_robot_cfg()}
    cfg.sim.nconmax = 45
    cfg.viewer.body_name = "pelvis_2_link"

    # -----------------------------------------------------------------
    # Robot-specific definitions
    # -----------------------------------------------------------------
    site_names = ("left_foot", "right_foot")
    geom_names = tuple(
        f"{side}_foot{i}_collision" for side in ("left", "right") for i in [0, 2, 4, 6, 8, 10]
    )
    
    locomotion_joints = (
        r"leg_.*_1_.*",
        r"leg_.*_2_.*",
        r"leg_.*_3_.*",
        r"leg_.*_length_.*",
        r"leg_.*_4_.*",
        r"leg_.*_5_.*",
        r"pelvis_.*",
    )
    left_arm_joints = (r"arm_left.*",)
    right_arm_joints = (r"arm_right.*",)

    # -----------------------------------------------------------------
    # Contact sensors (for locomotion)
    # -----------------------------------------------------------------
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
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
    body_ground_cfg = ContactSensorCfg(
        name="body_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(leg_left_femur_link|leg_right_femur_link|leg_left_knee_link|leg_right_knee_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, body_ground_cfg)

    # -----------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = KANGAROO_ACTION_SCALE
    joint_pos_action.actuator_names = KANGAROO_ACTUATOR_NAMES

    # -----------------------------------------------------------------
    # Commands: add velocity tracking to existing pose commands
    # -----------------------------------------------------------------
    cfg.commands["twist"] = loco_mdp.UniformVelocityCommandCfg(
      asset_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.1,
      rel_heading_envs=0.3,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=loco_mdp.UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.0, 1.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-0.5, 0.5),
        heading=(-math.pi, math.pi),
      ),
    )

    # Configure pose command ranges for Kangaroo workspace
    cfg.commands["pose_command_left"].ranges.pos_x = (-0.6, 0.6)
    cfg.commands["pose_command_left"].ranges.pos_y = (0.2, 0.6)
    cfg.commands["pose_command_left"].ranges.pos_z = (-0.6, 0.6)

    cfg.commands["pose_command_right"].ranges.pos_x = (-0.6, 0.6)
    cfg.commands["pose_command_right"].ranges.pos_y = (-0.2, -0.6)
    cfg.commands["pose_command_right"].ranges.pos_z = (-0.6, 0.6)

    # -----------------------------------------------------------------
    # Observations: add locomotion observations
    # -----------------------------------------------------------------
    # Add locomotion-specific observations to policy
    cfg.observations["policy"].terms["base_ang_vel"] = ObservationTermCfg(
        func=loco_mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
        noise=Unoise(n_min=-0.2, n_max=0.2),
    )
    cfg.observations["policy"].terms["base_lin_acc"] = ObservationTermCfg(
        func=loco_mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_acc"},
        noise=Unoise(n_min=-0.5, n_max=0.5),
    )
    cfg.observations["policy"].terms["twist_command"] = ObservationTermCfg(
        func=loco_mdp.generated_commands,
        params={"command_name": "twist"},
    )

    # Add locomotion-specific observations to critic
    cfg.observations["critic"].terms["base_ang_vel"] = ObservationTermCfg(
        func=loco_mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_ang_vel"},
    )
    cfg.observations["critic"].terms["base_lin_acc"] = ObservationTermCfg(
        func=loco_mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_acc"},
    )
    cfg.observations["critic"].terms["twist_command"] = ObservationTermCfg(
        func=loco_mdp.generated_commands,
        params={"command_name": "twist"},
    )
    cfg.observations["critic"].terms["foot_height"] = ObservationTermCfg(
        func=loco_mdp.foot_height,
        params={"asset_cfg": SceneEntityCfg("robot", site_names=site_names)},
    )
    cfg.observations["critic"].terms["foot_air_time"] = ObservationTermCfg(
        func=loco_mdp.foot_air_time,
        params={"sensor_name": "feet_ground_contact"},
    )
    cfg.observations["critic"].terms["foot_contact"] = ObservationTermCfg(
        func=loco_mdp.foot_contact,
        params={"sensor_name": "feet_ground_contact"},
    )
    cfg.observations["critic"].terms["foot_contact_forces"] = ObservationTermCfg(
        func=loco_mdp.foot_contact_forces,
        params={"sensor_name": "feet_ground_contact"},
    )

    # -----------------------------------------------------------------
    # Events: add locomotion events
    # -----------------------------------------------------------------
    cfg.events["reset_base"] = EventTermCfg(
        func=loco_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        },
    )
    cfg.events["push_robot"] = EventTermCfg(
        func=loco_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
    cfg.events["foot_friction"] = EventTermCfg(
        mode="startup",
        func=loco_mdp.randomize_field,
        domain_randomization=True,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
            "operation": "abs",
            "field": "geom_friction",
            "ranges": (0.3, 1.2),
        },
    )
    cfg.events["reset_robot_joints"].params["asset_cfg"].joint_names = (r"arm_.*",)
    cfg.events["reset_frictionloss"].params["asset_cfg"].joint_names = (r"arm_.*",)

    # -----------------------------------------------------------------
    # Rewards: add locomotion rewards to existing reaching rewards
    # -----------------------------------------------------------------
    # Velocity tracking
    cfg.rewards["track_lin_vel"] = RewardTermCfg(
        func=loco_mdp.track_linear_velocity,
        weight=2.0,
        params={"command_name": "twist", "std": math.sqrt(0.25)},
    )
    cfg.rewards["track_ang_vel"] = RewardTermCfg(
        func=loco_mdp.track_angular_velocity,
        weight=2.0,
        params={"command_name": "twist", "std": math.sqrt(0.5)},
    )
    
    # Stability
    cfg.rewards["upright"] = RewardTermCfg(
        func=loco_mdp.flat_orientation,
        weight=1.0,
        params={
            "std": math.sqrt(0.2),
            "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis_2_link",)),
        },
    )
    cfg.rewards["body_ang_vel"] = RewardTermCfg(
        func=loco_mdp.body_angular_velocity_penalty,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("pelvis_2_link",))},
    )
    cfg.rewards["angular_momentum"] = RewardTermCfg(
        func=loco_mdp.angular_momentum_penalty,
        weight=-0.02,
        params={"sensor_name": "robot/root_angmom"},
    )
    
    # Posture for locomotion joints only
    cfg.rewards["pose_loco"] = RewardTermCfg(
        func=loco_mdp.variable_posture,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=locomotion_joints),
            "command_name": "twist",
            "std_standing": {
                r"leg_.*_1_.*": 0.05,
                r"leg_.*_2_.*": 0.05,
                r"leg_.*_3_.*": 0.05,
                r"leg_.*_length_.*": 0.05,
                r"leg_.*_4_.*": 0.05,
                r"leg_.*_5_.*": 0.05,
                r"pelvis_.*": 0.05,
            },
            "std_walking": {
                r"leg_.*_1_.*": 0.15,
                r"leg_.*_2_.*": 0.3,
                r"leg_.*_3_.*": 0.15,
                r"leg_.*_length_.*": 0.15,
                r"leg_.*_4_.*": 0.25,
                r"leg_.*_5_.*": 0.1,
                r"pelvis_1.*": 0.08,
                r"pelvis_2.*": 0.2,
            },
            "std_running": {
                r"leg_.*_1_.*": 0.2,
                r"leg_.*_2_.*": 0.5,
                r"leg_.*_3_.*": 0.2,
                r"leg_.*_length_.*": 0.25,
                r"leg_.*_4_.*": 0.35,
                r"leg_.*_5_.*": 0.15,
                r"pelvis_1.*": 0.08,
                r"pelvis_2.*": 0.3,
            },
            "walking_threshold": 0.05,
            "running_threshold": 1.5,
        },
    )
    
    # Feet rewards
    cfg.rewards["air_time"] = RewardTermCfg(
        func=loco_mdp.feet_air_time,
        weight=0.25,
        params={
            "sensor_name": "feet_ground_contact",
            "threshold_min": 0.05,
            "threshold_max": 0.5,
            "command_name": "twist",
            "command_threshold": 0.5,
        },
    )
    cfg.rewards["foot_clearance"] = RewardTermCfg(
        func=loco_mdp.feet_clearance,
        weight=-2.0,
        params={
            "target_height": 0.1,
            "command_name": "twist",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
        },
    )
    cfg.rewards["foot_swing_height"] = RewardTermCfg(
        func=loco_mdp.feet_swing_height,
        weight=-0.25,
        params={
            "sensor_name": "feet_ground_contact",
            "target_height": 0.1,
            "command_name": "twist",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
        },
    )
    cfg.rewards["foot_slip"] = RewardTermCfg(
        func=loco_mdp.feet_slip,
        weight=-0.1,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
        },
    )
    cfg.rewards["soft_landing"] = RewardTermCfg(
        func=loco_mdp.soft_landing,
        weight=-1e-5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "command_threshold": 0.05,
        },
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=loco_mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )
    
    # Separate action rates for body vs arms
    cfg.rewards["action_rate_l2"].params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=locomotion_joints),
    }
    cfg.rewards["action_rate_l2"].weight = -0.1
    cfg.rewards["action_rate_arms_l2"] = RewardTermCfg(
        func=reach_mdp.action_rate_l2_louis,
        weight=-0.0001,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=left_arm_joints + right_arm_joints),
        },
    )

    # -----------------------------------------------------------------
    # Terminations: add locomotion terminations
    # -----------------------------------------------------------------
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=loco_mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
    )
    cfg.terminations["illegal_contacts"] = TerminationTermCfg(
        func=loco_mdp.illegal_contact,
        params={"sensor_name": "body_ground_contact"},
    )

    # -----------------------------------------------------------------
    # Curriculum: add velocity curriculum
    # -----------------------------------------------------------------
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
        func=loco_mdp.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
                {"step": 5000 * 24, "lin_vel_x": (-1.5, 2.0), "ang_vel_z": (-0.7, 0.7)},
            ],
        },
    )

    # -----------------------------------------------------------------
    # Play mode overrides
    # -----------------------------------------------------------------
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)

    return cfg


def pal_kangaroo_hands_flat_loco_reaching_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Kangaroo with hands (5 DoF arms) flat terrain locomotion + reaching."""
    cfg = pal_kangaroo_flat_loco_reaching_env_cfg(play=play)

    cfg.scene.entities = {"robot": get_kangaroo_hands_robot_cfg()}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = KANGAROO_HANDS_ACTION_SCALE
    joint_pos_action.actuator_names = KANGAROO_HANDS_ACTUATOR_NAMES

    return cfg


def pal_kangaroo_grippers_flat_loco_reaching_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Kangaroo with grippers (7 DoF arms) flat terrain locomotion + reaching."""
    cfg = pal_kangaroo_flat_loco_reaching_env_cfg(play=play)

    cfg.scene.entities = {"robot": get_kangaroo_grippers_robot_cfg()}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = KANGAROO_GRIPPERS_ACTION_SCALE
    joint_pos_action.actuator_names = KANGAROO_GRIPPERS_ACTUATOR_NAMES

    return cfg