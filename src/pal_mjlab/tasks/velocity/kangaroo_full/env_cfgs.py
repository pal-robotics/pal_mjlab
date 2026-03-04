"""PAL Robotics kangaroo_full velocity tracking environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers import MetricsTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise


from pal_mjlab.robots.pal_kangaroo_full.kangaroo_full_constants import (  # noqa: F401
    KANG_FULL_ACTUATOR_NAMES,
    KANG_FULL_ACTION_SCALE,
    get_kangaroo_full_robot_cfg,
    ANKLE_XY_CONVEX_HULL_POINTS,
    HIP_XY_CONVEX_HULL_POINTS,
)
from pal_mjlab.tasks.velocity.kangaroo_full import mdp


def pal_kangaroo_full_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics kangaroo_full rough terrain velocity configuration."""
    cfg = make_velocity_env_cfg()
    cfg.scene.entities = {"robot": get_kangaroo_full_robot_cfg("kangaroo_full")}
    cfg.sim.nconmax = None
    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.mujoco.timestep = 0.002
    cfg.decimation = 10

    site_names = ("left_foot", "right_foot")
    geom_names = tuple(
        f"{side}_foot_collision"
        for side in ("left", "right")
    )

    _ACTUATED_JOINT_RE = (
        r".*_hip_z_slider$"
        r"|.*_hip_xy_slider_l$"
        r"|.*_hip_xy_slider_r$"
        r"|.*_ankle_xy_slider_l$"
        r"|.*_ankle_xy_slider_r$"
        r"|.*_leg_length_slider$"
        r"|pelvis_1_joint$"
        r"|pelvis_2_joint$"
        r"|arm_.*_[1-4]_joint$"
    )

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_xy_foot|right_ankle_xy_foot)$",  # subtree so foot link is included
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
            pattern=r"^(left_leg_length_femur|right_leg_length_femur|left_leg_length_tibia|right_leg_length_tibia)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="baselink", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="baselink", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, body_ground_cfg)

    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = KANG_FULL_ACTION_SCALE
    joint_pos_action.actuator_names = KANG_FULL_ACTUATOR_NAMES

    cfg.viewer.body_name = "pelvis_2_link"

    assert cfg.commands is not None
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.15

    #-- Observations

    cfg.observations["actor"].terms["height_scan"] = None
    cfg.observations["critic"].terms["height_scan"] = None
    cfg.observations["actor"].terms["base_lin_vel"] = None
    cfg.observations["actor"].terms["projected_gravity"] = None
    cfg.observations["actor"].terms["base_ang_vel"] = ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/global_angvel"},
    )
    cfg.observations["actor"].terms["imu_projected_gravity"] = ObservationTermCfg(
        func=mdp.imu_projected_gravity,
        params={"sensor_name": "robot/orientation"},
        noise=Unoise(n_min=-0.5, n_max=0.5),
    )
    cfg.observations["actor"].terms["base_lin_acc"] = ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/local_linacc"},
        noise=Unoise(n_min=-0.5, n_max=0.5),
    )
    cfg.observations["critic"].terms["imu_projected_gravity"] = ObservationTermCfg(
        func=mdp.imu_projected_gravity,
        params={"sensor_name": "robot/orientation"},
    )
    cfg.observations["critic"].terms["base_lin_acc"] = ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/local_linacc"},
    )
    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/local_linvel"},
    )
    cfg.observations["critic"].terms["base_ang_vel"] = ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/global_angvel"},
    )
    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    ### Disabling the use of history length as we haven't seen much improvements with it
    ### Moreover, our best policy #62 doesn't use any history length
    # cfg.observations["actor"].history_length = 5  # Keep last 5 frames
    # cfg.observations["critic"].history_length = 5  # Keep last 5 frames
    
    #-- Events

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis_2_link",)
    cfg.events["joint_friction"] = EventTermCfg(
        mode="startup",
        func=mdp.randomize_field,
        domain_randomization=True,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "field": "dof_frictionloss",
            "ranges": (-0.008, 0.008),
            "operation": "add",
        },
    )
    cfg.events["encoder_bias"].params["asset_cfg"].joint_names = [r"^(?!.*_leg_length_slider$).*"]
    cfg.events["leg_length_encoder_bias"] = EventTermCfg(
        mode="startup",
        func=mdp.randomize_encoder_bias,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=[r".*_leg_length_slider$"]
            ),
            "bias_range": (-0.005, 0.005),
        },
    )

    #-- Rewards

    cfg.rewards["pose"].params["asset_cfg"].joint_names = (_ACTUATED_JOINT_RE,)
    cfg.rewards["pose"].params["std_standing"] = {_ACTUATED_JOINT_RE: 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        # Lower body.
        r".*_hip_z_slider": 0.01,
        r".*_hip_xy_slider_l": 0.01,
        r".*_hip_xy_slider_r": 0.01,
        r".*_leg_length_slider$": 0.05,
        r".*_ankle_xy_slider_l": 0.01,
        r".*_ankle_xy_slider_r": 0.01,
        # Waist.
        r"pelvis_1.*": 0.08,
        r"pelvis_2.*": 0.2,
        # Arms.
        r"arm_.*_1_.*": 0.2,  # pitch
        r"arm_.*_4_.*": 0.2,  # elbow
        r"arm_.*_(?![14]_joint)\d+_joint": 0.1,
    }
    cfg.rewards["pose"].params["std_running"] = {
        # Lower body.
        r".*_hip_z_slider": 0.015,
        r".*_hip_xy_slider_l": 0.015,
        r".*_hip_xy_slider_r": 0.015,
        r".*_leg_length_slider$": 0.08,
        r".*_ankle_xy_slider_l": 0.015,
        r".*_ankle_xy_slider_r": 0.015,
        # Waist.
        r"pelvis_1.*": 0.08,
        r"pelvis_2.*": 0.3,
        # Arms.
        r"arm_.*_1_.*": 0.4,
        r"arm_.*_4_.*": 0.35,
        r"arm_.*_(?![14]_joint)\d+_joint": 0.15,
    }
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("pelvis_2_link",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis_2_link",)
    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["air_time"].weight = 0.25
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    cfg.rewards["joint_velocity_limit"] = RewardTermCfg(
        func=mdp.joint_vel_limit,
        weight = -0.02,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]), "limit_scale": 1.0},
    )

    # The hull points should correspond to the respective joints defined in the joint_names_group order
    # leg_*_2_joint corresponds to Hip Pitch and leg_*_3_joint corresponds to Hip roll
    cfg.rewards["convex_hull_joint_limits_hip"] = RewardTermCfg(
        func=mdp.joint_limits_convex_hull,
        weight=-10.0,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
        "metrics_suffix": "hipXY",
        "joint_names_group": [
            [r"left_hip_xy_pitch", r"left_hip_xy_roll"],
            [r"right_hip_xy_pitch", r"right_hip_xy_roll"],
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
            [r"left_ankle_xy_pitch", r"left_ankle_xy_roll"],
            [r"right_ankle_xy_pitch", r"right_ankle_xy_roll"],
        ],
        "hull_points": ANKLE_XY_CONVEX_HULL_POINTS,
        },
    )

    cfg.rewards["electrical_power_cost"] = RewardTermCfg(
        func=mdp.electrical_power_cost,
        weight=-0.5,
        params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
        },
    )

    ## Metrics
    cfg.metrics = {"joint_vel_mag": MetricsTermCfg(func=mdp.joint_velocity_magnitude, params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
                "joint_acc_mag": MetricsTermCfg(func=mdp.joint_accelerations_magnitude, params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
                "joint_torque_mag": MetricsTermCfg(func=mdp.joint_torques_magnitude, params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
                "action_rate_l2": MetricsTermCfg(func=mdp.action_rate_l2, params={}),
                "action_acc_l2": MetricsTermCfg(func=mdp.action_acc_l2, params={})}

    # # All except leg length joints
    # cfg.rewards["joint_accel"] = RewardTermCfg(
    #     func=mdp.joint_acc_l2,
    #     weight=-1.0e-8,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    # )

    # cfg.curriculum["joint_accel"] = CurriculumTermCfg(
    #   func=mdp.reward_weight,
    #   params={"reward_name": "joint_accel",
    #           "weight_stages": [
    #               {"step": 0, "weight": 0.0 },
    #               {"step": 2000 * 24, "weight": -1.0e-8},
    #               {"step": 8000 * 24, "weight": -1.0e-7},
    #               {"step": 15000 * 24, "weight": -1.0e-6},
    #           ],
    #   },
    # )


    #-- Terminations

    cfg.terminations["illegal_contacts"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": "body_ground_contact"},
    )

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=mdp.randomize_terrain,
            mode="reset",
            params={},
        )

        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg

'''
def pal_kangaroo_full_hands_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics KANGAROO with hands (5 DoF per arms) rough terrain velocity configuration."""
    cfg = pal_kangaroo_full_rough_env_cfg(play=play)

    cfg.scene.entities = {"robot": get_kangaroo_full_hands_robot_cfg()}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = KANGAROO_HANDS_ACTION_SCALE
    joint_pos_action.actuator_names = KANGAROO_HANDS_ACTUATOR_NAMES

    return cfg


def pal_kangaroo_full_grippers_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics KANGAROO with grippers (7 DoF per arms) rough terrain velocity configuration."""
    cfg = pal_kangaroo_full_rough_env_cfg(play=play)

    cfg.scene.entities = {"robot": get_kangaroo_full_grippers_robot_cfg()}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = KANGAROO_GRIPPERS_ACTION_SCALE
    joint_pos_action.actuator_names = KANGAROO_GRIPPERS_ACTUATOR_NAMES

    return cfg
'''

def pal_kangaroo_full_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics KANGAROO flat terrain velocity configuration."""
    cfg = pal_kangaroo_full_rough_env_cfg(play=play)

    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None
    
    # Switch to flat terrain.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Disable terrain curriculum.
    assert cfg.curriculum is not None
    assert "terrain_levels" in cfg.curriculum
    del cfg.curriculum["terrain_levels"]

    if play:
        # Disable command curriculum.
        assert "command_vel" in cfg.curriculum
        del cfg.curriculum["command_vel"]

        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
        twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

    return cfg

'''
def pal_kangaroo_full_hands_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics KANGAROO with hands (5 DoF per arms) flat terrain velocity configuration."""
    cfg = pal_kangaroo_full_flat_env_cfg(play=play)

    cfg.scene.entities = {"robot": get_kangaroo_full_hands_robot_cfg()}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = KANGAROO_HANDS_ACTION_SCALE
    joint_pos_action.actuator_names = KANGAROO_HANDS_ACTUATOR_NAMES

    return cfg


def pal_kangaroo_full_grippers_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics KANGAROO with grippers (7 DoF per arms) flat terrain velocity configuration."""
    cfg = pal_kangaroo_full_flat_env_cfg(play=play)

    cfg.scene.entities = {"robot": get_kangaroo_full_grippers_robot_cfg()}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = KANGAROO_GRIPPERS_ACTION_SCALE
    joint_pos_action.actuator_names = KANGAROO_GRIPPERS_ACTUATOR_NAMES

    return cfg
'''