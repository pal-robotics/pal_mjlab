"""PAL Robotics KANGAROO locomotion + dual-arm reaching environment configurations.

Uses reaching task as base and adds velocity tracking for locomotion.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from pal_mjlab.robots import (
    KANG_FULL_ACTION_SCALE,
    KANG_FULL_ACTUATOR_NAMES,
    get_kangaroo_full_robot_cfg,
)
from pal_mjlab.tasks.reaching import mdp as reach_mdp
from pal_mjlab.tasks.reaching.reaching_env_cfg import make_reaching_env_cfg

# Base reaching task
# Locomotion mdp functions


def pal_kangaroo_full_reaching_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics KANGAROO fixed base + dual-leg reaching."""

    # Start from reaching base
    cfg = make_reaching_env_cfg()
    cfg.scene.entities = {"robot": get_kangaroo_full_robot_cfg("kangaroo_full_fixed")}
    cfg.sim.nconmax = 45
    cfg.viewer.body_name = "pelvis_2_link"

    # -----------------------------------------------------------------
    # Robot-specific definitions
    # -----------------------------------------------------------------

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="baselink", entity="robot"),
        secondary=ContactMatch(
            mode="subtree", pattern="baselink", entity="robot"
        ),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (self_collision_cfg,)

    # -----------------------------------------------------------------
    # Actions
    # -----------------------------------------------------------------
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = KANG_FULL_ACTION_SCALE
    joint_pos_action.actuator_names = KANG_FULL_ACTUATOR_NAMES

    # # Configure pose command ranges for Kangaroo workspace
    cfg.commands["pose_command_left"].site_name = "left_foot"
    cfg.commands["pose_command_left"].ranges.pos_x = (-0.3, 0.3)
    cfg.commands["pose_command_left"].ranges.pos_y = (0.0, 0.4)
    cfg.commands["pose_command_left"].ranges.pos_z = (-1.0, -0.6)
    cfg.commands["pose_command_left"].ranges.roll = (1.57, 1.57)
    cfg.commands["pose_command_left"].ranges.pitch = (0.0, 0.0)
    cfg.commands["pose_command_left"].ranges.yaw = (-1.57, -1.57)


    cfg.commands["pose_command_right"].site_name = "right_foot"
    cfg.commands["pose_command_right"].ranges.pos_x = (-0.3, 0.3)
    cfg.commands["pose_command_right"].ranges.pos_y = (0.0, -0.4)
    cfg.commands["pose_command_right"].ranges.pos_z = (-1.0, -0.6)
    cfg.commands["pose_command_right"].ranges.roll = (1.57, 1.57)
    cfg.commands["pose_command_right"].ranges.pitch = (0.0, 0.0)
    cfg.commands["pose_command_right"].ranges.yaw = (-1.57, -1.57)

    cfg.rewards = {}  # Clear reaching rewards since we'll add custom ones for locomotion + reaching
    cfg.curriculum = {}


    cfg.observations["actor"].terms["joint_pos"].params[
        "asset_cfg"
    ] = SceneEntityCfg("robot", joint_names=KANG_FULL_ACTUATOR_NAMES)
    cfg.observations["critic"].terms["joint_pos"].params[
        "asset_cfg"
    ] = SceneEntityCfg("robot", joint_names=KANG_FULL_ACTUATOR_NAMES)

    cfg.observations["critic"].terms["joint_vel"].params[
        "asset_cfg"
    ] = SceneEntityCfg("robot", joint_names=KANG_FULL_ACTUATOR_NAMES)
    cfg.observations["actor"].terms["joint_vel"].params[
        "asset_cfg"
    ] = SceneEntityCfg("robot", joint_names=KANG_FULL_ACTUATOR_NAMES)

    ############### Rewards ###############
    
    KANG_UPPER_BODY_NAMES = ('arm_.*_1_joint', 'arm_.*_2_joint', 'arm_.*_3_joint', 'arm_.*_4_joint', 'pelvis_1_joint', 'pelvis_2_joint')
    cfg.rewards["home_pose"] = RewardTermCfg(
            func=reach_mdp.posture,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=KANG_UPPER_BODY_NAMES),
                "std": {
                        # Full body.
                        r".*": 0.2,
                    },
                },
        )
    
    cfg.rewards["pos_left"] = RewardTermCfg(
            func=reach_mdp.position_command_error,
            weight=-2.0,
            params={
                "site_name": "left_foot",
                "command_name": "pose_command_left",
            },
        )
    cfg.rewards["pos_right"] = RewardTermCfg(
            func=reach_mdp.position_command_error,
            weight=-2.0,
            params={
                "site_name": "right_foot",
                "command_name": "pose_command_right",
            },
        )

    cfg.rewards["pos_left_fine_grained"] = RewardTermCfg(
            func=reach_mdp.position_command_error_tanh,
            weight=2.0,
            params={
                "site_name": "left_foot",
                "command_name": "pose_command_left",
                "std": 0.05,
            },
        )
    cfg.rewards["pos_right_fine_grained"] = RewardTermCfg(
            func=reach_mdp.position_command_error_tanh,
            weight=3.0,
            params={
                "site_name": "right_foot",
                "command_name": "pose_command_right",
                "std": 0.05,
            }
        )
    
    # cfg.rewards["ee_left_orientation"] = RewardTermCfg(
    #         func=reach_mdp.orientation_command_error,
    #         weight=-0.2,
    #         params={
    #             "site_name": "left_foot",
    #             "command_name": "pose_command_left",
    #         },
    #     )
    #     )
    # cfg.rewards["ee_right_orientation"] = RewardTermCfg(
    #         func=reach_mdp.orientation_command_error,
    #         weight=-0.2,
    #         params={
    #             "site_name": "right_foot",
    #             "command_name": "pose_command_right",
    #         },
    #     )

    cfg.rewards["action_rate_l2"] = RewardTermCfg(
            func=reach_mdp.action_rate_l2_louis,
            weight=-0.003,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=KANG_FULL_ACTUATOR_NAMES
                ),  
            },
        ),
    
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=reach_mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    # -----------------------------------------------------------------
    # Play mode overrides
    # -----------------------------------------------------------------
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        # cfg.events.pop("push_robot", None)

    return cfg

