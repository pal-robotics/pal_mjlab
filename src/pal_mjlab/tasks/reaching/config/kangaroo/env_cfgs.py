"""PAL Robotics KANGAROO reaching environment configurations."""

from pal_mjlab.robots import (
    get_kangaroo_robot_cfg,
    KANGAROO_ACTION_SCALE,
    get_kangaroo_hands_robot_cfg,
    KANGAROO_HANDS_ACTION_SCALE,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from pal_mjlab.tasks.reaching.reaching_env_cfg import make_reaching_env_cfg


def pal_kangaroo_flat_reaching_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics KANGAROO reaching configuration."""
    cfg = make_reaching_env_cfg()

    cfg.scene.entities = {"robot": get_kangaroo_robot_cfg()}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = KANGAROO_ACTION_SCALE

    cfg.viewer.body_name = "pelvis_2_link"

    cfg.commands["pose_command_left"].ranges.pos_x = (-0.6, 0.6)
    cfg.commands["pose_command_left"].ranges.pos_y = (0.2, 0.6)
    cfg.commands["pose_command_left"].ranges.pos_z = (-0.6, 0.6)

    cfg.rewards["pose"].params["asset_cfg"].joint_names = (
        # Lower body.
        r"leg_.*_1_.*",
        r"leg_.*_2_.*",
        r"leg_.*_3_.*",
        r"leg_.*_length_.*",
        r"leg_.*_4_.*",
        r"leg_.*_5_.*",
        # Waist.
        r"pelvis_.*",
        # Arms.
        r"arm_right.*",
    )
    cfg.rewards["pose"].params["std"] = {
        # Lower body.
        r"leg_.*_1_.*": 0.05,
        r"leg_.*_2_.*": 0.05,
        r"leg_.*_3_.*": 0.05,
        r"leg_.*_length_.*": 0.05,
        r"leg_.*_4_.*": 0.05,
        r"leg_.*_5_.*": 0.05,
        # Waist.
        r"pelvis_.*": 0.08,
        # Arms.
        r"arm_right_1_joint": 0.1,
        r"arm_right_2_joint": 0.15,
        r"arm_right_4_joint": 0.1,
        r"arm_right_(?![124]_joint)\d+_joint": 0.05,
    }
    cfg.rewards["action_rate_body_l2"].params["asset_cfg"].joint_names = (
        # Lower body.
        r"leg_.*_1_.*",
        r"leg_.*_2_.*",
        r"leg_.*_3_.*",
        r"leg_.*_length_.*",
        r"leg_.*_4_.*",
        r"leg_.*_5_.*",
        # Waist.
        r"pelvis_.*",
        # Arms.
        r"arm_right.*",
    )
    cfg.rewards["action_rate_left_arm_l2"].params["asset_cfg"].joint_names = (
        r"arm_left.*",
    )

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["policy"].enable_corruption = False

    return cfg


def pal_kangaroo_hands_flat_reaching_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics KANGAROO with hands reaching configuration."""
    cfg = make_reaching_env_cfg()

    cfg.scene.entities = {"robot": get_kangaroo_hands_robot_cfg()}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = KANGAROO_HANDS_ACTION_SCALE

    cfg.viewer.body_name = "pelvis_2_link"

    cfg.commands["pose_command_left"].ranges.pos_x = (-0.3, -0.2)
    cfg.commands["pose_command_left"].ranges.pos_y = (-0.3, 0.3)
    cfg.commands["pose_command_left"].ranges.pos_z = (-0.3, 0.3)

    cfg.rewards["pose"].params["asset_cfg"].joint_names = (
        # Lower body.
        r"leg_.*_1_.*",
        r"leg_.*_2_.*",
        r"leg_.*_3_.*",
        r"leg_.*_length_.*",
        r"leg_.*_4_.*",
        r"leg_.*_5_.*",
        # Waist.
        r"pelvis_.*",
        # Arms.
        r"arm_right.*",
    )
    cfg.rewards["pose"].params["std"] = {
        # Lower body.
        r"leg_.*_1_.*": 0.05,
        r"leg_.*_2_.*": 0.05,
        r"leg_.*_3_.*": 0.05,
        r"leg_.*_length_.*": 0.05,
        r"leg_.*_4_.*": 0.05,
        r"leg_.*_5_.*": 0.05,
        # Waist.
        r"pelvis_.*": 0.08,
        # Arms.
        r"arm_right_1_joint": 0.1,
        r"arm_right_2_joint": 0.15,
        r"arm_right_4_joint": 0.1,
        r"arm_right_(?![124]_joint)\d+_joint": 0.05,
    }
    cfg.rewards["action_rate_body_l2"].params["asset_cfg"].joint_names = (
        # Lower body.
        r"leg_.*_1_.*",
        r"leg_.*_2_.*",
        r"leg_.*_3_.*",
        r"leg_.*_length_.*",
        r"leg_.*_4_.*",
        r"leg_.*_5_.*",
        # Waist.
        r"pelvis_.*",
        # Arms.
        r"arm_right.*",
    )
    cfg.rewards["action_rate_left_arm_l2"].params["asset_cfg"].joint_names = (
        r"arm_left.*",
    )

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["policy"].enable_corruption = False

    return cfg
