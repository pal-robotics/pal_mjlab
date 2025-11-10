"""PAL Robotics Talos flat terrain tracking configuration.

This module provides factory functions that create complete ManagerBasedRlEnvCfg
instances for the Talos robot tracking task on flat terrain.
"""

from copy import deepcopy

from pal_mjlab.robots import TALOS_ACTION_SCALE, get_talos_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.tracking_env_cfg import create_tracking_env_cfg
from mjlab.utils.retval import retval


@retval
def PAL_TALOS_FLAT_TRACKING_ENV_CFG() -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics Talos flat terrain tracking configuration."""
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    return create_tracking_env_cfg(
        robot_cfg=get_talos_robot_cfg(),
        action_scale=TALOS_ACTION_SCALE,
        viewer_body_name="base_link",
        motion_file="",
        anchor_body_name="base_link",
        body_names=(
            "base_link",
            "torso_2_link",
            "leg_left_3_link",
            "leg_left_4_link",
            "leg_left_6_link",
            "leg_right_3_link",
            "leg_right_4_link",
            "leg_right_6_link",
            "arm_left_3_link",
            "arm_left_4_link",
            "arm_left_7_link",
            "arm_right_3_link",
            "arm_right_4_link",
            "arm_right_7_link",
        ),
        foot_friction_geom_names=(r"^(left|right)_foot_collision$",),
        ee_body_names=(
            "leg_left_6_link",
            "leg_right_6_link",
            "arm_left_7_link",
            "arm_right_7_link",
        ),
        base_com_body_name="torso_link",
        sensors=(self_collision_cfg,),
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range={
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.78, 0.78),
        },
        joint_position_range=(-0.1, 0.1),
    )


@retval
def PAL_TALOS_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG() -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics Talos flat terrain tracking config without state estimation.

    This variant disables motion_anchor_pos_b and base_lin_vel observations,
    simulating the lack of state estimation.
    """
    cfg = deepcopy(PAL_TALOS_FLAT_TRACKING_ENV_CFG)
    assert "policy" in cfg.observations
    cfg.observations["policy"].terms.pop("motion_anchor_pos_b")
    cfg.observations["policy"].terms.pop("base_lin_vel")
    return cfg
