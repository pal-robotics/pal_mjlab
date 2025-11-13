"""PAL Robotics KANGAROO reaching environment configurations."""

from pal_mjlab.robots import *
from mjlab.envs import ManagerBasedRlEnvCfg
from pal_mjlab.tasks.reaching.reaching_env_cfg import create_reaching_env_cfg
from mjlab.utils.retval import retval

@retval
def PAL_KANGAROO_ENV_CFG() -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics KANGAROO reaching configuration."""
    geom_names = ("left_foot_collision", "right_foot_collision")
    cfg = create_reaching_env_cfg(
        robot_cfg=get_kangaroo_robot_cfg(),
        action_scale=KANGAROO_ACTION_SCALE,
        viewer_body_name="pelvis_2_link",
        foot_friction_geom_names=geom_names,
        pos_x = (-0.6, 0.6),
        pos_y = (0.2, 0.6),
        pos_z = (-0.6, 0.6),
        posture_jn = (
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
        ),
        posture_std = {
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
        },
    )
    return cfg

@retval
def PAL_KANGAROO_HANDS_ENV_CFG() -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics KANGAROO reaching configuration."""
    geom_names = ("left_foot_collision", "right_foot_collision")
    cfg = create_reaching_env_cfg(
        robot_cfg=get_kangaroo_hands_robot_cfg(),
        action_scale=KANGAROO_HANDS_ACTION_SCALE,
        viewer_body_name="pelvis_2_link",
        foot_friction_geom_names=geom_names,
        pos_x = (-0.6, -0.2),
        pos_y = (-0.6, 0.6),
        pos_z = (-0.6, 0.6),
        posture_jn = (
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
        ),
        posture_std = {
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
        },
    )
    return cfg