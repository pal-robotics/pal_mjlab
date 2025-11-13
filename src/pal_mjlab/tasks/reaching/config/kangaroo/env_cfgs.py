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
        pos_x = (-1.0, -0.2),
        pos_y = (-0.5, 0.5),
        pos_z = (-0.3, 0.3),
    )
    return cfg