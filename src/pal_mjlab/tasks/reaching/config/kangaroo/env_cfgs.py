"""PAL Robotics KANGAROO reaching environment configurations."""

from pal_mjlab.robots import (
    KANGAROO_ACTION_SCALE,
    get_kangaroo_robot_cfg,
)
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
    )
    return cfg