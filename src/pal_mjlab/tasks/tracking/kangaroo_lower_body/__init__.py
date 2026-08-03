from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import pal_kangaroo_lower_body_flat_tracking_env_cfg
from .rl_cfg import pal_kangaroo_lower_body_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pal-Kangaroo_Lower_Body",
  env_cfg=pal_kangaroo_lower_body_flat_tracking_env_cfg(),
  play_env_cfg=pal_kangaroo_lower_body_flat_tracking_env_cfg(play=True),
  rl_cfg=pal_kangaroo_lower_body_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pal-Kangaroo-Lower-Body-No-State-Estimation",
  env_cfg=pal_kangaroo_lower_body_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=pal_kangaroo_lower_body_flat_tracking_env_cfg(
    has_state_estimation=False, play=True
  ),
  rl_cfg=pal_kangaroo_lower_body_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
