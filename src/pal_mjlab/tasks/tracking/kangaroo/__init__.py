from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from pal_mjlab.tasks.tracking.obstacle_onnx import ObstacleDropMotionTrackingRunner

from .env_cfgs import (
  pal_kangaroo_flat_tracking_env_cfg,
  pal_kangaroo_hands_flat_env_cfg,
  pal_kangaroo_obstacle_tracking_env_cfg,
)
from .rl_cfg import pal_kangaroo_tracking_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pal-Kangaroo",
  env_cfg=pal_kangaroo_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=pal_kangaroo_flat_tracking_env_cfg(
    has_state_estimation=False, play=True
  ),
  rl_cfg=pal_kangaroo_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Box-Pal-Kangaroo",
  env_cfg=pal_kangaroo_obstacle_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=pal_kangaroo_obstacle_tracking_env_cfg(
    has_state_estimation=False, play=True
  ),
  rl_cfg=pal_kangaroo_tracking_ppo_runner_cfg(),
  runner_cls=ObstacleDropMotionTrackingRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pal-Kangaroo-Hands",
  env_cfg=pal_kangaroo_hands_flat_env_cfg(has_state_estimation=False),
  play_env_cfg=pal_kangaroo_hands_flat_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=pal_kangaroo_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
