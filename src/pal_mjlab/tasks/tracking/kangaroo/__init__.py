from mjlab.tasks.registry import register_mjlab_task
from pal_mjlab.tasks.tracking.rl.encoder_history import (
    PalMotionTrackingOnPolicyRunner,
    PalStandardOnPolicyRunner,
)

from .env_cfgs import pal_kangaroo_flat_tracking_env_cfg
from .rl_cfg import pal_kangaroo_tracking_ppo_runner_cfg

# --- Default A-RMA Configuration ---
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pal-Kangaroo",
  env_cfg=pal_kangaroo_flat_tracking_env_cfg(),
  play_env_cfg=pal_kangaroo_flat_tracking_env_cfg(play=True),
  rl_cfg=pal_kangaroo_tracking_ppo_runner_cfg(),
  runner_cls=PalStandardOnPolicyRunner,
)

# --- No State Estimation ---
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pal-Kangaroo-No-State-Estimation",
  env_cfg=pal_kangaroo_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=pal_kangaroo_flat_tracking_env_cfg(
    has_state_estimation=False, play=True
  ),
  rl_cfg=pal_kangaroo_tracking_ppo_runner_cfg(),
  runner_cls=PalStandardOnPolicyRunner,
)

