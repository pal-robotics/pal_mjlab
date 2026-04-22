from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from pal_mjlab.tasks.tracking.rl.encoder_history import PalMotionTrackingOnPolicyRunner

from .env_cfgs import pal_kangaroo_flat_tracking_env_cfg
from .rl_cfg import pal_kangaroo_tracking_ppo_runner_cfg

# --- Default (No History) ---
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pal-Kangaroo",
  env_cfg=pal_kangaroo_flat_tracking_env_cfg(use_history=False),
  play_env_cfg=pal_kangaroo_flat_tracking_env_cfg(play=True, use_history=False),
  rl_cfg=pal_kangaroo_tracking_ppo_runner_cfg(use_history_encoder=False),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# --- History Encoder ---
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pal-Kangaroo-History",
  env_cfg=pal_kangaroo_flat_tracking_env_cfg(use_history=True),
  play_env_cfg=pal_kangaroo_flat_tracking_env_cfg(play=True, use_history=True),
  rl_cfg=pal_kangaroo_tracking_ppo_runner_cfg(use_history_encoder=True),
  runner_cls=PalMotionTrackingOnPolicyRunner,
)

# --- No State Estimation (No History) ---
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pal-Kangaroo-No-State-Estimation",
  env_cfg=pal_kangaroo_flat_tracking_env_cfg(has_state_estimation=False, use_history=False),
  play_env_cfg=pal_kangaroo_flat_tracking_env_cfg(
    has_state_estimation=False, play=True, use_history=False
  ),
  rl_cfg=pal_kangaroo_tracking_ppo_runner_cfg(use_history_encoder=False),
  runner_cls=MotionTrackingOnPolicyRunner,
)

# --- No State Estimation (With History) ---
register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pal-Kangaroo-No-State-Estimation-History",
  env_cfg=pal_kangaroo_flat_tracking_env_cfg(has_state_estimation=False, use_history=True),
  play_env_cfg=pal_kangaroo_flat_tracking_env_cfg(
    has_state_estimation=False, play=True, use_history=True
  ),
  rl_cfg=pal_kangaroo_tracking_ppo_runner_cfg(use_history_encoder=True),
  runner_cls=PalMotionTrackingOnPolicyRunner,
)
