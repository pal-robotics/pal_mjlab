from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  PAL_KANGAROO_ENV_CFG,
  PAL_KANGAROO_HANDS_ENV_CFG,
)
from .rl_cfg import PAL_KANGAROO_PPO_RUNNER_CFG

register_mjlab_task(
  task_id="Mjlab-Reaching-Pal-Kangaroo",
  env_cfg=PAL_KANGAROO_ENV_CFG,
  rl_cfg=PAL_KANGAROO_PPO_RUNNER_CFG,
)

register_mjlab_task(
  task_id="Mjlab-Reaching-Pal-Kangaroo-Hands",
  env_cfg=PAL_KANGAROO_HANDS_ENV_CFG,
  rl_cfg=PAL_KANGAROO_PPO_RUNNER_CFG,
)