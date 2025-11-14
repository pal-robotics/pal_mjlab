from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  PAL_KANGAROO_ROUGH_ENV_CFG,
  PAL_KANGAROO_FLAT_ENV_CFG,
)
from .rl_cfg import PalKangarooPPORunnerCfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Pal-Kangaroo",
  env_cfg=PAL_KANGAROO_ROUGH_ENV_CFG,
  rl_cfg=PalKangarooPPORunnerCfg,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pal-Kangaroo",
  env_cfg=PAL_KANGAROO_FLAT_ENV_CFG,
  rl_cfg=PalKangarooPPORunnerCfg,
)