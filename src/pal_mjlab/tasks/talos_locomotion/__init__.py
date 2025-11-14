from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  PAL_TALOS_ROUGH_ENV_CFG,
  PAL_TALOS_FLAT_ENV_CFG,
)
from .rl_cfg import PalTalosPPORunnerCfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Pal-Talos",
  env_cfg=PAL_TALOS_ROUGH_ENV_CFG,
  rl_cfg=PalTalosPPORunnerCfg,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pal-Talos",
  env_cfg=PAL_TALOS_FLAT_ENV_CFG,
  rl_cfg=PalTalosPPORunnerCfg,
)