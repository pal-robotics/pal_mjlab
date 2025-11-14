from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  PAL_TALOS_FLAT_TRACKING_ENV_CFG,
  PAL_TALOS_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG,
)
from .rl_cfg import PalTalosFlatPPORunnerCfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pal-Talos",
  env_cfg=PAL_TALOS_FLAT_TRACKING_ENV_CFG,
  rl_cfg=PalTalosFlatPPORunnerCfg,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Pal-Talos-No-State-Estimation",
  env_cfg=PAL_TALOS_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG,
  rl_cfg=PalTalosFlatPPORunnerCfg,
)