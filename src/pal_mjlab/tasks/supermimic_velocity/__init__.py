"""PAL Kangaroo flat-ground velocity tracking trained with SuperMimic (roadmap M5 v1)."""

from mjlab.tasks.registry import register_mjlab_task

from pal_mjlab.rl.supermimic import SuperMimicRunner

from .env_cfgs import pal_kangaroo_flat_supermimic_env_cfg
from .rl_cfg import pal_kangaroo_velocity_supermimic_runner_cfg

register_mjlab_task(
  task_id="Mjlab-SuperMimic-Velocity-Flat-Pal-Kangaroo",
  env_cfg=pal_kangaroo_flat_supermimic_env_cfg(),
  play_env_cfg=pal_kangaroo_flat_supermimic_env_cfg(play=True),
  rl_cfg=pal_kangaroo_velocity_supermimic_runner_cfg(),
  runner_cls=SuperMimicRunner,
)
