from mjlab.tasks.registry import register_mjlab_task

from pal_mjlab.tasks.supermimic.rl.runner import SuperMimicRunner

from .env_cfgs import pal_kangaroo_grippers_supermimic_env_cfg
from .rl_cfg import pal_kangaroo_supermimic_runner_cfg

register_mjlab_task(
  task_id="Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers",
  env_cfg=pal_kangaroo_grippers_supermimic_env_cfg(),
  play_env_cfg=pal_kangaroo_grippers_supermimic_env_cfg(play=True),
  rl_cfg=pal_kangaroo_supermimic_runner_cfg(),
  runner_cls=SuperMimicRunner,
)
