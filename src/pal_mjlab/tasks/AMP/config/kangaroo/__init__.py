from mjlab.tasks.registry import register_mjlab_task
from pal_mjlab.tasks.AMP.rl import AmpOnPolicyRunner

from .env_cfgs import kangaroo_flat_amp_env_cfg
from .rl_cfg import kangaroo_amp_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-AMP-Flat-Pal-Kangaroo",
  env_cfg=kangaroo_flat_amp_env_cfg(),
  play_env_cfg=kangaroo_flat_amp_env_cfg(play=True),
  rl_cfg=kangaroo_amp_ppo_runner_cfg(),
  runner_cls=AmpOnPolicyRunner,
)
