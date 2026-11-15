from mjlab.tasks.registry import register_mjlab_task

from pal_mjlab.tasks.stand_up.rl import StandUpOnPolicyRunner

from .env_cfgs import pal_kangaroo_stand_up_env_cfg
from .rl_cfg import pal_kangaroo_stand_up_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-StandUp-Flat-Pal-Kangaroo",
  env_cfg=pal_kangaroo_stand_up_env_cfg(),
  play_env_cfg=pal_kangaroo_stand_up_env_cfg(play=True),
  rl_cfg=pal_kangaroo_stand_up_ppo_runner_cfg(),
  runner_cls=StandUpOnPolicyRunner,
)
