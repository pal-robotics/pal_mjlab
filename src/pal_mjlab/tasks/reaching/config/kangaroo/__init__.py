from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  pal_kangaroo_flat_reaching_env_cfg,
  pal_kangaroo_hands_flat_reaching_env_cfg,
)
from .rl_cfg import pal_kangaroo_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Reaching-Pal-Kangaroo",
  env_cfg=pal_kangaroo_flat_reaching_env_cfg(),
  play_env_cfg=pal_kangaroo_flat_reaching_env_cfg(play=True),
  rl_cfg=pal_kangaroo_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Reaching-Pal-Kangaroo-Hands",
  env_cfg=pal_kangaroo_hands_flat_reaching_env_cfg(),
  play_env_cfg=pal_kangaroo_hands_flat_reaching_env_cfg(play=True),
  rl_cfg=pal_kangaroo_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)