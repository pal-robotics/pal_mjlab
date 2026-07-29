from mjlab.tasks.registry import register_mjlab_task

from pal_mjlab.tasks.box_lifting.rl import BoxLiftingOnPolicyRunner

from .env_cfgs import (
  pal_kangaroo_box_lifting_flat_env_cfg,
  pal_kangaroo_box_lifting_rough_env_cfg,
)
from .rl_cfg import pal_kangaroo_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Box-Lifting-Rough-Pal-Kangaroo",
  env_cfg=pal_kangaroo_box_lifting_rough_env_cfg(),
  play_env_cfg=pal_kangaroo_box_lifting_rough_env_cfg(play=True),
  rl_cfg=pal_kangaroo_ppo_runner_cfg(),
  runner_cls=BoxLiftingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Box-Lifting-Flat-Pal-Kangaroo",
  env_cfg=pal_kangaroo_box_lifting_flat_env_cfg(),
  play_env_cfg=pal_kangaroo_box_lifting_flat_env_cfg(play=True),
  rl_cfg=pal_kangaroo_ppo_runner_cfg(),
  runner_cls=BoxLiftingOnPolicyRunner,
)
