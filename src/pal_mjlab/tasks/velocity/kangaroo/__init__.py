from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  pal_kangaroo_flat_env_cfg,
  pal_kangaroo_grippers_flat_env_cfg,
  pal_kangaroo_grippers_rough_env_cfg,
  pal_kangaroo_hands_flat_env_cfg,
  pal_kangaroo_hands_rough_env_cfg,
  pal_kangaroo_rough_env_cfg,
  pal_kangaroo_pebbles_env_cfg,
  pal_kangaroo_grippers_pebbles_env_cfg,
  pal_kangaroo_hands_pebbles_env_cfg
)
from .rl_cfg import pal_kangaroo_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Pal-Kangaroo",
  env_cfg=pal_kangaroo_rough_env_cfg(),
  play_env_cfg=pal_kangaroo_rough_env_cfg(play=True),
  rl_cfg=pal_kangaroo_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pal-Kangaroo",
  env_cfg=pal_kangaroo_flat_env_cfg(),
  play_env_cfg=pal_kangaroo_flat_env_cfg(play=True),
  rl_cfg=pal_kangaroo_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Pal-Kangaroo-Hands",
  env_cfg=pal_kangaroo_hands_rough_env_cfg(),
  play_env_cfg=pal_kangaroo_hands_rough_env_cfg(play=True),
  rl_cfg=pal_kangaroo_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pal-Kangaroo-Hands",
  env_cfg=pal_kangaroo_hands_flat_env_cfg(),
  play_env_cfg=pal_kangaroo_hands_flat_env_cfg(play=True),
  rl_cfg=pal_kangaroo_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Pal-Kangaroo-Grippers",
  env_cfg=pal_kangaroo_grippers_rough_env_cfg(),
  play_env_cfg=pal_kangaroo_grippers_rough_env_cfg(play=True),
  rl_cfg=pal_kangaroo_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pal-Kangaroo-Grippers",
  env_cfg=pal_kangaroo_grippers_flat_env_cfg(),
  play_env_cfg=pal_kangaroo_grippers_flat_env_cfg(play=True),
  rl_cfg=pal_kangaroo_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Pebbles-Pal-Kangaroo",
  env_cfg=pal_kangaroo_pebbles_env_cfg(),
  play_env_cfg=pal_kangaroo_pebbles_env_cfg(play=True),
  rl_cfg=pal_kangaroo_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Pebbles-Pal-Kangaroo-Grippers",
  env_cfg=pal_kangaroo_grippers_pebbles_env_cfg(),
  play_env_cfg=pal_kangaroo_grippers_pebbles_env_cfg(play=True),
  rl_cfg=pal_kangaroo_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Pebbles-Pal-Kangaroo-Hands",
  env_cfg=pal_kangaroo_hands_pebbles_env_cfg(),
  play_env_cfg=pal_kangaroo_hands_pebbles_env_cfg(play=True),
  rl_cfg=pal_kangaroo_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
