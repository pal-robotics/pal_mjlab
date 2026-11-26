from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  pal_kangaroo_full_flat_env_cfg,
  # pal_kangaroo_full_grippers_flat_env_cfg,
  # pal_kangaroo_full_grippers_rough_env_cfg,
  # pal_kangaroo_full_hands_flat_env_cfg,
  # pal_kangaroo_full_hands_rough_env_cfg,
  pal_kangaroo_full_rough_env_cfg,
)
from .rl_cfg import pal_kangaroo_full_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Pal-Kangaroo-Full",
  env_cfg=pal_kangaroo_full_rough_env_cfg(),
  play_env_cfg=pal_kangaroo_full_rough_env_cfg(play=True),
  rl_cfg=pal_kangaroo_full_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Pal-Kangaroo-Full",
  env_cfg=pal_kangaroo_full_flat_env_cfg(),
  play_env_cfg=pal_kangaroo_full_flat_env_cfg(play=True),
  rl_cfg=pal_kangaroo_full_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
"""
register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Pal-Kangaroo-Full-Hands",
    env_cfg=pal_kangaroo_full_hands_rough_env_cfg(),
    play_env_cfg=pal_kangaroo_full_hands_rough_env_cfg(play=True),
    rl_cfg=pal_kangaroo_full_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Pal-Kangaroo-Full-Hands",
    env_cfg=pal_kangaroo_full_hands_flat_env_cfg(),
    play_env_cfg=pal_kangaroo_full_hands_flat_env_cfg(play=True),
    rl_cfg=pal_kangaroo_full_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Pal-Kangaroo-Full-Grippers",
    env_cfg=pal_kangaroo_full_grippers_rough_env_cfg(),
    play_env_cfg=pal_kangaroo_full_grippers_rough_env_cfg(play=True),
    rl_cfg=pal_kangaroo_full_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Pal-Kangaroo-Full-Grippers",
    env_cfg=pal_kangaroo_full_grippers_flat_env_cfg(),
    play_env_cfg=pal_kangaroo_full_grippers_flat_env_cfg(play=True),
    rl_cfg=pal_kangaroo_full_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
"""
