from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from pal_mjlab.tasks.registry import register_pal_mjlab_task
from pal_mjlab.rl.fast_sac import FastSACRunner

from .env_cfgs import (
    pal_kangaroo_full_reaching_env_cfg,
)
from .rl_cfg import pal_kangaroo_ppo_runner_cfg, pal_kangaroo_fast_sac_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Reaching-Pal-Kangaroo-Full",
    env_cfg=pal_kangaroo_full_reaching_env_cfg(),
    play_env_cfg=pal_kangaroo_full_reaching_env_cfg(play=True),
    rl_cfg=pal_kangaroo_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_pal_mjlab_task(
    task_id="Mjlab-Reaching-FastSAC-Pal-Kangaroo-Full",
    env_cfg=pal_kangaroo_full_reaching_env_cfg(),
    play_env_cfg=pal_kangaroo_full_reaching_env_cfg(play=True),
    rl_cfg=pal_kangaroo_fast_sac_runner_cfg(),
    runner_cls=FastSACRunner,
)