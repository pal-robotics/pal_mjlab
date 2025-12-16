from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
    pal_tiago_reaching_env_cfg,
)
from .rl_cfg import pal_tiago_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Reaching-Pal-Tiago",
    env_cfg=pal_tiago_reaching_env_cfg(),
    play_env_cfg=pal_tiago_reaching_env_cfg(play=True),
    rl_cfg=pal_tiago_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)


