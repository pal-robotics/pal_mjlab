from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import pal_reemc_flat_env_cfg, pal_reemc_rough_env_cfg
from .rl_cfg import pal_reemc_ppo_runner_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Pal-Reemc",
    env_cfg=pal_reemc_rough_env_cfg(),
    play_env_cfg=pal_reemc_rough_env_cfg(play=True),
    rl_cfg=pal_reemc_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Pal-Reemc",
    env_cfg=pal_reemc_flat_env_cfg(),
    play_env_cfg=pal_reemc_flat_env_cfg(play=True),
    rl_cfg=pal_reemc_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
