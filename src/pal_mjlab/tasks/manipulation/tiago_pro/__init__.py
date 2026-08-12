from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import lift_env_cfg
from .rl_cfg import lift_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0",
  env_cfg=lift_env_cfg(),
  play_env_cfg=lift_env_cfg(play=True),
  rl_cfg=lift_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
