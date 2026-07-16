from mjlab.tasks.registry import register_mjlab_task
from pal_mjlab.tasks.grippers_manipulation.rl import HandManipulationOnPolicyRunner

from .env_cfgs import (
  pal_kangaroo_grippers_manipulation_flat_env_cfg,
)
from .rl_cfg import pal_kangaroo_ppo_runner_cfg


register_mjlab_task(
  task_id="Mjlab-Manipulation-Flat-Pal-Kangaroo-Hands",
  env_cfg=pal_kangaroo_grippers_manipulation_flat_env_cfg(),
  play_env_cfg=pal_kangaroo_grippers_manipulation_flat_env_cfg(play=True),
  rl_cfg=pal_kangaroo_ppo_runner_cfg(),
  runner_cls=HandManipulationOnPolicyRunner,
)