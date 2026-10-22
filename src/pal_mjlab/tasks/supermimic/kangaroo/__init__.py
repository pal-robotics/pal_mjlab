from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import pal_kangaroo_grippers_supermimic_env_cfg
from .rl_cfg import PPO_NUM_ENVS, pal_kangaroo_supermimic_ppo_runner_cfg

# PPO-Sim baseline: the SuperMimic env trained with model-free PPO.
_ppo_env_cfg = pal_kangaroo_grippers_supermimic_env_cfg()
_ppo_env_cfg.scene.num_envs = PPO_NUM_ENVS

register_mjlab_task(
  task_id="Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Grippers",
  env_cfg=_ppo_env_cfg,
  play_env_cfg=pal_kangaroo_grippers_supermimic_env_cfg(play=True),
  rl_cfg=pal_kangaroo_supermimic_ppo_runner_cfg(),
)
