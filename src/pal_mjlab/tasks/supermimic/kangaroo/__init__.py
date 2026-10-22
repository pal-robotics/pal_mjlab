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

# Gain experiment (D27): arms and waist at 1 Hz (MAC/WBC's setting), legs at 3.5 Hz.
_ppo_1hz_env_cfg = pal_kangaroo_grippers_supermimic_env_cfg(
  arm_waist_natural_freq_hz=1.0
)
_ppo_1hz_env_cfg.scene.num_envs = PPO_NUM_ENVS

register_mjlab_task(
  task_id="Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Grippers-ArmWaist1Hz",
  env_cfg=_ppo_1hz_env_cfg,
  play_env_cfg=pal_kangaroo_grippers_supermimic_env_cfg(
    play=True, arm_waist_natural_freq_hz=1.0
  ),
  rl_cfg=pal_kangaroo_supermimic_ppo_runner_cfg(),
)
