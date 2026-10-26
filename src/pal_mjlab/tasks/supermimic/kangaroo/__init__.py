from mjlab.tasks.registry import register_mjlab_task

from pal_mjlab.tasks.supermimic.rl.runner import SuperMimicRunner

from .env_cfgs import (
  pal_kangaroo_grippers_supermimic_env_cfg,
  pal_kangaroo_hands_supermimic_env_cfg,
)
from .rl_cfg import (
  PPO_NUM_ENVS,
  SUPERMIMIC_NUM_ENVS,
  pal_kangaroo_supermimic_ppo_runner_cfg,
  pal_kangaroo_supermimic_runner_cfg,
)

# Task name, env config and log folder prefix per Kangaroo variant: 7-DoF arms with grippers,
# 5-DoF arms with Seed Robotics hands. The hands runs log apart (kangaroo_hands_supermimic*), so
# resuming by run name never picks up a checkpoint of the other robot.
VARIANTS = (
  ("Grippers", pal_kangaroo_grippers_supermimic_env_cfg, "kangaroo"),
  ("Hands", pal_kangaroo_hands_supermimic_env_cfg, "kangaroo_hands"),
)


def _supermimic_env_cfg(env_cfg, **kwargs):
  """Training env of the SuperMimic runner: terminal states are kept (auto_reset off)."""
  cfg = env_cfg(**kwargs)
  cfg.scene.num_envs = SUPERMIMIC_NUM_ENVS
  cfg.auto_reset = False
  return cfg


def _ppo_env_cfg(env_cfg, **kwargs):
  cfg = env_cfg(**kwargs)
  cfg.scene.num_envs = PPO_NUM_ENVS
  return cfg


def _runner_cfg(make, log_prefix):
  cfg = make()
  cfg.experiment_name = cfg.experiment_name.replace("kangaroo", log_prefix, 1)
  return cfg


for name, env_cfg, log_prefix in VARIANTS:
  # SuperMimic (Algorithm A.1).
  register_mjlab_task(
    task_id=f"Mjlab-SuperMimic-Flat-Pal-Kangaroo-{name}",
    env_cfg=_supermimic_env_cfg(env_cfg),
    play_env_cfg=env_cfg(play=True),
    rl_cfg=_runner_cfg(pal_kangaroo_supermimic_runner_cfg, log_prefix),
    runner_cls=SuperMimicRunner,
  )

  # Arms and waist at 1 Hz (D27), the gains of the PPO-Sim baseline run.
  register_mjlab_task(
    task_id=f"Mjlab-SuperMimic-Flat-Pal-Kangaroo-{name}-ArmWaist1Hz",
    env_cfg=_supermimic_env_cfg(env_cfg, arm_waist_natural_freq_hz=1.0),
    play_env_cfg=env_cfg(play=True, arm_waist_natural_freq_hz=1.0),
    rl_cfg=_runner_cfg(pal_kangaroo_supermimic_runner_cfg, log_prefix),
    runner_cls=SuperMimicRunner,
  )

  # PPO-Sim baseline: the SuperMimic env trained with model-free PPO.
  register_mjlab_task(
    task_id=f"Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-{name}",
    env_cfg=_ppo_env_cfg(env_cfg),
    play_env_cfg=env_cfg(play=True),
    rl_cfg=_runner_cfg(pal_kangaroo_supermimic_ppo_runner_cfg, log_prefix),
  )

  # Gain experiment (D27): arms and waist at 1 Hz (MAC/WBC's setting), legs at 3.5 Hz.
  register_mjlab_task(
    task_id=f"Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-{name}-ArmWaist1Hz",
    env_cfg=_ppo_env_cfg(env_cfg, arm_waist_natural_freq_hz=1.0),
    play_env_cfg=env_cfg(play=True, arm_waist_natural_freq_hz=1.0),
    rl_cfg=_runner_cfg(pal_kangaroo_supermimic_ppo_runner_cfg, log_prefix),
  )
