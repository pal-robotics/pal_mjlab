from mjlab.tasks.registry import register_mjlab_task

from pal_mjlab.tasks.supermimic.rl.runner import SuperMimicRunner
from pal_mjlab.tasks.supermimic.rl.velocity import VelocityRunner

from .env_cfgs import (
  pal_kangaroo_grippers_supermimic_env_cfg,
  pal_kangaroo_hands_supermimic_env_cfg,
  pal_kangaroo_hands_velocity_env_cfg,
)
from .rl_cfg import (
  PPO_NUM_ENVS,
  SUPERMIMIC_NUM_ENVS,
  pal_kangaroo_supermimic_ppo_runner_cfg,
  pal_kangaroo_supermimic_runner_cfg,
  pal_kangaroo_velocity_runner_cfg,
)

# Task name, env config and robot name per Kangaroo variant: 7-DoF arms with grippers, 5-DoF arms with Seed
# Robotics hands. Both log to one folder per algorithm (supermimic, supermimic_ppo); the default run name is
# the robot's (D49, D52).
VARIANTS = (
  ("Grippers", pal_kangaroo_grippers_supermimic_env_cfg, "grippers"),
  ("Hands", pal_kangaroo_hands_supermimic_env_cfg, "hands"),
)


def _supermimic_env_cfg(env_cfg):
  """Training env of the SuperMimic runner: terminal states are kept (auto_reset off)."""
  cfg = env_cfg()
  cfg.scene.num_envs = SUPERMIMIC_NUM_ENVS
  cfg.auto_reset = False
  return cfg


def _ppo_env_cfg(env_cfg):
  cfg = env_cfg()
  cfg.scene.num_envs = PPO_NUM_ENVS
  return cfg


def _runner_cfg(make, robot):
  cfg = make()
  cfg.run_name = robot
  return cfg


for name, env_cfg, robot in VARIANTS:
  # SuperMimic (Algorithm A.1).
  register_mjlab_task(
    task_id=f"Mjlab-SuperMimic-Flat-Pal-Kangaroo-{name}",
    env_cfg=_supermimic_env_cfg(env_cfg),
    play_env_cfg=env_cfg(play=True),
    rl_cfg=_runner_cfg(pal_kangaroo_supermimic_runner_cfg, robot),
    runner_cls=SuperMimicRunner,
  )

  # PPO-Sim baseline: the same env trained with model-free PPO.
  register_mjlab_task(
    task_id=f"Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-{name}",
    env_cfg=_ppo_env_cfg(env_cfg),
    play_env_cfg=env_cfg(play=True),
    rl_cfg=_runner_cfg(pal_kangaroo_supermimic_ppo_runner_cfg, robot),
  )

# The velocity interface, post-trained on a Hands motion policy (D56):
#   uv run train Mjlab-SuperMimic-Velocity-Flat-Pal-Kangaroo-Hands --agent.motion-checkpoint <run>/model_N.pt
register_mjlab_task(
  task_id="Mjlab-SuperMimic-Velocity-Flat-Pal-Kangaroo-Hands",
  env_cfg=_supermimic_env_cfg(pal_kangaroo_hands_velocity_env_cfg),
  play_env_cfg=pal_kangaroo_hands_velocity_env_cfg(play=True),
  rl_cfg=_runner_cfg(pal_kangaroo_velocity_runner_cfg, "hands"),
  runner_cls=VelocityRunner,
)
