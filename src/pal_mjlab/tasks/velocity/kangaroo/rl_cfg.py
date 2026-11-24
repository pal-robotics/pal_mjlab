"""RL configuration for PAL Robotics' KANGAROO velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def pal_kangaroo_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for PAL Kangaroo velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="kangaroo_velocity",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def pal_kangaroo_run_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """RL runner configuration for fine-tuning the street-running task.

  Same as the velocity runner (same experiment name, so ``--agent.resume`` finds the
  rough-terrain runs), with fewer iterations (counted from the resumed iteration):
  resuming at 15000 runs up to 27000, i.e. 2000 iterations at 3 m/s, 3000 at 4 m/s,
  4000 at 4.5 m/s and 3000 at 5 m/s. Override with ``--agent.max-iterations``.
  """
  cfg = pal_kangaroo_ppo_runner_cfg()
  cfg.max_iterations = 12_000
  return cfg
