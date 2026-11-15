"""RL configuration for the PAL Robotics KANGAROO safe-fall task."""

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class SafeFallRunnerCfg(RslRlOnPolicyRunnerCfg):
  init_actor_checkpoint: str | None = None
  """rsl_rl checkpoint (e.g. a walking policy) to initialize the actor from.
  None trains from scratch."""
  init_action_std: float = 0.5
  """Action std after loading ``init_actor_checkpoint``."""
  actor_freeze_iterations: int = 0
  """Initial iterations in which only the critic is updated."""


def pal_kangaroo_safe_fall_ppo_runner_cfg() -> SafeFallRunnerCfg:
  """Same networks as the velocity task, so a walking actor can be loaded."""
  return SafeFallRunnerCfg(
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
      # Half the velocity task's: a robot at rest loses little by acting
      # noisily, and with 0.01 the exploration noise kept growing.
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="kangaroo_safe_fall",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=4_000,
  )
