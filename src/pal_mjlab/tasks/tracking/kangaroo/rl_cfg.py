"""RL configuration for PAL Robotics' Kangaroo motion imitation task."""

import os

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


from dataclasses import dataclass

@dataclass
class ArmaOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration specific to the A-RMA 3-phase training process."""
    p1_iterations: int = 25000
    """Total iterations for Phase 1 (privileged PPO)."""
    p2_iterations: int = 5000
    """Total iterations for Phase 2 (DAgger TCN)."""
    p3_iterations: int = 10000
    """Total iterations for Phase 3 (PPO finetune)."""
    p2_batch_size: int = 8192
    """Batch size for TCN supervised learning."""
    p2_learning_rate: float = 1e-3
    """Learning rate for TCN parameter updates."""


def pal_kangaroo_tracking_ppo_runner_cfg() -> ArmaOnPolicyRunnerCfg:
  """Create RL runner configuration for PAL Kangaroo tracking task."""
  actor_kwargs = {
      "class_name": "pal_mjlab.tasks.tracking.kangaroo.custom_models:ArmaActorModel",
      "hidden_dims": (512, 256, 128),
      "activation": "elu",
      "obs_normalization": True,
      "distribution_cfg": {
          "class_name": "GaussianDistribution",
          "init_std": 1.0,
          "std_type": "scalar",
      },
  }
  critic_kwargs = {
      "class_name": "pal_mjlab.tasks.tracking.kangaroo.custom_models:ArmaCriticModel",
      "hidden_dims": (512, 256, 128),
      "activation": "elu",
      "obs_normalization": True,
  }

  return ArmaOnPolicyRunnerCfg(
    actor=RslRlModelCfg(**actor_kwargs),
    critic=RslRlModelCfg(**critic_kwargs),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
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
    experiment_name="kangaroo_tracking",
    save_interval=500,
    num_steps_per_env=24,
    # max_iterations becomes the fallback/meta metric. We'll use p1,p2,p3 in the runner.
    max_iterations=30_000, 
  )


