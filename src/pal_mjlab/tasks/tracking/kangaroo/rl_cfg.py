"""RL configuration for PAL Robotics' Kangaroo motion imitation task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def pal_kangaroo_tracking_ppo_runner_cfg(
  use_history_encoder: bool = False,
) -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for PAL Kangaroo tracking task."""
  actor_kwargs = {
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
      "hidden_dims": (512, 256, 128),
      "activation": "elu",
      "obs_normalization": True,
  }

  if use_history_encoder:
    actor_kwargs["class_name"] = "pal_mjlab.tasks.tracking.kangaroo.custom_models:HistoryEncoderModel"

  return RslRlOnPolicyRunnerCfg(
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
    max_iterations=30_000,
  )
