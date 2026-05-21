"""RL configuration for PAL Robotics' KANGAROO velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from pal_mjlab.rl.flash_sac import FlashSACRunnerCfg


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


def pal_kangaroo_flashsac_runner_cfg() -> FlashSACRunnerCfg:
    """Create FlashSAC runner configuration for PAL Kangaroo velocity task."""
    return FlashSACRunnerCfg(
        experiment_name="kangaroo_velocity",
        asymmetric_observation=True,
        # Network depth matches (512, 256, 128) ladder
        actor_num_blocks=3,
        actor_hidden_dim=512,
        critic_num_blocks=3,
        critic_hidden_dim=512,
        # Tuned for locomotion
        gamma=0.99,
        n_step=3,
        normalize_reward=True,
        temp_initial_value=1.0,
        temp_target_sigma=0.5,
        buffer_min_length=10_000,
        buffer_max_length=1_000_000,
        sample_batch_size=256,
        actor_update_period=2,
        critic_target_update_tau=0.005,
    )