"""RL configuration for PAL Robotics' KANGAROO velocity task."""

from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
    RslRlPpoAlgorithmCfg,
)

from pal_mjlab.rl import (
  RslRlFastSacAlgorithmCfg,
  RslRlFastSacRunnerCfg,
)

def pal_kangaroo_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for PAL Kangaroo velocity task."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            stochastic=True,
            init_noise_std=1.0,
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            stochastic=False,
            init_noise_std=1.0,
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
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=30_000,
    )


def pal_kangaroo_fast_sac_runner_cfg() -> RslRlFastSacRunnerCfg:
  """Create FastSAC runner configuration for PAL Robotics's KANGAROO velocity task."""
  return RslRlFastSacRunnerCfg(
    algorithm=RslRlFastSacAlgorithmCfg(
      critic_lr=3e-4,
      actor_lr=3e-4,
      alpha_lr=3e-4,
      gamma=0.97,
      tau=0.125,
      batch_size=8192,
      learning_starts=10,
      policy_frequency=4,
      num_updates=8,
      target_entropy_ratio=0.0,
      alpha_init=0.001,
      use_autotune=True,
      actor_hidden_dim=512,
      critic_hidden_dim=768,
      num_atoms=101,
      v_min=-20.0,
      v_max=20.0,
      num_q_networks=2,
      use_layer_norm=True,
      use_tanh=True,
      log_std_min=-5.0,
      log_std_max=0.0,
      buffer_size=1024,
      num_steps=1,
      obs_normalization=True,
      max_grad_norm=0.0,
      weight_decay=0.001,
      compile=True,
      amp=True,
      amp_dtype="bf16",
    ),
    experiment_name="kangaroo_velocity_fast_sac",
    save_interval=1000,
    num_steps_per_env=1,
    max_iterations=50_000,
  )