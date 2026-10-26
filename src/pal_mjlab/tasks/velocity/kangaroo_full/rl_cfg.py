"""RL configuration for PAL Robotics' kangaroo_full velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from dataclasses import dataclass

@dataclass
class ProjectorRunnerCfg(RslRlOnPolicyRunnerCfg):
  projector_hidden_in: tuple[int, ...] = (256, 256)
  """Hidden widths of the residual MLP in head_in (env obs -> backbone obs)."""
  projector_hidden_out: tuple[int, ...] = (128,)
  """Hidden widths of the residual MLP in head_out (backbone action -> env action)."""


def pal_kangaroo_full_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for PAL kangaroo_full velocity task."""
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
    experiment_name="kangaroo_full_velocity",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=30_000,
  )

def pal_kangaroo_full_projector_ppo_runner_cfg() -> ProjectorRunnerCfg:
  """Projector heads on top of a frozen pretrained base policy.

  The backbone's architecture, normalizer and exploration std come from the
  base checkpoint (--pretrained-checkpoint), not from this cfg.
  """
  return ProjectorRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),  # only used to read obs/act dims; replaced by the backbone
      activation="elu",  # must match the base policy: the backbone is rebuilt with this
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.001,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="kangaroo_full_projector_velocity",
    save_interval=500,
    num_steps_per_env=24,
    max_iterations=500,
    projector_hidden_in=(256, 256),
    projector_hidden_out=(256, 256),
  )