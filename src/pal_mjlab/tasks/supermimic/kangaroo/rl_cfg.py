"""Runner configurations: SuperMimic (Table A.5) and the PPO-Sim baseline (Section 4.1).

PPO-Sim shares SuperMimic's motions, observations, reward, termination, domain
randomization, actor and critic network sizes and evaluation protocol; it uses
BeyondMimic's PPO hyperparameters (as in mjlab's G1 tracking task) and 8192 parallel
environments per GPU, twice SuperMimic's.
"""

from dataclasses import dataclass

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from pal_mjlab.tasks.supermimic.rl.runner import SuperMimicRunnerCfg

PPO_NUM_ENVS = 8192
SUPERMIMIC_NUM_ENVS = 4096


def pal_kangaroo_supermimic_runner_cfg() -> SuperMimicRunnerCfg:
  """The paper's hyperparameters (Table A.5); see ``rl/supermimic.py`` for the values."""
  return SuperMimicRunnerCfg(
    experiment_name="kangaroo_supermimic",
    logger="tensorboard",
    max_iterations=320_000,
    save_interval=2000,
    obs_groups={"actor": ("goal", "proprio"), "critic": ("critic",)},
  )


@dataclass
class EncoderDecoderModelCfg(RslRlModelCfg):
  """Actor of Table A.4: goal encoder + decoder on (z, proprioceptive history)."""

  encoder_dims: tuple[int, ...] = (2048, 1024, 512, 256)
  latent_dim: int = 32
  goal_group: str = "goal"
  class_name: str = "pal_mjlab.tasks.supermimic.rl.models:EncoderDecoderModel"


def pal_kangaroo_supermimic_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=EncoderDecoderModelCfg(
      hidden_dims=(2048, 1024, 512, 256),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128),
      activation="elu",
      obs_normalization=True,
    ),
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
    obs_groups={"actor": ("goal", "proprio"), "critic": ("critic",)},
    experiment_name="kangaroo_supermimic_ppo",
    logger="tensorboard",
    save_interval=200,
    num_steps_per_env=24,
    max_iterations=1_000_000,  # bounded by wall-clock (gpu-run --max), see the docs
  )
