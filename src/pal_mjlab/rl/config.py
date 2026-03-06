"""RSL-RL configuration."""

from dataclasses import dataclass, field
from mjlab.rl import RslRlBaseRunnerCfg, RslRlOnPolicyRunnerCfg

@dataclass
class RslRlFastSacAlgorithmCfg:
  """Configuration for the FastSAC algorithm."""

  class_name: str = "pal_mjlab.rl.fast_sac.fast_sac:FastSAC"
  """Fully qualified class name resolved by rsl-rl's resolve_callable."""

  # Learning rates
  critic_lr: float = 3e-4
  """Critic learning rate."""
  actor_lr: float = 3e-4
  """Actor learning rate."""
  alpha_lr: float = 3e-4
  """Entropy temperature learning rate."""

  # SAC hyper-parameters
  gamma: float = 0.97
  """Discount factor."""
  tau: float = 0.125
  """Soft target update coefficient."""
  batch_size: int = 8192
  """Global batch size (split across envs)."""
  learning_starts: int = 10
  """Number of env steps before training begins."""
  policy_frequency: int = 4
  """Actor update frequency relative to critic updates."""
  num_updates: int = 8
  """Number of gradient updates per env step (UTD ratio)."""
  target_entropy_ratio: float = 0.0
  """Target entropy as a ratio of -n_act."""
  alpha_init: float = 0.001
  """Initial entropy temperature."""
  use_autotune: bool = True
  """Whether to auto-tune the entropy temperature."""

  # Network architecture
  actor_hidden_dim: int = 512
  """Actor hidden layer base width (halving: 512->256->128)."""
  critic_hidden_dim: int = 768
  """Critic hidden layer base width (halving: 768->384->192)."""
  num_atoms: int = 101
  """Number of atoms for the distributional critic (C51)."""
  v_min: float = -20.0
  """Minimum support value for C51."""
  v_max: float = 20.0
  """Maximum support value for C51."""
  num_q_networks: int = 2
  """Number of Q-networks in the critic ensemble."""
  use_layer_norm: bool = True
  """Whether to use LayerNorm in actor and critic."""
  use_tanh: bool = True
  """Whether to use tanh squashing on actor output."""
  log_std_min: float = -5.0
  """Minimum log standard deviation for the actor."""
  log_std_max: float = 0.0
  """Maximum log standard deviation for the actor."""

  # Replay buffer
  buffer_size: int = 1024
  """Per-env replay buffer capacity."""
  num_steps: int = 1
  """Number of n-step returns."""

  # Observation normalization
  obs_normalization: bool = True
  """Whether to use empirical observation normalization."""

  # Optimization
  max_grad_norm: float = 0.0
  """Maximum gradient norm (0 = no clipping)."""
  weight_decay: float = 0.001
  """AdamW weight decay."""

  # Performance
  compile: bool = True
  """Whether to use torch.compile for update functions."""
  amp: bool = True
  """Whether to use automatic mixed precision."""
  amp_dtype: str = "bf16"
  """AMP dtype: 'bf16' or 'fp16'."""
  rnd_cfg: dict | None = None

@dataclass
class RslRlFastSacRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner configuration for FastSAC.

  FastSAC is off-policy, so we use num_steps_per_env=1 and let the
  algorithm handle replay internally.
  """

  class_name: str = "FastSACRunner"
  """
  Runner class name. 
  """
  num_steps_per_env: int = 1
  """Must be 1 for off-policy SAC."""
  algorithm: RslRlFastSacAlgorithmCfg = field(default_factory=RslRlFastSacAlgorithmCfg)
  """The FastSAC algorithm configuration."""