"""RSL-RL configuration."""

from dataclasses import dataclass, field
from typing import Any, Literal, Tuple


@dataclass
class SupermimicEncoderModelCfg:
  """Config for a single encoder model."""

@dataclass
class SupermimicDecoderModelCfg:
  """Config for a single encoder model."""

@dataclass
class SupermimicCriticModelCfg:
  """Config for a single encoder model."""


@dataclass
class SupermimicAlgorithmCfg:
  """Config for the Supermimic learning loop."""


@dataclass
class SupermimicBaseRunnerCfg:
  seed: int = 42
  """The seed for the experiment. Default is 42."""
  num_steps_per_env: int = 24
  """The number of steps per environment update."""
  max_iterations: int = 300
  """The maximum number of iterations."""
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"actor": ("actor",), "critic": ("critic",)},
  )
  save_interval: int = 50
  """The number of iterations between saves."""
  experiment_name: str = "exp1"
  """Directory name used to group runs under ``{log_root}/{experiment_name}/``.
  The log root defaults to ``logs/rsl_rl`` and can be overridden with
  ``--log-root`` on the CLI."""
  run_name: str = ""
  """Optional label appended to the timestamped run directory
  (e.g. ``2025-01-27_14-30-00_{run_name}``). Also becomes the
  display name for the run in wandb."""
  logger: Literal["wandb", "tensorboard"] = "wandb"
  """The logger to use. Default is wandb."""
  wandb_project: str = "mjlab"
  """The wandb project name."""
  wandb_tags: Tuple[str, ...] = ()
  """Tags for the wandb run. Default is empty tuple."""
  resume: bool = False
  """Whether to resume the experiment. Default is False."""
  load_run: str = ".*"
  """The run directory to load. Default is ".*" which means all runs. If regex
  expression, the latest (alphabetical order) matching run will be loaded.
  """
  load_checkpoint: str = "model_.*.pt"
  """The checkpoint file to load. Default is "model_.*.pt" (all). If regex expression,
  the latest (alphabetical order) matching file will be loaded.
  """
  clip_actions: float | None = None
  """The clipping range for action values. If None (default), no clipping is applied."""
  upload_model: bool = True
  """Whether to upload model files (.pt, .onnx) to W&B on save. Set to
  False to keep metric logging but avoid storage usage. Default is True."""


@dataclass
class SupermimicRunnerCfg(SupermimicBaseRunnerCfg):
  class_name: str = "OnPolicyRunner"
  """The runner class name. Default is OnPolicyRunner."""
  encoders: dict[str, SupermimicEncoderModelCfg] = field(
    default_factory=lambda: {
      "encoder": SupermimicEncoderModelCfg(
      ),
    }
  )
  decoder: SupermimicDecoderModelCfg = field(
    default_factory=lambda: SupermimicDecoderModelCfg(
    )
  )
  """The actor configuration."""
  critic: SupermimicCriticModelCfg = field(default_factory=SupermimicCriticModelCfg)
  """The critic configuration."""
  algorithm: SupermimicAlgorithmCfg = field(default_factory=SupermimicAlgorithmCfg)
  """The algorithm configuration."""