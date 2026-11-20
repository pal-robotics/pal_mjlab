"""Configuration of the SuperMimic runner."""

from dataclasses import dataclass, field

from mjlab.rl import RslRlBaseRunnerCfg

from .names import DEFAULT_CRITIC_GROUP, EnvNamesCfg
from .supermimic import SuperMimicCfg


@dataclass
class SuperMimicRunnerCfg(RslRlBaseRunnerCfg):
  algorithm: SuperMimicCfg = field(default_factory=SuperMimicCfg)
  env_names: EnvNamesCfg = field(default_factory=EnvNamesCfg)
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {
      "actor": EnvNamesCfg().actor_groups,
      "critic": (DEFAULT_CRITIC_GROUP,),
    },
  )
  action_noise: float = 0.1
  """Std of the Gaussian collection noise on each action dimension."""
  replay_steps: int = 1000
  """Replay buffer size per env."""
  env_steps_per_iteration: float = 2.5
  """Vector env steps per iteration: update-to-data ratio 0.4."""
  warmup_steps: int = 32
  """Env steps collected before the first update."""
  compile: bool = True
  """Compile the kinematics, the imagination step and the world model's window loss."""
  log_interval: int = 10
