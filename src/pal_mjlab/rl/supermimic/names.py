"""What SuperMimic reads from the env, configured in the env itself.

mjlab evaluates observation terms while it builds the env, before any runner exists, so
the env's SuperMimic configuration lives in the env config (``SuperMimicEnvCfg``), not in
the runner's. The runner and the heads read it from there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mjlab.envs import ManagerBasedRlEnvCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

DEFAULT_CRITIC_GROUP = "critic"
"""The critic's observation group unless the runner's ``obs_groups`` say otherwise."""


@dataclass
class ImaginationCfg:
  """The env's terms and observation groups that SuperMimic reads, by name.

  Only what the env knows by name is configured here. Observation terms are recognized by
  their function instead (``heads.py``), and the actor and critic groups are the runner's
  ``obs_groups``: ``{"actor": (goal_group, proprio_group), "critic": (<critic group>,)}``.
  """

  command: str = "motion"
  """Command term holding the reference and the robot's state."""
  failure_terms: tuple[str, ...] = ("anchor_failure",)
  """Termination terms that label a transition as a failure (any of them)."""
  goal_group: str = "goal"
  """Actor group of the goal, encoded by the actor's goal encoder."""
  proprio_group: str = "proprio"
  """Actor group of the proprioception, predicted by the analytic observation head."""
  action_term: str = "joint_pos"
  """Joint position action term the policy drives."""

  @property
  def actor_groups(self) -> tuple[str, str]:
    """The actor's input groups, in the order the policy concatenates them."""
    return (self.goal_group, self.proprio_group)


@dataclass(kw_only=True)
class SuperMimicEnvCfg(ManagerBasedRlEnvCfg):
  """An mjlab env config with SuperMimic's configuration."""

  imagination: ImaginationCfg = field(default_factory=ImaginationCfg)


def imagination_cfg(env: ManagerBasedRlEnv) -> ImaginationCfg:
  """The env's SuperMimic configuration (its config must be a ``SuperMimicEnvCfg``)."""
  cfg = getattr(env.cfg, "imagination", None)
  if not isinstance(cfg, ImaginationCfg):
    raise TypeError(
      f"SuperMimic needs a SuperMimicEnvCfg env config, got {type(env.cfg).__name__}."
    )
  return cfg


def critic_group(train_cfg: dict, names: ImaginationCfg) -> str:
  """The critic's single observation group, after checking ``obs_groups`` agrees with
  ``names`` (the policy concatenates goal then proprioception)."""
  groups = train_cfg["obs_groups"]
  actor, critic = tuple(groups["actor"]), tuple(groups["critic"])
  if actor != names.actor_groups:
    raise ValueError(
      f"obs_groups['actor'] must be (goal_group, proprio_group) = {names.actor_groups}, "
      f"got {actor}."
    )
  if len(critic) != 1:
    raise ValueError(f"obs_groups['critic'] must be a single group, got {critic}.")
  return critic[0]
