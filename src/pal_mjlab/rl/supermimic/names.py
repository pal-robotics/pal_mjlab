"""Names of the env's terms and groups that SuperMimic reads (no other dependency, so
every module of the package can import it)."""

from dataclasses import dataclass

DEFAULT_CRITIC_GROUP = "critic"
"""The critic's observation group unless the runner's ``obs_groups`` say otherwise."""


@dataclass
class EnvNamesCfg:
  """Names of the env's terms and observation groups that SuperMimic reads.

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


def env_names_from_dict(train_cfg: dict) -> EnvNamesCfg:
  """The runner receives its config as a dict (``asdict``); rebuild the names from it."""
  names = dict(train_cfg.get("env_names") or {})
  if "failure_terms" in names:
    names["failure_terms"] = tuple(names["failure_terms"])
  return EnvNamesCfg(**names)


def critic_group(train_cfg: dict, names: EnvNamesCfg) -> str:
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
