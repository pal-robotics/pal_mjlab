"""SuperMimic runner configuration for the Kangaroo velocity task."""

from pal_mjlab.rl.supermimic import SuperMimicRunnerCfg
from pal_mjlab.rl.supermimic.supermimic import CriticUpdateCfg, SuperMimicCfg
from pal_mjlab.tasks.supermimic_velocity.env_cfgs import (
  AIR_TIME_WEIGHT,
  TRACK_LIN_FINE_WEIGHT,
  TRACK_LIN_WEIGHT,
)

# The largest return the rewards allow: their positive weights (linear tracking, wide and
# fine, yaw tracking 2, upright 2, pose 1, air time x 2 feet; each term <= 1 per foot)
# x dt 0.02, over 1 / (1 - gamma) = 100 steps. Overriding a positive weight on the command
# line? Also pass --agent.algorithm.critic.max-return.
MAX_RETURN = (
  (TRACK_LIN_WEIGHT + TRACK_LIN_FINE_WEIGHT + 2.0 + 2.0 + 1.0 + 2 * AIR_TIME_WEIGHT)
  * 0.02
  / (1.0 - 0.99)
)


def pal_kangaroo_velocity_supermimic_runner_cfg() -> SuperMimicRunnerCfg:
  """The paper's hyperparameters (the defaults of ``rl/supermimic``)."""
  return SuperMimicRunnerCfg(
    experiment_name="kangaroo_supermimic_velocity",
    logger="tensorboard",
    max_iterations=50_000,
    save_interval=1000,
    obs_groups={"actor": ("goal", "proprio"), "critic": ("critic",)},
    # The critic's values ran away from the returns on this task (laptop runs v3-v5,
    # roadmap M5): a target critic and a return bound keep them anchored.
    algorithm=SuperMimicCfg(critic=CriticUpdateCfg(target_update=0.05, max_return=MAX_RETURN)),
    # Spread the 20 s time-outs over training (otherwise they come in waves, every 400
    # iterations, and the episode curves only show the episodes that end early).
    random_episode_start=True,
  )
