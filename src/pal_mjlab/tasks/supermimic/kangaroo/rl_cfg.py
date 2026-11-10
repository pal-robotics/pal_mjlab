"""RL configuration for PAL Robotics' Kangaroo SuperMimic task."""

from pal_mjlab.tasks.supermimic.rl.runner import SuperMimicRunnerCfg


def pal_kangaroo_supermimic_runner_cfg() -> SuperMimicRunnerCfg:
  """Create the SuperMimic runner configuration: the paper's hyperparameters (Table A.5),
  which are the defaults of ``rl/supermimic.py``."""
  return SuperMimicRunnerCfg(
    experiment_name="kangaroo_supermimic",
    logger="tensorboard",
    max_iterations=320_000,
    save_interval=2000,
    obs_groups={"actor": ("goal", "proprio"), "critic": ("critic",)},
  )
