"""scripts/supermimic/train.py: one log folder for both robots, run names that start with the robot (D49)."""

from mjlab.scripts.train import TrainConfig
from pal_mjlab.scripts.supermimic.train import robot_run_name

GRIPPERS = "Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers-ArmWaist1Hz"
HANDS = "Mjlab-SuperMimic-Flat-Pal-Kangaroo-Hands-ArmWaist1Hz"


def test_both_robots_log_to_one_folder():
  for task in (GRIPPERS, HANDS, "Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers"):
    assert TrainConfig.from_task(task).agent.experiment_name == "supermimic"
  for task in (
    "Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Grippers",
    "Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Hands",
  ):
    assert TrainConfig.from_task(task).agent.experiment_name == "supermimic_ppo"


def test_run_names_start_with_the_robot():
  assert robot_run_name(GRIPPERS, "full-24h") == "grippers-full-24h"
  assert robot_run_name(HANDS, "full-24h") == "hands-full-24h"
  assert robot_run_name(GRIPPERS, "grippers-dev") == "grippers-dev"
  assert robot_run_name(HANDS, "") == "hands"
