"""The IMU linear acceleration is critic-only for PAL tasks.

The lower-body Kangaroo velocity tasks are the exception: their deployed policies
still rely on it as an actor observation.
"""

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
from mjlab.tasks.registry import list_tasks, load_env_cfg

PAL_TASKS = [task for task in list_tasks() if "-Pal-" in task]
ACTOR_LIN_ACC_TASKS = [
  t for t in PAL_TASKS if t.startswith("Mjlab-Velocity-") and "Lower-Body" in t
]


@pytest.mark.parametrize("play", [False, True])
@pytest.mark.parametrize("task", PAL_TASKS)
def test_base_lin_acc_in_actor_only_for_lower_body_velocity(task, play):
  actor_terms = load_env_cfg(task, play=play).observations["actor"].terms
  assert ("base_lin_acc" in actor_terms) == (task in ACTOR_LIN_ACC_TASKS)


@pytest.mark.parametrize("task", [t for t in PAL_TASKS if "Kangaroo" in t])
def test_base_lin_acc_kept_in_critic(task):
  assert "base_lin_acc" in load_env_cfg(task).observations["critic"].terms
