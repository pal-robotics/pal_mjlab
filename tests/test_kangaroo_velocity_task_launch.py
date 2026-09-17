"""Smoke tests launching train and play for the Kangaroo velocity tasks."""

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.scripts.train import TrainConfig, run_train
from mjlab.tasks.registry import list_tasks

KANGAROO_VELOCITY_TASKS = (
  "Mjlab-Velocity-Flat-Pal-Kangaroo",
  "Mjlab-Velocity-Flat-Pal-Kangaroo-Grippers",
  "Mjlab-Velocity-Flat-Pal-Kangaroo-Hands",
  "Mjlab-Velocity-Flat-Pal-Kangaroo-Lower-Body",
  "Mjlab-Velocity-Rough-Pal-Kangaroo",
  "Mjlab-Velocity-Rough-Pal-Kangaroo-Lower-Body",
)

NUM_ENVS = 2
NUM_STEPS = 4


class StepOnlyViewer:
  """Viewer stub that steps the policy instead of opening a window."""

  def __init__(self, env, policy, **kwargs):
    del kwargs
    self.env = env
    self.policy = policy

  def run(self) -> None:
    obs = self.env.get_observations()
    for _ in range(NUM_STEPS):
      with torch.inference_mode():
        actions = self.policy(obs)
      obs = self.env.step(actions)[0]


def test_registered_kangaroo_velocity_tasks():
  registered = [
    task
    for task in list_tasks()
    if task.startswith("Mjlab-Velocity-") and "Pal-Kangaroo" in task
  ]
  assert set(registered) == set(KANGAROO_VELOCITY_TASKS)


@pytest.mark.parametrize("task", KANGAROO_VELOCITY_TASKS)
def test_train_launches(task, tmp_path, monkeypatch):
  device = get_test_device()
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0" if device == "cuda" else "")

  cfg = TrainConfig.from_task(task)
  cfg.env.scene.num_envs = NUM_ENVS
  cfg.agent.logger = "tensorboard"
  cfg.agent.num_steps_per_env = NUM_STEPS
  cfg.agent.algorithm.num_mini_batches = 1
  cfg.agent.max_iterations = 1

  log_dir = tmp_path / "train"
  run_train(task, cfg, log_dir)

  assert (log_dir / "model_0.pt").exists()


@pytest.mark.parametrize("task", KANGAROO_VELOCITY_TASKS)
def test_play_launches(task, monkeypatch):
  monkeypatch.setattr("mjlab.scripts.play.NativeMujocoViewer", StepOnlyViewer)

  run_play(
    task,
    PlayConfig(
      agent="zero",
      num_envs=NUM_ENVS,
      device=get_test_device(),
      viewer="native",
    ),
  )
