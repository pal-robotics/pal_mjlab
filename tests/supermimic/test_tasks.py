"""The registered SuperMimic tasks (D52): one SuperMimic and one PPO-Sim task per robot, run with mjlab's
``train`` and ``play``; each algorithm logs to one folder for both robots, run names default to the robot."""

from pathlib import Path

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.scripts.train import TrainConfig, run_train
from mjlab.tasks import registry
from mjlab.tasks.registry import list_tasks, load_runner_cls
from pal_mjlab.tasks.supermimic.rl.runner import SuperMimicRunner, SuperMimicRunnerCfg
from pal_mjlab.tasks.supermimic.rl.supermimic import (
  CriticUpdateCfg,
  PolicyUpdateCfg,
  SuperMimicCfg,
  WorldModelUpdateCfg,
)

FIXTURES = Path(__file__).parent / "fixtures"

TASKS = {
  "Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers": ("supermimic", "grippers"),
  "Mjlab-SuperMimic-Flat-Pal-Kangaroo-Hands": ("supermimic", "hands"),
  "Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Grippers": ("supermimic_ppo", "grippers"),
  "Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Hands": ("supermimic_ppo", "hands"),
}


def test_registered_supermimic_tasks():
  assert {t for t in list_tasks() if "SuperMimic" in t} == set(TASKS)


def test_log_folders_and_run_names():
  for task, (experiment, robot) in TASKS.items():
    agent = TrainConfig.from_task(task).agent
    assert (agent.experiment_name, agent.run_name) == (experiment, robot)


def test_supermimic_tasks_train_with_the_supermimic_runner():
  for task in TASKS:
    expected = SuperMimicRunner if "-PPO-" not in task else None
    assert load_runner_cls(task) is expected


class StepOnlyViewer:
  """Viewer stub that steps the policy instead of opening a window."""

  def __init__(self, env, policy, **kwargs):
    del kwargs
    self.env, self.policy = env, policy

  def run(self) -> None:
    obs = self.env.get_observations()
    for _ in range(4):
      with torch.inference_mode():
        obs = self.env.step(self.policy(obs))[0]


def _tiny(cfg: TrainConfig, task: str) -> TrainConfig:
  """CPU-sized: few envs, the tiny library, small networks, one iteration."""
  robot = TASKS[task][1]
  library = FIXTURES / ("library_tiny" if robot == "grippers" else "library_tiny_hands")
  cfg.env.scene.num_envs = 4
  motion = cfg.env.commands["motion"]
  motion.library_path = str(library)
  motion.compile_kinematics = False
  agent = cfg.agent
  agent.max_iterations = 1
  if isinstance(agent, SuperMimicRunnerCfg):
    agent.compile, agent.warmup_steps, agent.replay_steps = False, 12, 60
    agent.eval_interval = 0
    agent.algorithm = SuperMimicCfg(
      world_model=WorldModelUpdateCfg(batches=1, batch_size=16, hidden=(32,)),
      policy=PolicyUpdateCfg(
        horizon=4, num_rollouts=8, hidden_dims=(32,), encoder_dims=(32,), latent_dim=8
      ),
      critic=CriticUpdateCfg(hidden_dims=(16,)),
    )
  else:
    agent.num_steps_per_env, agent.algorithm.num_mini_batches = 4, 1
  return cfg


@pytest.mark.parametrize("task", TASKS)
def test_train_launches(task, tmp_path, monkeypatch):
  """``uv run train <task> --env.commands.motion.library-path <library>``: one iteration, a checkpoint."""
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0" if get_test_device() == "cuda" else "")
  cfg = _tiny(TrainConfig.from_task(task), task)
  cfg.agent.logger = "tensorboard"
  run_train(task, cfg, tmp_path / "train")
  assert sorted(p.name for p in (tmp_path / "train").glob("model_*.pt"))


@pytest.mark.parametrize("task", TASKS)
def test_play_launches(task, monkeypatch):
  """``uv run play <task>``: the registered play config (default library) with a dummy agent."""
  monkeypatch.setattr("mjlab.scripts.play.NativeMujocoViewer", StepOnlyViewer)
  library = FIXTURES / (
    "library_tiny" if TASKS[task][1] == "grippers" else "library_tiny_hands"
  )
  play_cfg = registry._REGISTRY[task].play_env_cfg
  motion = play_cfg.commands["motion"]
  saved = motion.library_path, motion.compile_kinematics
  motion.library_path, motion.compile_kinematics = str(library), False
  try:
    run_play(
      task,
      PlayConfig(agent="zero", num_envs=2, device=get_test_device(), viewer="native"),
    )
  finally:
    motion.library_path, motion.compile_kinematics = saved
