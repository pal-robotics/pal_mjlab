"""The registered SuperMimic task, driven by mjlab's ``train`` and ``play``."""

from pathlib import Path

import mjlab
import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import torch
import tyro
from conftest import get_test_device
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.scripts.train import TrainConfig, run_train
from mjlab.tasks.registry import list_tasks, load_runner_cls
from pal_mjlab.robots import kangaroo_passive_leg_joints
from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import LIBRARY_PATH
from pal_mjlab.tasks.supermimic.rl.runner import SuperMimicRunner
from pal_mjlab.tasks.supermimic.rl.supermimic import (
  CriticUpdateCfg,
  PolicyUpdateCfg,
  SuperMimicCfg,
  WorldModelUpdateCfg,
)

LIBRARY = Path(__file__).parent / "fixtures" / "library_tiny"
TASK = "Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers"


def test_registered_task():
  assert [t for t in list_tasks() if "SuperMimic" in t] == [TASK]
  assert load_runner_cls(TASK) is SuperMimicRunner


def test_train_cli_overrides():
  """``uv run train <task> --env.commands.motion.library-path <library> ...``."""
  cfg = tyro.cli(
    TrainConfig,
    args=[
      "--env.commands.motion.library-path",
      "some/library",
      "--env.scene.num-envs",
      "4096",
    ],
    default=TrainConfig.from_task(TASK),
    config=mjlab.TYRO_FLAGS,
  )
  motion = cfg.env.commands["motion"]
  assert (motion.library_path, cfg.env.scene.num_envs) == ("some/library", 4096)
  assert motion.passive_joint_state is kangaroo_passive_leg_joints


def test_train_launches(tmp_path, monkeypatch):
  """One iteration with few envs, the test library and small networks: a checkpoint and
  the policy as ONNX."""
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0" if get_test_device() == "cuda" else "")
  cfg = TrainConfig.from_task(TASK)
  cfg.env.scene.num_envs = 4
  motion = cfg.env.commands["motion"]
  motion.library_path = str(LIBRARY)
  motion.compile_kinematics = False
  agent = cfg.agent
  agent.max_iterations = 1
  agent.compile, agent.warmup_steps, agent.replay_steps = False, 12, 60
  agent.algorithm = SuperMimicCfg(
    world_model=WorldModelUpdateCfg(batches=1, batch_size=16, hidden=(32,)),
    policy=PolicyUpdateCfg(
      horizon=4, num_rollouts=8, hidden_dims=(32,), encoder_dims=(32,), latent_dim=8
    ),
    critic=CriticUpdateCfg(hidden_dims=(16,)),
  )
  run_train(TASK, cfg, tmp_path)
  assert (tmp_path / "model_1.pt").exists() and (tmp_path / "model_1.onnx").exists()


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


def test_play_launches(tmp_path, monkeypatch):
  """``uv run play <task>``: the registered play configuration, which reads the library at
  its default path under the working directory."""
  monkeypatch.setattr("mjlab.scripts.play.NativeMujocoViewer", StepOnlyViewer)
  monkeypatch.chdir(tmp_path)
  (tmp_path / LIBRARY_PATH).parent.mkdir(parents=True)
  (tmp_path / LIBRARY_PATH).symlink_to(LIBRARY)
  run_play(
    TASK,
    PlayConfig(agent="zero", num_envs=2, device=get_test_device(), viewer="native"),
  )
