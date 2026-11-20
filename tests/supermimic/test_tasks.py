"""The registered SuperMimic task, driven by mjlab's ``train`` and ``play``."""

import shutil
from dataclasses import asdict
from pathlib import Path

import mjlab
import pytest
import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import torch
import tyro
from conftest import get_test_device
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.scripts.train import TrainConfig, run_train
from mjlab.tasks.registry import list_tasks, load_runner_cls
from pal_mjlab.robots import kangaroo_passive_leg_joints
from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import LIBRARY_PATH
from pal_mjlab.rl.supermimic import SuperMimicRunner
from pal_mjlab.rl.supermimic.names import EnvNamesCfg, critic_group
from pal_mjlab.rl.supermimic.supermimic import (
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



def _renamed(terms: dict, names: dict) -> dict:
  """``terms`` with the keys in ``names`` renamed, in the same order."""
  return {names.get(k, k): v for k, v in terms.items()}


def test_train_with_other_env_names(tmp_path, monkeypatch):
  """SuperMimic reads the env's command, failure term, groups and action term by the names
  in ``agent.env_names`` / ``agent.obs_groups``, and the observation terms by function: an
  env where every one of them is named differently trains and exports the same way."""
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0" if get_test_device() == "cuda" else "")
  cfg = TrainConfig.from_task(TASK)
  env = cfg.env
  env.scene.num_envs = 4
  env.commands = _renamed(env.commands, {"motion": "reference"})
  motion = env.commands["reference"]
  motion.library_path = str(LIBRARY)
  motion.compile_kinematics = False
  env.observations = _renamed(
    env.observations, {"goal": "target", "proprio": "sensors", "critic": "value"}
  )
  for group in env.observations.values():
    group.terms = {f"{name}_x": term for name, term in group.terms.items()}
  env.terminations = _renamed(env.terminations, {"anchor_failure": "fell"})
  env.actions = _renamed(env.actions, {"joint_pos": "pd_targets"})
  terms = [
    *(t for g in env.observations.values() for t in g.terms.values()),
    *env.rewards.values(),
    *env.terminations.values(),
  ]
  for term in terms:
    if term is not None and term.params.get("command_name") == "motion":
      term.params["command_name"] = "reference"

  agent = cfg.agent
  agent.env_names = EnvNamesCfg(
    command="reference",
    failure_terms=("fell",),
    goal_group="target",
    proprio_group="sensors",
    action_term="pd_targets",
  )
  agent.obs_groups = {"actor": ("target", "sensors"), "critic": ("value",)}
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


def test_obs_groups_must_match_env_names():
  """The actor concatenates goal then proprioception: ``obs_groups`` saying otherwise is
  refused instead of silently feeding the encoder the wrong group."""
  agent = TrainConfig.from_task(TASK).agent
  agent.obs_groups = {"actor": ("proprio", "goal"), "critic": ("critic",)}
  with pytest.raises(ValueError, match="goal_group, proprio_group"):
    critic_group(asdict(agent), agent.env_names)


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
  try:
    (tmp_path / LIBRARY_PATH).symlink_to(LIBRARY)
  except OSError:  # Windows without symlink privileges: the library is tiny
    shutil.copytree(LIBRARY, tmp_path / LIBRARY_PATH)
  run_play(
    TASK,
    PlayConfig(agent="zero", num_envs=2, device=get_test_device(), viewer="native"),
  )
