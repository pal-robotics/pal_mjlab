"""The motion command's twin vs the simulator env (and vs the heads it will replace).

The model must evolve the command, end episodes and build the goal and critic features
exactly as the env's command term does, on the states the env logs.
"""

from pathlib import Path

import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.rl.command_state import command_state_dicts, load_command_state_dicts
from pal_mjlab.rl.supermimic.command_model import CommandModel, build_command_model
from pal_mjlab.rl.supermimic.names import EnvNamesCfg
from pal_mjlab.rl.supermimic.state import heading_pose, heading_state
from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import (
  pal_kangaroo_grippers_supermimic_env_cfg,
)
from pal_mjlab.tasks.supermimic.kangaroo.rl_cfg import pal_kangaroo_supermimic_runner_cfg
from pal_mjlab.tasks.supermimic.mdp.command_model import MotionLibraryCommandModel

LIBRARY = Path(__file__).parent / "fixtures" / "library_tiny"
NUM_ENVS = 4
MOTION_MODEL = pal_kangaroo_supermimic_runner_cfg().command_model


def _env() -> ManagerBasedRlEnv:
  cfg = pal_kangaroo_grippers_supermimic_env_cfg()
  cfg.scene.num_envs = NUM_ENVS
  motion = cfg.commands["motion"]
  motion.library_path = str(LIBRARY)
  motion.compile_kinematics = False
  return ManagerBasedRlEnv(cfg=cfg, device=get_test_device())


def _state(env, cmd):
  qpos = cmd.robot_qpos.clone()
  qpos[:, :3] -= env.scene.env_origins
  return heading_state(qpos, cmd.robot_qvel), heading_pose(qpos)


@pytest.fixture(scope="module")
def env():
  return _env()


def test_task_config_names_the_motion_model(env):
  model = build_command_model(MOTION_MODEL, env, "motion")
  assert isinstance(model, MotionLibraryCommandModel)
  assert isinstance(model, CommandModel)


def test_model_matches_the_env(env):
  model = MotionLibraryCommandModel(env, "motion")
  cmd = model.cmd
  tm = env.termination_manager
  gen = torch.Generator(device=env.device).manual_seed(0)
  obs, _ = env.reset()
  resets = 0
  for _ in range(80):
    c = model.state()
    state, pose = _state(env, cmd)

    # The goal and critic features the env observes.
    goal = model.goal(c, state, pose)
    torch.testing.assert_close(goal, obs["goal"], atol=1e-5, rtol=0)
    extras = model.critic_extras(c, state, pose)
    torch.testing.assert_close(extras, obs["critic"][:, -extras.shape[-1] :], atol=1e-4, rtol=0)


    action = 2.0 * torch.randn(
      NUM_ENVS, env.action_manager.total_action_dim, device=env.device, generator=gen
    )
    obs, _, terminated, truncated, _ = env.step(action)

    # The episode end and the command's evolution over the step.
    assert torch.equal(model.done(c), tm.get_term("motion_end"))
    nxt, after = model.advance(c), model.state()
    for k in c:
      assert torch.equal(nxt[k], after[k]), k

    fresh = terminated | truncated
    if fresh.any():
      obs, _ = env.reset(env_ids=fresh.nonzero().squeeze(-1))
      resets += int(fresh.sum())
  assert resets > 0


def test_command_terms_checkpoint_their_sampler(env):
  """The motion command's sampler is saved and restored through the command term."""
  saved = command_state_dicts(env)
  assert saved.keys() == {"motion"}
  saved = {"motion": {"sampler": {k: v.clone() for k, v in saved["motion"]["sampler"].items()}}}
  sampler = env.command_manager.get_term("motion").sampler
  sampler.update(torch.tensor([0], device=env.device), torch.tensor([0.5], device=env.device))
  load_command_state_dicts(env, saved)
  for k, v in command_state_dicts(env)["motion"]["sampler"].items():
    assert torch.equal(v, saved["motion"]["sampler"][k]), k


def test_command_checkpoint_mismatch_fails_clearly(env):
  state = command_state_dicts(env)["motion"]
  with pytest.raises(KeyError, match="missing from the checkpoint"):
    load_command_state_dicts(env, {})
  with pytest.raises(KeyError, match="not in the env"):
    load_command_state_dicts(env, {"motion": state, "other": {}})
  load_command_state_dicts(env, {}, strict=False)  # non-strict: nothing to restore


@pytest.mark.parametrize(
  ("cls", "error", "match"),
  [
    (None, ValueError, "not set"),
    ("MotionLibraryCommandModel", TypeError, "must be a class"),
    (EnvNamesCfg, TypeError, "not a CommandModel"),
  ],
)
def test_bad_command_models_fail_clearly(env, cls, error, match):
  with pytest.raises(error, match=match):
    build_command_model(cls, env, "motion")


def test_collector_reads_the_state_from_the_robot(env):
  """The collector's state (action joints on the action term's entity, no command) is the
  one the motion command gives, bit for bit, over a rollout."""
  from pal_mjlab.rl.supermimic.runner import SimCollector

  context_cfg = env.observation_manager.get_term_cfg("critic", "sim_context")
  model = MotionLibraryCommandModel(env, "motion")
  collector = SimCollector(env, EnvNamesCfg(), context_cfg, model)
  cmd = env.command_manager.get_term("motion")
  gen = torch.Generator(device=env.device).manual_seed(1)
  env.reset()
  for _ in range(10):
    for got, want in zip(collector.state(), _state(env, cmd), strict=True):
      assert torch.equal(got, want)
    action = torch.randn(
      NUM_ENVS, env.action_manager.total_action_dim, device=env.device, generator=gen
    )
    _, _, terminated, truncated, _ = env.step(action)
    done = terminated | truncated
    if done.any():
      env.reset(env_ids=done.nonzero().squeeze(-1))
