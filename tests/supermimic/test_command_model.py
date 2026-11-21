"""The motion command's twin vs the simulator env (and vs the heads it will replace).

The model must evolve the command, end episodes and build the goal and critic features
exactly as the env's command term does, on the states the env logs.
"""

from pathlib import Path

import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.rl.supermimic.command_model import CommandModel, build_command_model
from pal_mjlab.rl.supermimic.heads import AnalyticHeads
from pal_mjlab.rl.supermimic.names import EnvNamesCfg
from pal_mjlab.rl.supermimic.state import heading_pose, heading_state, state_to_qpos_qvel
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
  heads = AnalyticHeads(env, compile_fk=False)
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

    # Identical to the heads it replaces (step 3 must be a pure move).
    root_quat = state_to_qpos_qvel(state, pose)[0][:, 3:7]  # as imagination calls it
    assert torch.equal(goal, heads.goal(root_quat, c["clip"], c["frame"]))
    context = torch.zeros(NUM_ENVS, 0, device=env.device)
    critic = heads.critic_input(state, pose, goal, context, c["clip"], c["frame"])
    assert torch.equal(extras, critic[:, -extras.shape[-1] :])
    assert torch.equal(model.termination_ref(c), heads.ref_root(c["frame"]))
    ref, ref_heads = model.reference(c), heads.reference(c["frame"])
    for a, b in zip(ref, ref_heads, strict=True):
      torch.testing.assert_close(a, b, atol=1e-6, rtol=0)

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


def test_state_dict_round_trip(env):
  model = MotionLibraryCommandModel(env, "motion")
  saved = {k: v.clone() for k, v in model.state_dict()["sampler"].items()}
  sampler = model.cmd.sampler
  sampler.update(torch.tensor([0], device=env.device), torch.tensor([0.5], device=env.device))
  model.load_state_dict({"sampler": saved})
  for k, v in model.state_dict()["sampler"].items():
    assert torch.equal(v, saved[k]), k


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
  collector = SimCollector(env, EnvNamesCfg(), context_cfg)
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
