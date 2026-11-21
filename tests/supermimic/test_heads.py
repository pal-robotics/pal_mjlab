"""Analytic heads vs the simulator env: observation, reward, and imagination's critic input.

The policy must see the same inputs, and optimize the same reward, in imagination as in
simulation. These checks run the env with random actions (episodes end and reset) and
evaluate the heads on the true states.
"""

from pathlib import Path

import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg
from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import (
  pal_kangaroo_grippers_supermimic_env_cfg,
)
from pal_mjlab.tasks.supermimic.mdp import MotionLibraryCommand
from pal_mjlab.rl.supermimic.command_model import command_model_of
from pal_mjlab.rl.supermimic.heads import AnalyticHeads
from pal_mjlab.rl.supermimic.supermimic import critic_input
from pal_mjlab.rl.supermimic.state import heading_pose, heading_state

LIBRARY = Path(__file__).parent / "fixtures" / "library_tiny"
NUM_ENVS = 4


def _env() -> ManagerBasedRlEnv:
  cfg = pal_kangaroo_grippers_supermimic_env_cfg()
  cfg.scene.num_envs = NUM_ENVS
  cfg.observations["proprio"].enable_corruption = False  # compare noise-free ō
  motion = cfg.commands["motion"]
  motion.library_path = str(LIBRARY)
  motion.compile_kinematics = False
  return ManagerBasedRlEnv(cfg=cfg, device=get_test_device())


def _state(env: ManagerBasedRlEnv, cmd: MotionLibraryCommand):
  qpos = cmd.robot_qpos.clone()
  qpos[:, :3] -= env.scene.env_origins
  return heading_state(qpos, cmd.robot_qvel), heading_pose(qpos)


def test_heads_match_the_env():
  env = _env()
  model = command_model_of(env, "motion")
  heads = AnalyticHeads(env, model, compile_fk=False)
  cmd: MotionLibraryCommand = env.command_manager.get_term("motion")
  context_cfg = env.observation_manager.get_term_cfg("critic", "sim_context")
  gen = torch.Generator(device=env.device).manual_seed(0)
  obs, _ = env.reset()
  history = None
  fresh = torch.ones(NUM_ENVS, dtype=torch.bool, device=env.device)
  resets = 0
  for _ in range(80):
    state, pose = _state(env, cmd)
    context = context_cfg.func(env, **context_cfg.params)
    c = model.state()
    goal = model.goal(c, state, pose)
    torch.testing.assert_close(goal, obs["goal"], atol=1e-5, rtol=0)
    critic = critic_input(model, state, pose, goal, context, c)
    torch.testing.assert_close(critic, obs["critic"], atol=1e-4, rtol=0)

    # Proprioception in the env's layout. A reset env's history is backfilled with its
    # first frame (mjlab's history buffer); otherwise the newest frame is appended.
    last = env.action_manager.action.clone()
    frame = heads.observation(state, heads.kinematics(state, pose), context, last)
    filled = {n: f[:, None].expand(-1, heads.history, -1) for n, f in frame.items()}
    if history is None:
      history = filled
    else:
      history = {
        n: torch.where(
          fresh[:, None, None], filled[n], torch.cat([history[n][:, 1:], f[:, None]], 1)
        )
        for n, f in frame.items()
      }
    torch.testing.assert_close(
      heads.flatten_history(history), obs["proprio"], atol=1e-4, rtol=0
    )

    action = 2.0 * torch.randn(
      NUM_ENVS, env.action_manager.total_action_dim, device=env.device, generator=gen
    )
    prev = last.clone()
    obs, reward, terminated, truncated, _ = env.step(action)
    next_state, next_pose = _state(env, cmd)
    # The step's resulting state, its action and the command state it targeted.
    b = heads.source.imagined(next_state, next_pose, c, action, prev, context)
    terms = heads.reward_terms(b)
    torch.testing.assert_close(sum(terms.values()), reward, atol=1e-5, rtol=1e-5)
    rm = env.reward_manager  # each term as the env weighed it (before the dt scaling)
    for i, name in enumerate(rm.active_terms):
      if name in terms:
        torch.testing.assert_close(
          terms[name] / heads.reward_scale, rm._step_reward[:, i], atol=1e-5, rtol=1e-5
        )
    assert set(terms) == {n for n in rm.active_terms if rm.get_term_cfg(n).weight}

    fresh = terminated | truncated
    if fresh.any():
      obs, _ = env.reset(env_ids=fresh.nonzero().squeeze(-1))
      resets += int(fresh.sum())
  assert resets > 0
  env.close()


def test_rewards_without_a_single_definition_fail_clearly():
  """A reward term imagination cannot compute is refused, every such term listed, rather
  than silently left out of the imagined objective."""
  from mjlab.envs.mdp import rewards as mj_rewards
  from mjlab.managers.reward_manager import RewardTermCfg

  cfg = pal_kangaroo_grippers_supermimic_env_cfg()
  cfg.scene.num_envs = 2
  cfg.commands["motion"].library_path = str(LIBRARY)
  cfg.commands["motion"].compile_kinematics = False
  cfg.rewards["mjlab_rate"] = RewardTermCfg(func=mj_rewards.action_rate_l2, weight=-0.1)
  cfg.rewards["alive"] = RewardTermCfg(func=mj_rewards.is_alive, weight=1.0)
  cfg.rewards["unweighted"] = RewardTermCfg(func=mj_rewards.is_terminated, weight=0.0)
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  with pytest.raises(NotImplementedError, match=r"mjlab_rate .*, alive .*@imaginable") as e:
    AnalyticHeads(env, command_model_of(env, "motion"), compile_fk=False)
  assert "unweighted" not in str(e.value)  # weight 0: the env does not compute it either
  env.close()


def test_heads_refuse_other_joints_than_the_library():
  """The predicted state holds the library's joints, so a joint term on any other set of
  joints (here every joint, passive ones included) has no analytic head."""
  cfg = pal_kangaroo_grippers_supermimic_env_cfg()
  cfg.commands["motion"].library_path = str(LIBRARY)
  cfg.commands["motion"].compile_kinematics = False
  cfg.observations["proprio"].terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg(
    "robot"
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  with pytest.raises(ValueError, match="motion library's joints"):
    AnalyticHeads(env, command_model_of(env, "motion"), compile_fk=False)
  env.close()
