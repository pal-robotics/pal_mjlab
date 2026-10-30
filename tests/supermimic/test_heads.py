"""Analytic heads vs the simulator env: observation, goal, critic input, reward, failure.

The policy must see the same inputs, and optimize the same reward, in imagination as in
simulation. These checks run the env with random actions (episodes end and reset) and
evaluate the heads on the true states.
"""

from pathlib import Path

import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import (
  pal_kangaroo_grippers_supermimic_env_cfg,
  pal_kangaroo_hands_supermimic_env_cfg,
  tight_tracking,
)
from pal_mjlab.tasks.supermimic.mdp import MotionLibraryCommand
from pal_mjlab.tasks.supermimic.rl.heads import AnalyticHeads
from pal_mjlab.tasks.supermimic.state import heading_pose, heading_state

FIXTURES = Path(__file__).parent / "fixtures"
NUM_ENVS = 4
# The paper's reward (grippers), and the experimental one with the feet tracked on their own (hands).
VARIANTS = {
  "grippers": (pal_kangaroo_grippers_supermimic_env_cfg, "library_tiny"),
  "hands_tight": (
    lambda: tight_tracking(pal_kangaroo_hands_supermimic_env_cfg()),
    "library_tiny_hands",
  ),
}


def _env(variant: str = "grippers") -> ManagerBasedRlEnv:
  make_cfg, library = VARIANTS[variant]
  cfg = make_cfg()
  cfg.scene.num_envs = NUM_ENVS
  cfg.auto_reset = False
  cfg.observations["proprio"].enable_corruption = False  # compare noise-free ō
  motion = cfg.commands["motion"]
  motion.library_path = str(FIXTURES / library)
  motion.compile_kinematics = False
  return ManagerBasedRlEnv(cfg=cfg, device=get_test_device())


def _state(env: ManagerBasedRlEnv, cmd: MotionLibraryCommand):
  qpos = cmd.robot_qpos.clone()
  qpos[:, :3] -= env.scene.env_origins
  return heading_state(qpos, cmd.robot_qvel), heading_pose(qpos)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_heads_match_the_env(variant):
  env = _env(variant)
  heads = AnalyticHeads(env, compile_fk=False)
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
    goal = heads.goal(cmd.robot_qpos[:, 3:7], cmd.clip, cmd.frame)
    torch.testing.assert_close(goal, obs["goal"], atol=1e-5, rtol=0)
    critic = heads.critic_input(state, pose, goal, context, cmd.clip, cmd.frame)
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
    target, clip, prev = cmd.frame.clone(), cmd.clip.clone(), last.clone()
    obs, reward, terminated, truncated, _ = env.step(action)
    next_state, next_pose = _state(env, cmd)
    kin = heads.kinematics(next_state, next_pose)
    ref = heads.reference(target)
    r = heads.reward(next_state, kin, ref, action, prev)
    torch.testing.assert_close(r, reward, atol=1e-5, rtol=1e-5)
    tm = env.termination_manager
    assert torch.equal(
      heads.failure_margin(kin, ref) > 1.0, tm.get_term("anchor_failure")
    )
    assert torch.equal(heads.clip_end(clip, target), tm.get_term("motion_end"))

    fresh = terminated | truncated
    if fresh.any():
      obs, _ = env.reset(env_ids=fresh.nonzero().squeeze(-1))
      resets += int(fresh.sum())
  assert resets > 0
  env.close()
