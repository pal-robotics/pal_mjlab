"""SuperMimic Kangaroo env on a 3-clip real-data library (2 s each, 50 Hz), for both variants
(7-DoF arms with grippers, 5-DoF arms with hands; fixtures/make_fixtures.py)."""

import re
from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import (
  pal_kangaroo_grippers_supermimic_env_cfg,
  pal_kangaroo_hands_supermimic_env_cfg,
)
from pal_mjlab.tasks.supermimic.mdp import MotionLibraryCommand

FIXTURES = Path(__file__).parent / "fixtures"
VARIANTS = {  # env, library, joints (all, actuated)
  "grippers": (pal_kangaroo_grippers_supermimic_env_cfg, "library_tiny", 32, 28),
  "hands": (pal_kangaroo_hands_supermimic_env_cfg, "library_tiny_hands", 32, 28),
}
NUM_ENVS = 3
CLIP_LENGTH = 100


def _make_env(variant: str, play: bool = False, **cmd) -> ManagerBasedRlEnv:
  make_cfg, library, _, _ = VARIANTS[variant]
  cfg = make_cfg(play=play)
  cfg.scene.num_envs = NUM_ENVS
  motion = cfg.commands["motion"]
  motion.library_path = str(FIXTURES / library)
  motion.compile_kinematics = False
  for k, v in cmd.items():
    setattr(motion, k, v)
  return ManagerBasedRlEnv(cfg=cfg, device=get_test_device())


@pytest.fixture(scope="module", params=list(VARIANTS))
def variant(request) -> str:
  return request.param


@pytest.fixture(scope="module")
def env(variant):
  env = _make_env(variant)
  env.reset()
  yield env
  env.close()


def test_observation_layout(env, variant):
  obs = env.observation_manager.compute()
  k, (nj, na) = 10, VARIANTS[variant][2:]
  frame = na + na + 6  # reference q, q̇ and root orientation (6D)
  assert obs["goal"].shape == (NUM_ENVS, k * frame)
  assert obs["proprio"].shape == (NUM_ENVS, 10 * (na + na + 3 + 3 + na))
  state, xi = 13 + 2 * nj, 1 + na + 3
  assert obs["critic"].shape == (NUM_ENVS, state + k * frame + xi + 4 + 1)


def test_torch_kinematics_match_the_simulator(env):
  """FK on the gathered qpos/qvel equals mjlab's body data (indexing, origins, frames).

  Checked after stepping with random actions, so the robot is away from the reference.
  """
  gen = torch.Generator(device=env.device).manual_seed(0)
  for _ in range(15):
    action = torch.randn(
      NUM_ENVS, env.action_manager.total_action_dim, device=env.device, generator=gen
    )
    env.step(action)
  env.sim.forward()
  command: MotionLibraryCommand = env.command_manager.get_term("motion")
  ref = command.ref_bodies.pos
  assert (
    command.robot_bodies.pos - ref
  ).abs().max() > 1e-3  # robot moved off the reference
  command._version += 1  # drop the cache
  robot = env.scene["robot"]
  ids = robot.find_bodies(command.cfg.body_names, preserve_order=True)[0]
  bodies = command.robot_bodies
  data = robot.data
  torch.testing.assert_close(
    bodies.pos, data.body_link_pos_w[:, ids], atol=1e-4, rtol=0
  )
  dots = (bodies.quat * data.body_link_quat_w[:, ids]).sum(-1).abs()
  torch.testing.assert_close(dots, torch.ones_like(dots), atol=1e-4, rtol=0)
  torch.testing.assert_close(
    bodies.lin_vel, data.body_link_lin_vel_w[:, ids], atol=1e-3, rtol=0
  )
  torch.testing.assert_close(
    bodies.ang_vel, data.body_link_ang_vel_w[:, ids], atol=1e-3, rtol=0
  )


def test_hand_does_not_collide_with_its_elbow(env, variant):
  """D46: the elbow capsule (arm_*_4_collision, copied from the 7-DoF forearm) reaches the Seed palm when
  the wrist rolls. That pair is excluded, so no wrist pose makes it a contact on the same arm. Seeded arm
  and wrist poses, both roll limits included."""
  if variant != "hands":
    pytest.skip("Seed hands only")
  m = env.sim.mj_model
  d = mujoco.MjData(m)

  def body(g: int) -> str:
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])

  joints = [
    j
    for j in range(m.njnt)
    if re.search(
      r"(arm|hand)_(left|right)_", mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    )
  ]
  rng = np.random.default_rng(0)
  for k in range(200):
    mujoco.mj_resetData(m, d)
    for j in joints:
      lo, hi = m.jnt_range[j]
      name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
      q = (lo, hi)[k % 2] if "palm_roll" in name else rng.uniform(lo, hi)
      d.qpos[m.jnt_qposadr[j]] = q
    mujoco.mj_forward(m, d)
    for g1, g2 in zip(
      d.contact.geom1[: d.ncon], d.contact.geom2[: d.ncon], strict=True
    ):
      pair = sorted((body(g1), body(g2)))  # crossed arms may still touch the other hand
      assert not any(
        pair == [f"robot/arm_{s}_4_link", f"robot/hand_{s}_palm_link"]
        for s in ("left", "right")
      ), pair


def test_sim_context_ranges(env, variant):
  nj, na = VARIANTS[variant][2:]
  start = (13 + 2 * nj) + 10 * (na + na + 6)
  xi = env.observation_manager.compute_group("critic")[:, start : start + 1 + na + 3]
  assert ((xi[:, 0] >= 0.3) & (xi[:, 0] <= 1.2)).all()  # foot friction
  assert (
    xi[:, 1 : 1 + na].abs() <= 0.01 + 1e-6
  ).all()  # calibration offsets, actuated joints
  com = xi[:, 1 + na :]
  assert (com[:, 0].abs() <= 0.025 + 1e-6).all() and (
    com[:, 1:].abs() <= 0.05 + 1e-6
  ).all()
  # Passive four-bar joints are not measured: no calibration offset.
  robot = env.scene["robot"]
  passive = robot.find_joints((".*_femur_joint", ".*_knee_joint"))[0]
  assert (robot.data.encoder_bias[:, passive] == 0).all()


@pytest.mark.parametrize("variant", VARIANTS)
def test_episode_ends_at_the_clip_end(variant):
  """Each env plays one clip from frame 0; with failures disabled it ends after L-1 steps."""
  env = _make_env(variant, play=True)
  env.cfg.terminations.pop("anchor_failure", None)
  env.termination_manager._term_names.remove("anchor_failure")
  command: MotionLibraryCommand = env.command_manager.get_term("motion")
  command.set_eval_clips(torch.arange(NUM_ENVS))
  env.reset()
  assert (command.clip == torch.arange(NUM_ENVS, device=env.device)).all()
  ends = torch.zeros(NUM_ENVS, dtype=torch.long)
  rewards = torch.zeros(NUM_ENVS)
  zero = torch.zeros(NUM_ENVS, env.action_manager.total_action_dim, device=env.device)
  for step in range(1, CLIP_LENGTH + 2):
    _, reward, terminated, _, _ = env.step(zero)
    assert torch.isfinite(reward).all()
    rewards += torch.where(ends == 0, reward.cpu(), 0.0)
    ends = torch.where((ends == 0) & terminated.cpu(), step, ends)
  assert (ends == CLIP_LENGTH - 1).all(), ends
  # Positive terms are bounded by their weights: 5 per step, times dt.
  assert (rewards <= 5 * env.step_dt * (CLIP_LENGTH - 1) + 1e-4).all()
  env.close()


@pytest.mark.parametrize("variant", VARIANTS)
def test_falls_terminate(variant):
  """Large random actions make the robot fall; the anchor rule must catch it."""
  env = _make_env(variant)
  env.reset()
  gen = torch.Generator(device=env.device).manual_seed(1)
  failures = 0
  for _ in range(100):
    action = 3.0 * torch.randn(
      NUM_ENVS, env.action_manager.total_action_dim, device=env.device, generator=gen
    )
    env.step(action)
    failures += int(env.termination_manager.get_term("anchor_failure").sum())
  assert failures > 0
  env.close()
