"""SuperMimic Kangaroo env on the 3-clip test library (2 s each, 50 Hz;
fixtures/make_fixtures.py)."""

import math
from pathlib import Path

import mujoco
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.robots import KANGAROO_GRIPPERS_ACTION_SCALE
from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import (
  ARM_WAIST_ACTION_SCALE_FREQ_HZ,
  pal_kangaroo_grippers_supermimic_env_cfg,
)
from pal_mjlab.tasks.supermimic.mdp import MotionLibraryCommand

LIBRARY = Path(__file__).parent / "fixtures" / "library_tiny"
NUM_ENVS = 3
CLIP_LENGTH = 100
NUM_JOINTS = 28  # the library's joints: the actuated ones
NUM_BODIES = 12


def _make_env(play: bool = False, failure: bool = True) -> ManagerBasedRlEnv:
  cfg = pal_kangaroo_grippers_supermimic_env_cfg(play=play)
  cfg.scene.num_envs = NUM_ENVS
  cfg.auto_reset = True
  motion = cfg.commands["motion"]
  motion.library_path = str(LIBRARY)
  motion.compile_kinematics = False
  if not failure:
    cfg.terminations.pop("anchor_failure")
  return ManagerBasedRlEnv(cfg=cfg, device=get_test_device())


@pytest.fixture(scope="module")
def env():
  env = _make_env()
  env.reset()
  yield env
  env.close()


def _step_randomly(env: ManagerBasedRlEnv, steps: int, scale: float = 1.0) -> None:
  gen = torch.Generator(device=env.device).manual_seed(0)
  shape = (NUM_ENVS, env.action_manager.total_action_dim)
  for _ in range(steps):
    env.step(scale * torch.randn(shape, device=env.device, generator=gen))


def test_observation_layout(env):
  obs = env.observation_manager.compute()
  k, nj = 10, NUM_JOINTS
  frame = nj + nj + 6  # reference q, q̇ and root orientation (6D)
  assert obs["goal"].shape == (NUM_ENVS, k * frame)
  assert obs["proprio"].shape == (NUM_ENVS, 10 * (nj + nj + 3 + 3 + nj))
  state, xi = 13 + 2 * nj, 1 + nj + 3
  assert obs["critic"].shape == (NUM_ENVS, state + k * frame + xi + 4 + 1)


def test_torch_kinematics_match_the_simulator(env):
  """FK on the gathered qpos/qvel equals mjlab's body data (indexing, origins, frames).

  Checked after stepping with random actions, so the robot is away from the reference.
  """
  _step_randomly(env, 15)
  env.sim.forward()
  command: MotionLibraryCommand = env.command_manager.get_term("motion")
  assert command.ref_bodies.pos.shape == (NUM_ENVS, NUM_BODIES, 3)
  bodies = command.bodies(command.robot_qpos, command.robot_qvel)
  assert (bodies.pos - command.ref_bodies.pos).abs().max() > 1e-3  # off the reference
  robot = env.scene["robot"]
  ids = robot.find_bodies(command.cfg.body_names, preserve_order=True)[0]
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


def test_sim_context_ranges(env):
  nj = NUM_JOINTS
  start = (13 + 2 * nj) + 10 * (nj + nj + 6)
  xi = env.observation_manager.compute_group("critic")[:, start : start + 1 + nj + 3]
  assert ((xi[:, 0] >= 0.3) & (xi[:, 0] <= 1.2)).all()  # foot friction
  assert (xi[:, 1 : 1 + nj].abs() <= 0.01 + 1e-6).all()  # calibration offsets
  com = xi[:, 1 + nj :]
  assert (com[:, 0].abs() <= 0.025 + 1e-6).all() and (
    com[:, 1:].abs() <= 0.05 + 1e-6
  ).all()
  # The passive knee linkage is not measured: no calibration offset.
  robot = env.scene["robot"]
  passive = robot.find_joints((".*_femur_joint", ".*_knee_joint"))[0]
  assert (robot.data.encoder_bias[:, passive] == 0).all()


def test_reset_writes_the_reference_with_the_legs_closed():
  """A reset puts the robot on the reference (no noise in play) and sets the passive knee
  linkage from the leg length: both anchors of each leg's connect constraint coincide."""
  env = _make_env(play=True)
  command: MotionLibraryCommand = env.command_manager.get_term("motion")
  command.set_eval_clips(torch.arange(NUM_ENVS))
  env.reset()
  start = command.library.qpos[command.start_frame].clone()
  start[:, :3] += env.scene.env_origins
  torch.testing.assert_close(command.robot_qpos, start, atol=1e-6, rtol=0)

  model = env.sim.mj_model
  data = mujoco.MjData(model)
  for qpos in env.sim.data.qpos.cpu().numpy():
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    for side in ("left", "right"):
      eq = model.equality(f"robot/leg_{side}_closed_kin_joint")
      a = data.xpos[eq.obj1id] + data.xmat[eq.obj1id].reshape(3, 3) @ eq.data[:3]
      b = data.xpos[eq.obj2id] + data.xmat[eq.obj2id].reshape(3, 3) @ eq.data[3:6]
      assert abs(a - b).max() < 1e-4
  env.close()


def test_episode_ends_at_the_clip_end():
  """Each env plays one clip from frame 0; with failures disabled it ends after L-1 steps."""
  env = _make_env(play=True, failure=False)
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


def test_falls_terminate():
  """Large random actions make the robot fall; the anchor rule must catch it."""
  env = _make_env()
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


def test_action_scales():
  """The robot's gains; the legs' action scales are the robot's, the arms' and waist's are
  those of gains at ``ARM_WAIST_ACTION_SCALE_FREQ_HZ``: larger by the frequency ratio squared."""
  cfg = pal_kangaroo_grippers_supermimic_env_cfg()
  scales = cfg.actions["joint_pos"].scale
  assert set(scales) == set(KANGAROO_GRIPPERS_ACTION_SCALE)
  ratio = (3.5 / ARM_WAIST_ACTION_SCALE_FREQ_HZ) ** 2
  for name, robot_scale in KANGAROO_GRIPPERS_ACTION_SCALE.items():
    expected = robot_scale if name.startswith("leg_") else ratio * robot_scale
    assert math.isclose(scales[name], expected, rel_tol=1e-3), name
