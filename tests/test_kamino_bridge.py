"""Bring-up checks for the Kamino bridge, on the real kangaroo_full model."""

import mjlab.tasks  # noqa: F401
import mujoco
import pal_mjlab.tasks  # noqa: F401
import pytest
import torch
import warp as wp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from pal_mjlab.robots.pal_kangaroo_full.kangaroo_full_constants import (
  KANG_FULL_TRANSMISSION_AT_HOME,
)
from pal_mjlab.sim.kamino.simulation import KaminoSimulation

TASK = "Mjlab-Velocity-Flat-Pal-Kangaroo-Full-Kamino"
COORDINATES = {
  "hip_yaw": "left_hip_z",
  "hip_pitch": "left_hip_xy_cross",
  "hip_roll": "left_hip_xy",
  "ankle_pitch": "leg_left_4_joint",
  "ankle_roll": "leg_left_5_joint",
}


def _make_env(gravity: bool = True, height: float = 0.0) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(TASK)
  cfg.scene.num_envs = 2
  # The default matrix-free solver has GPU-only kernels; dense LLT runs on CPU.
  cfg.sim.linear_solver = "LLTB"
  cfg.sim.max_iterations = 200
  cfg.sim.rho_0 = 0.01
  cfg.events = {k: v for k, v in cfg.events.items() if v.mode == "reset"}  # No DR.
  cfg.observations["actor"].enable_corruption = False
  if not gravity:
    cfg.sim.mujoco.gravity = (0.0, 0.0, 0.0)
  robot = cfg.scene.entities["robot"]
  robot.init_state.pos = (*robot.init_state.pos[:2], robot.init_state.pos[2] + height)
  env = ManagerBasedRlEnv(cfg, device="cpu")
  env.reset()
  return env


def _site_gap_mm(env: ManagerBasedRlEnv) -> torch.Tensor:
  m, sites = env.sim.mj_model, wp.to_torch(env.sim.wp_data.site_xpos)
  return (sites[:, m.eq_obj1id] - sites[:, m.eq_obj2id]).norm(dim=-1) * 1e3


def _joint(env: ManagerBasedRlEnv, name: str) -> torch.Tensor:
  return env.sim.data.qpos[
    :, env.sim.mj_model.joint(f"robot/{name}").qposadr[0]
  ].clone()


@pytest.fixture(scope="module")
def floating():
  """No gravity and no contact, so only the mechanism's own kinematics act."""
  env = _make_env(gravity=False, height=0.5)
  yield env
  env.close()


def test_uses_the_kamino_simulation(floating):
  assert isinstance(floating.sim, KaminoSimulation)


def test_reset_pose_is_on_the_constraint_manifold(floating):
  assert _site_gap_mm(floating).max() < 0.05


def test_state_round_trips_through_kamino(floating):
  """One step at rest must hand back the same qpos, within what the step moves."""
  floating.reset()
  before = floating.sim.data.qpos.clone()
  floating.step(torch.zeros(2, floating.action_manager.total_action_dim))
  assert (floating.sim.data.qpos - before).abs().max() < 2e-3


@pytest.mark.parametrize("screw", ["1", "2", "3", "4", "5"])
def test_screws_drive_the_joints_like_mujoco(floating, screw):
  """Kamino's closed chain must reproduce the ratios measured on MuJoCo's constraints.

  The screws hold through a PD, so a driven screw also nudges its neighbours: the
  prediction uses how far every screw actually moved.
  """
  env = floating
  env.reset()
  term = env.action_manager.get_term("joint_pos")
  name = f"leg_left_{screw}_actuator"
  index = term._target_names.index(name)
  screws = {s: f"leg_left_{s}_actuator" for s in "12345"}
  start = {j: _joint(env, j) for j in (*screws.values(), *COORDINATES.values())}
  action = torch.zeros(2, env.action_manager.total_action_dim)
  action[:, index] = 2e-3 / term._scale[0, index]  # A 2 mm screw target from home.
  for _ in range(40):
    env.step(action)
  moved = {s: _joint(env, n) - start[n] for s, n in screws.items()}
  assert moved[screw].abs().min() > 5e-4, "the screw barely moved"
  for coordinate, joint in COORDINATES.items():
    predicted = sum(
      KANG_FULL_TRANSMISSION_AT_HOME[s].get(coordinate, 0.0) * moved[s] for s in screws
    )
    actual = _joint(env, joint) - start[joint]
    assert torch.isclose(actual, predicted, rtol=0.05, atol=2e-3).all(), (
      joint,
      actual,
      predicted,
    )


def test_stands_and_carries_its_weight():
  env = _make_env()
  action = torch.zeros(2, env.action_manager.total_action_dim)
  for _ in range(25):
    env.step(action)
  robot = env.scene["robot"]
  assert robot.data.root_link_pos_w[:, 2].min() > 0.7
  assert _site_gap_mm(env).max() < 1.0
  force = env.scene.sensors["feet_ground_contact"].data.force.norm(dim=-1).sum(-1)
  weight = mujoco.mj_getTotalmass(env.sim.mj_model) * 9.81
  assert torch.isclose(force, torch.full_like(force, weight), rtol=0.3).all()
  env.close()
