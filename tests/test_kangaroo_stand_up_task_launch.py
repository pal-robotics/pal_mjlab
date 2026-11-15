"""Smoke and integration tests for the Kangaroo stand-up task."""

import math

import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.scripts.train import TrainConfig, run_train
from mjlab.tasks.registry import list_tasks, load_env_cfg
from pal_mjlab.robots import NATURAL_FREQ
from pal_mjlab.tasks.stand_up import mdp

TASK = "Mjlab-StandUp-Flat-Pal-Kangaroo"
NUM_ENVS = 2
NUM_STEPS = 4
ACTION_DIM = 22


def _quick_start_states(cfg, num_states: int = NUM_ENVS) -> None:
  """Make the start state library small and quick to generate."""
  start_cfg = cfg.events["generate_start_states"].params["start_cfg"]
  start_cfg.num_states = num_states
  start_cfg.ramp_time_s = 0.2
  start_cfg.settle_time_s = 0.6
  start_cfg.max_joint_speed = math.inf
  start_cfg.max_root_speed = math.inf
  cfg.events["reset_to_start_state"].params["start_cfg"] = start_cfg


def _load_cfg(play: bool = False, num_envs: int = NUM_ENVS):
  cfg = load_env_cfg(TASK, play=play)
  cfg.scene.num_envs = num_envs
  _quick_start_states(cfg, num_states=num_envs)
  return cfg


class StepOnlyViewer:
  """Viewer stub that steps the policy instead of opening a window."""

  def __init__(self, env, policy, **kwargs):
    del kwargs
    self.env = env
    self.policy = policy

  def run(self) -> None:
    obs = self.env.get_observations()
    for _ in range(NUM_STEPS):
      with torch.inference_mode():
        actions = self.policy(obs)
      obs = self.env.step(actions)[0]


@pytest.fixture(scope="module")
def env():
  cfg = _load_cfg(play=True, num_envs=8)
  environment = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  yield environment
  environment.close()


def _set_standing_fraction(env, fraction: float) -> None:
  params = env.event_manager.get_term_cfg("reset_to_start_state").params
  params["standing_fraction"] = fraction


def _stand_cfg(env) -> mdp.StandCfg:
  return env.reward_manager.get_term_cfg("standing").params["stand_cfg"]


def test_task_is_registered():
  assert TASK in list_tasks()


def test_train_launches(tmp_path, monkeypatch):
  device = get_test_device()
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0" if device == "cuda" else "")

  cfg = TrainConfig.from_task(TASK)
  cfg.env.scene.num_envs = NUM_ENVS
  _quick_start_states(cfg.env)
  cfg.agent.logger = "tensorboard"
  cfg.agent.num_steps_per_env = NUM_STEPS
  cfg.agent.algorithm.num_mini_batches = 1
  cfg.agent.max_iterations = 1

  log_dir = tmp_path / "train"
  run_train(TASK, cfg, log_dir)

  # The checkpoint keeps where the curriculum stands.
  infos = torch.load(log_dir / "model_0.pt", weights_only=False)["infos"]
  assert infos["curriculum_state"]["stand_up"]["assist_level"] == 1.0
  assert infos["curriculum_state"]["stand_up"]["care_scale"] == 1.0


def test_play_launches(monkeypatch):
  monkeypatch.setattr("mjlab.scripts.play.NativeMujocoViewer", StepOnlyViewer)
  monkeypatch.setattr(
    "mjlab.scripts.play.load_env_cfg",
    lambda task, play=False: _load_cfg(play=play),
  )

  run_play(
    TASK,
    PlayConfig(
      agent="zero",
      num_envs=NUM_ENVS,
      device=get_test_device(),
      viewer="native",
    ),
  )


def test_arm_action_scale_is_that_of_a_one_hertz_actuator(env):
  term = env.action_manager.get_term("joint_pos")
  scale = dict(zip(term.target_names, term.scale[0].tolist(), strict=True))
  one_hertz = (2.0 * math.pi) ** 2
  assert scale["arm_left_1_joint"] == pytest.approx(
    0.25 * 50 / (121**2 * 3.3e-05 * one_hertz), rel=1e-3
  )
  assert scale["arm_right_4_joint"] == pytest.approx(
    0.25 * 25 / (101**2 * 2.99e-05 * one_hertz), rel=1e-3
  )
  # The legs keep the walking scale, and the arms their gains.
  assert scale["leg_left_length_joint"] == pytest.approx(
    0.25 * 1100 / (4.1 * NATURAL_FREQ**2), rel=1e-3
  )
  model = env.sim.mj_model
  stiffness = model.actuator("robot/arm_left_1_joint").gainprm[0]
  assert stiffness == pytest.approx(121**2 * 3.3e-05 * NATURAL_FREQ**2, rel=1e-3)


def test_fallen_starts_are_at_rest_and_held_by_the_start_action(env):
  _set_standing_fraction(env, 0.0)
  obs, _ = env.reset()
  robot = env.scene["robot"]
  term = env.action_manager.get_term("joint_pos")
  params = env.event_manager.get_term_cfg("reset_to_start_state").params
  library = mdp.get_start_state_library(env, params["start_cfg"])

  assert torch.count_nonzero(robot.data.joint_vel) == 0
  assert torch.count_nonzero(robot.data.root_link_vel_w) == 0
  distance = torch.cdist(robot.data.joint_pos, library.joint_pos).min(dim=-1).values
  assert bool((distance < 1e-5).all())

  # The policy is shown the start action as its last one, and applying it
  # commands the joint positions the robot is in, at no cost.
  start_action = mdp.get_start_action(env).clone()
  assert obs["actor"].shape == (env.num_envs, 80)
  assert torch.allclose(obs["actor"][:, 55:77], start_action)
  joint_pos = robot.data.joint_pos[:, term.target_ids].clone()
  env.step(start_action)
  target = robot.data.joint_pos_target[:, term.target_ids]
  assert torch.allclose(target, joint_pos, atol=1e-5)
  assert torch.count_nonzero(mdp.episode_action_rate_l2(env)) == 0


def test_standing_starts_are_up(env):
  _set_standing_fraction(env, 1.0)
  env.reset()
  zero = torch.zeros(env.num_envs, ACTION_DIM, device=env.device)
  for _ in range(5):
    env.step(zero)
  assert bool(mdp.is_standing(env, _stand_cfg(env)).all())


def test_lift_assist_carries_a_robot_on_the_ground_only(env):
  stand_cfg = _stand_cfg(env)
  params = env.event_manager.get_term_cfg("lift_assist").params
  zero = torch.zeros(env.num_envs, ACTION_DIM, device=env.device)
  weight = mdp.get_ground_contact(env, stand_cfg.contact_cfg).weight

  # Off in play.
  assert params["level"] == 0.0
  _set_standing_fraction(env, 0.0)
  env.reset()
  env.step(zero)
  assert torch.count_nonzero(env.sim.data.xfrc_applied) == 0

  params["level"] = 0.5
  env.step(zero)
  lift = env.sim.data.xfrc_applied[..., 2].sum(dim=-1)
  missing = 1.0 - mdp.height_progress(env, stand_cfg)
  assert torch.allclose(lift, 0.5 * weight * missing, rtol=0.05)

  _set_standing_fraction(env, 1.0)
  env.reset()
  for _ in range(3):
    env.step(zero)
  lift = env.sim.data.xfrc_applied[..., 2].sum(dim=-1)
  assert bool((lift < 0.1 * weight).all())
  params["level"] = 0.0
