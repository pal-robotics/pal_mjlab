"""Single-definition terms' batches (``imagination.py``) vs the simulator env.

The simulator batch must hold exactly what the env and the collector see; an
``@imaginable`` term must give, through the env, its function's value on that batch; and a
batch built from the same state as a prediction must give the same kinematics.
"""

from pathlib import Path

import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from pal_mjlab.rl.supermimic.command_model import command_model_of
from pal_mjlab.rl.supermimic.imagination import (
  Batch,
  batch_source,
  imaginable,
  imagined_fn,
  sim_batch,
)
from pal_mjlab.rl.supermimic.names import imagination_cfg
from pal_mjlab.rl.supermimic.runner import SimCollector
from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import (
  pal_kangaroo_grippers_supermimic_env_cfg,
)

LIBRARY = Path(__file__).parent / "fixtures" / "library_tiny"
NUM_ENVS = 4


@imaginable
def probe(b: Batch, scale: float) -> torch.Tensor:
  """A toy single-definition term: scaled root height and first site's x."""
  return scale * torch.stack([b.state[:, 0], b.kin.site_quat[:, 0, 0]], dim=-1)


@pytest.fixture(scope="module")
def env():
  cfg = pal_kangaroo_grippers_supermimic_env_cfg()
  cfg.scene.num_envs = NUM_ENVS
  motion = cfg.commands["motion"]
  motion.library_path = str(LIBRARY)
  motion.compile_kinematics = False
  cfg.observations["probe"] = ObservationGroupCfg(
    terms={"x": ObservationTermCfg(func=probe, params={"scale": 2.0})},
    concatenate_terms=True,
    enable_corruption=False,
  )
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  yield env
  env.close()


def _rollout(env, steps: int, seed: int):
  """Random actions, resetting finished episodes; yields after each reset/step."""
  gen = torch.Generator(device=env.device).manual_seed(seed)
  env.reset()
  yield
  for _ in range(steps):
    action = 2.0 * torch.randn(
      env.num_envs, env.action_manager.total_action_dim, device=env.device, generator=gen
    )
    _, _, terminated, truncated, _ = env.step(action)
    yield
    done = terminated | truncated
    if done.any():
      env.reset(env_ids=done.nonzero().squeeze(-1))
      yield


def test_sim_batch_holds_what_the_env_sees(env):
  names = imagination_cfg(env)
  model = command_model_of(env, names.command)
  context_cfg = env.observation_manager.get_term_cfg("critic", "sim_context")
  collector = SimCollector(env, names, context_cfg, model)
  cmd = env.command_manager.get_term("motion")
  tracked = [cmd.fk.body_names.index(f"robot/{n}") for n in names.bodies]
  for _ in _rollout(env, 12, seed=0):
    b = sim_batch(env)
    state, pose = collector.state()
    assert torch.equal(b.state, state) and torch.equal(b.pose, pose)
    for k, v in model.state().items():
      assert torch.equal(b.c[k], v), k
    assert torch.equal(b.context, context_cfg.func(env, **context_cfg.params))
    assert torch.equal(b.action, env.action_manager.action)
    assert torch.equal(b.prev_action, env.action_manager.prev_action)
    # FK in the env-local frame = the command's world-frame bodies minus the env origin.
    robot = cmd.robot_bodies
    world = robot.pos[:, tracked] - env.scene.env_origins[:, None]
    torch.testing.assert_close(b.kin.body_pos, world, atol=1e-5, rtol=0)
    torch.testing.assert_close(b.kin.body_quat, robot.quat[:, tracked], atol=1e-5, rtol=0)


def test_sim_batch_is_shared_within_a_state(env):
  env.reset()
  first = sim_batch(env)
  assert sim_batch(env) is first  # the step's terms share one batch (one FK)
  action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
  env.step(action)
  stepped = sim_batch(env)
  assert stepped is not first
  env.reset(env_ids=torch.tensor([0], device=env.device))  # same env step, new state
  after_reset = sim_batch(env)
  assert after_reset is not stepped
  assert not torch.equal(after_reset.state[0], stepped.state[0])
  torch.testing.assert_close(after_reset.state[1:], stepped.state[1:])


def test_imaginable_terms_compute_their_function_in_the_env(env):
  assert imagined_fn(probe) is not None and imagined_fn(probe) is not probe
  assert imagined_fn(env.observation_manager.get_term_cfg("critic", "sim_context").func) is None
  source = batch_source(env)
  for _ in _rollout(env, 6, seed=1):
    obs = env.observation_manager.compute()["probe"]
    b = sim_batch(env)
    torch.testing.assert_close(obs, imagined_fn(probe)(b, scale=2.0), atol=0, rtol=0)
    # The same state as a prediction: the same kinematics (up to the quaternion round
    # trip of the heading state), so the same term value.
    imagined = source.imagined(b.state, b.pose, b.c, b.action, b.prev_action, b.context)
    for got, want in zip(imagined.kin, b.kin, strict=True):
      torch.testing.assert_close(got, want, atol=1e-5, rtol=0)
    torch.testing.assert_close(
      imagined_fn(probe)(imagined, scale=2.0), obs, atol=1e-5, rtol=0
    )


def test_spec_maps_env_joints_to_state_columns(env):
  spec = batch_source(env).spec
  robot = env.scene["robot"]
  assert spec.joint_cols(spec.joint_ids) == list(range(len(spec.joint_ids)))
  assert spec.joint_cols(spec.joint_ids[3:1:-1]) == [3, 2]
  with pytest.raises(ValueError, match="not in the state"):
    spec.joint_cols(slice(None))  # every joint: the passive ones are not in the state
  actuated = env.observation_manager.get_term_cfg("proprio", "joint_pos").params["asset_cfg"]
  assert isinstance(actuated, SceneEntityCfg)
  assert spec.joint_cols(actuated.joint_ids) == list(range(len(spec.joint_ids)))
  assert spec.site("imu") == 0 and spec.body(spec.body_names[-1]) == len(spec.body_names) - 1
  with pytest.raises(KeyError, match="ImaginationCfg.bodies"):
    spec.body("nope")
  assert len(robot.joint_names) == spec.num_robot_joints
