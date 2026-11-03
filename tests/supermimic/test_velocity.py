"""The velocity interface (Appendix G, D56): the reference command, the analytic heads against the
env, post-training on a frozen motion policy, and its ONNX policy."""

from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from onnx.reference import ReferenceEvaluator
from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import (
  pal_kangaroo_hands_supermimic_env_cfg,
  pal_kangaroo_hands_velocity_env_cfg,
)
from pal_mjlab.tasks.supermimic.kangaroo.rl_cfg import (
  pal_kangaroo_supermimic_runner_cfg,
  pal_kangaroo_velocity_runner_cfg,
)
from pal_mjlab.tasks.supermimic.library import MotionLibrary
from pal_mjlab.tasks.supermimic.mdp.velocity import reference_twist
from pal_mjlab.tasks.supermimic.rl.runner import SuperMimicRunner
from pal_mjlab.tasks.supermimic.rl.supermimic import (
  CriticUpdateCfg,
  PolicyUpdateCfg,
  WorldModelUpdateCfg,
)
from pal_mjlab.tasks.supermimic.rl.velocity import VelocityHeads, VelocityRunner
from pal_mjlab.tasks.supermimic.state import heading_pose, heading_state

LIBRARY = Path(__file__).parent / "fixtures" / "library_tiny_hands"
NUM_ENVS = 8


def _env_cfg(make_cfg, num_envs: int = NUM_ENVS):
  cfg = make_cfg()
  cfg.scene.num_envs = num_envs
  cfg.auto_reset = False
  motion = cfg.commands["motion"]
  motion.library_path = str(LIBRARY)
  motion.compile_kinematics = False
  return cfg


def _tiny(cfg, **runner):
  """CPU-sized networks; the motion and velocity runs share the actor's shapes."""
  cfg.algorithm = type(cfg.algorithm)(
    world_model=WorldModelUpdateCfg(
      batches=2, batch_size=16, hidden=(32, 32), termination_hidden=(16,)
    ),
    policy=PolicyUpdateCfg(
      horizon=4, num_rollouts=16, hidden_dims=(32,), encoder_dims=(32,), latent_dim=8
    ),
    critic=CriticUpdateCfg(hidden_dims=(16,)),
  )
  settings = dict(
    compile=False, warmup_steps=12, replay_steps=60, log_interval=1,
    save_interval=3, eval_interval=3, eval_envs=4,
  )  # fmt: skip
  return asdict(replace(cfg, **{**settings, **runner}))


def test_reference_command_is_the_centered_mean_within_the_clip():
  lib = MotionLibrary(LIBRARY)
  qp, qv = lib.qpos.double().numpy(), lib.qvel.double().numpy()
  w, x, y, z = qp[:, 3:7].T
  yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  c, s = np.cos(yaw), np.sin(yaw)
  z_row = np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
  twist = np.stack(
    [
      c * qv[:, 0] + s * qv[:, 1],
      -s * qv[:, 0] + c * qv[:, 1],
      (z_row.T * qv[:, 3:6]).sum(-1),  # world z of the body-frame angular velocity
    ],
    axis=-1,
  )
  got = reference_twist(lib, window=25).numpy()
  kernel = np.ones(25)
  for start, n in zip(lib.clip_start.tolist(), lib.clip_length.tolist(), strict=True):
    seg = twist[start : start + n]
    count = np.convolve(np.ones(n), kernel, "same")[:, None]
    mean = np.stack([np.convolve(seg[:, i], kernel, "same") for i in range(3)], -1)
    np.testing.assert_allclose(got[start : start + n], mean / count, atol=1e-5)


def _state(env, motion):
  qpos = motion.robot_qpos.clone()
  qpos[:, :3] -= env.scene.env_origins
  return heading_state(qpos, motion.robot_qvel), heading_pose(qpos)


def test_heads_match_the_env():
  """Command, critic input and reward in imagination equal the env's, held and reference
  commands alike, through falls and resets."""
  cfg = _env_cfg(pal_kangaroo_hands_velocity_env_cfg, 4)
  cfg.observations["proprio"].enable_corruption = False
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  heads = VelocityHeads(env, compile_fk=False)
  motion = env.command_manager.get_term("motion")
  velocity = env.command_manager.get_term("velocity")
  context_cfg = env.observation_manager.get_term_cfg("critic", "sim_context")
  gen = torch.Generator(device=env.device).manual_seed(0)
  obs, _ = env.reset()
  sources, resets = set(), 0
  for _ in range(80):
    row, frame = velocity.replay_row(), motion.frame.clone()
    command = heads.command(row, frame)
    torch.testing.assert_close(command, obs["velocity_command"], atol=1e-6, rtol=0)
    state, pose = _state(env, motion)
    context = context_cfg.func(env, **context_cfg.params)
    critic = heads.critic_input(state, pose, command, context, motion.clip, frame)
    torch.testing.assert_close(critic, obs["critic"], atol=1e-4, rtol=0)
    sources.update(row[:, 3].tolist())

    prev = env.action_manager.action.clone()
    action = 2.0 * torch.randn(
      4, env.action_manager.total_action_dim, device=env.device, generator=gen
    )
    obs, reward, terminated, truncated, _ = env.step(action)
    next_state, next_pose = _state(env, motion)
    kin = heads.kinematics(next_state, next_pose)
    r = heads.reward(next_state, kin, command, action, prev)
    torch.testing.assert_close(r, reward, atol=1e-5, rtol=1e-5)
    done = terminated | truncated
    if done.any():
      obs, _ = env.reset(env_ids=done.nonzero().squeeze(-1))
      resets += int(done.sum())
  assert sources == {0.0, 1.0} and resets > 0
  env.close()


@pytest.fixture(scope="module")
def motion_checkpoint(tmp_path_factory):
  """A tiny Hands motion-tracking run: the policy to post-train on."""
  cfg = _env_cfg(pal_kangaroo_hands_supermimic_env_cfg)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=get_test_device()))
  out = tmp_path_factory.mktemp("motion")
  runner = SuperMimicRunner(
    env,
    _tiny(pal_kangaroo_supermimic_runner_cfg(), eval_interval=0),
    str(out),
    env.device,
  )
  runner.learn(num_learning_iterations=3)
  env.close()
  return out / "model_3.pt"


@pytest.fixture(scope="module")
def velocity_env():
  cfg = _env_cfg(pal_kangaroo_hands_velocity_env_cfg)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=get_test_device()))
  yield env
  env.close()


def test_post_training_trains_only_the_velocity_encoder(
  velocity_env, motion_checkpoint, tmp_path
):
  env = velocity_env
  cfg = _tiny(
    pal_kangaroo_velocity_runner_cfg(), motion_checkpoint=str(motion_checkpoint)
  )
  runner = VelocityRunner(env, cfg, str(tmp_path), env.device)
  encoder = [p.detach().clone() for p in runner.alg.actor.encoder.parameters()]
  runner.learn(num_learning_iterations=6)  # with an imagined-vs-sim check at 3
  assert (tmp_path / "model_6.pt").exists() and (tmp_path / "model_6.onnx").exists()

  motion = torch.load(motion_checkpoint, map_location=env.device, weights_only=False)[
    "alg"
  ]
  frozen = runner.alg.actor.motion.state_dict()
  for name, value in motion["actor"].items():
    assert torch.equal(frozen[name], value), name  # policy and normalizer unchanged
  for p0, p in zip(encoder, runner.alg.actor.encoder.parameters(), strict=True):
    assert not torch.equal(p0, p)
  fitted = runner.alg.world_model.state_dict()
  assert any(
    not torch.equal(fitted[k], v)
    for k, v in motion["world_model"]["state_dict"].items()
  )  # loaded, then fit online

  # Resuming needs no motion checkpoint; a new run does.
  resumed = VelocityRunner(
    env, _tiny(pal_kangaroo_velocity_runner_cfg()), None, env.device
  )
  resumed.load(str(tmp_path / "model_6.pt"))
  assert resumed.current_learning_iteration == 6
  for a, b in zip(
    runner.alg.actor.state_dict().values(),
    resumed.alg.actor.state_dict().values(),
    strict=True,
  ):
    torch.testing.assert_close(a, b)
  fresh = VelocityRunner(
    env, _tiny(pal_kangaroo_velocity_runner_cfg()), None, env.device
  )
  with pytest.raises(ValueError, match="motion-checkpoint"):
    fresh.learn(num_learning_iterations=1)

  # The ONNX policy: [velocity_command, proprio] -> the actor's actions.
  path = tmp_path / "model_6.onnx"
  meta = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
  om = env.unwrapped.observation_manager
  names = [t for g in ("velocity_command", "proprio") for t in om.active_terms[g]]
  assert meta["observation_names"].split(",") == names and "goal_root_body" not in meta
  obs = runner.collector.observations()
  x = torch.cat([obs["velocity_command"], obs["proprio"]], dim=-1)
  torch.backends.cuda.matmul.allow_tf32 = False  # learn()'s TF32; ONNX runs FP32
  with torch.no_grad():
    expected = runner.alg.actor(obs).cpu().numpy()
  (got,) = ReferenceEvaluator(str(path)).run(None, {"obs": x[:1].cpu().numpy()})
  np.testing.assert_allclose(got[0], expected[0], atol=1e-4)

  # The replay's command rows (held command, reference flag) and the policy step's statistics.
  rows = runner.replay.command[: int(runner.replay.count.min())]
  assert rows.shape[-1] == 4 and set(rows[..., 3].unique().tolist()) <= {0.0, 1.0}
  stats = runner.alg.update_policy(runner.replay, runner.generator)
  assert np.isfinite([stats["align"], stats["objective"], stats["critic_loss"]]).all()
