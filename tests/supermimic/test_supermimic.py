"""SuperMimic's updates and runner on the tiny library (CPU-sized networks)."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import (
  pal_kangaroo_grippers_supermimic_env_cfg,
)
from pal_mjlab.tasks.supermimic.kangaroo.rl_cfg import (
  pal_kangaroo_supermimic_runner_cfg,
)
from pal_mjlab.tasks.supermimic.rl.runner import SuperMimicRunner
from pal_mjlab.tasks.supermimic.rl.supermimic import (
  CriticUpdateCfg,
  PolicyUpdateCfg,
  SuperMimicCfg,
  WorldModelUpdateCfg,
  imagine,
  lambda_returns,
)

LIBRARY = Path(__file__).parent / "fixtures" / "library_tiny"
NUM_ENVS = 8


def _tiny_cfg(**runner):
  cfg = pal_kangaroo_supermimic_runner_cfg()
  cfg.algorithm = SuperMimicCfg(
    world_model=WorldModelUpdateCfg(
      batches=2, batch_size=16, hidden=(32, 32), termination_hidden=(16,)
    ),
    policy=PolicyUpdateCfg(
      horizon=4, num_rollouts=16, hidden_dims=(32,), encoder_dims=(32,), latent_dim=8
    ),
    critic=CriticUpdateCfg(hidden_dims=(16,)),
  )
  settings = dict(
    compile_kinematics=False, warmup_steps=12, replay_steps=60, log_interval=1,
    save_interval=3, eval_interval=3, eval_envs=4,
  )  # fmt: skip
  return replace(cfg, **{**settings, **runner})


@pytest.fixture(scope="module")
def env():
  cfg = pal_kangaroo_grippers_supermimic_env_cfg()
  cfg.scene.num_envs = NUM_ENVS
  cfg.auto_reset = False
  motion = cfg.commands["motion"]
  motion.library_path = str(LIBRARY)
  motion.compile_kinematics = False
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device=get_test_device()))
  yield env
  env.close()


def test_lambda_returns_limits():
  g = torch.Generator().manual_seed(0)
  b, h, gamma = 5, 6, 0.9
  r, v = torch.rand(b, h, generator=g), torch.rand(b, h + 1, generator=g)
  cont = (torch.rand(b, h, generator=g) > 0.2).float()
  # λ = 0: one-step TD targets.
  torch.testing.assert_close(
    lambda_returns(r, cont, v, gamma, 0.0), r + gamma * cont * v[:, 1:]
  )
  # λ = 1: the discounted return to the horizon, bootstrapped with V_H.
  ret = v[:, -1]
  for t in reversed(range(h)):
    ret = r[:, t] + gamma * cont[:, t] * ret
  torch.testing.assert_close(lambda_returns(r, cont, v, gamma, 1.0)[:, 0], ret)


def test_runner_trains_saves_and_resumes(env, tmp_path):
  cfg = asdict(_tiny_cfg())
  runner = SuperMimicRunner(env, cfg, str(tmp_path), env.device)
  runner.learn(num_learning_iterations=6)
  assert (tmp_path / "model_3.pt").exists() and (tmp_path / "model_6.pt").exists()
  resumed = SuperMimicRunner(env, asdict(_tiny_cfg()), None, env.device)
  resumed.load(str(tmp_path / "model_6.pt"))
  assert resumed.current_learning_iteration == 6
  for a, b in zip(
    runner.alg.actor.state_dict().values(),
    resumed.alg.actor.state_dict().values(),
    strict=True,
  ):
    torch.testing.assert_close(a, b)
  obs = runner.collector.observations()
  action = resumed.get_inference_policy(env.device)(obs)
  assert action.shape == (NUM_ENVS, env.num_actions) and action.isfinite().all()


@pytest.fixture(scope="module")
def trained(env, tmp_path_factory):
  """A runner after a few iterations: a filled replay buffer and non-trivial networks."""
  runner = SuperMimicRunner(env, asdict(_tiny_cfg(eval_interval=0)), None, env.device)
  runner.learn(num_learning_iterations=3)
  return runner


def test_policy_step_updates_only_the_actor(trained):
  alg = trained.alg
  alg.cfg.critic.epochs = 0  # isolate the policy step from the critic fit
  try:
    before = {
      name: [p.detach().clone() for p in m.parameters()]
      for name, m in (
        ("actor", alg.actor),
        ("critic", alg.critic),
        ("world_model", alg.world_model),
      )
    }
    for prm in (*alg.critic.parameters(), *alg.world_model.parameters()):
      prm.grad = None  # leftovers of earlier fits
    alg.update_policy(trained.replay, trained.generator)
  finally:
    alg.cfg.critic.epochs = 5
  for name, m in (("critic", alg.critic), ("world_model", alg.world_model)):
    for p0, p in zip(before[name], m.parameters(), strict=True):
      assert torch.equal(p0, p), name
      assert p.grad is None, name
  changed = [
    not torch.equal(p0, p)
    for p0, p in zip(before["actor"], alg.actor.parameters(), strict=True)
  ]
  assert any(changed)


def test_objective_gradient_matches_finite_differences(trained):
  """BPTT through the world model: the directional derivative of the imagined return
  along its gradient equals finite differences (Eq. 2, no sensor noise)."""
  alg = trained.alg
  pc = replace(alg.cfg.policy, sensor_noise=False)
  start = trained.replay.sample_starts(
    16, trained.heads.history, torch.Generator(device=trained.device).manual_seed(1)
  )
  params = list(alg.actor.parameters())

  def objective() -> torch.Tensor:
    roll = imagine(trained.heads, alg.world_model, alg.actor, alg.critic, start, pc)
    return roll.objective.sum()

  grads = torch.autograd.grad(objective(), params)
  norm = torch.sqrt(sum(g.square().sum() for g in grads))
  eps = 1e-3
  with torch.no_grad():
    for p, g in zip(params, grads, strict=True):
      p.add_(eps * g / norm)
    plus = objective()
    for p, g in zip(params, grads, strict=True):
      p.sub_(2 * eps * g / norm)
    minus = objective()
    for p, g in zip(params, grads, strict=True):
      p.add_(eps * g / norm)
  fd = (plus - minus) / (2 * eps)
  torch.testing.assert_close(fd, norm, rtol=0.05, atol=1e-4)
