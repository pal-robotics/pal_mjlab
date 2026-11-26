"""The Kangaroo flat velocity task trained with SuperMimic (roadmap M5 v1)."""

import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.train import TrainConfig, run_train
from pal_mjlab.rl.supermimic.contracts import check_contract
from pal_mjlab.rl.supermimic.imagination import batch_source
from pal_mjlab.rl.supermimic.supermimic import (
  CriticUpdateCfg,
  PolicyUpdateCfg,
  SuperMimicCfg,
  WorldModelUpdateCfg,
)
from pal_mjlab.tasks.supermimic_velocity.env_cfgs import (
  pal_kangaroo_flat_supermimic_env_cfg,
)

TASK = "Mjlab-SuperMimic-Velocity-Flat-Pal-Kangaroo"


@pytest.fixture(scope="module")
def env():
  cfg = pal_kangaroo_flat_supermimic_env_cfg()
  cfg.scene.num_envs = 16
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  yield env
  env.close()


def test_the_velocity_task_meets_the_contract(env):
  """Every reward and observation, the stateful foot terms included, as imagination
  computes it; the command is resampled now and then, which imagination does not foresee."""
  report = check_contract(env, steps=120, action_std=0.5, resampled_commands=True)
  assert report.missing == []
  assert report.resets > 0
  for prefix, tol in {"reward/": 1e-4, "obs/": 1e-4, "command/": 0.0}.items():
    name, err = report.worst(prefix)
    assert err <= tol, f"{name}: {err:.2e} > {tol:.0e}"


def test_the_state_is_the_action_joints_and_xi_holds_the_joint_randomization(env):
  source = batch_source(env)
  assert len(source.spec.joint_ids) == env.action_manager.total_action_dim
  parts = source.spec.context_parts
  assert {"friction", "encoder_bias", "com_offset", "frictionloss", "damping", "armature"} <= set(parts)
  xi = source.context()
  model = env.sim.model
  dofs = env.scene["robot"].indexing.joint_v_adr[parts["damping"].joint_ids]
  damping = model.dof_damping.expand(env.num_envs, model.dof_damping.shape[-1])[:, dofs]
  torch.testing.assert_close(xi[:, parts["damping"].columns], damping)


def test_train_launches(tmp_path, monkeypatch):
  monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0" if get_test_device() == "cuda" else "")
  cfg = TrainConfig.from_task(TASK)
  cfg.env.scene.num_envs = 4
  agent = cfg.agent
  agent.max_iterations = 1
  agent.compile, agent.warmup_steps, agent.replay_steps = False, 12, 60
  agent.algorithm = SuperMimicCfg(
    world_model=WorldModelUpdateCfg(batches=1, batch_size=16, hidden=(32,)),
    policy=PolicyUpdateCfg(
      horizon=4, num_rollouts=8, hidden_dims=(32,), encoder_dims=(32,), latent_dim=8
    ),
    critic=CriticUpdateCfg(hidden_dims=(16,)),
  )
  run_train(TASK, cfg, tmp_path)
  assert (tmp_path / "model_1.pt").exists() and (tmp_path / "model_1.onnx").exists()


def test_failure_probability_has_finite_gradients_upright(env):
  """The imagined failures are differentiated through: their gradient must be finite,
  upright included (where the tilt angle's acos is not differentiable)."""
  from pal_mjlab.rl.supermimic.heads import AnalyticHeads
  from pal_mjlab.rl.supermimic.command_model import command_model_of
  from pal_mjlab.rl.supermimic.imagination import sim_batch

  heads = AnalyticHeads(env, command_model_of(env, "twist"), compile_fk=False)
  # Every failure is a single definition: imagination does not rely on the learned head.
  assert set(heads.failures) == {"fell_over", "base_too_low", "illegal_contacts"}
  assert heads.all_failures_analytic
  env.reset()
  b = sim_batch(env)
  state = b.state.clone()
  state[:, 1:7] = torch.tensor([1.0, 0, 0, 0, 1.0, 0], device=env.device)  # upright
  state.requires_grad_(True)
  imagined = heads.source.imagined(state, b.pose, b.c, b.action, b.prev_action, b.context)
  p = heads.failure_probability(imagined, torch.zeros(env.num_envs, device=env.device))
  (grad,) = torch.autograd.grad(p.sum(), state)
  assert torch.isfinite(grad).all()


def test_episodes_start_at_random_points(env):
  """``train`` asks for random episode starts (as for PPO): the envs' time-outs then
  spread over training instead of all landing at once."""
  from dataclasses import asdict

  from mjlab.rl import RslRlVecEnvWrapper
  from pal_mjlab.rl.supermimic import SuperMimicRunner
  from pal_mjlab.tasks.supermimic_velocity.rl_cfg import (
    pal_kangaroo_velocity_supermimic_runner_cfg,
  )

  agent = pal_kangaroo_velocity_supermimic_runner_cfg()
  agent.compile, agent.warmup_steps, agent.replay_steps = False, 12, 60
  agent.algorithm.world_model = WorldModelUpdateCfg(batches=1, batch_size=16, hidden=(32,))
  agent.algorithm.policy = PolicyUpdateCfg(
    horizon=4, num_rollouts=8, hidden_dims=(32,), encoder_dims=(32,), latent_dim=8
  )
  agent.algorithm.critic = CriticUpdateCfg(hidden_dims=(16,), target_update=0.05, max_return=15.0)
  runner = SuperMimicRunner(RslRlVecEnvWrapper(env), asdict(agent), None, env.device)
  runner.learn(1, init_at_random_ep_len=True)
  lengths = env.episode_length_buf.float()
  assert lengths.std() > 50  # spread over the episode (1000 steps), not all near 0
  assert int(lengths.max()) < env.max_episode_length


def test_step_logs_cover_all_envs_every_step(env, capsys):
  """The step logs (read like PPO's curves) are means over all envs of each step: the
  reward terms add up to the env's reward, and the command reports its tracking error."""
  from dataclasses import asdict

  from mjlab.rl import RslRlVecEnvWrapper
  from pal_mjlab.rl.supermimic import SuperMimicRunner
  from pal_mjlab.tasks.supermimic_velocity.rl_cfg import (
    pal_kangaroo_velocity_supermimic_runner_cfg,
  )

  agent = pal_kangaroo_velocity_supermimic_runner_cfg()
  agent.compile, agent.replay_steps = False, 60
  agent.algorithm.world_model = WorldModelUpdateCfg(batches=1, batch_size=16, hidden=(32,))
  agent.algorithm.policy = PolicyUpdateCfg(
    horizon=4, num_rollouts=8, hidden_dims=(32,), encoder_dims=(32,), latent_dim=8
  )
  runner = SuperMimicRunner(RslRlVecEnvWrapper(env), asdict(agent), None, env.device)
  runner.writer, runner.generator = None, torch.Generator(device=env.device)
  obs = runner.start_collection()
  obs, t = runner._collect(obs)
  log = t["step_log"]
  for name in env.reward_manager.active_terms:
    assert f"reward/{name}" in log
  for name in env.termination_manager.active_terms:
    assert f"terminations_per_min/{name}" in log
  for key in ("error_vel_xy", "error_vel_yaw", "speed_xy", "standing_error_xy"):
    assert f"command/twist/{key}" in log
  assert log["reward_total"] * env.step_dt == pytest.approx(float(t["reward"].mean()), rel=1e-4, abs=1e-6)
  runner._log(1)
  assert "time_to_failure" in capsys.readouterr().out
