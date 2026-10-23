"""World-model pieces: state round trip, replay windows, loss, gradients, collection parity."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pal_mjlab.tasks.supermimic.library import MotionLibrary
from pal_mjlab.tasks.supermimic.rl.replay import RolloutData
from pal_mjlab.tasks.supermimic.rl.world_model import (
  WorldModel,
  WorldModelCfg,
  window_loss,
)
from pal_mjlab.tasks.supermimic.state import (
  heading_pose,
  heading_state,
  integrate_pose,
  pose_delta,
  state_to_qpos_qvel,
)

LIBRARY = Path(__file__).parent / "fixtures" / "library_tiny"
NJ = 32


def _random_qpos_qvel(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
  g = torch.Generator().manual_seed(seed)
  qpos = torch.randn(n, 7 + NJ, generator=g, dtype=torch.float64)
  qpos[:, 3:7] /= qpos[:, 3:7].norm(
    dim=-1, keepdim=True
  )  # any orientation, upside down too
  return qpos, torch.randn(n, 6 + NJ, generator=g, dtype=torch.float64)


@pytest.mark.parametrize("seed", range(3))
def test_state_round_trip(seed):
  qpos, qvel = _random_qpos_qvel(256, seed)
  qpos2, qvel2 = state_to_qpos_qvel(heading_state(qpos, qvel), heading_pose(qpos))
  sign = torch.sign((qpos2[:, 3:7] * qpos[:, 3:7]).sum(-1, keepdim=True))
  qpos2 = torch.cat([qpos2[:, :3], qpos2[:, 3:7] * sign, qpos2[:, 7:]], dim=-1)
  torch.testing.assert_close(qpos2, qpos, atol=1e-10, rtol=0)
  torch.testing.assert_close(qvel2, qvel, atol=1e-10, rtol=0)
  pose, other = heading_pose(qpos), heading_pose(qpos.roll(1, 0))
  torch.testing.assert_close(
    integrate_pose(pose, pose_delta(pose, other)), other, atol=1e-10, rtol=0
  )


def _write_rollouts(folder: Path, T: int, E: int, seed: int) -> dict[str, np.ndarray]:
  """Synthetic rollouts with random episode ends, consistent with the tiny library."""
  rng = np.random.default_rng(seed)
  qpos, qvel = _random_qpos_qvel(T * E, seed)
  arrays = {
    "qpos": qpos.float().numpy().reshape(T, E, -1),
    "qvel": qvel.float().numpy().reshape(T, E, -1),
    "action": rng.normal(size=(T, E, 28)).astype(np.float32),
    "context": rng.normal(size=(T, E, 32)).astype(np.float32),
    "clip": rng.integers(0, 3, (T, E)).astype(np.int32),
    "frame": rng.integers(0, 300, (T, E)).astype(np.int32),
    "reward": rng.random((T, E)).astype(np.float32),
    "failure": rng.random((T, E)) < 0.05,
    "done": rng.random((T, E)) < 0.15,
  }
  arrays["next_qpos"] = np.roll(arrays["qpos"], -1, axis=0)
  arrays["next_qvel"] = np.roll(arrays["qvel"], -1, axis=0)
  folder.mkdir(parents=True)
  for k, v in arrays.items():
    np.save(folder / f"{k}.npy", v)
  (folder / "meta.json").write_text(json.dumps({"checkpoint": f"{folder.name}.pt"}))
  return arrays


@pytest.mark.parametrize("k", [1, 3, 8])
def test_windows_stay_inside_episodes(tmp_path, k):
  arrays = _write_rollouts(tmp_path / "model_1", T=40, E=6, seed=k)
  data = RolloutData([tmp_path / "model_1"], MotionLibrary(LIBRARY), "cpu")
  starts = data.valid_starts(k)
  t0, e = starts // data.E, starts % data.E
  done = arrays["done"]
  # Valid iff no episode ends before the window's last transition.
  expected = {
    (t, ee)
    for t in range(40 - k + 1)
    for ee in range(6)
    if not done[t : t + k - 1, ee].any()
  }
  assert set(zip(t0.tolist(), e.tolist(), strict=True)) == expected
  w = data.windows(starts, k)
  for i in range(len(starts)):
    t, ee = int(t0[i]), int(e[i])
    for j in range(k):
      torch.testing.assert_close(w["states"][i, j], data.state[t + j, ee])
    torch.testing.assert_close(w["states"][i, k], data.next_state[t + k - 1, ee])


def _tiny_model(dtype=torch.float32, predict_delta=True) -> WorldModel:
  torch.manual_seed(0)
  cfg = WorldModelCfg(
    state_dim=13 + 2 * NJ,
    hidden=(32, 32),
    termination_hidden=(16,),
    predict_delta=predict_delta,
  )
  return WorldModel(cfg).to(dtype)


def _batch(n: int, k: int, dtype=torch.float32):
  g = torch.Generator().manual_seed(1)
  qpos, qvel = _random_qpos_qvel(n * (k + 1), 2)
  states = heading_state(qpos, qvel).to(dtype).reshape(n, k + 1, -1)
  poses = heading_pose(qpos).to(dtype).reshape(n, k + 1, -1)
  return dict(
    states=states,
    poses=poses,
    actions=torch.randn(n, k, 28, generator=g, dtype=dtype),
    context=torch.randn(n, 32, generator=g, dtype=dtype),
    ref_root=torch.randn(n, k, 2, generator=g, dtype=dtype),
    failure=torch.rand(n, k, generator=g) < 0.1,
  )


@pytest.mark.parametrize("predict_delta", [True, False])
def test_model_overfits_a_batch(predict_delta):
  model = _tiny_model(predict_delta=predict_delta)
  b = _batch(16, 4)
  opt = torch.optim.Adam(model.parameters(), lr=3e-3)
  first = None
  for _ in range(300):
    loss, _ = window_loss(model, **b)
    first = first if first is not None else loss.item()
    opt.zero_grad()
    loss.backward()
    opt.step()
  assert loss.item() < 0.2 * first


def test_rollout_gradient_matches_finite_differences():
  """BPTT through the model: d(objective)/d(actions) equals finite differences (float64)."""
  model = _tiny_model(torch.float64)
  b = _batch(2, 3, torch.float64)
  actions = b["actions"].clone().requires_grad_()

  def objective(acts: torch.Tensor) -> torch.Tensor:
    s, p = b["states"][:, 0], b["poses"][:, 0]
    total = 0.0
    for j in range(acts.shape[1]):
      s, p = model.step(s, p, acts[:, j], b["context"])
      total = total + s.square().sum() + p.square().sum()
    return total

  objective(actions).backward()
  eps = 1e-6
  for idx in [(0, 0, 0), (1, 2, 5), (0, 1, 27)]:
    plus, minus = actions.detach().clone(), actions.detach().clone()
    plus[idx] += eps
    minus[idx] -= eps
    fd = (objective(plus) - objective(minus)) / (2 * eps)
    assert torch.isclose(actions.grad[idx], fd, rtol=1e-5, atol=1e-7), idx


def test_collected_reward_matches_the_analytic_reward():
  """Reward parity: the pure tracking reward on logged terminal-aware states reproduces
  the env's reward, so the world model can compute rewards instead of learning them."""
  from conftest import get_test_device
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
    get_kangaroo_grippers_spec,
  )
  from pal_mjlab.scripts.supermimic.collect_rollouts import collect
  from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import (
    TRACKED_BODIES,
    pal_kangaroo_grippers_supermimic_env_cfg,
  )
  from pal_mjlab.tasks.supermimic.kinematics import (
    TorchKinematics,
    compile_kinematic_model,
  )
  from pal_mjlab.tasks.supermimic.mdp.rewards import tracking_reward_fn
  from pal_mjlab.tasks.supermimic.mdp.tracking import BodyState

  cfg = pal_kangaroo_grippers_supermimic_env_cfg()
  cfg.scene.num_envs = 4
  cfg.auto_reset = False
  cfg.commands["motion"].library_path = str(LIBRARY)
  cfg.commands["motion"].compile_kinematics = False
  device = get_test_device()
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  vec = RslRlVecEnvWrapper(env)
  gen = torch.Generator(device=device).manual_seed(0)

  def policy(obs):
    return 2.0 * torch.randn(
      4, env.action_manager.total_action_dim, device=device, generator=gen
    )

  data = collect(env, vec, policy, steps=60, action_noise=0.1, generator=gen)
  env.close()
  assert data["done"].any()  # episodes ended, so terminal states were exercised

  lib = MotionLibrary(LIBRARY)
  fk = TorchKinematics(
    compile_kinematic_model(get_kangaroo_grippers_spec()), TRACKED_BODIES
  )
  reward = tracking_reward_fn(cfg.rewards)

  def bodies(qpos, qvel):
    k = fk(
      torch.from_numpy(qpos).reshape(-1, 39), torch.from_numpy(qvel).reshape(-1, 38)
    )
    return BodyState(k.body_pos, k.body_quat, k.body_lin_vel, k.body_ang_vel)

  frame = torch.from_numpy(data["frame"]).long().reshape(-1)
  ref = fk(lib.qpos[frame], lib.qvel[frame])
  ref = BodyState(ref.body_pos, ref.body_quat, ref.body_lin_vel, ref.body_ang_vel)
  r = reward(bodies(data["next_qpos"], data["next_qvel"]), ref).reshape(
    data["reward"].shape
  )
  np.testing.assert_allclose(r.numpy(), data["reward"], atol=1e-4)
  # Terminal states are kept: after an episode end the next step starts elsewhere.
  t, e = np.nonzero(data["done"][:-1])
  jumps = np.abs(data["qpos"][t + 1, e, :3] - data["next_qpos"][t, e, :3]).max(-1)
  assert (jumps > 1e-3).all()
