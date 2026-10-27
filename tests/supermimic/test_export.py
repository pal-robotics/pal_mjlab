"""ONNX export (D48): the file computes the actor's actions on recorded env observations, and its
metadata is enough to build those observations (goal by state.reference_goal, proprio oldest first)."""

from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from onnx.reference import ReferenceEvaluator
from tensordict import TensorDict

from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import (
  pal_kangaroo_grippers_supermimic_env_cfg,
  pal_kangaroo_hands_supermimic_env_cfg,
)
from pal_mjlab.tasks.supermimic.kinematics import quat_rotate_inv
from pal_mjlab.tasks.supermimic.rl.export import OBS_GROUPS, export_policy
from pal_mjlab.tasks.supermimic.rl.models import EncoderDecoderModel
from pal_mjlab.tasks.supermimic.state import reference_goal

FIXTURES = Path(__file__).parent / "fixtures"
VARIANTS = {
  "grippers": (pal_kangaroo_grippers_supermimic_env_cfg, "library_tiny"),
  "hands": (pal_kangaroo_hands_supermimic_env_cfg, "library_tiny_hands"),
}
NUM_ENVS, STEPS = 2, 12


@pytest.fixture(scope="module", params=list(VARIANTS))
def rollout(request, tmp_path_factory):
  """A play env (no noise) driven by a small random actor; observations, metadata, ONNX path."""
  make_cfg, library = VARIANTS[request.param]
  cfg = make_cfg(play=True)
  cfg.scene.num_envs = NUM_ENVS
  cfg.commands["motion"].library_path = str(FIXTURES / library)
  cfg.commands["motion"].compile_kinematics = False
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  obs = TensorDict(env.observation_manager.compute(), batch_size=[NUM_ENVS])
  torch.manual_seed(0)
  actor = EncoderDecoderModel(
    obs, {"actor": list(OBS_GROUPS)}, "actor", env.action_manager.total_action_dim,
    hidden_dims=(64, 32), encoder_dims=(64, 32), latent_dim=8, obs_normalization=True,
  ).to(env.device)  # fmt: skip
  steps = []
  for _ in range(STEPS):
    x = torch.cat([obs[g] for g in OBS_GROUPS], dim=-1)
    actor.obs_normalizer.update(x)  # a normalizer away from identity
    with torch.no_grad():
      action = actor.act(x)
    steps.append((obs, x, action))
    env.step(action)
    obs = TensorDict(env.observation_manager.compute(), batch_size=[NUM_ENVS])
  path = tmp_path_factory.mktemp(request.param) / "model_1.onnx"
  actor.eval()
  export_policy(env, actor, path, "model_1.pt")
  meta = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
  yield env, actor, steps, obs, meta, path
  env.close()


def _list(meta, key, cast=str):
  return [cast(v) for v in meta[key].split(",")]


def test_onnx_matches_the_actor(rollout):
  _, actor, steps, _, _, path = rollout
  session = ReferenceEvaluator(str(path))
  for _, x, _ in steps:
    with torch.no_grad():
      expected = actor.act(x).cpu().numpy()
    for i in range(NUM_ENVS):
      (got,) = session.run(None, {"obs": x[i : i + 1].cpu().numpy()})
      np.testing.assert_allclose(got[0], expected[i], atol=1e-4)


def test_metadata_layout_sums_to_the_observation(rollout):
  env, _, steps, _, meta, _ = rollout
  dims = _list(meta, "observation_terms_dim", lambda v: int(float(v)))
  history = _list(meta, "observation_terms_history_length", lambda v: int(float(v)))
  assert sum(d * h for d, h in zip(dims, history, strict=True)) == steps[0][1].shape[-1]
  assert _list(meta, "observation_names") == [
    t for g in OBS_GROUPS for t in env.observation_manager.active_terms[g]
  ]
  assert len(_list(meta, "action_joint_names")) == env.action_manager.total_action_dim
  assert len(_list(meta, "action_scale", float)) == env.action_manager.total_action_dim
  assert meta["checkpoint"] == "model_1.pt" and meta["goal_root_body"] == "base_link"


def test_goal_from_the_reference(rollout):
  """The goal rebuilt from the library's clip arrays and the robot's root quaternion."""
  env, _, _, obs, meta, _ = rollout
  cmd = env.command_manager.get_term("motion")
  robot = env.scene["robot"]
  k = int(float(meta["goal_num_future_frames"]))
  joints = [robot.joint_names.index(n) for n in _list(meta, "goal_joint_names")]
  for e in range(NUM_ENVS):
    last = int(cmd.library.clip_end[cmd.clip[e]]) - 1
    idx = torch.clamp(cmd.frame[e] + torch.arange(k, device=env.device), max=last)
    qpos, qvel = cmd.library.qpos[idx].cpu(), cmd.library.qvel[idx].cpu()
    root_quat = robot.data.root_link_quat_w[e].cpu()
    goal = reference_goal(qpos, qvel, root_quat, torch.tensor(joints))
    torch.testing.assert_close(goal, obs["goal"][e].cpu(), atol=1e-5, rtol=0)


def test_proprio_from_robot_state(rollout):
  """Newest slot of each proprio term from the robot state as described; history oldest first."""
  env, _, steps, obs, meta, _ = rollout
  robot = env.scene["robot"]
  names = _list(meta, "observation_names")
  dims = _list(meta, "observation_terms_dim", lambda v: int(float(v)))
  history = _list(meta, "observation_terms_history_length", lambda v: int(float(v)))
  joints = [robot.joint_names.index(n) for n in _list(meta, "observation_joint_names")]
  default = torch.tensor(_list(meta, "default_joint_pos", float), device=env.device)
  site = robot.find_sites(meta["imu_site"])[0][0]
  gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(NUM_ENVS, 3)
  expected = {
    "joint_pos": robot.data.joint_pos[:, joints] - default[joints],
    "joint_vel": robot.data.joint_vel[:, joints],
    "imu_projected_gravity": quat_rotate_inv(robot.data.site_quat_w[:, site], gravity),
    "actions": steps[-1][2],
  }
  start = len(names) - len(env.observation_manager.active_terms["proprio"])
  offsets = np.cumsum([0] + [d * h for d, h in zip(dims, history, strict=True)])[start:]
  prev = steps[-1][0]["proprio"]
  for i, name in enumerate(names[start:]):
    d, h = dims[start + i], history[start + i]
    lo = offsets[i] - offsets[0]
    term = obs["proprio"][:, lo : lo + d * h].reshape(NUM_ENVS, h, d)
    before = prev[:, lo : lo + d * h].reshape(NUM_ENVS, h, d)
    torch.testing.assert_close(
      term[:, :-1], before[:, 1:]
    )  # shifted by one step: oldest first
    if name in expected:
      torch.testing.assert_close(term[:, -1], expected[name], atol=1e-5, rtol=0)
