"""The task's scripts: replay configuration and its held failure, the benchmark and the
motion viewer's frames."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pal_mjlab.tasks  # noqa: F401  # Populates the task registry.
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
from pal_mjlab.tasks.supermimic import mdp
from pal_mjlab.tasks.supermimic.library import MotionLibrary
from pal_mjlab.tasks.supermimic.mdp import terminations
from pal_mjlab.tasks.supermimic.scripts import evaluate, play, view_motion

LIBRARY = Path(__file__).parent / "fixtures" / "library_tiny"
TASK = "Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers"


def test_held_failure_waits_for_its_duration(monkeypatch):
  """Ends an episode only after the failure lasted `seconds` without a break; reset per env."""
  env = SimpleNamespace(num_envs=2, device="cpu", step_dt=0.5)
  failed = iter(torch.tensor(v) for v in ([1, 0], [1, 1], [0, 1], [1, 1], [1, 1]))
  monkeypatch.setattr(
    terminations, "anchor_failure", lambda *a, **k: next(failed).bool()
  )
  term = terminations.anchor_failure_held(cfg=None, env=env)
  out = [term(env, "motion", 0.25, 0.8, 1.0).tolist() for _ in range(4)]
  assert out == [[False, False], [True, False], [False, True], [False, True]]
  term.reset(env_ids=torch.tensor([1]))
  assert term(env, "motion", 0.25, 0.8, 1.0).tolist() == [True, False]


def test_replay_sets_library_clips_and_held_failure():
  """The replay configuration, and its held failure in a real env (large random actions
  make the robot fall)."""
  registered = load_env_cfg(TASK, play=True)
  cfg = play.replay_env_cfg(
    TASK,
    play.ReplayConfig(
      checkpoint_file="model.pt", library_path=str(LIBRARY), clips=("squat",)
    ),
  )
  motion = cfg.commands["motion"]
  assert (motion.library_path, motion.clips) == (str(LIBRARY), ("squat",))
  failure = cfg.terminations["anchor_failure"]
  assert failure.func is mdp.anchor_failure_held
  assert failure.params == {
    **registered.terminations["anchor_failure"].params,
    "seconds": play.FALL_HOLD_S,
  }

  failure.params["seconds"] = 0.2
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  command = env.command_manager.get_term("motion")
  assert command.library.clip_names == ("squat",)
  gen = torch.Generator(device=env.device).manual_seed(1)
  fired = 0
  for _ in range(100):
    action = 3.0 * torch.randn(
      2, env.action_manager.total_action_dim, device=env.device, generator=gen
    )
    env.step(action)
    fired += int(env.termination_manager.get_term("anchor_failure").sum())
  env.close()
  assert fired > 0


def test_benchmark_plays_every_clip_once():
  benchmark = evaluate.Benchmark(TASK, str(LIBRARY), get_test_device())
  actions = benchmark.env.action_manager.total_action_dim
  per_clip = benchmark.run(
    lambda obs: torch.zeros(obs.batch_size[0], actions, device=benchmark.device)
  )
  report = benchmark.report(per_clip, Path("run/model_1.pt"))
  benchmark.env.close()
  assert report["clips"]["name"] == ["squat", "stand", "turn"]
  assert report["by_category"]["upper_body"]["clips"] == 2
  for name in ("rmpkpe_legs", "joint_err_arms", "torque_rate_waist", "saturation_legs"):
    assert np.isfinite(per_clip[name]).all() and (per_clip[name] >= 0).all(), name
  assert ((per_clip["tracking_return"] > 0) & (per_clip["tracking_return"] <= 1)).all()
  # A clip ends at its last frame or at the robot's failure.
  ended = (per_clip["steps"] == 99) | (per_clip["failed"] == 1)
  assert ended.all() and (per_clip["steps"] <= 99).all()


def test_viewer_frames_close_the_legs():
  """The library's frames as the scene model's qpos: the library's joints in their place and
  the passive knee linkage closed."""
  cfg = load_env_cfg(TASK, play=True)
  motion = cfg.commands["motion"]
  library = MotionLibrary(LIBRARY)
  model = Scene(cfg.scene, "cpu").compile()
  qpos = view_motion.model_qpos(model, library, motion)
  assert qpos.shape == (300, model.nq)
  np.testing.assert_allclose(
    qpos[:, model.joint("robot/arm_left_4_joint").qposadr[0]],
    library.qpos[:, 7 + library.joint_names.index("arm_left_4_joint")],
  )
  data = mujoco.MjData(model)
  for frame in qpos[::25]:
    data.qpos[:] = frame
    mujoco.mj_kinematics(model, data)
    for side in ("left", "right"):
      eq = model.equality(f"robot/leg_{side}_closed_kin_joint")
      a = data.xpos[eq.obj1id] + data.xmat[eq.obj1id].reshape(3, 3) @ eq.data[:3]
      b = data.xpos[eq.obj2id] + data.xmat[eq.obj2id].reshape(3, 3) @ eq.data[3:6]
      assert abs(a - b).max() < 1e-4
