"""Motion-library converter: MuJoCo qvel layout, subsets, and loading a library or chosen
clips of it."""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch
from pal_mjlab.scripts.bones_to_library import (
  FPS,
  clip_qpos_qvel,
  load_manifest,
  select_clips,
)
from pal_mjlab.scripts.supermimic.view_motion import robot_model
from pal_mjlab.tasks.supermimic.library import MotionLibrary

FIXTURES = Path(__file__).parent / "fixtures"
LIBRARY = FIXTURES / "library_tiny"


@pytest.mark.parametrize("seed", range(5))
def test_constant_motion_is_recovered_exactly(seed):
  """Constant body-frame angular velocity, linear root motion and linear joint ramps
  are differentiated exactly, including the one-sided first and last frames."""
  rng = np.random.default_rng(seed)
  fps, n, nj = 50.0, 40, 32
  t = np.arange(n) / fps
  omega_b = rng.normal(0, 2.0, 3)
  v_w = rng.normal(0, 1.0, 3)
  joint_rate = rng.normal(0, 1.0, nj)

  q0 = rng.normal(size=4)
  q0 /= np.linalg.norm(q0)
  quat = np.zeros((n, 4))
  for k in range(n):
    # q(t) = q0 * exp(omega_b t): rotation at constant body-frame rate.
    dq = np.zeros(4)
    mujoco.mju_axisAngle2Quat(
      dq, omega_b / np.linalg.norm(omega_b), np.linalg.norm(omega_b) * t[k]
    )
    mujoco.mju_mulQuat(quat[k], q0, dq)
  quat[1::2] *= -1.0  # sign flips must not matter
  root_pos = rng.normal(size=3) + v_w * t[:, None]
  dof = rng.normal(size=nj) + joint_rate * t[:, None]

  qpos, qvel = clip_qpos_qvel(root_pos, quat, dof, fps)
  assert qpos.shape == (n, 7 + nj) and qvel.shape == (n, 6 + nj)
  np.testing.assert_allclose(qvel[:, :3], np.broadcast_to(v_w, (n, 3)), atol=1e-9)
  np.testing.assert_allclose(qvel[:, 3:6], np.broadcast_to(omega_b, (n, 3)), atol=1e-9)
  np.testing.assert_allclose(
    qvel[:, 6:], np.broadcast_to(joint_rate, (n, nj)), atol=1e-9
  )
  # Stored quaternions are unit and hemisphere-continuous.
  np.testing.assert_allclose(np.linalg.norm(qpos[:, 3:7], axis=1), 1.0, atol=1e-12)
  assert np.all(np.sum(qpos[1:, 3:7] * qpos[:-1, 3:7], axis=1) > 0)


def _manifest(seed: int = 0) -> dict[str, list]:
  rng = np.random.default_rng(seed)
  n = 2000  # about 12 h
  return {
    "clip": [f"clip_{i}" for i in range(n)],
    "category": [f"cat_{i % 7}" for i in range(n)],
    "move_name": ["walk"] * n,
    "g1_n_frames": rng.integers(100, 2000, n).tolist(),
  }


def test_dev_is_2_5_hours_of_every_category():
  m = _manifest()
  hours = np.asarray(m["g1_n_frames"]) / FPS / 3600
  dev = select_clips(m, "dev")
  assert dev == select_clips(m, "dev")  # deterministic
  assert 2.5 <= hours[dev].sum() < 3.0
  assert len({m["category"][i] for i in dev}) == 7


def test_a_missing_library_names_the_flag(tmp_path):
  with pytest.raises(FileNotFoundError, match="library-path"):
    MotionLibrary(tmp_path / "nowhere")


def test_reviewed_clips_leave_subsets_without_reshuffling(tmp_path):
  """A clip removed by review stays in the selection pool (flagged), so subsets keep the
  same other clips."""
  import pyarrow as pa
  import pyarrow.parquet as pq

  n = 6
  table = {
    "clip": [f"c{i}" for i in range(n)],
    "category": ["a"] * n,
    "move_name": ["walk"] * n,
    "g1_n_frames": [100] * n,
    "kangaroo_path": [f"motions/c{i}.pkl" for i in range(n)],
    "final_kept": [True, True, False, True, False, True],
    "review_excluded": [False] * n,
  }
  pq.write_table(pa.table(table), tmp_path / "manifest.parquet")
  old = load_manifest(tmp_path)
  assert old["clip"] == ["c0", "c1", "c3", "c5"] and not any(old["excluded"])
  # c4 failed the automatic filters; c3 passed them and was removed by review.
  table["final_kept"][3] = False
  table["review_excluded"] = [False, False, False, True, False, False]
  pq.write_table(pa.table(table), tmp_path / "manifest.parquet")
  new = load_manifest(tmp_path)
  assert new["clip"] == old["clip"]
  assert new["excluded"] == [False, False, True, False]
  assert select_clips(new, "full") == select_clips(old, "full")


def test_chosen_clips_of_a_library():
  """``clips`` loads only those clips, in the order asked, with their own frames."""
  full = MotionLibrary(LIBRARY)
  names = full.clip_names
  lib = MotionLibrary(LIBRARY, clips=(names[2], names[0]))
  assert lib.clip_names == (names[2], names[0])
  assert lib.clip_start.tolist() == [0, 100] and lib.clip_length.tolist() == [100, 100]
  assert lib.categories == ("Basic Locomotion Neutral", "Stunts")
  torch.testing.assert_close(lib.qpos, torch.cat([full.qpos[200:], full.qpos[:100]]))
  torch.testing.assert_close(lib.qvel, torch.cat([full.qvel[200:], full.qvel[:100]]))
  with pytest.raises(KeyError):
    MotionLibrary(LIBRARY, clips=("221011/not_a_clip",))


@pytest.mark.parametrize(
  "library, robot_joints", [("library_tiny", 32), ("library_tiny_hands", 32)]
)
def test_view_motion_picks_the_robot_of_the_library(library, robot_joints):
  lib = MotionLibrary(FIXTURES / library)
  model = robot_model(lib.joint_names)
  assert model.nq == 7 + robot_joints == lib.qpos.shape[1]
