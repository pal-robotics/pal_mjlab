"""Motion-library converter: MuJoCo qvel layout, subsets, and library path resolution."""

import mujoco
import numpy as np
import pytest
from pal_mjlab.scripts.bones_to_library import (
  FPS,
  clip_qpos_qvel,
  load_manifest,
  select_clips,
  subset_name,
)
from pal_mjlab.tasks.supermimic.library import resolve_library


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


@pytest.mark.parametrize(
  "subset, name",
  [("dev", "dev"), ("full", "full"), ("10", "10h"), ("10h", "10h"), ("0.5h", "0.5h")],
)
def test_subset_names(subset, name):
  assert subset_name(subset) == name


@pytest.mark.parametrize("subset", ["", "abc", "-1", "0h"])
def test_bad_subsets_raise(subset):
  with pytest.raises(ValueError):
    subset_name(subset)


def test_hour_subsets_are_nested_and_dev_is_2_5_hours():
  m = _manifest()
  hours = np.asarray(m["g1_n_frames"]) / FPS / 3600
  small, large = (set(select_clips(m, s)) for s in ("1h", "3h"))
  assert small < large
  assert hours[sorted(small)].sum() >= 1.0 and hours[sorted(large)].sum() >= 3.0
  assert select_clips(m, "dev") == select_clips(m, "2.5h")
  assert len({m["category"][i] for i in small}) == 7  # every category represented


def test_library_paths_resolve(tmp_path, monkeypatch):
  packed = tmp_path / "data" / "libraries" / "4h"
  packed.mkdir(parents=True)
  (packed / "clips.json").write_text("{}")
  monkeypatch.chdir(tmp_path)
  assert resolve_library("", "data", "4") == packed  # relative dataset, hour subset
  assert resolve_library("data/libraries/4h", "elsewhere", "dev") == packed
  with pytest.raises(ValueError):
    resolve_library("", "", "dev")


def test_reviewed_clips_leave_subsets_without_reshuffling(tmp_path):
  """A clip removed by review stays in the selection pool (flagged), so subsets keep the
  same other clips; manifests without the review column still load."""
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
