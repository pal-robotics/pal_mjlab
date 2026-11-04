"""The closed-chain Kangaroo with the 7-DoF arms: the arms are kangaroo_grippers', loops closed."""

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import KANGAROO_GRIPPERS_XML
from pal_mjlab.robots.pal_kangaroo_full.kangaroo_full_constants import (
  KANG_FULL_GRIPPERS_ARTICULATION,
  REGEX_FULL_ACTUATED_JOINTS,
  get_kangaroo_full_grippers_robot_cfg,
  get_kangaroo_full_grippers_spec,
)

SIDES = ("left", "right")
AUDIT = (
  Path(__file__).parents[2] / "docs" / "supermimic" / "audit" / "kangaroo_7dof_urdf.py"
)


@pytest.fixture(scope="module")
def full():
  return get_kangaroo_full_grippers_spec().compile()


@pytest.fixture(scope="module")
def grippers():
  spec = mujoco.MjSpec.from_file(str(KANGAROO_GRIPPERS_XML))
  for exclude in list(spec.excludes):
    spec.delete(exclude)
  return spec.compile()


def _arm_poses(model, data, side, q):
  """Arm links in the torso frame, for arm joint values q[0..6]."""
  data.qpos[:] = model.qpos0
  for k in range(7):
    data.qpos[model.joint(f"arm_{side}_{k + 1}_joint").qposadr[0]] = q[k]
  mujoco.mj_kinematics(model, data)
  torso = model.body("pelvis_2_link").id
  rot, pos = data.xmat[torso].reshape(3, 3), data.xpos[torso]
  out = {}
  for name in (*(f"arm_{side}_{k}_link" for k in range(1, 8)), f"arm_{side}_tip_link"):
    b = model.body(name).id
    out[name] = (rot.T @ (data.xpos[b] - pos), rot.T @ data.xmat[b].reshape(3, 3))
  return out


def test_size_and_mass(full, grippers):
  # 78 joints without the root: 14 arm, 2 waist, 62 of the legs (with the hip crosses locked).
  assert (full.nq, full.nv, full.neq) == (91, 90, 24)
  for side in SIDES:
    arm = f"arm_{side}_1_link"
    assert full.body_subtreemass[full.body(arm).id] == pytest.approx(
      grippers.body_subtreemass[grippers.body(arm).id]
    )


@pytest.mark.parametrize("side", SIDES)
def test_arm_joints_are_those_of_grippers(full, grippers, side):
  for k in range(1, 8):
    name = f"arm_{side}_{k}_joint"
    np.testing.assert_array_equal(full.joint(name).axis, grippers.joint(name).axis)
    np.testing.assert_array_equal(full.joint(name).range, grippers.joint(name).range)
  # D3, D54: the signs a mix-up has already cost GPU hours on.
  axes = [full.joint(f"arm_{side}_{k}_joint").axis[2] for k in range(1, 8)]
  assert axes == [-1, 1, -1, 1, 1, -1, 1]


@pytest.mark.parametrize("side", SIDES)
def test_arm_forward_kinematics_equals_grippers(full, grippers, side):
  df, dg = mujoco.MjData(full), mujoco.MjData(grippers)
  rng = np.random.default_rng(0)
  ranges = np.array(
    [grippers.joint(f"arm_{side}_{k}_joint").range for k in range(1, 8)]
  )
  for q in rng.uniform(ranges[:, 0], ranges[:, 1], size=(3000, 7)):
    a, b = _arm_poses(full, df, side, q), _arm_poses(grippers, dg, side, q)
    for name in a:
      assert np.linalg.norm(a[name][0] - b[name][0]) < 1e-6, name  # 0.001 mm
      c = (np.trace(a[name][1].T @ b[name][1]) - 1) / 2
      assert np.degrees(np.arccos(np.clip(c, -1, 1))) < 1e-3, name


@pytest.mark.parametrize("side", SIDES)
def test_arm_matches_the_urdf_audit(full, side):
  """The chain of PAL's xacros, as in docs/supermimic/audit/kangaroo_7dof_urdf.py (D54)."""
  module_spec = importlib.util.spec_from_file_location("audit", AUDIT)
  audit = importlib.util.module_from_spec(module_spec)
  module_spec.loader.exec_module(audit)
  mj = audit.Mj.__new__(audit.Mj)
  mj.m, mj.d = full, mujoco.MjData(full)
  mj.torso = full.body("pelvis_2_link").id
  chain = {c[0]: c for c in audit.urdf_chain(side)}
  lows = np.array([chain[str(k)][3][0] for k in range(1, 8)])
  highs = np.array([chain[str(k)][3][1] for k in range(1, 8)])
  rng = np.random.default_rng(0)
  for q7 in rng.uniform(lows, highs, size=(300, 7)):
    q = np.concatenate([[0.0], q7])
    u, x = audit.urdf_fk(side, q), mj.fk(side, q)
    for k in [*map(str, range(1, 8)), "tool"]:
      assert np.linalg.norm(u[k][:3, 3] - x[k][:3, 3]) < 1e-5, k
      assert audit.rot_deg(u[k], x[k]) < 1e-2, k


def test_gripper_contacts_are_excluded():
  spec = get_kangaroo_full_grippers_spec()
  excludes = {(e.bodyname1, e.bodyname2) for e in spec.excludes}
  for side in SIDES:
    assert (f"arm_{side}_4_link", f"arm_{side}_7_link") in excludes
    for tip in SIDES:
      assert (f"arm_{side}_7_link", f"gripper_{side}_fingertip_{tip}_link") in excludes


def test_robot_config_covers_the_actuated_joints(full):
  import re

  names = [full.joint(j).name for j in range(1, full.njnt)]
  actuated = [n for n in names if re.match(REGEX_FULL_ACTUATED_JOINTS, n)]
  assert len(actuated) == 28
  driven = [
    n
    for n in names
    if any(
      re.match(expr, n)
      for a in KANG_FULL_GRIPPERS_ARTICULATION.actuators
      for expr in a.target_names_expr
    )
  ]
  assert sorted(driven) == sorted(actuated)
  cfg = get_kangaroo_full_grippers_robot_cfg()
  assert cfg.articulation is KANG_FULL_GRIPPERS_ARTICULATION


def test_waist_and_arms_have_the_serial_robots_dynamics():
  by_joint = {}
  for a in KANG_FULL_GRIPPERS_ARTICULATION.actuators:
    for expr in a.target_names_expr:
      by_joint[expr] = a
  shoulder = by_joint["arm_.*_1_joint"]
  assert shoulder.armature == pytest.approx(121**2 * 3.3e-05)
  assert by_joint["pelvis_1_joint"] is shoulder
  # The legs keep the 5 Hz gains from the generated reflected inertia.
  screw = by_joint[r"leg_(left|right)_2_actuator$"]
  assert screw.stiffness == pytest.approx(72.09 * (2 * 3.1415926535 * 5) ** 2, abs=1e-2)
