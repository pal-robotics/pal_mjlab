"""The Kangaroo's passive femur and knee joints as a function of the leg length."""

import mujoco
import numpy as np
import pytest
import torch
from mjlab.scene import Scene, SceneCfg
from pal_mjlab.robots import (
  get_kangaroo_grippers_robot_cfg,
  kangaroo_passive_leg_joints,
)

SIDES = ("left", "right")


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
  cfg = SceneCfg(num_envs=1, entities={"robot": get_kangaroo_grippers_robot_cfg()})
  return Scene(cfg, "cpu").compile()


def _lengths(model: mujoco.MjModel, n: int) -> torch.Tensor:
  lo, hi = model.joint("robot/leg_left_length_joint").range
  return torch.linspace(float(lo), float(hi), n, dtype=torch.float64)


def test_passive_joints_close_the_leg(model):
  """Over the whole slide range, both anchors of each leg's connect constraint coincide."""
  data = mujoco.MjData(model)
  lengths = _lengths(model, 50)
  zeros = torch.zeros_like(lengths)
  joints = {f"leg_{side}_length_joint": lengths for side in SIDES}
  passive = kangaroo_passive_leg_joints(joints, {name: zeros for name in joints})
  for i in range(len(lengths)):
    for name, value in joints.items():
      data.qpos[model.joint(f"robot/{name}").qposadr] = value[i].item()
    for name, (pos, _) in passive.items():
      joint = model.joint(f"robot/{name}")
      assert joint.range[0] - 1e-3 <= pos[i] <= joint.range[1] + 1e-3
      data.qpos[joint.qposadr] = pos[i].item()
    mujoco.mj_kinematics(model, data)
    for side in SIDES:
      eq = model.equality(f"robot/leg_{side}_closed_kin_joint")
      anchors = [
        data.xpos[body] + data.xmat[body].reshape(3, 3) @ anchor
        for body, anchor in ((eq.obj1id, eq.data[:3]), (eq.obj2id, eq.data[3:6]))
      ]
      assert np.linalg.norm(anchors[0] - anchors[1]) < 1e-4


def test_passive_joint_velocities(model):
  """Velocities are the time derivative of the positions."""
  lengths = _lengths(model, 20)[1:-1]
  rate, dt = torch.full_like(lengths, 0.3), 1e-6
  names = [f"leg_{side}_length_joint" for side in SIDES]

  def passive(length):
    return kangaroo_passive_leg_joints(
      {name: length for name in names}, {name: rate for name in names}
    )

  now, later = passive(lengths), passive(lengths + rate * dt)
  for name, (pos, vel) in now.items():
    assert torch.allclose((later[name][0] - pos) / dt, vel, rtol=1e-4)
