"""Motion library of the SuperMimic tests: three synthetic 2 s clips for the Kangaroo with
grippers, in ``library_tiny/`` (the format of ``pal_mjlab/tasks/supermimic/library.py``).

  uv run python tests/supermimic/fixtures/make_fixtures.py

  squat  both legs shorten and extend on their slides
  stand  the default pose with the arms and the waist swinging
  turn   the same away from the origin, the root turning about the vertical

The root height keeps the lowest foot on the ground. The motions are kinematic: they are
smooth and within the joint ranges, not dynamically consistent.
"""

import json
import re
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.scene import Scene
from pal_mjlab.robots import REGEX_ALL_ACTUATED_JOINTS, kangaroo_passive_leg_joints
from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import (
  FOOT_GEOMS,
  pal_kangaroo_grippers_supermimic_env_cfg,
)

FPS, FRAMES = 50, 100
OUT = Path(__file__).parent / "library_tiny"


def _foot_height(model: mujoco.MjModel, data: mujoco.MjData) -> float:
  """Height of the lowest foot collision geom above the floor."""
  (floor,) = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
  mujoco.mj_kinematics(model, data)
  return min(
    mujoco.mj_geomDistance(model, data, floor, model.geom(f"robot/{g}").id, 10.0, None)
    for g in FOOT_GEOMS
  )


def main() -> None:
  cfg = pal_kangaroo_grippers_supermimic_env_cfg()
  model = Scene(cfg.scene, "cpu").compile()
  data = mujoco.MjData(model)
  robot_joints = [
    model.joint(j).name.removeprefix("robot/") for j in range(1, model.njnt)
  ]
  joint_names = [n for n in robot_joints if re.fullmatch(REGEX_ALL_ACTUATED_JOINTS, n)]
  column = {name: i for i, name in enumerate(joint_names)}
  init = cfg.scene.entities["robot"].init_state.joint_pos
  default = np.array(
    [
      next((v for pattern, v in init.items() if re.fullmatch(pattern, name)), 0.0)
      for name in joint_names
    ]
  )

  t = np.arange(FRAMES) / FPS
  phase = 2.0 * np.pi * t / t[-1]
  swing, bump = np.sin(phase), 0.5 * (1.0 - np.cos(phase))  # both start and end at 0

  def clip(joints: dict[str, np.ndarray], xy=(0.0, 0.0), yaw=0.0) -> np.ndarray:
    """(FRAMES, 7 + nj) qpos: the default pose plus per-joint offsets, on the ground."""
    qpos = np.zeros((FRAMES, 7 + len(joint_names)))
    qpos[:, :2] = xy
    qpos[:, 3], qpos[:, 6] = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    qpos[:, 7:] = default
    for pattern, offset in joints.items():
      for name in joint_names:
        if re.fullmatch(pattern, name):
          qpos[:, 7 + column[name]] += offset
    positions = {n: torch.from_numpy(qpos[:, 7 + column[n]]) for n in joint_names}
    rates = {n: torch.zeros(FRAMES) for n in joint_names}
    passive = kangaroo_passive_leg_joints(positions, rates)
    for frame in range(FRAMES):
      data.qpos[:7] = qpos[frame, :7]
      for name in joint_names:
        data.qpos[model.joint(f"robot/{name}").qposadr] = qpos[frame, 7 + column[name]]
      for name, (position, _) in passive.items():
        data.qpos[model.joint(f"robot/{name}").qposadr] = position[frame].item()
      qpos[frame, 2] = -_foot_height(model, data)
    return qpos

  clips = {
    "squat": clip({"leg_.*_length_joint": -0.1 * bump}),
    "stand": clip(
      {
        "arm_left_1_joint": 0.3 * swing,
        "arm_right_1_joint": 0.3 * swing,
        "arm_.*_4_joint": 0.3 * bump,
        "pelvis_1_joint": 0.2 * swing,
      }
    ),
    "turn": clip(
      {"arm_.*_2_joint": -0.2 * bump, "pelvis_2_joint": 0.1 * swing},
      xy=(0.4, -0.3),
      yaw=0.7 + 0.3 * bump,
    ),
  }
  categories = {"squat": "legs", "stand": "upper_body", "turn": "upper_body"}

  qpos = np.concatenate(list(clips.values()))
  qvel = np.concatenate(
    [
      np.concatenate(
        [
          np.gradient(q[:, :3], 1.0 / FPS, axis=0),
          np.zeros((FRAMES, 2)),  # the root only yaws, so its angular velocity is the
          np.gradient(2.0 * np.arctan2(q[:, 6], q[:, 3]), 1.0 / FPS)[
            :, None
          ],  # yaw rate
          np.gradient(q[:, 7:], 1.0 / FPS, axis=0),
        ],
        axis=1,
      )
      for q in clips.values()
    ]
  )
  OUT.mkdir(exist_ok=True)
  np.save(OUT / "qpos.npy", qpos.astype(np.float32))
  np.save(OUT / "qvel.npy", qvel.astype(np.float32))
  meta = {
    "fps": FPS,
    "joint_names": joint_names,
    "clips": [
      {
        "name": name,
        "category": categories[name],
        "start": i * FRAMES,
        "length": FRAMES,
      }
      for i, name in enumerate(clips)
    ],
  }
  (OUT / "clips.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
  main()
