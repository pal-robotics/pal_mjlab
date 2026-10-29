"""Real-data fixtures of the SuperMimic tests: 3 BONES-SEED clips (idle turn, dance, stunt).

  library_tiny<suffix>/      the first 100 frames (2 s) of each clip, packed as bones_to_library
                             does (test_env, test_heads, test_supermimic, test_world_model)
  bones_3clips<suffix>.npz   every 4th frame (12.5 Hz) of each clip with GMR's position of every
                             body both models have (test_kinematics: the clips load unchanged)

Suffix "" for grippers (7-DoF arms, from kangaroo_bones_seed/dataset), "_hands" for hands
(5-DoF arms, from dataset_5dof). GMR's model is the one the variant was retargeted with.

  python tests/supermimic/fixtures/make_fixtures.py --variant hands \
    --motions ~/Projects/supermimic/kangaroo_bones_seed/dataset_5dof/motions \
    --gmr-xml <GMR bones-seed-kangaroo>/assets/pal_kangaroo_hands/kangaroo_hands.xml
"""

import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
import pal_mjlab.tasks  # noqa: F401  (before pal_mjlab.robots, which it imports: a cycle otherwise)
import tyro
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  get_kangaroo_grippers_spec,
  get_kangaroo_hands_spec,
)
from pal_mjlab.scripts.bones_to_library import clip_qpos_qvel
from pal_mjlab.tasks.supermimic.kinematics import compile_kinematic_model

CLIPS = {
  "221011/idle_turn_270_002__A051": "Basic Locomotion Neutral",
  "230414/dance_retro_jackson_jump_on_R_loop_002__A323": "Dancing",
  "231009/burning_start_R_001__A468": "Stunts",
}
VARIANTS = {
  "grippers": ("", get_kangaroo_grippers_spec),
  "hands": ("_hands", get_kangaroo_hands_spec),
}
FPS, LIBRARY_FRAMES, STRIDE = 50, 100, 4


def main(variant: str, motions: Path, gmr_xml: Path) -> None:
  suffix, spec_fn = VARIANTS[variant]
  out = Path(__file__).parent
  clips = {c: pickle.load(open(motions.expanduser() / f"{c}.pkl", "rb")) for c in CLIPS}
  joint_names = list(next(iter(clips.values()))["joint_names"])

  # library
  lib = out / f"library_tiny{suffix}"
  lib.mkdir(exist_ok=True)
  qpos, qvel = zip(
    *(
      clip_qpos_qvel(
        np.asarray(m["root_pos"][:LIBRARY_FRAMES], np.float64),
        np.asarray(m["root_rot"][:LIBRARY_FRAMES], np.float64)[:, [3, 0, 1, 2]],
        np.asarray(m["dof_pos"][:LIBRARY_FRAMES], np.float64),
        FPS,
      )
      for m in clips.values()
    ),
    strict=True,
  )
  np.save(lib / "qpos.npy", np.concatenate(qpos).astype(np.float32))
  np.save(lib / "qvel.npy", np.concatenate(qvel).astype(np.float32))
  meta = {
    "fps": FPS,
    "joint_names": joint_names,
    "subset": "test-fixture",
    "clips": [
      {
        "name": c,
        "category": cat,
        "start": k * LIBRARY_FRAMES,
        "length": LIBRARY_FRAMES,
      }
      for k, (c, cat) in enumerate(CLIPS.items())
    ],
  }
  (lib / "clips.json").write_text(json.dumps(meta, indent=1))

  # GMR body positions
  gmr = mujoco.MjModel.from_xml_path(str(gmr_xml.expanduser()))
  pal = compile_kinematic_model(spec_fn())
  names = lambda m: [m.body(i).name for i in range(1, m.nbody)]  # noqa: E731
  bodies = [b for b in names(pal) if b in names(gmr)]
  data = mujoco.MjData(gmr)
  sub = {k: [] for k in ("root_pos", "root_rot_xyzw", "dof_pos", "gmr_body_pos")}
  for m in clips.values():
    for t in range(0, len(m["dof_pos"]), STRIDE):
      data.qpos[:] = np.concatenate(
        [m["root_pos"][t], m["root_rot"][t][[3, 0, 1, 2]], m["dof_pos"][t]]
      )
      mujoco.mj_kinematics(gmr, data)
      sub["root_pos"].append(m["root_pos"][t])
      sub["root_rot_xyzw"].append(m["root_rot"][t])
      sub["dof_pos"].append(m["dof_pos"][t])
      sub["gmr_body_pos"].append(data.xpos[[gmr.body(b).id for b in bodies]].copy())
  np.savez_compressed(
    out / f"bones_3clips{suffix}.npz",
    clip_names=list(CLIPS),
    lengths=[len(range(0, len(m["dof_pos"]), STRIDE)) for m in clips.values()],
    fps_subsampled=FPS / STRIDE,
    joint_names=joint_names,
    body_names=bodies,
    **{k: np.asarray(v) for k, v in sub.items()},
  )
  print(f"{variant}: {lib.name}/, bones_3clips{suffix}.npz ({len(bodies)} bodies)")


if __name__ == "__main__":
  tyro.cli(main)
