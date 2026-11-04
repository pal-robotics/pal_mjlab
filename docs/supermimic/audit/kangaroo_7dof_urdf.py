"""Audit the 7-DoF arm of pal_mjlab's kangaroo_grippers.xml (and GMR's kangaroo_7dof.xml) against PAL's URDF.

The URDF chain torso_link -> arm_<side>_base_link -> arm_1..7 -> tool_link is written out from the xacros of
pal_sea_arm (humble-devel: urdf/arm/arm_properties/humanoid/{4dof,5dof,7dof}_arm.urdf.xacro, arm.urdf.xacro) and
kangaroo_robot (kangaroo.urdf.xacro: the arm mounts). Checks: joint ranges; world axes and anchors at the zero pose;
the pose of every arm link and of the tool link over library frames and uniform samples of the ranges.

  uv run python docs/supermimic/audit/kangaroo_7dof_urdf.py \
    src/pal_mjlab/robots/pal_kangaroo/xmls/kangaroo_grippers.xml <GMR>/assets/pal_kangaroo_7dof/kangaroo_7dof.xml \
    <7-DoF library folder> [<out.json>]

Results: kangaroo_7dof.md.
"""

import json
import sys

import mujoco
import numpy as np

PI = 3.14159  # the xacros' value
D2R = np.pi / 180


def rpy(r, p, y):
  cr, sr, cp, sp, cy, sy = (
    np.cos(r),
    np.sin(r),
    np.cos(p),
    np.sin(p),
    np.cos(y),
    np.sin(y),
  )
  rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
  ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
  rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
  return rz @ ry @ rx


def tf(xyz, r):
  t = np.eye(4)
  t[:3, :3], t[:3, 3] = r, xyz
  return t


def axis_angle(axis, q):
  a = np.asarray(axis, float)
  k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
  return np.eye(3) + np.sin(q) * k + (1 - np.cos(q)) * k @ k


def urdf_chain(side):
  """[(joint, origin 4x4, axis, (lower, upper))] from torso_link; the tool link last (fixed)."""
  mount = {
    "left": ((0, 0.105954, 0.384396), (1.3963, PI, PI)),
    "right": ((0, -0.105954, 0.384396), (1.3963, PI, 0)),
  }
  arm5 = {"left": (-210 * D2R, 90 * D2R), "right": (-90 * D2R, 210 * D2R)}[side]
  return [
    ("base", tf(mount[side][0], rpy(*mount[side][1])), None, None),
    ("1", tf((0, 0, 0), rpy(3.1416, 0, 3.1416)), (0, 0, -1), (-150 * D2R, 150 * D2R)),
    (
      "2",
      tf((0.03711, 0.02, -0.17011), rpy(0, 1.5708, 0)),
      (0, 0, 1),
      (-65 * D2R, 140 * D2R),
    ),
    (
      "3",
      tf((0.15989, -0.02, -0.03711), rpy(0, -1.5708, 0)),
      (0, 0, -1),
      (-150 * D2R, 150 * D2R),
    ),
    (
      "4",
      tf((-0.03711, -0.02, -0.17011), rpy(0, -1.5708, 0)),
      (0, 0, 1),
      (-65 * D2R, 140 * D2R),
    ),  # 5dof_arm
    (
      "5",
      tf((-0.17011, 0.02, -0.03711), rpy(0, -1.5708, 0)),
      (0, 0, 1),
      arm5,
    ),  # 7dof_arm
    (
      "6",
      tf((0.0422, 0, 0.1654), rpy(0, 1.5708, 0)),
      (0, 0, -1),
      (-108 * D2R, 172 * D2R),
    ),
    (
      "7",
      tf((-0.0534, -0.07, -0.0422), rpy(0, -1.5708, 0)),
      (0, 0, 1),
      (-150 * D2R, 150 * D2R),
    ),
    ("tool", tf((0, 0, 0.017), np.eye(3)), None, None),
  ]


def urdf_fk(side, q):
  """Link frames in torso_link for joint values q[1..7]; also the world axis and anchor of each joint."""
  t = np.eye(4)
  out = {}
  for name, origin, axis, _ in urdf_chain(side):
    t = t @ origin
    if axis is not None:
      out[f"axis{name}"] = t[:3, :3] @ np.asarray(axis, float)
      out[f"anchor{name}"] = t[:3, 3].copy()
      t = t @ tf((0, 0, 0), axis_angle(axis, q[int(name)]))
    out[name] = t.copy()
  return out


class Mj:
  def __init__(self, path):
    spec = mujoco.MjSpec.from_file(path)
    for e in list(
      spec.excludes
    ):  # excludes of bodies mjlab adds or renames (terrain, ...) are not kinematics
      if spec.body(e.bodyname1) is None or spec.body(e.bodyname2) is None:
        spec.delete(e)
    self.m = spec.compile()
    self.d = mujoco.MjData(self.m)
    self.torso = self.m.body(
      "pelvis_2_link"
    ).id  # torso_link is fixed at z 0.0061 in it (the torso mesh's pos)

  def qadr(self, name):
    return self.m.jnt_qposadr[self.m.joint(name).id]

  def fk(self, side, q):
    self.d.qpos[:] = self.m.qpos0
    for k in range(1, 8):
      self.d.qpos[self.qadr(f"arm_{side}_{k}_joint")] = q[k]
    mujoco.mj_kinematics(self.m, self.d)
    tt = tf(self.d.xpos[self.torso], self.d.xmat[self.torso].reshape(3, 3)) @ tf(
      (0, 0, 0.0061), np.eye(3)
    )
    inv = np.linalg.inv(tt)
    out = {}
    for k in range(1, 8):
      b = self.m.body(f"arm_{side}_{k}_link").id
      out[str(k)] = inv @ tf(self.d.xpos[b], self.d.xmat[b].reshape(3, 3))
      j = self.m.joint(f"arm_{side}_{k}_joint").id
      out[f"axis{k}"] = inv[:3, :3] @ self.d.xaxis[j]
      out[f"anchor{k}"] = (inv @ np.append(self.d.xanchor[j], 1))[:3]
    out["tool"] = out["7"] @ tf((0, 0, 0.017), np.eye(3))
    return out


def rot_deg(a, b):
  c = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
  return np.degrees(np.arccos(np.clip(c, -1, 1)))


def main():
  pal_xml, gmr_xml, lib = sys.argv[1:4]
  models = {"pal": Mj(pal_xml), "gmr": Mj(gmr_xml)}
  rng = np.random.default_rng(0)
  clips = json.load(open(f"{lib}/clips.json"))
  qpos = np.load(f"{lib}/qpos.npy", mmap_mode="r")
  names = clips["joint_names"]
  frames = np.sort(rng.choice(len(qpos), 3000, replace=False))
  report = {}

  for side in ("left", "right"):
    chain = {c[0]: c for c in urdf_chain(side)}
    print(f"\n== arm {side}")
    for mname, mj in models.items():
      print(f"-- {mname}: ranges (URDF vs XML, rad)")
      for k in range(1, 8):
        lo, hi = chain[str(k)][3]
        r = mj.m.jnt_range[mj.m.joint(f"arm_{side}_{k}_joint").id]
        flag = (
          "" if abs(r[0] - lo) < 1e-4 and abs(r[1] - hi) < 1e-4 else "  <-- MISMATCH"
        )
        print(
          f"   arm_{k}: URDF [{lo:+.5f}, {hi:+.5f}]  XML [{r[0]:+.5f}, {r[1]:+.5f}]{flag}"
        )
      zero = np.zeros(8)
      u, x = urdf_fk(side, zero), mj.fk(side, zero)
      print(f"-- {mname}: zero pose, world axis / anchor in torso_link")
      for k in range(1, 8):
        da = np.degrees(np.arccos(np.clip(u[f"axis{k}"] @ x[f"axis{k}"], -1, 1)))
        dp = np.linalg.norm(u[f"anchor{k}"] - x[f"anchor{k}"]) * 1000
        print(
          f"   arm_{k}: URDF axis {np.round(u[f'axis{k}'], 4)}  XML axis {np.round(x[f'axis{k}'], 4)}  "
          f"angle {da:.3f} deg, anchor {dp:.3f} mm"
        )
      # poses over library frames and uniform samples of the URDF ranges
      cols = [names.index(f"arm_{side}_{k}_joint") for k in range(1, 8)]
      lib_q = np.asarray(qpos[frames][:, 7:])[:, cols]
      lows = np.array([chain[str(k)][3][0] for k in range(1, 8)])
      highs = np.array([chain[str(k)][3][1] for k in range(1, 8)])
      uni_q = rng.uniform(lows, highs, size=(3000, 7))
      for label, qs in (("library", lib_q), ("uniform", uni_q)):
        errs = {k: [0.0, 0.0] for k in [*map(str, range(1, 8)), "tool"]}
        for q7 in qs:
          q = np.concatenate([[0.0], q7])
          u, x = urdf_fk(side, q), mj.fk(side, q)
          for k in errs:
            errs[k][0] = max(
              errs[k][0], np.linalg.norm(u[k][:3, 3] - x[k][:3, 3]) * 1000
            )
            errs[k][1] = max(errs[k][1], rot_deg(u[k], x[k]))
        print(
          f"-- {mname}: {label} ({len(qs)} poses), max link error vs URDF: "
          + ", ".join(f"{k} {e[0]:.3f} mm / {e[1]:.3f} deg" for k, e in errs.items())
        )
        report[f"{side}/{mname}/{label}"] = {
          k: [round(e[0], 4), round(e[1], 4)] for k, e in errs.items()
        }
      # the sign check: arm 5 / 7 with the opposite axis
      for k in (5, 7):
        q = np.zeros(8)
        q[k] = 0.5
        u, x = urdf_fk(side, q), mj.fk(side, q)
        print(
          f"   arm_{k} at +0.5 rad: tool {np.linalg.norm(u['tool'][:3, 3] - x['tool'][:3, 3]) * 1000:.3f} mm, "
          f"{rot_deg(u['tool'], x['tool']):.3f} deg from the URDF"
        )

  # all bodies the two XMLs share, full qpos of the library frames
  pal, gmr = models["pal"], models["gmr"]
  shared = [
    pal.m.body(i).name
    for i in range(1, pal.m.nbody)
    if mujoco.mj_name2id(gmr.m, mujoco.mjtObj.mjOBJ_BODY, pal.m.body(i).name) >= 0
  ]
  jp = {n: pal.qadr(n) for n in names}
  jg = {n: gmr.qadr(n) for n in names}
  worst = (0.0, 0.0, "")
  for f in frames:
    row = np.asarray(qpos[f])
    for mj, adr in ((pal, jp), (gmr, jg)):
      mj.d.qpos[:] = mj.m.qpos0
      mj.d.qpos[:7] = row[:7]
      for i, n in enumerate(names):
        mj.d.qpos[adr[n]] = row[7 + i]
      mujoco.mj_kinematics(mj.m, mj.d)
    for b in shared:
      bp, bg = pal.m.body(b).id, gmr.m.body(b).id
      dp = np.linalg.norm(pal.d.xpos[bp] - gmr.d.xpos[bg]) * 1000
      dr = rot_deg(
        tf(0, pal.d.xmat[bp].reshape(3, 3)), tf(0, gmr.d.xmat[bg].reshape(3, 3))
      )
      if dp > worst[0] or dr > worst[1]:
        worst = (max(dp, worst[0]), max(dr, worst[1]), b)
  print(
    f"\n== pal vs GMR: {len(shared)} shared bodies over {len(frames)} library frames: "
    f"max {worst[0]:.4f} mm, {worst[1]:.4f} deg (last worst body {worst[2]})"
  )
  report["pal_vs_gmr"] = {
    "bodies": len(shared),
    "max_mm": worst[0],
    "max_deg": worst[1],
  }
  json.dump(
    report, open(sys.argv[4] if len(sys.argv) > 4 else "/dev/null", "w"), indent=1
  )


if __name__ == "__main__":
  main()
