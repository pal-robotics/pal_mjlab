# Kangaroo 7-DoF model vs PAL's URDF (2026-11-02, D54)

**Question.** Do the training model (`kangaroo_grippers.xml`) and the retargeting model (GMR `bones-seed-kangaroo`,
`assets/pal_kangaroo_7dof/kangaroo_7dof.xml`) have the robot's arm kinematics? In particular, do they have the
axes of arm joints 5 and 7, which `main` has as `0 0 -1`? A sign flip turns the wrist the other way for the same
joint value, so the data, the policy and the robot would disagree.

**Reference.** The robot's URDF, from PAL's xacros:
- `pal_sea_arm` (humble-devel), `urdf/arm/arm_properties/humanoid/`:
  - `4dof_arm.urdf.xacro`: arms 1–3;
  - `5dof_arm.urdf.xacro`: arm 4, as attached for the 5- and 7-DoF arms;
  - `7dof_arm.urdf.xacro`: arms 5–7;
  - `arm.urdf.xacro`: how they attach, and the tool link at z 0.017 m on arm 7.
- `kangaroo_robot` (`kangaroo.urdf.xacro`): the arm mounts on `torso_link`.

The URDF says:

| joint | axis | range | origin in the parent link |
|---|---|---|---|
| arm 5 | `0 0 1` | left −210°..90°, right −90°..210° | xyz (−0.17011, 0.02, −0.03711), rpy (0, −π/2, 0) |
| arm 6 | `0 0 -1` | −108°..172° | xyz (0.0422, 0, 0.1654), rpy (0, π/2, 0) |
| arm 7 | `0 0 1` | −150°..150° | xyz (−0.0534, −0.07, −0.0422), rpy (0, −π/2, 0) |

**Method** (`kangaroo_7dof_urdf.py`; the 5-DoF audit's method, D51). The URDF chain from `torso_link` through
the arm mount, arms 1–7 and the tool link is written out from those xacros. It is compared, per side, with each
MuJoCo model:
- joint ranges;
- each joint's axis and anchor in `torso_link` at the zero pose;
- the pose of every arm link and of the tool link:
  - over 3,000 random frames of the 7-DoF `dev` library;
  - over 3,000 uniform samples of the URDF ranges.
- Last, the two MuJoCo models against each other: every shared body over the same 3,000 library frames, full
  qpos.

**Results** (both models, both sides):
- Ranges equal for arms 1–7.
- Zero-pose axes and anchors equal (0.000°, 0.000 mm).
- Every arm link and the tool link within **0.001 mm and 0.000°** of the URDF, on the library frames and on the
  uniform samples. The 0.001 mm is the xacros' π ≈ 3.14159 and π/2 ≈ 1.5708.
- `kangaroo_grippers.xml` and `kangaroo_7dof.xml` are identical on all 47 shared bodies (0.0000 mm, 0.0000°).
- **Negative control:** `main`'s `kangaroo_grippers.xml` (arms 5 and 7 about −z) fails:
  - axes 180° off;
  - arm 7 up to 176 mm and the tool up to 198 mm off;
  - at arm 5 = +0.5 rad, the tool is 67 mm and 57° off.

**Not compared:**
- Actuator values: the URDF gives 43 N·m for arms 1–7, while `kangaroo_constants.py` keeps PAL's identified values
  (50 N·m for arms 1–2, 25 N·m for 3–7, #119).
- Inertias: in the URDF, arm 7 is 0.17 kg; the MuJoCo models lump the gripper base into it, 0.76 kg.

**Guard.** `tests/test_kangaroo_arm_axes.py` checks the arm 5–7 axes of `kangaroo_grippers.xml`, and arm 5 and the
mirrored wrist roll of `kangaroo_hands.xml`, against the URDF, so a merge cannot flip them back.
